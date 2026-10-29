from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.corpora import build_kld_corpora as builder


def candidate(repo: str, path: str, content: bytes, *, test: bool = False) -> builder.Candidate:
    return builder.Candidate(
        repo=repo,
        sha=("a" if repo == "one/repo" else "b") * 40,
        path=path,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        is_test=test,
    )


class KldCorpusBuilderTest(unittest.TestCase):
    def test_normalisation_and_rejection_rules(self) -> None:
        self.assertEqual(builder._normalise_content(b"a\r\nb\rc\n"), b"a\nb\nc\n")
        self.assertIsNone(builder._normalise_content(b"prefix\0suffix"))
        self.assertIsNone(builder._normalise_content(b"\xff"))
        self.assertTrue(builder._is_excluded("src/generated/table.rs", "rust"))
        self.assertTrue(builder._is_excluded("src/types.d.ts", "js-ts"))
        self.assertTrue(builder._is_excluded("src/code.min.js", "js-ts"))
        self.assertFalse(builder._is_excluded("tests/client.test.ts", "js-ts"))

    def test_exact_dedup_keeps_lexicographically_first_path(self) -> None:
        content = b" ".join([b"token"] * 30)
        rows = builder.deduplicate(
            [
                candidate("two/repo", "z.rs", content),
                candidate("one/repo", "a.rs", content),
            ]
        )
        self.assertEqual([(row.repo, row.path) for row in rows], [("one/repo", "a.rs")])

    def test_minhash_drops_near_duplicate(self) -> None:
        original = " ".join(f"token{index}" for index in range(100)).encode()
        near_copy = original.replace(b"token50", b"changed50")
        rows = builder.deduplicate(
            [
                candidate("two/repo", "z.rs", near_copy),
                candidate("one/repo", "a.rs", original),
            ]
        )
        self.assertEqual([row.path for row in rows], ["a.rs"])

    def test_linguist_attributes_are_honoured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(
                ["git", "init", "--quiet"], cwd=root, check=True, timeout=10
            )
            (root / ".gitattributes").write_text(
                "generated.rs linguist-generated\n"
                "vendored.rs linguist-vendored=true\n"
                "hidden.rs linguist-detectable=false\n",
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "add", ".gitattributes"], cwd=root, check=True, timeout=10
            )
            source = builder.Source("fixture/repo", "a" * 40, root)
            self.assertEqual(
                builder._attribute_exclusions(
                    source,
                    ["generated.rs", "vendored.rs", "hidden.rs", "plain.rs"],
                    10,
                ),
                {"generated.rs", "vendored.rs", "hidden.rs"},
            )

    def test_round_robin_serialization_offsets_and_test_cap(self) -> None:
        groups = [
            [candidate("one/repo", "a.rs", b"aaa"), candidate("one/repo", "c.rs", b"ccc")],
            [
                candidate("two/repo", "b.rs", b"bbb"),
                candidate("two/repo", "d.rs", b"d" * 20, test=True),
            ],
        ]
        artifact = builder.serialize_candidates(builder.round_robin(groups), 100)
        self.assertEqual(artifact.data, b"aaa\n\nbbb\n\nccc\n")
        rows = [json.loads(line) for line in artifact.manifest.splitlines()]
        self.assertEqual([row["path"] for row in rows], ["a.rs", "b.rs", "c.rs"])
        for row in rows:
            span = artifact.data[row["byte_start"] : row["byte_end"]]
            self.assertEqual(hashlib.sha256(span).hexdigest(), row["sha256"])

    def test_record_uses_only_supplied_timestamp(self) -> None:
        sources = json.loads((ROOT / "corpora" / "sources.json").read_text())
        artifact = builder.CorpusArtifact(b"x\n", b"{}\n", 1)
        record = builder.build_record(
            sources,
            {"general": artifact, "rust": artifact, "js-ts": artifact},
            "literal-time",
        )
        self.assertEqual(record["timestamp"], "literal-time")
        self.assertEqual(record["corpora"]["rust"]["byte_count"], 2)

    def test_verify_detects_manifest_mismatch(self) -> None:
        artifact = builder.CorpusArtifact(b"x\n", b"{}\n", 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "BUILD.json").write_text(
                json.dumps(
                    {
                        "corpora": {
                            name: {
                                "sha256": hashlib.sha256(artifact.data).hexdigest(),
                                "byte_count": 2,
                                "file_count": 1,
                            }
                            for name in ("general", "rust", "js-ts")
                        }
                    }
                )
            )
            for name in ("general", "rust", "js-ts"):
                (root / f"{name}.manifest.jsonl").write_bytes(artifact.manifest)
            self.assertEqual(
                builder._verify(root, {name: artifact for name in ("general", "rust", "js-ts")}),
                [],
            )
            (root / "rust.manifest.jsonl").write_text("corrupt\n")
            self.assertIn("rust: rebuilt manifest differs", builder._verify(root, {"rust": artifact}))


if __name__ == "__main__":
    unittest.main()
