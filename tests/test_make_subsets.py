from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.corpora import make_subsets


class FrozenSubsetsTest(unittest.TestCase):
    def test_codegen_is_ranked_per_language_and_runner_compatible(self) -> None:
        tasks = json.loads((ROOT / "subsets" / "polyglot.json").read_text())
        self.assertEqual(len(tasks), 78)
        self.assertEqual(Counter(task["language"] for task in tasks), {"rust": 30, "javascript": 48})
        for language in ("rust", "javascript"):
            selected = [task for task in tasks if task["language"] == language]
            self.assertEqual([task["rank"] for task in selected], sorted(task["rank"] for task in selected))
            for task in selected:
                self.assertEqual(
                    task["rank"], make_subsets._rank("codegen", task["task_id"])
                )
                self.assertTrue(task["task_id"].startswith(language + "/"))
                self.assertIn("solution_path", task)
                self.assertIn("example_path", task)
        self.assertTrue(
            all(task["test_cmd"] == ["cargo", "test", "--", "--include-ignored"] for task in tasks[:20])
        )

    def test_diffxyz_has_two_targets_and_caps_each_repo(self) -> None:
        rows = [json.loads(line) for line in (ROOT / "subsets" / "diffxyz.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 50)
        self.assertEqual(Counter(row["language"] for row in rows), {"rust": 25, "javascript": 25})
        for language in ("rust", "javascript"):
            language_rows = [row for row in rows if row["language"] == language]
            self.assertLessEqual(
                max(Counter(row["repo"].casefold() for row in language_rows).values()), 3
            )
        for row in rows:
            self.assertIn("udiff", row)
            self.assertIn("search-replace", row)
            self.assertEqual(row["rank"], make_subsets._rank("diffxyz", row["task_id"]))
            self.assertTrue(200 <= len(row["old_code"]) <= 12_000)

    def test_agentic_ids_and_images_are_exact(self) -> None:
        rows = [json.loads(line) for line in (ROOT / "subsets" / "swebench.jsonl").read_text().splitlines()]
        self.assertEqual([row["instance_id"] for row in rows], make_subsets.EXPECTED_AGENTIC_IDS)
        for row in rows:
            expected = (
                "swebench/sweb.eval.x86_64."
                + row["instance_id"].replace("__", "_1776_")
                + ":latest"
            )
            self.assertEqual(row["image"], expected)
            self.assertTrue(0.25 <= row["solve_rate"] <= 0.80)

    def test_build_hashes_cover_frozen_outputs(self) -> None:
        build = json.loads((ROOT / "subsets" / "BUILD.json").read_text())
        for name, recorded in build["outputs"].items():
            data = (ROOT / "subsets" / name).read_bytes()
            self.assertEqual(recorded["byte_count"], len(data))
            self.assertEqual(recorded["sha256"], hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    unittest.main()
