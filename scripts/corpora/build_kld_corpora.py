#!/usr/bin/env python3
"""Build reproducible general, Rust, and JavaScript/TypeScript KLD corpora."""

from __future__ import annotations

import argparse
from collections import defaultdict
import dataclasses
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

from scripts import ROOT
import tempfile
from typing import Any, Iterable, Iterator
import urllib.request
import zipfile


BUILDER_VERSION = "1"
DEFAULT_TARGET_BYTES = 1_290_590
MIN_BYTES = 512
MAX_BYTES = 262_144
MIN_LINES = 20
MAX_MEAN_LINE_BYTES = 200
MINHASH_PERMUTATIONS = 128
MINHASH_THRESHOLD = 0.80
_U64_MASK = (1 << 64) - 1
_TOKEN_RE = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[^\s]")
_EXCLUDED_DIRS = {
    "target",
    "dist",
    "build",
    "out",
    ".next",
    "coverage",
    "node_modules",
    "vendor",
    "third_party",
    "generated",
    "gen",
    "__snapshots__",
    "fixtures",
    "__fixtures__",
    "testdata",
}
_JS_EXTENSIONS = {".ts", ".tsx", ".js", ".mjs", ".cjs"}


@dataclasses.dataclass(frozen=True, slots=True)
class Source:
    repo: str
    sha: str
    root: Path


@dataclasses.dataclass(frozen=True, slots=True)
class Candidate:
    repo: str
    sha: str
    path: str
    content: bytes
    sha256: str
    is_test: bool


@dataclasses.dataclass(frozen=True, slots=True)
class CorpusArtifact:
    data: bytes
    manifest: bytes
    file_count: int


def _run(
    command: list[str],
    *,
    cwd: Path | None,
    timeout: float,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        cwd=cwd,
        input=input_bytes,
        capture_output=True,
        timeout=timeout,
        check=True,
    )


def _safe_cache_name(repo: str, sha: str) -> str:
    return repo.replace("/", "--") + "-" + sha


def _clone_exact(repo: str, sha: str, cache_dir: Path, timeout: float) -> Source:
    root = cache_dir / _safe_cache_name(repo, sha)
    if root.exists():
        head = _run(["git", "rev-parse", "HEAD"], cwd=root, timeout=timeout).stdout.strip()
        if head.decode("ascii") != sha:
            raise ValueError(f"cached checkout has wrong revision: {root}")
        return Source(repo, sha, root)

    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kld-clone-", dir=cache_dir) as temp_dir:
        staged = Path(temp_dir) / "repo"
        staged.mkdir()
        _run(["git", "init", "--quiet"], cwd=staged, timeout=timeout)
        _run(
            ["git", "remote", "add", "origin", f"https://github.com/{repo}.git"],
            cwd=staged,
            timeout=timeout,
        )
        _run(["git", "fetch", "--depth=1", "origin", sha], cwd=staged, timeout=timeout)
        _run(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=staged, timeout=timeout)
        os.replace(staged, root)
    return Source(repo, sha, root)


def _download_general(config: dict[str, Any], cache_dir: Path, timeout: float) -> bytes:
    expected_archive = str(config["archive_sha256"])
    archive = cache_dir / f"wikitext-2-raw-v1-{expected_archive}.zip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        request = urllib.request.Request(str(config["url"]), headers={"User-Agent": "kld-corpus-builder/1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected_archive:
            raise ValueError("downloaded Wikitext archive SHA-256 mismatch")
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, archive)
    archive_data = archive.read_bytes()
    if hashlib.sha256(archive_data).hexdigest() != expected_archive:
        raise ValueError("cached Wikitext archive SHA-256 mismatch")
    with zipfile.ZipFile(archive) as bundle:
        data = bundle.read(str(config["member"]))
    if hashlib.sha256(data).hexdigest() != str(config["file_sha256"]):
        raise ValueError("wiki.test.raw SHA-256 mismatch")
    return data


def _is_excluded(path: str, language: str) -> bool:
    pure = PurePosixPath(path)
    parts = pure.parts
    if any(part in _EXCLUDED_DIRS for part in parts[:-1]):
        return True
    if any(parts[index : index + 2] == ("tests", "baselines") for index in range(len(parts) - 1)):
        return True
    name = pure.name
    if name.endswith((".min.js", ".bundle.js", "_pb.js", ".pb.rs")):
        return True
    if language == "rust" and name.startswith("unicode_tables") and name.endswith(".rs"):
        return True
    if language == "rust":
        return pure.suffix != ".rs"
    return pure.suffix not in _JS_EXTENSIONS or name.endswith(".d.ts")


def _attribute_exclusions(source: Source, paths: list[str], timeout: float) -> set[str]:
    if not paths:
        return set()
    payload = b"\0".join(os.fsencode(path) for path in paths) + b"\0"
    completed = _run(
        [
            "git",
            "check-attr",
            "-z",
            "--stdin",
            "linguist-generated",
            "linguist-vendored",
            "linguist-detectable",
        ],
        cwd=source.root,
        timeout=timeout,
        input_bytes=payload,
    )
    fields = completed.stdout.split(b"\0")
    if fields and not fields[-1]:
        fields.pop()
    if len(fields) % 3:
        raise ValueError(f"unexpected git check-attr output for {source.repo}")
    excluded: set[str] = set()
    for index in range(0, len(fields), 3):
        path = os.fsdecode(fields[index])
        attribute = fields[index + 1].decode("ascii")
        value = fields[index + 2].decode("utf-8", "strict").casefold()
        if attribute in {"linguist-generated", "linguist-vendored"} and value in {"set", "true"}:
            excluded.add(path)
        if attribute == "linguist-detectable" and value == "false":
            excluded.add(path)
    return excluded


def _normalise_content(raw: bytes) -> bytes | None:
    if b"\0" in raw[:8192]:
        return None
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError:
        return None
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _is_test_file(path: str, content: bytes) -> bool:
    pure = PurePosixPath(path)
    lowered_parts = {part.casefold() for part in pure.parts[:-1]}
    name = pure.name.casefold()
    return bool(
        lowered_parts.intersection({"test", "tests", "__tests__"})
        or ".test." in name
        or ".spec." in name
        or name.startswith(("test_", "tests_"))
        or b"#[test]" in content
        or b"#[cfg(test)]" in content
    )


def collect_candidates(source: Source, language: str, timeout: float) -> list[Candidate]:
    tracked = _run(["git", "ls-files", "-z"], cwd=source.root, timeout=timeout).stdout
    paths = [os.fsdecode(value) for value in tracked.split(b"\0") if value]
    paths = [path for path in paths if not _is_excluded(path, language)]
    excluded = _attribute_exclusions(source, paths, timeout)
    candidates: list[Candidate] = []
    for path in sorted(paths, key=os.fsencode):
        if path in excluded:
            continue
        raw = (source.root / path).read_bytes()
        if not MIN_BYTES <= len(raw) <= MAX_BYTES:
            continue
        normalised = _normalise_content(raw)
        if normalised is None:
            continue
        lines = normalised.splitlines()
        if len(lines) < MIN_LINES:
            continue
        if sum(map(len, lines)) / len(lines) > MAX_MEAN_LINE_BYTES:
            continue
        emitted = normalised.rstrip(b"\n")
        digest = hashlib.sha256(emitted).hexdigest()
        candidates.append(
            Candidate(source.repo, source.sha, path, emitted, digest, _is_test_file(path, emitted))
        )
    return candidates


def _shingle_hashes(content: bytes) -> frozenset[int]:
    tokens = _TOKEN_RE.findall(content)
    if len(tokens) < 5:
        tokens = [content]
    shingles: set[int] = set()
    windows: Iterable[bytes]
    if len(tokens) == 1:
        windows = tokens
    else:
        windows = (b"\0".join(tokens[index : index + 5]) for index in range(len(tokens) - 4))
    for window in windows:
        shingles.add(int.from_bytes(hashlib.blake2b(window, digest_size=8).digest(), "big"))
    return frozenset(shingles)


def minhash_signature(shingles: frozenset[int]) -> tuple[int, ...]:
    minima = [_U64_MASK] * MINHASH_PERMUTATIONS
    for value in shingles:
        second = ((value << 17) | (value >> 47)) & _U64_MASK
        second |= 1
        for permutation in range(MINHASH_PERMUTATIONS):
            permuted = (value + permutation * second + permutation * permutation * 0x9E3779B97F4A7C15) & _U64_MASK
            if permuted < minima[permutation]:
                minima[permutation] = permuted
    return tuple(minima)


def deduplicate(candidates: Iterable[Candidate]) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda item: (os.fsencode(item.path), os.fsencode(item.repo)))
    exact: set[str] = set()
    signatures: list[tuple[int, ...]] = []
    accepted: list[Candidate] = []
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    rows_per_band = 4
    for candidate in ordered:
        if candidate.sha256 in exact:
            continue
        shingles = _shingle_hashes(candidate.content)
        signature = minhash_signature(shingles)
        possible: set[int] = set()
        for band in range(MINHASH_PERMUTATIONS // rows_per_band):
            start = band * rows_per_band
            possible.update(buckets[(band, signature[start : start + rows_per_band])])
        duplicate = any(
            sum(left == right for left, right in zip(signature, signatures[index]))
            / MINHASH_PERMUTATIONS
            >= MINHASH_THRESHOLD
            for index in possible
        )
        if duplicate:
            continue
        index = len(accepted)
        accepted.append(candidate)
        signatures.append(signature)
        exact.add(candidate.sha256)
        for band in range(MINHASH_PERMUTATIONS // rows_per_band):
            start = band * rows_per_band
            buckets[(band, signature[start : start + rows_per_band])].append(index)
    return accepted


def round_robin(groups: list[list[Candidate]]) -> Iterator[Candidate]:
    positions = [0] * len(groups)
    while True:
        yielded = False
        for index, group in enumerate(groups):
            if positions[index] < len(group):
                yield group[positions[index]]
                positions[index] += 1
                yielded = True
        if not yielded:
            return


def serialize_candidates(candidates: Iterable[Candidate], target_bytes: int) -> CorpusArtifact:
    data = bytearray()
    rows: list[dict[str, Any]] = []
    test_bytes = 0
    for candidate in candidates:
        separator_size = 2 if data else 0
        projected_size = len(data) + separator_size + len(candidate.content) + 1
        if projected_size > target_bytes:
            break
        projected_test = test_bytes + (len(candidate.content) if candidate.is_test else 0)
        if candidate.is_test and projected_test * 4 > projected_size:
            continue
        if data:
            data.extend(b"\n\n")
        start = len(data)
        data.extend(candidate.content)
        end = len(data)
        rows.append(
            {
                "repo": candidate.repo,
                "sha": candidate.sha,
                "path": candidate.path,
                "sha256": candidate.sha256,
                "byte_start": start,
                "byte_end": end,
            }
        )
        test_bytes = projected_test
    data.extend(b"\n")
    manifest = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    return CorpusArtifact(bytes(data), manifest, len(rows))


def build_code_corpus(
    source_configs: list[dict[str, Any]],
    language: str,
    cache_dir: Path,
    timeout: float,
    target_bytes: int,
) -> CorpusArtifact:
    sources = [
        _clone_exact(str(config["repo"]), str(config["sha"]), cache_dir, timeout)
        for config in source_configs
    ]
    candidates = deduplicate(
        candidate
        for source in sources
        for candidate in collect_candidates(source, language, timeout)
    )
    by_repo: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_repo[candidate.repo].append(candidate)
    groups = [sorted(by_repo[source.repo], key=lambda item: os.fsencode(item.path)) for source in sources]
    return serialize_candidates(round_robin(groups), target_bytes)


def _general_artifact(config: dict[str, Any], data: bytes) -> CorpusArtifact:
    row = {
        "repo": str(config["repo"]),
        "sha": str(config["archive_sha256"]),
        "path": str(config["member"]),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_start": 0,
        "byte_end": len(data),
    }
    manifest = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return CorpusArtifact(data, manifest, 1)


def build_all(
    sources: dict[str, Any], cache_dir: Path, timeout: float, target_bytes: int
) -> dict[str, CorpusArtifact]:
    general_data = _download_general(sources["general"], cache_dir, timeout)
    return {
        "general": _general_artifact(sources["general"], general_data),
        "rust": build_code_corpus(sources["rust"], "rust", cache_dir, timeout, target_bytes),
        "js-ts": build_code_corpus(sources["js-ts"], "js-ts", cache_dir, timeout, target_bytes),
    }


def _source_revisions(sources: dict[str, Any]) -> dict[str, Any]:
    return {
        "general": {
            "url": sources["general"]["url"],
            "archive_sha256": sources["general"]["archive_sha256"],
            "file_sha256": sources["general"]["file_sha256"],
        },
        "rust": [{"repo": item["repo"], "sha": item["sha"]} for item in sources["rust"]],
        "js-ts": [{"repo": item["repo"], "sha": item["sha"]} for item in sources["js-ts"]],
    }


def build_record(
    sources: dict[str, Any], artifacts: dict[str, CorpusArtifact], timestamp: str
) -> dict[str, Any]:
    return {
        "builder_version": BUILDER_VERSION,
        "timestamp": timestamp,
        "sources": _source_revisions(sources),
        "corpora": {
            name: {
                "sha256": hashlib.sha256(artifact.data).hexdigest(),
                "byte_count": len(artifact.data),
                "file_count": artifact.file_count,
            }
            for name, artifact in artifacts.items()
        },
    }


def _write_outputs(output_dir: Path, artifacts: dict[str, CorpusArtifact], record: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, artifact in artifacts.items():
        (output_dir / f"{name}.txt").write_bytes(artifact.data)
        (output_dir / f"{name}.manifest.jsonl").write_bytes(artifact.manifest)
    (output_dir / "BUILD.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _verify(output_dir: Path, artifacts: dict[str, CorpusArtifact]) -> list[str]:
    with (output_dir / "BUILD.json").open(encoding="utf-8") as handle:
        recorded = json.load(handle)
    failures: list[str] = []
    for name, artifact in artifacts.items():
        actual = {
            "sha256": hashlib.sha256(artifact.data).hexdigest(),
            "byte_count": len(artifact.data),
            "file_count": artifact.file_count,
        }
        if recorded.get("corpora", {}).get(name) != actual:
            failures.append(f"{name}: rebuilt metadata differs from BUILD.json")
        manifest_path = output_dir / f"{name}.manifest.jsonl"
        if not manifest_path.is_file() or manifest_path.read_bytes() != artifact.manifest:
            failures.append(f"{name}: rebuilt manifest differs")
    return failures


def build_parser() -> argparse.ArgumentParser:
    root = ROOT
    parser = argparse.ArgumentParser(
        description=(
            "Clone exact revisions, filter and deduplicate source files, interleave repositories, "
            "and build byte-reproducible KLD corpora. Normal builds write ignored .txt bytes plus "
            "committable manifests and BUILD.json; --verify rebuilds without overwriting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python scripts/build_kld_corpora.py --timestamp "
            "2026-08-20T00:00:00Z; then run the same command with --verify and no timestamp."
        ),
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=root / "corpora" / "sources.json",
        help="pinned source manifest",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "corpora", help="corpus and manifest directory")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=root / ".cache" / "kld-corpora",
        help="reusable exact-revision checkouts and downloads",
    )
    parser.add_argument(
        "--target-bytes",
        type=int,
        default=DEFAULT_TARGET_BYTES,
        help="maximum serialized bytes for each code corpus; files are never truncated",
    )
    parser.add_argument("--timeout", type=float, default=300, help="seconds allowed for each git or HTTP operation")
    parser.add_argument(
        "--timestamp",
        help="literal reproducible timestamp stored in BUILD.json; required unless --verify",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="rebuild and compare SHA-256 metadata and manifests with existing BUILD.json; write nothing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.target_bytes <= 0:
        raise SystemExit("--target-bytes must be positive")
    if not args.verify and args.timestamp is None:
        raise SystemExit("--timestamp is required for a normal build")
    with args.sources.open(encoding="utf-8") as handle:
        sources = json.load(handle)
    artifacts = build_all(sources, args.cache_dir.resolve(), args.timeout, args.target_bytes)
    if args.verify:
        failures = _verify(args.output_dir, artifacts)
        if failures:
            for failure in failures:
                print(f"VERIFY FAILED: {failure}", file=sys.stderr)
            return 1
        print("KLD corpus verification passed", file=sys.stderr)
        return 0
    record = build_record(sources, artifacts, args.timestamp)
    _write_outputs(args.output_dir, artifacts, record)
    for name, metadata in record["corpora"].items():
        print(f"{name}: {metadata['byte_count']} bytes {metadata['sha256']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
