#!/usr/bin/env python3
"""Derive frozen Polyglot, Diff-XYZ, and SWE-bench Multilingual subsets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts import ROOT
import tempfile
from typing import Any, Iterable
import urllib.parse
import urllib.request


GENERATOR_VERSION = "2"
SEED_PREFIX = "bloosh-quant-eval-v1"
# The first ten, kept so the legs already run against them stay comparable. Any
# selection change that drops one of these invalidates that data, so it fails loudly.
ORIGINAL_AGENTIC_IDS = [
    "uutils__coreutils-6682",
    "tokio-rs__axum-734",
    "tokio-rs__axum-2096",
    "tokio-rs__axum-1730",
    "astral-sh__ruff-15356",
    "preactjs__preact-4245",
    "preactjs__preact-4182",
    "facebook__docusaurus-9897",
    "babel__babel-15445",
    "babel__babel-16130",
]

# Not a knob: a tripwire. The selection is a pure function of the pinned sources and
# the seed, so a diff here means a source moved or the policy changed, and the reader
# is told which rather than handed a quietly different subset.
EXPECTED_AGENTIC_IDS = [
    "apache__druid-13704",
    "apache__druid-14136",
    "apache__druid-16875",
    "apache__lucene-12022",
    "apache__lucene-12212",
    "apache__lucene-13704",
    "astral-sh__ruff-15356",
    "babel__babel-15445",
    "babel__babel-16130",
    "briannesbitt__carbon-3005",
    "caddyserver__caddy-4774",
    "caddyserver__caddy-6345",
    "facebook__docusaurus-9897",
    "faker-ruby__faker-2705",
    "fastlane__fastlane-19304",
    "fastlane__fastlane-21857",
    "fluent__fluentd-3328",
    "fluent__fluentd-3640",
    "fluent__fluentd-4311",
    "gin-gonic__gin-1957",
    "gin-gonic__gin-3741",
    "gohugoio__hugo-12204",
    "hashicorp__terraform-35611",
    "jekyll__jekyll-8047",
    "laravel__framework-46234",
    "laravel__framework-48573",
    "laravel__framework-51195",
    "micropython__micropython-12158",
    "php-cs-fixer__php-cs-fixer-8075",
    "phpoffice__phpspreadsheet-3463",
    "phpoffice__phpspreadsheet-4114",
    "phpoffice__phpspreadsheet-4313",
    "preactjs__preact-3562",
    "preactjs__preact-4182",
    "preactjs__preact-4245",
    "projectlombok__lombok-2792",
    "projectlombok__lombok-3594",
    "projectlombok__lombok-3674",
    "prometheus__prometheus-14861",
    "prometheus__prometheus-9248",
    "redis__redis-11510",
    "redis__redis-12272",
    "redis__redis-13115",
    "rubocop__rubocop-13579",
    "rubocop__rubocop-13627",
    "rubocop__rubocop-13668",
    "tokio-rs__axum-1730",
    "tokio-rs__axum-2096",
    "tokio-rs__axum-734",
    "uutils__coreutils-6682",
    "valkey-io__valkey-790",
    "valkey-io__valkey-928",
]
# The dataset names its languages only through the log parser it grades with.
# Grading runs inside the instance's own published image, so admitting a language
# costs nothing here -- the earlier rust/jsts restriction left a clean pool of 14,
# which cannot carry a 50-instance subset.
_PARSER_LANGUAGE = {
    "parse_log_cargo": "rust",
    "parse_log_gotest": "go",
    "parse_log_phpunit": "php",
    "parse_log_jest": "jsts",
    "parse_log_karma": "jsts",
    "parse_log_vitest": "jsts",
    "parse_log_immutable_js": "jsts",
    "parse_log_tap": "jsts",
    "parse_log_maven": "java",
    "parse_log_ant": "java",
    "parse_log_gradle_custom": "java",
    "parse_log_rspec_transformed_json": "ruby",
    "parse_log_ruby_unit": "ruby",
    "parse_log_jekyll": "ruby",
    "parse_log_googletest": "cpp",
    "parse_log_doctest": "cpp",
    "parse_log_redis": "c",
    "parse_log_jq": "c",
    "parse_log_micropython_test": "c",
}

# Per language, then per repo within it. Both caps exist so no single repo's idioms
# decide the score; the language cap is not binding for rust or jsts, whose clean
# pools are smaller than it.
_AGENTIC_PER_LANGUAGE = 10
_AGENTIC_PER_REPO = 3


def _run(command: list[str], *, cwd: Path | None, timeout: float) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        timeout=timeout,
        check=True,
    )


def _cache_name(label: str, sha: str) -> str:
    return label.replace("/", "--") + "-" + sha


def ensure_repo(
    url: str,
    label: str,
    sha: str,
    cache_dir: Path,
    timeout: float,
    *,
    checkout: bool,
) -> Path:
    root = cache_dir / _cache_name(label, sha)
    if not root.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="subset-source-", dir=cache_dir) as temp_dir:
            staged = Path(temp_dir) / "repo"
            staged.mkdir()
            _run(["git", "init", "--quiet"], cwd=staged, timeout=timeout)
            _run(["git", "remote", "add", "origin", url], cwd=staged, timeout=timeout)
            _run(
                ["git", "fetch", "--filter=blob:none", "--depth=1", "origin", sha],
                cwd=staged,
                timeout=timeout,
            )
            _run(["git", "update-ref", "refs/subsets/pinned", sha], cwd=staged, timeout=timeout)
            os.replace(staged, root)
    _run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=root, timeout=timeout)
    if checkout:
        _run(["git", "checkout", "--quiet", "--detach", sha], cwd=root, timeout=timeout)
    return root


def _canonical_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _validate_dataset(rows: list[dict[str, Any]], config: dict[str, Any], label: str) -> None:
    if len(rows) != int(config["row_count"]):
        raise ValueError(f"{label}: expected {config['row_count']} rows, got {len(rows)}")
    digest = hashlib.sha256(_canonical_jsonl(rows)).hexdigest()
    if digest != str(config["jsonl_sha256"]):
        raise ValueError(f"{label}: canonical JSONL SHA-256 mismatch ({digest})")


def fetch_dataset_rows(
    config: dict[str, Any], cache_dir: Path, timeout: float, supplied: Path | None = None
) -> list[dict[str, Any]]:
    label = str(config["dataset"])
    if supplied is not None:
        rows = _load_jsonl(supplied)
        _validate_dataset(rows, config, label)
        return rows
    cache = cache_dir / (_cache_name(label, str(config["revision"])) + ".jsonl")
    if cache.is_file():
        rows = _load_jsonl(cache)
        _validate_dataset(rows, config, label)
        return rows

    revision_url = "https://huggingface.co/api/datasets/" + label + "/revision/" + str(config["revision"])
    request = urllib.request.Request(revision_url, headers={"User-Agent": "frozen-subset-builder/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        revision = json.load(response)
    if revision.get("sha") != config["revision"]:
        raise ValueError(f"{label}: Hugging Face did not resolve the pinned revision")

    rows: list[dict[str, Any]] = []
    expected_count = int(config["row_count"])
    for offset in range(0, expected_count, 100):
        query = urllib.parse.urlencode(
            {
                "dataset": label,
                "config": "default",
                "split": "test",
                "offset": offset,
                "length": min(100, expected_count - offset),
            }
        )
        page_request = urllib.request.Request(
            "https://datasets-server.huggingface.co/rows?" + query,
            headers={"User-Agent": "frozen-subset-builder/1"},
        )
        with urllib.request.urlopen(page_request, timeout=timeout) as response:
            page = json.load(response)
        if int(page.get("num_rows_total", -1)) != expected_count:
            raise ValueError(f"{label}: dataset row count changed")
        for expected_index, wrapped in enumerate(page.get("rows", []), offset):
            if wrapped.get("row_idx") != expected_index:
                raise ValueError(f"{label}: non-contiguous row indices")
            if wrapped.get("truncated_cells"):
                raise ValueError(f"{label}: row {expected_index} contains truncated cells")
            row = wrapped.get("row")
            if not isinstance(row, dict):
                raise ValueError(f"{label}: row {expected_index} is not an object")
            rows.append(row)
    _validate_dataset(rows, config, label)
    cache.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_jsonl(rows)
    with tempfile.NamedTemporaryFile(dir=cache.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, cache)
    return rows


def _rank(namespace: str, task_id: str) -> str:
    return hashlib.sha256(f"{SEED_PREFIX}|{namespace}|{task_id}".encode()).hexdigest()


def derive_codegen(
    polyglot_root: Path,
    manifest_path: Path,
    source_repo: str,
    source_sha: str,
    excluded: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Select every *viable* exercise in the two languages, ordered by seeded rank.

    Viability is not a judgement call: some Exercism exercises ship a reference
    example that cannot compile as given (crates used but never declared in
    Cargo.toml), so neither a model nor the reference can pass them. They are
    identified by running selftest_polyglot.py over the full pool and recorded in
    subsets/excluded.json, which keeps the selection deterministic and auditable
    rather than silently scoring unscoreable tasks.

    It took the first 20 per language, which resolves a 1.3x cost difference and
    nothing finer -- the measured per-task log-ratio spread needs ~160 paired tasks
    for 1.15x. Taking the whole pool is the only free way to buy that, and it extends
    rather than resamples: every task the first 40 held is still here, so the legs
    already run against them stay comparable.
    """

    tasks: list[dict[str, Any]] = []
    for language in ("rust", "javascript"):
        exercise_root = polyglot_root / language / "exercises" / "practice"
        ranked = sorted(
            (
                (_rank("codegen", f"{language}/{path.name}"), f"{language}/{path.name}", path)
                for path in exercise_root.iterdir()
                if path.is_dir()
            ),
            key=lambda item: (item[0], item[1]),
        )
        if len(ranked) != {"rust": 30, "javascript": 49}[language]:
            raise ValueError(f"unexpected {language} Polyglot pool size: {len(ranked)}")
        viable = [item for item in ranked if item[1] not in excluded]
        if len(viable) < 20:
            raise ValueError(
                f"only {len(viable)} viable {language} exercises after exclusions; need 20"
            )
        for rank, task_id, source in viable:
            with (source / ".meta" / "config.json").open(encoding="utf-8") as handle:
                files = json.load(handle)["files"]
            tasks.append(
                {
                    "task_id": task_id,
                    "language": language,
                    "repo": source_repo,
                    "source_sha": source_sha,
                    "source_dir": os.path.relpath(source, manifest_path.resolve().parent),
                    "solution_path": files["solution"],
                    "example_path": files["example"],
                    "test_cmd": (
                        ["cargo", "test", "--", "--include-ignored"]
                        if language == "rust"
                        else ["npm", "test", "--", "--runInBand"]
                    ),
                    "rank": rank,
                }
            )
    return tasks


def _repo_key(repo: str) -> str:
    return repo.casefold()


def derive_diffxyz(
    rows: Iterable[dict[str, Any]], excluded_repos: set[str]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    excluded = {_repo_key(repo) for repo in excluded_repos}
    for language in ("rust", "javascript"):
        ranked: list[tuple[str, str, str, dict[str, Any]]] = []
        for row in rows:
            if row.get("lang") != language or _repo_key(str(row.get("repo", ""))) in excluded:
                continue
            old_code = row.get("old_code")
            if not isinstance(old_code, str) or not 200 <= len(old_code) <= 12_000:
                continue
            identity = f"{row['repo']}@{row['commit']}:{row['path']}"
            ranked.append((_rank("diffxyz", identity), str(row["repo"]), str(row["path"]), row))
        counts: Counter[str] = Counter()
        language_rows: list[dict[str, Any]] = []
        for rank, repo, path, row in sorted(ranked, key=lambda item: (item[0], item[1], item[2])):
            repo_key = _repo_key(repo)
            if counts[repo_key] >= 3:
                continue
            identity = f"{repo}@{row['commit']}:{path}"
            language_rows.append(
                {
                    "task_id": identity,
                    "language": language,
                    "repo": repo,
                    "commit": row["commit"],
                    "path": path,
                    "old_code": row["old_code"],
                    "new_code": row["new_code"],
                    "udiff": row["udiff"],
                    "search-replace": row["search-replace"],
                    "rank": rank,
                }
            )
            counts[repo_key] += 1
            if len(language_rows) == 25:
                break
        if len(language_rows) != 25:
            raise ValueError(f"Diff-XYZ produced only {len(language_rows)} {language} rows")
        selected.extend(language_rows)
    return selected


def multi_swe_repo_set(root: Path, sha: str, timeout: float) -> set[str]:
    # Every language directory. A repo appears under exactly one of them, so naming a
    # subset of the directories silently admits every other language's contaminated repos.
    completed = _run(
        ["git", "ls-tree", "-r", "--name-only", sha],
        cwd=root,
        timeout=timeout,
    )
    repos: set[str] = set()
    for raw_path in completed.stdout.decode("utf-8").splitlines():
        filename = Path(raw_path).name
        suffix = "_dataset.jsonl"
        if filename.endswith(suffix) and "__" in filename:
            owner, repo = filename[: -len(suffix)].split("__", 1)
            repos.add(f"{owner}/{repo}")
    if not repos:
        raise ValueError("Multi-SWE-bench repository set is empty")
    return repos


def load_submission_results(root: Path, expected_count: int) -> list[dict[str, Any]]:
    paths = sorted((root / "evaluation" / "multilingual").glob("*/per_instance_details.json"))
    if len(paths) != expected_count:
        raise ValueError(f"expected {expected_count} multilingual submissions, found {len(paths)}")
    results: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected a JSON object")
        results.append(value)
    return results


def derive_agentic(
    rows: Iterable[dict[str, Any]],
    submissions: list[dict[str, Any]],
    leakage_repos: set[str],
) -> list[dict[str, Any]]:
    """Select the instances by seeded rank, capped per language and per repo.

    The 0.25-0.80 solve-rate window is the whole design: an instance every submitted
    system solves and one none of them solve both score the same for every quant, so
    neither can separate anything. It is deliberately unchanged here -- what widened
    is the set of languages, which the window then filters as before.
    """
    leakage = {_repo_key(repo) for repo in leakage_repos}
    pools: dict[str, list[tuple[str, str, dict[str, Any], float]]] = defaultdict(list)
    for row in rows:
        language = _PARSER_LANGUAGE.get(row.get("log_parser"))
        if language is None or _repo_key(str(row.get("repo", ""))) in leakage:
            continue
        instance_id = str(row["instance_id"])
        solve_rate = sum(
            bool(result.get(instance_id, {}).get("resolved")) for result in submissions
        ) / len(submissions)
        if 0.25 <= solve_rate <= 0.80:
            pools[language].append((_rank("agentic", instance_id), instance_id, row, solve_rate))

    chosen: dict[str, tuple[dict[str, Any], float, str, str]] = {}
    for language in sorted(pools):
        counts: Counter[str] = Counter()
        accepted = 0
        for rank, instance_id, row, solve_rate in sorted(pools[language], key=lambda item: (item[0], item[1])):
            repo_key = _repo_key(str(row["repo"]))
            if counts[repo_key] >= _AGENTIC_PER_REPO:
                continue
            chosen[instance_id] = (row, solve_rate, rank, language)
            counts[repo_key] += 1
            accepted += 1
            if accepted == _AGENTIC_PER_LANGUAGE:
                break
    if set(chosen) != set(EXPECTED_AGENTIC_IDS):
        missing = sorted(set(EXPECTED_AGENTIC_IDS) - set(chosen))
        extra = sorted(set(chosen) - set(EXPECTED_AGENTIC_IDS))
        raise AssertionError(f"agentic derivation changed; missing={missing}, extra={extra}")
    # The 10-instance subset must stay a prefix of this one, or the legs already run
    # against it stop being comparable with anything measured from here on.
    if not set(ORIGINAL_AGENTIC_IDS) <= set(chosen):
        raise AssertionError(
            f"lost original instances: {sorted(set(ORIGINAL_AGENTIC_IDS) - set(chosen))}"
        )

    output: list[dict[str, Any]] = []
    for instance_id in EXPECTED_AGENTIC_IDS:
        row, solve_rate, rank, language = chosen[instance_id]
        output.append(
            {
                "instance_id": instance_id,
                "repo": row["repo"],
                "base_commit": row["base_commit"],
                "problem_statement": row["problem_statement"],
                "image": "swebench/sweb.eval.x86_64."
                + instance_id.replace("__", "_1776_")
                + ":latest",
                "language": language,
                "solve_rate": solve_rate,
                "rank": rank,
            }
        )
    return output


def _write_json(path: Path, value: Any) -> bytes:
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> bytes:
    data = _canonical_jsonl(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def build_parser() -> argparse.ArgumentParser:
    root = ROOT
    parser = argparse.ArgumentParser(
        description=(
            "Fetch exact source snapshots and deterministically materialize 79 Polyglot tasks, "
            "50 Diff-XYZ rows with both generation targets, and 52 SWE-bench Multilingual instances. "
            "Fetched JSON rows and Git checkouts are cached, so reruns are offline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python scripts/make_subsets.py. After the first cached run no network is "
            "needed; --diffxyz-jsonl and --swebench-jsonl accept pre-fetched canonical exports."
        ),
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=root / "subsets" / "sources.json",
        help="pinned upstream revisions and canonical dataset hashes",
    )
    parser.add_argument(
        "--corpus-sources",
        type=Path,
        default=root / "corpora" / "sources.json",
        help="L1 corpus source manifest used by the Diff-XYZ leakage filter",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "subsets",
        help="destination for polyglot.json, diffxyz.jsonl, swebench.jsonl, and BUILD.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=root / ".cache" / "subsets",
        help="reusable exact-revision Git and canonical JSONL cache",
    )
    parser.add_argument(
        "--diffxyz-jsonl",
        type=Path,
        help="offline canonical Diff-XYZ JSONL export; its pinned SHA-256 is required",
    )
    parser.add_argument(
        "--swebench-jsonl",
        type=Path,
        help="offline canonical SWE-bench Multilingual JSONL export; its pinned SHA-256 is required",
    )
    parser.add_argument("--timeout", type=float, default=300, help="seconds allowed for each Git or HTTP operation")
    parser.add_argument(
        "--exclude",
        help="JSON file with task_ids proven unscoreable by the harness self-test "
             "(default: subsets/excluded.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    with args.sources.open(encoding="utf-8") as handle:
        sources = json.load(handle)
    cache = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()

    poly_config = sources["polyglot"]
    polyglot = ensure_repo(
        poly_config["url"],
        poly_config["repo"],
        poly_config["sha"],
        cache,
        args.timeout,
        checkout=True,
    )
    experiment_config = sources["swebench_experiments"]
    experiments = ensure_repo(
        experiment_config["url"],
        experiment_config["repo"],
        experiment_config["sha"],
        cache,
        args.timeout,
        checkout=True,
    )
    multi_config = sources["multi_swe_bench"]
    multi_swe = ensure_repo(
        multi_config["url"],
        multi_config["dataset"],
        multi_config["sha"],
        cache,
        args.timeout,
        checkout=False,
    )

    diff_rows = fetch_dataset_rows(sources["diffxyz"], cache, args.timeout, args.diffxyz_jsonl)
    swe_rows = fetch_dataset_rows(sources["swebench_multilingual"], cache, args.timeout, args.swebench_jsonl)
    leakage_repos = multi_swe_repo_set(multi_swe, multi_config["sha"], args.timeout)
    submissions = load_submission_results(experiments, int(experiment_config["submission_count"]))

    polyglot_path = output_dir / "polyglot.json"
    excluded_path = Path(args.exclude) if getattr(args, "exclude", None) else (
        ROOT / "subsets" / "excluded.json"
    )
    excluded = frozenset(
        json.loads(excluded_path.read_text(encoding="utf-8")).get("task_ids", [])
        if excluded_path.is_file()
        else []
    )
    codegen = derive_codegen(
        polyglot,
        polyglot_path,
        str(poly_config["repo"]),
        str(poly_config["sha"]),
        excluded,
    )
    if excluded:
        print(f"excluded {len(excluded)} unscoreable exercise(s): {sorted(excluded)}", file=sys.stderr)
    agentic = derive_agentic(swe_rows, submissions, leakage_repos)
    with args.corpus_sources.open(encoding="utf-8") as handle:
        corpus_sources = json.load(handle)
    l1_repos = {
        item["repo"]
        for language in ("rust", "js-ts")
        for item in corpus_sources[language]
    }
    diffxyz = derive_diffxyz(diff_rows, l1_repos | {row["repo"] for row in agentic})

    outputs = {
        "polyglot.json": _write_json(polyglot_path, codegen),
        "diffxyz.jsonl": _write_jsonl(output_dir / "diffxyz.jsonl", diffxyz),
        "swebench.jsonl": _write_jsonl(output_dir / "swebench.jsonl", agentic),
    }
    build = {
        "generator_version": GENERATOR_VERSION,
        "seed_prefix": SEED_PREFIX,
        "sources": sources,
        "outputs": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "byte_count": len(data)}
            for name, data in outputs.items()
        },
    }
    _write_json(output_dir / "BUILD.json", build)
    print(f"generated {len(codegen)} codegen tasks, {len(diffxyz)} Diff-XYZ rows, "
          f"and {len(agentic)} agentic instances", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
