#!/usr/bin/env python3
"""Run frozen Polyglot/Exercism tasks with one bounded repair round."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
from pathlib import Path
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterable

from scripts import ROOT
from scripts.instruments import emit
from scripts.instruments.longctx import build_padding, wrap_prompt
from scripts.instruments.emit import (
    CodexStats,
    codex_base_url,
    add_context_arguments,
    emit,
    ContextSampler,
    apply_server_probe,
    invoke_codex,
    metrics_delta,
    scrape_metrics,
    make_context,
    server_alive,
)


DEFAULT_SERVER_URL = "http://127.0.0.1:8000/v1"


class LegAborted(RuntimeError):
    """The leg cannot produce valid rows any more; stop rather than record failures."""



@dataclasses.dataclass(frozen=True, slots=True)
class TaskResult:
    task_id: str
    measures: dict[str, Any]


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, list):
        raise ValueError("manifest must be a JSON array")
    return manifest


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired("task", 0)
    return remaining


def _test_command(value: Any) -> list[str]:
    if isinstance(value, str):
        command = shlex.split(value)
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        command = value
    else:
        raise ValueError("test_cmd must be a command string or an array of arguments")
    if not command:
        raise ValueError("test_cmd must not be empty")
    return command


def task_test_command(task: dict[str, Any]) -> list[str]:
    """Return the task command with language-required test activation."""

    command = _test_command(task["test_cmd"])
    if str(task.get("language", "")).lower() != "rust":
        return command
    if Path(command[0]).name != "cargo" or len(command) < 2 or command[1] != "test":
        raise ValueError("Rust test_cmd must invoke cargo test")
    if "--include-ignored" in command:
        return command
    if "--" in command:
        separator = command.index("--")
        return command[: separator + 1] + ["--include-ignored"] + command[separator + 1 :]
    return command + ["--", "--include-ignored"]


def _path_values(value: Any, field: str) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
        return value
    raise ValueError(f"{field} must be a non-empty path string or array of path strings")


def _exercise_path(root: Path, value: str, field: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{field} paths must be relative to source_dir")
    resolved_root = root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{field} path escapes source_dir: {value}") from exc
    return resolved


def task_file_paths(
    task: dict[str, Any], workdir: Path, manifest_field: str, config_field: str
) -> list[Path]:
    """Resolve manifest path overrides or Exercism metadata inside a scratch copy."""

    value = task.get(manifest_field)
    if value is None:
        config_path = workdir / ".meta" / "config.json"
        try:
            with config_path.open(encoding="utf-8") as handle:
                config = json.load(handle)
            value = config["files"][config_field]
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"cannot derive {manifest_field} from {config_path}: {exc}"
            ) from exc
    return [
        _exercise_path(workdir, item, manifest_field)
        for item in _path_values(value, manifest_field)
    ]


def _javascript_test_paths(task: dict[str, Any], workdir: Path) -> list[Path]:
    # Contract: resolvable after .meta is deleted. prepare_test_harness runs once more
    # before each test cycle, but _strip_reference_material has removed .meta by then,
    # so the config-derived answer is cached on first use rather than re-read.
    cached = task.get("_resolved_test_paths")
    if cached is not None:
        return [workdir / value for value in cached]
    config_path = workdir / ".meta" / "config.json"
    if config_path.is_file():
        try:
            with config_path.open(encoding="utf-8") as handle:
                config = json.load(handle)
            values = _path_values(config["files"]["test"], "files.test")
            paths = [_exercise_path(workdir, value, "files.test") for value in values]
            task["_resolved_test_paths"] = [
                str(path.relative_to(workdir)) for path in paths
            ]
            return paths
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"cannot derive JavaScript test files from {config_path}: {exc}"
            ) from exc

    extensions = {".js", ".cjs", ".mjs"}
    paths: list[Path] = []
    for path in workdir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        name = path.name.lower()
        parts = {part.lower() for part in path.relative_to(workdir).parts[:-1]}
        if (
            ".test." in name
            or ".spec." in name
            or name.startswith("test")
            or parts.intersection({"test", "tests", "__tests__"})
        ):
            paths.append(path)
    return paths


def _strip_reference_material(workdir: Path) -> None:
    """Remove Exercism's .meta/ from the copy the model is given.

    .meta/ carries the reference solution (proof.ci.js, example.rs). Left in place the
    model can simply read the answer, so task_success measures retrieval, not solving.
    Deleting beats sandboxing here: Codex's workspace-write sandbox restricts writes,
    not reads, and a file that is absent cannot be read by any policy.

    Contract: every consumer of .meta must have cached what it needs before this runs.
    """

    shutil.rmtree(workdir / ".meta", ignore_errors=True)


def _link_shared_node_modules(workdir: Path) -> None:
    """Point a JS exercise at one shared dependency tree.

    Each Exercism exercise declares jest in its own devDependencies; installing per
    exercise is ~20x redundant, so one install is linked in instead. Without it every
    JS task self-tests as BROKEN with "jest: not found" -- technically correct and
    useless, so a missing install is raised loudly rather than scored.
    """

    shared = Path(
        os.environ.get(
            "POLYGLOT_NODE_MODULES",
            str(ROOT / ".cache" / "js-deps" / "node_modules"),
        )
    )
    if not shared.is_dir():
        raise FileNotFoundError(
            f"shared node_modules not found at {shared}; run `npm install` in "
            ".cache/js-deps or set POLYGLOT_NODE_MODULES"
        )
    target = workdir / "node_modules"
    if not target.exists():
        target.symlink_to(shared, target_is_directory=True)


def _declare_reference_deps(task: dict[str, Any], workdir: Path) -> None:
    """Declare crates the shipped reference example uses but Cargo.toml omits.

    Affects the scratch copy only. The stub and the tests use no external crate, so
    the task posed to the model is unchanged -- this repairs the reference so the
    self-test can distinguish a solvable exercise from a broken one. Dropping these
    exercises instead would bias selection toward easy tasks.
    """

    table = ROOT / "subsets" / "rust-deps.json"
    if not table.is_file():
        return
    deps = json.loads(table.read_text(encoding="utf-8"))["deps"].get(task.get("task_id"))
    if not deps:
        return
    cargo = workdir / "Cargo.toml"
    text = cargo.read_text(encoding="utf-8")
    # Contract: idempotent. This runs again before each test cycle, and a repeated
    # key makes cargo reject the manifest outright -- which scores the task failed
    # however good the solution is.
    declared = set(re.findall(r"(?m)^\s*([A-Za-z0-9_-]+)\s*=", text))
    lines = [
        f'{name} = "{version}"'
        for name, version in sorted(deps.items())
        if name not in declared
    ]
    if not lines:
        return
    if "[dependencies]" in text:
        text = text.replace("[dependencies]", "[dependencies]\n" + "\n".join(lines), 1)
    else:
        text = text.rstrip("\n") + "\n\n[dependencies]\n" + "\n".join(lines) + "\n"
    cargo.write_text(text, encoding="utf-8")


def prepare_test_harness(task: dict[str, Any], workdir: Path) -> int:
    """Enable tests disabled by Exercism in a disposable exercise copy."""

    if str(task.get("language", "")).lower() in {"rust", "rs"}:
        _declare_reference_deps(task, workdir)
        return 0
    if str(task.get("language", "")).lower() not in {"javascript", "js"}:
        return 0
    _link_shared_node_modules(workdir)
    replacements = 0
    for path in _javascript_test_paths(task, workdir):
        text = path.read_text(encoding="utf-8")
        updated, count = re.subn(r"\bxtest\(", "test(", text)
        if count:
            path.write_text(updated, encoding="utf-8")
            replacements += count
    return replacements


def _run_tests(command: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _merge_stats(stats: Iterable[CodexStats]) -> dict[str, int | None]:
    values = list(stats)

    def total(field: str) -> int | None:
        items = [getattr(item, field) for item in values]
        return sum(items) if items and all(item is not None for item in items) else None

    return {
        "agent_turns": total("agent_turns"),
        "tool_calls": total("tool_calls"),
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "cached_input_tokens": total("cached_input_tokens"),
        # Billable on some routes even though cache reads are near-free, so dropping it
        # understates cost rather than merely losing detail.
        "cache_write_tokens": total("cache_write_tokens"),
        "reasoning_tokens": total("reasoning_tokens"),
    }


def _prompt(task: dict[str, Any]) -> str:
    return (
        f"Solve the {task['language']} exercise in this working directory. "
        "Read the exercise instructions and existing files, implement the solution, and "
        "run or inspect tests as useful. Do not modify files outside this working directory."
    )


def run_task(
    task: dict[str, Any],
    manifest_dir: Path,
    lane: str,
    reasoning_effort: str,
    timeout: float,
    dry_run: bool,
    harness_verified: bool = False,
    transcript: Path | None = None,
    metrics_url: str | None = None,
    context_padding: int = 0,
) -> TaskResult:
    task_id = "<missing-task-id>"
    wall_seconds = 0.0
    stats: list[CodexStats] = []
    deadline = time.monotonic() + timeout
    failed_cycles = 0
    metrics_before = scrape_metrics(metrics_url) if metrics_url else {}
    sampler = None
    if metrics_url:
        sampler = ContextSampler(metrics_url)
        sampler.start()
    try:
        if not isinstance(task, dict):
            raise ValueError("each manifest entry must be a JSON object")
        task_id = str(task.get("task_id", ""))
        for field in ("task_id", "language", "source_dir", "test_cmd"):
            if field not in task:
                raise ValueError(f"task is missing required field {field!r}")
        if not task_id:
            raise ValueError("task_id must be non-empty")
        source = Path(task["source_dir"])
        if not source.is_absolute():
            source = manifest_dir / source
        command = task_test_command(task)

        with tempfile.TemporaryDirectory(prefix="polyglot-") as temp_dir:
            workdir = Path(temp_dir) / "exercise"
            shutil.copytree(source, workdir)
            prepare_test_harness(task, workdir)
            _strip_reference_material(workdir)
            prompt = _prompt(task)
            padding_tokens = 0
            if context_padding and not dry_run:
                padding, padding_tokens = build_padding(task, context_padding, metrics_url)
                prompt = wrap_prompt(prompt, padding)
            if not dry_run:
                invocation_started = time.monotonic()
                try:
                    turn_stats, _ = invoke_codex(
                        lane, reasoning_effort, prompt, workdir,
                        _remaining(deadline), transcript,
                    )
                finally:
                    wall_seconds += time.monotonic() - invocation_started
                stats.append(turn_stats)

            prepare_test_harness(task, workdir)
            test = _run_tests(command, workdir, _remaining(deadline))
            if test.returncode != 0:
                failed_cycles = 1
                repair_prompt = (
                    prompt
                    + "\n\nThe test command failed. Diagnose and repair the implementation, then "
                    + "re-run the tests. Test output follows:\n\n"
                    + test.stdout
                    + test.stderr
                )
                if not dry_run:
                    invocation_started = time.monotonic()
                    try:
                        turn_stats, _ = invoke_codex(
                            lane, reasoning_effort, repair_prompt, workdir,
                            _remaining(deadline), transcript,
                        )
                    finally:
                        wall_seconds += time.monotonic() - invocation_started
                    stats.append(turn_stats)
                prepare_test_harness(task, workdir)
                test = _run_tests(command, workdir, _remaining(deadline))

            passed = test.returncode == 0
            measures: dict[str, Any] = {
                "task_success": passed,
                "tests_passed": passed,
                "timed_out": False,
                "failed_test_cycles": failed_cycles,
                "harness_verified": harness_verified,
                "wall_seconds": round(wall_seconds, 6),
                "context_padding_tokens": padding_tokens,
                "notes": None if passed else "tests failed after one repair round",
            }
            measures.update(_merge_stats(stats))
            if transcript is not None:
                measures["transcript"] = str(transcript)
            if metrics_url:
                measures.update(metrics_delta(metrics_before, scrape_metrics(metrics_url)))
                measures["peak_context_tokens"] = sampler.stop()
            return TaskResult(task_id, measures)
    except subprocess.TimeoutExpired:
        known_stats = _merge_stats(stats)
        return TaskResult(
            task_id or "<missing-task-id>",
            {
                "task_success": False,
                "tests_passed": False,
                # A wall-clock cut-off is a harness parameter, not a model verdict,
                # so it has to be separable from a genuine test failure downstream.
                "timed_out": True,
                "failed_test_cycles": failed_cycles,
                "harness_verified": harness_verified,
                "wall_seconds": round(wall_seconds, 6),
                "notes": f"task timed out after {timeout:g}s",
                **known_stats,
                **({"transcript": str(transcript)} if transcript else {}),
                **(metrics_delta(metrics_before, scrape_metrics(metrics_url))
                   if metrics_url else {}),
                **({"peak_context_tokens": sampler.stop()} if sampler else {}),
            },
        )
    except Exception as exc:
        known_stats = _merge_stats(stats)
        note = (
            f"Codex exited with status {exc.returncode}"
            if isinstance(exc, subprocess.CalledProcessError)
            else f"{type(exc).__name__}: {exc}"
        )
        return TaskResult(
            task_id or "<missing-task-id>",
            {
                "task_success": False,
                "tests_passed": False,
                "timed_out": False,
                "failed_test_cycles": failed_cycles,
                "harness_verified": harness_verified,
                "wall_seconds": round(wall_seconds, 6),
                "notes": note,
                **known_stats,
                **({"transcript": str(transcript)} if transcript else {}),
                **(metrics_delta(metrics_before, scrape_metrics(metrics_url))
                   if metrics_url else {}),
                **({"peak_context_tokens": sampler.stop()} if sampler else {}),
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy and run frozen Exercism/Polyglot tasks, allow exactly one repair after "
            "a failed test cycle, and append one durable result row per task."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help=(
            "JSON array of {task_id, language, source_dir, test_cmd, optional "
            "example_path and solution_path} objects"
        ),
    )
    selftest = parser.add_mutually_exclusive_group()
    selftest.add_argument(
        "--selftest-report",
        type=Path,
        help="successful report produced by selftest_polyglot.py",
    )
    selftest.add_argument(
        "--skip-selftest",
        action="store_true",
        help="development only: run with an unverified harness",
    )
    add_context_arguments(parser)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip tasks already recorded for this quant and run-index in --out",
    )
    parser.add_argument(
        "--context-padding",
        type=int,
        default=0,
        help="tokens of unrelated real code prepended to each task prompt, to measure "
        "usable context as opposed to the hostable context of section 12.5",
    )
    parser.add_argument(
        "--transcripts",
        type=Path,
        help="directory for raw Codex event streams (default: <out dir>/transcripts)",
    )
    parser.add_argument(
        "--no-transcripts",
        action="store_true",
        help="do not persist raw Codex event streams",
    )
    parser.add_argument(
        "--metrics-url",
        default=DEFAULT_SERVER_URL,
        help="llama-server base URL scraped before and after each task for "
        "per-task prefill/decode/cache counters; empty string disables",
    )
    parser.add_argument("--jobs", type=int, default=1, help="concurrent task workers")
    parser.add_argument(
        "--timeout",
        type=float,
        default=2400,
        help="total seconds allowed per task; must clear the slowest quant's decode "
        "rate or the cut-off itself becomes the difference between quants",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="copy tasks and run tests, but skip every Codex invocation",
    )
    return parser


def validate_selftest_report(path: Path, tasks: list[dict[str, Any]]) -> None:
    try:
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"POLYGLOT HARNESS SELF-TEST REFUSED: cannot read {path}: {exc}") from exc
    if not isinstance(report, dict) or not isinstance(report.get("tasks"), list):
        raise SystemExit("POLYGLOT HARNESS SELF-TEST REFUSED: report must contain a tasks array")

    manifest_ids: list[str] = []
    for task in tasks:
        if not isinstance(task, dict) or not task.get("task_id"):
            raise SystemExit("POLYGLOT HARNESS SELF-TEST REFUSED: manifest task_id is missing")
        manifest_ids.append(str(task["task_id"]))
    if len(set(manifest_ids)) != len(manifest_ids):
        raise SystemExit("POLYGLOT HARNESS SELF-TEST REFUSED: manifest task_id values are not unique")

    results: dict[str, dict[str, Any]] = {}
    for item in report["tasks"]:
        if not isinstance(item, dict) or not item.get("task_id"):
            raise SystemExit("POLYGLOT HARNESS SELF-TEST REFUSED: invalid task entry in report")
        task_id = str(item["task_id"])
        if task_id in results:
            raise SystemExit(
                f"POLYGLOT HARNESS SELF-TEST REFUSED: duplicate report task_id {task_id!r}"
            )
        results[task_id] = item

    failures: list[str] = []
    for task_id in manifest_ids:
        item = results.get(task_id)
        if item is None:
            failures.append(f"{task_id}: missing from report")
            continue
        failed_checks = [
            name
            for name in ("reference_passes", "stub_fails")
            if item.get(name) is not True
        ]
        if failed_checks:
            failures.append(f"{task_id}: failed {', '.join(failed_checks)}")
    if failures:
        details = "\n".join(f"  BROKEN {failure}" for failure in failures)
        raise SystemExit(f"POLYGLOT HARNESS SELF-TEST REFUSED:\n{details}")


def completed_task_ids(out: Path, quant: str, run_index: int) -> set[str]:
    """Task ids already recorded for this exact leg.

    Scoped by quant and run_index because one output file holds every leg; matching
    on task_id alone would treat a different quant's result as this leg's work.
    """

    done: set[str] = set()
    if not out.exists():
        return done
    for line in out.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # a torn last line from a killed writer
            continue
        if row.get("quant") == quant and row.get("run_index") == run_index:
            done.add(row.get("task_id"))
    return done


def run_leg(
    tasks: list[dict[str, Any]],
    worker: "Callable[[dict[str, Any]], TaskResult]",
    record: "Callable[[TaskResult], None]",
    metrics_url: str,
    jobs: int,
) -> int:
    """Run the tasks, checking the endpoint between them.

    The check is between tasks, not just at startup: when the server died at task 8 of 40
    the loop ran on and wrote 33 rows with null token counts blaming the model, exited 0,
    and the driver's EXIT trap destroyed the box holding the server log.
    """

    aborted = threading.Event()

    def gated(task: dict[str, Any]) -> TaskResult:
        task_id = (task.get("task_id") or task.get("instance_id")) if isinstance(task, dict) else None
        if aborted.is_set():
            raise LegAborted(f"skipped {task_id}")
        if metrics_url and not server_alive(metrics_url):
            aborted.set()
            raise LegAborted(
                f"the server at {metrics_url} stopped answering before {task_id}. "
                f"Rows already written are valid; the rest of this leg is not. "
                f"Re-run with --resume once the server is back."
            )
        return worker(task)

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        try:
            for result in executor.map(gated, tasks):
                record(result)
        except LegAborted as exc:
            aborted.set()
            print(f"LEG ABORTED: {exc}", file=sys.stderr, flush=True)
            return 5
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    tasks = load_manifest(args.manifest)
    if args.skip_selftest:
        print(
            "*** WARNING: POLYGLOT HARNESS SELF-TEST SKIPPED; RESULTS ARE UNVERIFIED ***",
            file=sys.stderr,
        )
        harness_verified = False
    else:
        if args.selftest_report is None:
            raise SystemExit(
                "POLYGLOT HARNESS SELF-TEST REQUIRED: pass --selftest-report PATH "
                "or --skip-selftest for development only"
            )
        validate_selftest_report(args.selftest_report, tasks)
        harness_verified = True
    ctx = make_context(args)
    if args.resume:
        done = completed_task_ids(Path(args.out), args.quant, args.run_index)
        tasks = [t for t in tasks if t.get("task_id") not in done]
        print(f"resuming: {len(done)} task(s) already recorded, {len(tasks)} to run", flush=True)
        if not tasks:
            # Exiting 0 here reads as a completed leg. A driver that just rented a box,
            # downloaded the weights and served the model then tears it down reporting
            # success, having measured nothing -- which is what a spent run-index does,
            # including one spent by an earlier *void* attempt.
            raise SystemExit(
                f"NOTHING TO RUN: every task is already recorded for quant "
                f"{args.quant!r} at run-index {args.run_index}. Use a new --run-index "
                f"for a new configuration, or remove the rows if that leg was void."
            )
    # --dry-run must stay offline: it is the no-GPU harness check. External lanes
    # never touch llama-server, so scraping it would attribute local traffic to them.
    local_lane = args.lane in {"qwen", "quant"}
    # The probe and the runner must interrogate the same endpoint. When they diverged --
    # tunnel on 8010, Codex pinned to 8000 -- the fail-closed quant guard confirmed a
    # healthy server on 8010 while every one of 40 tasks died calling 8000.
    if local_lane and not args.dry_run:
        codex_url = codex_base_url()
        if codex_url is None:
            raise SystemExit(
                "cannot read Codex's config to learn which endpoint it will call; "
                "the server probe would be verifying an endpoint nothing uses"
            )
        if args.metrics_url and args.metrics_url.rstrip("/") != codex_url.rstrip("/"):
            raise SystemExit(
                f"ENDPOINT MISMATCH: Codex calls {codex_url} but --metrics-url is "
                f"{args.metrics_url}. Tunnel to the port Codex is configured for, or "
                "change the provider's base_url; do not measure one and serve the other."
            )
        args.metrics_url = args.metrics_url or codex_url
    # Telemetry may be switched off; the identity guard may not. Emptying
    # --metrics-url used to skip apply_server_probe too, so three legs ran with the
    # fail-closed wrong-quant guard silently off.
    metrics_url = args.metrics_url if (local_lane and not args.dry_run) else ""
    if local_lane and not args.dry_run:
        ctx = apply_server_probe(ctx, args.metrics_url or DEFAULT_SERVER_URL)

    transcript_root = args.transcripts
    if transcript_root is None and not args.no_transcripts:
        transcript_root = Path(args.out).resolve().parent / "transcripts"

    def record(result: TaskResult) -> None:
        emit(ctx, result.task_id, args.run_index, **result.measures)

    def worker(task: dict[str, Any]) -> TaskResult:
        transcript = None
        if transcript_root is not None:
            task_id = str(task.get("task_id", "unknown")) if isinstance(task, dict) else "unknown"
            safe = task_id.replace("/", "__")
            transcript = (transcript_root
                          / f"{args.quant}__run{args.run_index}__{safe}.jsonl")
        return run_task(
            task,
            args.manifest.resolve().parent,
            args.lane,
            args.reasoning_effort,
            args.timeout,
            args.dry_run,
            harness_verified,
            transcript,
            metrics_url,
            args.context_padding,
        )

    return run_leg(tasks, worker, record, metrics_url, args.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
