#!/usr/bin/env python3
"""Create SWE-bench prediction patches without running evaluation containers."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

from scripts.instruments.emit import add_context_arguments, emit, invoke_codex, make_context
from scripts.instruments.run_polyglot import DEFAULT_SERVER_URL, completed_task_ids, run_leg


@dataclasses.dataclass(frozen=True, slots=True)
class InstanceResult:
    task_id: str
    patch: str
    measures: dict[str, Any]
    # The run never reached the model. Kept off the row -- it is not a measurement --
    # and used to withhold the prediction, so the harness reports the instance
    # incomplete instead of binning a patch that was never attempted.
    infra_failure: bool = False


def load_instances(path: Path) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            instances.append(value)
    return instances


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired("task", 0)
    return remaining


def _git(args: list[str], cwd: Path | None, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )


def _cache_name(repo: str) -> str:
    stem = Path(repo.rstrip("/")).name.removesuffix(".git") or "repo"
    safe_stem = "".join(character if character.isalnum() else "-" for character in stem)
    digest = hashlib.sha256(repo.encode("utf-8")).hexdigest()[:16]
    return f"{safe_stem}-{digest}.git"


def _ensure_cache(repo: str, cache_dir: Path, deadline: float) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / _cache_name(repo)
    lock_path = cache_dir / (_cache_name(repo) + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if not cached.exists():
                with tempfile.TemporaryDirectory(prefix="clone-", dir=cache_dir) as temp_dir:
                    staged = Path(temp_dir) / "repo.git"
                    _git(["clone", "--mirror", repo, str(staged)], None, _remaining(deadline))
                    os.replace(staged, cached)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return cached


def _resolve_repo(repo: str, instances_dir: Path) -> str:
    candidate = Path(repo)
    if not candidate.is_absolute() and (instances_dir / candidate).exists():
        return str((instances_dir / candidate).resolve())
    if "://" not in repo and repo.count("/") == 1:
        return f"https://github.com/{repo.removesuffix('.git')}.git"
    return repo


def run_instance(
    instance: dict[str, Any],
    instances_dir: Path,
    cache_dir: Path,
    lane: str,
    reasoning_effort: str,
    timeout: float,
    dry_run: bool,
) -> InstanceResult:
    task_id = "<missing-instance-id>"
    wall_seconds = 0.0
    deadline = time.monotonic() + timeout
    try:
        if not isinstance(instance, dict):
            raise ValueError("each instances line must be a JSON object")
        task_id = str(instance.get("instance_id", ""))
        for field in ("instance_id", "repo", "base_commit", "problem_statement"):
            if field not in instance:
                raise ValueError(f"instance is missing required field {field!r}")
        if not task_id:
            raise ValueError("instance_id must be non-empty")
        repo = _resolve_repo(str(instance["repo"]), instances_dir)
        base_commit = str(instance["base_commit"])
        cached = _ensure_cache(repo, cache_dir, deadline)

        with tempfile.TemporaryDirectory(prefix="swebench-") as temp_dir:
            workdir = Path(temp_dir) / "repo"
            _git(["clone", "--shared", str(cached), str(workdir)], None, _remaining(deadline))
            _git(["checkout", "--detach", base_commit], workdir, _remaining(deadline))
            _git(["clean", "-xfd"], workdir, _remaining(deadline))

            stats = None
            if not dry_run:
                invocation_started = time.monotonic()
                try:
                    stats, _ = invoke_codex(
                        lane,
                        reasoning_effort,
                        str(instance["problem_statement"]),
                        workdir,
                        _remaining(deadline),
                    )
                finally:
                    wall_seconds += time.monotonic() - invocation_started

            _git(["add", "-N", "--all"], workdir, _remaining(deadline))
            diff = _git(
                ["diff", "--binary", base_commit, "--"], workdir, _remaining(deadline)
            ).stdout
            success = bool(diff)
            # A patch is not a solve. Only grade_swebench.py, reading the evaluation
            # containers' verdict, may set these -- anything else reads "the model
            # edited a file" as "the tests pass", which on this dataset is wrong about
            # half the time even for a strong model.
            return InstanceResult(
                task_id,
                diff,
                {
                    "task_success": None,
                    "tests_passed": None,
                    "failed_test_cycles": 0,
                    "agent_turns": stats.agent_turns if stats else None,
                    "tool_calls": stats.tool_calls if stats else None,
                    "prompt_tokens": stats.prompt_tokens if stats else None,
                    "completion_tokens": stats.completion_tokens if stats else None,
                    "wall_seconds": round(wall_seconds, 6),
                    "notes": (
                        None
                        if success
                        else "dry-run produced an empty diff"
                        if dry_run
                        else "Codex produced an empty diff"
                    ),
                },
            )
    except subprocess.TimeoutExpired:
        return InstanceResult(
            task_id or "<missing-instance-id>",
            "",
            {
                "task_success": False,
                "tests_passed": None,
                "failed_test_cycles": 0,
                "agent_turns": None,
                "tool_calls": None,
                "wall_seconds": round(wall_seconds, 6),
                "notes": f"task timed out after {timeout:g}s",
            },
        )
    except Exception as exc:
        note = (
            f"subprocess exited with status {exc.returncode}"
            if isinstance(exc, subprocess.CalledProcessError)
            else f"{type(exc).__name__}: {exc}"
        )
        # Not False: a clone that failed or a disk that filled says nothing about the
        # quant, and counting it as an unsolved instance charges the arm that happened
        # to be running at the time. The timeout above stays False because running out
        # of the configured budget *is* an outcome of the configuration.
        return InstanceResult(
            task_id or "<missing-instance-id>",
            "",
            {
                "task_success": None,
                "tests_passed": None,
                "failed_test_cycles": 0,
                "agent_turns": None,
                "tool_calls": None,
                "wall_seconds": round(wall_seconds, 6),
                "notes": note,
            },
            infra_failure=True,
        )


def infra_failed_ids(out: Path, quant: str, run_index: int) -> set[str]:
    """Instances this leg recorded without ever reaching the model.

    --resume must retry these. They were written to keep the failure visible, not because
    anything was measured, and skipping them turns one bad clone into a permanent hole in
    the arm -- which a paired design then drops from *both* arms.
    """
    failed: set[str] = set()
    if not out.exists():
        return failed
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # a torn last line from a killed writer
            continue
        if (row.get("quant") == quant and row.get("run_index") == run_index
                and row.get("task_success") is None
                and not str(row.get("notes") or "").startswith("Codex produced")):
            failed.add(row.get("task_id"))
    return failed


def _append_prediction(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SWE-bench-format prediction patches only. This script does not run "
            "SWE-bench evaluation containers; evaluate the predictions separately."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--instances",
        required=True,
        type=Path,
        help="JSONL input with instance_id, repo, base_commit, and problem_statement",
    )
    add_context_arguments(parser)
    parser.add_argument(
        "--predictions", required=True, type=Path, help="append-only SWE-bench prediction JSONL"
    )
    parser.add_argument(
        "--repo-cache",
        type=Path,
        default=Path("./.repo-cache"),
        help="reusable bare repository cache shared across configurations",
    )
    parser.add_argument("--jobs", type=int, default=1, help="concurrent instance workers")
    parser.add_argument("--timeout", type=float, default=900, help="total seconds allowed per instance")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="prepare repos and produce empty-diff predictions, but skip Codex",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip instances already recorded for this quant and run-index in --out",
    )
    parser.add_argument(
        "--metrics-url",
        default=DEFAULT_SERVER_URL,
        help="endpoint checked between instances; empty disables the check",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    instances = load_instances(args.instances)
    ctx = make_context(args)
    if args.resume:
        done = completed_task_ids(Path(args.out), args.quant, args.run_index)
        done -= infra_failed_ids(Path(args.out), args.quant, args.run_index)
        instances = [i for i in instances if str(i.get("instance_id")) not in done]
        print(f"resuming: {len(done)} instance(s) already recorded, "
              f"{len(instances)} to run", flush=True)
        if not instances:
            # Exiting 0 here reads as a completed leg, having measured nothing.
            raise SystemExit(
                f"NOTHING TO RUN: every instance is already recorded for quant "
                f"{args.quant!r} at run-index {args.run_index}. Use a new --run-index "
                f"for a new configuration, or remove the rows if that leg was void."
            )

    def worker(instance: dict[str, Any]) -> InstanceResult:
        return run_instance(
            instance,
            args.instances.resolve().parent,
            args.repo_cache.resolve(),
            args.lane,
            args.reasoning_effort,
            args.timeout,
            args.dry_run,
        )

    def record(result: InstanceResult) -> None:
        # An instance that never reached the model has no patch to submit. Submitting the
        # empty one anyway puts it in the harness's empty_patch_ids, which grades as a
        # model failure -- charging a failed clone to the quant under test.
        if not result.infra_failure:
            _append_prediction(
                args.predictions,
                {
                    "instance_id": result.task_id,
                    # The harness names its report and its logs after this, so every leg
                    # needs its own: two quants writing `qwen3.8-27b` would grade the
                    # second against the first's report directory.
                    "model_name_or_path": f"{ctx.model}-{ctx.quant}-r{args.run_index}",
                    "model_patch": result.patch,
                },
            )
        emit(ctx, result.task_id, args.run_index, **result.measures)

    # --dry-run must stay offline: it is the no-GPU harness check. External lanes never
    # touch llama-server, so probing it would attribute local traffic to them.
    local_lane = args.lane in {"qwen", "quant"}
    metrics_url = args.metrics_url if (local_lane and not args.dry_run) else ""
    return run_leg(instances, worker, record, metrics_url, args.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
