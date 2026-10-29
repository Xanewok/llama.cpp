#!/usr/bin/env python3
"""Verify that every Polyglot exercise distinguishes a solution from its stub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from scripts.instruments.run_polyglot import (
    _run_tests,
    load_manifest,
    prepare_test_harness,
    task_file_paths,
    task_test_command,
)


_OUTPUT_LIMIT = 8_000


def _captured_output(completed: subprocess.CompletedProcess[str]) -> str:
    output = completed.stdout + completed.stderr
    if len(output) <= _OUTPUT_LIMIT:
        return output
    return "[output truncated]\n" + output[-_OUTPUT_LIMIT:]


def _source_path(task: dict[str, Any], manifest_dir: Path) -> Path:
    source = Path(task["source_dir"])
    return source if source.is_absolute() else manifest_dir / source


def _overlay_reference(task: dict[str, Any], workdir: Path) -> None:
    examples = task_file_paths(task, workdir, "example_path", "example")
    solutions = task_file_paths(task, workdir, "solution_path", "solution")
    if len(examples) != len(solutions):
        matched: list[Path] = []
        for example in examples:
            same_suffix = [solution for solution in solutions if solution.suffix == example.suffix]
            if len(same_suffix) != 1 or same_suffix[0] in matched:
                raise ValueError(
                    "example_path and solution_path cannot be matched uniquely by suffix"
                )
            matched.append(same_suffix[0])
        solutions = matched
    for example, solution in zip(examples, solutions):
        if not example.is_file():
            raise FileNotFoundError(f"reference example does not exist: {example}")
        solution.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(example, solution)


def _check(
    task: dict[str, Any], manifest_dir: Path, timeout: float, reference: bool
) -> tuple[bool, dict[str, Any]]:
    try:
        command = task_test_command(task)
        source = _source_path(task, manifest_dir)
        with tempfile.TemporaryDirectory(prefix="polyglot-selftest-") as temp_dir:
            workdir = Path(temp_dir) / "exercise"
            shutil.copytree(source, workdir)
            if reference:
                _overlay_reference(task, workdir)
            replacements = prepare_test_harness(task, workdir)
            completed = _run_tests(command, workdir, timeout)
        check_passed = completed.returncode == 0 if reference else completed.returncode != 0
        return check_passed, {
            "returncode": completed.returncode,
            "javascript_tests_enabled": replacements,
            "output": _captured_output(completed),
        }
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        return False, {
            "returncode": None,
            "javascript_tests_enabled": None,
            "output": f"{type(exc).__name__}: {exc}",
        }


def check_task(
    task: dict[str, Any], manifest_dir: Path, timeout: float
) -> dict[str, Any]:
    task_id = (
        str(task.get("task_id", "<missing-task-id>"))
        if isinstance(task, dict)
        else "<missing-task-id>"
    )
    if not isinstance(task, dict):
        message = "ValueError: each manifest entry must be a JSON object"
        failed = {"returncode": None, "javascript_tests_enabled": None, "output": message}
        return {
            "task_id": task_id,
            "reference_passes": False,
            "stub_fails": False,
            "broken": True,
            "reference": failed,
            "stub": dict(failed),
        }
    missing = [
        field
        for field in ("task_id", "language", "source_dir", "test_cmd")
        if field not in task
    ]
    if missing or not task_id:
        message = f"ValueError: missing required field(s): {', '.join(missing) or 'task_id'}"
        failed = {"returncode": None, "javascript_tests_enabled": None, "output": message}
        return {
            "task_id": task_id or "<missing-task-id>",
            "reference_passes": False,
            "stub_fails": False,
            "broken": True,
            "reference": failed,
            "stub": dict(failed),
        }

    reference_passes, reference_details = _check(task, manifest_dir, timeout, True)
    stub_fails, stub_details = _check(task, manifest_dir, timeout, False)
    return {
        "task_id": task_id,
        "reference_passes": reference_passes,
        "stub_fails": stub_fails,
        "broken": not (reference_passes and stub_fails),
        "reference": reference_details,
        "stub": stub_details,
    }


def _diagnostic(result: dict[str, Any]) -> str:
    failures: list[str] = []
    if not result["reference_passes"]:
        failures.append(
            f"reference solution did not pass ({result['reference']['output'].strip() or 'no output'})"
        )
    if not result["stub_fails"]:
        returncode = result["stub"]["returncode"]
        if returncode == 0:
            failures.append("untouched stub passed (tests do not reject the stub)")
        else:
            failures.append(
                f"untouched stub check could not run ({result['stub']['output'].strip() or 'no output'})"
            )
    return f"BROKEN {result['task_id']}: " + "; ".join(failures)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Require the reference to pass and the untouched stub to fail for every task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="JSON self-test report")
    parser.add_argument("--timeout", type=float, default=300, help="seconds allowed per check")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    tasks = load_manifest(args.manifest)
    manifest_dir = args.manifest.resolve().parent
    results = [check_task(task, manifest_dir, args.timeout) for task in tasks]
    report = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "harness_verified": bool(results) and all(not result["broken"] for result in results),
        "tasks": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    broken = [result for result in results if result["broken"]]
    if broken:
        for result in broken:
            print(_diagnostic(result), file=sys.stderr)
        print(
            f"POLYGLOT HARNESS SELF-TEST FAILED: {len(broken)} of {len(results)} task(s) BROKEN",
            file=sys.stderr,
        )
        return 1
    print(f"POLYGLOT HARNESS VERIFIED: {len(results)} task(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
