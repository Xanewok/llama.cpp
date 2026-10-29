#!/usr/bin/env python3
"""Attach the evaluation containers' verdict to a SWE-bench leg's rows.

run_swebench.py produces patches and leaves `task_success` null, because a patch is
not a solve. This reads the report the official harness writes and fills that in.

Nothing here decides whether a patch is correct. The harness runs the instance's own
FAIL_TO_PASS and PASS_TO_PASS tests inside the published image and says so; this maps
its answer onto the rows and, where it has no answer, leaves the row ungraded rather
than guessing. That distinction is the point: an image that would not build and a
patch that broke the tests are the same empty result to a caller that only counts
successes, and only one of them is the quant's fault.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

# The report's own partition of the instances it was given. `unresolved` and
# `error` overlap the two failure-classification lists, so those are checked first.
_RESOLVED = "resolved_ids"
_MODEL_FAILURE = ("unresolved_ids", "empty_patch_ids")
# Only confirmed environment faults. The harness separates these from
# `ambiguous_failure_ids` precisely so the ambiguous ones are "never mistaken for a
# confirmed environment fault" (swebench/harness/infra_failure.py) -- it keeps them in
# unresolved_ids on purpose, and excluding them here would discard failures the patch
# may well have caused.
_NO_VERDICT = ("infra_failure_ids", "error_ids", "incomplete_ids")


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or _RESOLVED not in report:
        raise SystemExit(f"{path} is not a SWE-bench run report")
    return report


def verdicts(report: dict[str, Any]) -> dict[str, tuple[bool | None, str | None]]:
    """Instance id -> (resolved, reason it has no verdict).

    A no-verdict class is checked before the failure classes it overlaps: the harness
    leaves an infra failure in `unresolved_ids` too, and reading that first would
    charge a container that never built to the model.
    """
    out: dict[str, tuple[bool | None, str | None]] = {}
    for key in _NO_VERDICT:
        for instance_id in report.get(key, []):
            out[instance_id] = (None, key.removesuffix("_ids"))
    for key in _MODEL_FAILURE:
        for instance_id in report.get(key, []):
            out.setdefault(instance_id, (False, None))
    for instance_id in report.get(_RESOLVED, []):
        # Resolved wins outright: the tests passed, whatever else was logged.
        out[instance_id] = (True, None)
    return out


def grade(rows: list[dict[str, Any]], report: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    table = verdicts(report)
    tally = {"resolved": 0, "unresolved": 0, "ungraded": 0}
    graded = []
    for row in rows:
        row = dict(row)
        resolved, reason = table.get(row.get("task_id"), (None, "absent from the report"))
        row["task_success"] = resolved
        row["tests_passed"] = resolved
        if resolved is None:
            note = f"ungraded: {reason}"
            row["notes"] = f"{row['notes']}; {note}" if row.get("notes") else note
            tally["ungraded"] += 1
        else:
            tally["resolved" if resolved else "unresolved"] += 1
        graded.append(row)
    return graded, tally


def run_harness(args: argparse.Namespace) -> Path:
    """Invoke the official harness, which needs docker and the swebench package.

    Shelled out rather than imported so this script stays stdlib-only and the harness
    can live in its own environment; --python names that interpreter.
    """
    # Scoped to what was submitted. The pinned export holds all 300 instances, and
    # without this the harness counts the 248 nobody asked for as incomplete -- which
    # buries a genuinely missing prediction in a pile of expected ones.
    submitted = sorted({
        json.loads(line)["instance_id"]
        for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    })
    if not submitted:
        raise SystemExit(f"{args.predictions} holds no predictions")
    command = [
        args.python, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", str(args.dataset),
        "--predictions_path", str(args.predictions),
        "--instance_ids", *submitted,
        "--run_id", args.run_id,
        "--max_workers", str(args.max_workers),
        "--report_dir", str(args.report_dir),
    ]
    print(" ".join(command), file=sys.stderr, flush=True)
    subprocess.run(command, check=True)
    reports = sorted(Path(args.report_dir).glob(f"*.{args.run_id}.json"))
    if len(reports) != 1:
        raise SystemExit(
            f"expected one report matching *.{args.run_id}.json in {args.report_dir}, "
            f"found {[p.name for p in reports]}"
        )
    return reports[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", required=True, type=Path, help="leg output from run_swebench.py")
    parser.add_argument("--quant", required=True, help="which leg in --rows to grade")
    parser.add_argument("--run-index", required=True, type=int)
    parser.add_argument(
        "--out", type=Path,
        help="graded copy (default: <rows stem>.graded.jsonl). The input is never "
             "rewritten: it is the append-only record of what the run measured.",
    )
    parser.add_argument("--report", type=Path, help="an existing harness report to ingest")
    parser.add_argument("--run", action="store_true", help="run the harness to produce one")
    parser.add_argument(
        "--dataset", type=Path,
        default=Path(".cache/subsets/SWE-bench--SWE-bench_Multilingual-"
                     "846e647b9f33c0b51b739d005d13d85493c9af09.jsonl"),
        help="full instance records; the subset in subsets/ carries no tests, so the "
             "harness needs the pinned canonical export",
    )
    parser.add_argument("--predictions", type=Path, help="prediction JSONL from the same leg")
    parser.add_argument("--run-id", help="harness run id; defaults to <quant>-r<run-index>")
    parser.add_argument("--report-dir", type=Path, default=Path("./.swebench-reports"))
    parser.add_argument("--python", default=sys.executable, help="interpreter holding swebench")
    parser.add_argument("--max-workers", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.report) == bool(args.run):
        raise SystemExit("pass exactly one of --report and --run")
    args.run_id = args.run_id or f"{args.quant}-r{args.run_index}"

    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    leg = [r for r in rows if r.get("quant") == args.quant and r.get("run_index") == args.run_index]
    if not leg:
        raise SystemExit(f"no rows for quant {args.quant!r} at run-index {args.run_index}")

    if args.run:
        if not args.predictions:
            raise SystemExit("--run needs --predictions")
        report_path = run_harness(args)
    else:
        report_path = args.report
    report = load_report(report_path)

    graded, tally = grade(leg, report)
    out = args.out or args.rows.with_suffix(".graded.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in graded:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False) + "\n")

    total = tally["resolved"] + tally["unresolved"]
    rate = f"{tally['resolved'] / total:.3f}" if total else "n/a"
    print(f"{out}: {tally['resolved']} resolved, {tally['unresolved']} unresolved, "
          f"{tally['ungraded']} ungraded -- pass rate {rate} over {total} graded")
    if tally["ungraded"]:
        # Loud, because the denominator is now smaller than the subset and a pass rate
        # over a shrunken denominator is not comparable with one over the full subset.
        print(f"WARNING: {tally['ungraded']} instance(s) have no verdict; compare quants "
              f"only on instances graded in every arm", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
