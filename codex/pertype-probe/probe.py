#!/usr/bin/env python3
"""Reuse the recorded-artifact MMVQ harness with per-type freezes and head/tail cases."""
import importlib.util
from pathlib import Path
import signal
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location("mmvq_probe", ROOT / "codex/w2r4-general-probe/probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
probe.UP_PREFIX = "b6b1c7aae"
probe.FROZEN = {"up": ROOT / "codex/frozen/mmvq-w2r4-general", "patch": ROOT / "codex/frozen/mmvq-pertype"}
base_cases = probe.cases
base_selftest = probe.selftest
probe.STAGE_S = 60
canary = False


def cases(kind):
    if kind == "partial":
        return {(t, n, m, 512) for t in probe.TYPES for n in (5, 6, 7, 8) for m in ((1, 6, 13) if canary else range(1, 14))}
    result = base_cases(kind)
    if kind == "perf":
        result |= {("q6_K", n, 124160, 5120) for n in (5, 6, 7, 8)}
    return result


def selftest():
    probe.cases = base_cases
    try:
        base_selftest()
    finally:
        probe.cases = cases
    # The same recorded GPU SQL must also pass the head-geometry validator.
    raw = (probe.HERE / "fixtures/real-mmvq.sql").read_text()
    rows = probe.sql_rows(raw)
    row = next(r for r in rows if "gateup:q6_K:n6," in r["params"])
    original = next(line for line in raw.splitlines() if row["params"] in line)
    head = original.replace(row["params"], probe.case_params("q6_K", 6, 124160, 5120))
    assert len(probe.validate(head, "perf", "94cceb8b8", {("q6_K", 6, 124160, 5120)})) == 1
    assert all(len(probe.case_line(*c).split()) == 27 for c in cases("partial") | cases("perf"))


probe.cases = cases
probe.selftest = selftest


def main():
    global canary
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "canary", "run", "selftest", "gen"))
    parser.add_argument("--out", type=Path, default=ROOT / "codex/runs/pertype-preflight")
    parser.add_argument("--budget-s", type=int, default=3600)
    parser.add_argument("--only", choices=("all", "correctness", "perf"), default="all")
    args = parser.parse_args()
    canary = args.mode == "canary"
    if args.mode == "selftest":
        selftest()
    elif args.mode == "gen":
        probe.generate(args.out)
    else:
        probe.run(args.mode, args.out, args.budget_s, args.only)


if __name__ == "__main__":
    def stop(signum, frame):
        raise probe.Gate(7, "queue stopped the item")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        main()
    except probe.Gate as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(exc.code)
