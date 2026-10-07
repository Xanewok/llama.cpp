"""Trace writer latency: the following V begin minus the preceding next_begin stamp."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile

RECORDED = Path("/home/xanewok/repos/qwen38-eval/results/2026-10-05-stack6-depth-j5q/full-a1/1-s6/round-trace.jsonl")


def pairs(path, lo=0, hi=2**63 - 1, k=5, limit=None):
    previous = {}
    result = []
    with Path(path).open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            phases = row.get("phases", [])
            if not phases:
                continue
            tid = row["tid"]
            prev = previous.get(tid)
            previous[tid] = row
            if prev is None or row["round"] != prev["round"] + 1:
                continue
            pp = prev["phases"]
            if [p["phase"] for p in pp] != ["V", "C", *[f"D{i}" for i in range(1, k + 1)]]:
                continue
            if pp[0]["n_tokens"] != k + 1 or phases[0]["phase"] != "V":
                continue
            if not lo <= pp[0]["begin_ns"] < phases[0]["begin_ns"] <= hi:
                continue
            stamp = prev["next_begin_ns"]
            if not stamp:
                continue
            gap = phases[0]["begin_ns"] - stamp
            if gap < 0:
                raise ValueError("next_begin is later than the following V begin")
            result.append(gap / 1e6)
            if limit and len(result) >= limit:
                break
    return result


def fixtures():
    # Reuse the harness's recorded SSE, memory, journal and identity negative controls.
    from harness import depth
    depth.fixture_test("/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures")
    gaps = pairs(RECORDED, limit=64)
    if len(gaps) != 64 or not 0.2 < statistics.median(gaps) < 3:
        raise ValueError("recorded stack6 trace did not reproduce the writer gap")
    from harness import lib
    # A bounded sample of the actual trace also exercises counter parsing and incomplete-line handling.
    with RECORDED.open() as stream:
        rows = [json.loads(next(stream)) for _ in range(96)]
    lo, hi = rows[0]["phases"][0]["begin_ns"], rows[-1]["phases"][0]["begin_ns"]
    with tempfile.TemporaryDirectory(prefix="trace-fixture-", dir=Path(__file__).resolve().parent) as directory:
        sample = Path(directory) / "round-trace.jsonl"
        sample.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{incomplete\n')
        counts = lib.trace_window(sample, lo, hi, ["input", "ar_call"])
        if counts["rounds"] != len(rows) or counts["input"]["count"] <= 0 or counts["ar_call"]["count"] <= 0:
            raise ValueError("recorded counter parser lost rounds or counter groups")
        clean = pairs(sample)
        if len(clean) < 40:
            raise ValueError("recorded sample has too few writer pairs")
        broken = json.loads(json.dumps(rows))
        for row in broken:
            row["next_begin_ns"] = 0
        sample.write_text("".join(json.dumps(row) + "\n" for row in broken))
        if pairs(sample):
            raise ValueError("missing-next-begin negative control escaped")
    print(json.dumps({"recorded_trace": str(RECORDED), "pairs_M": len(gaps), "median_ms_M": statistics.median(gaps)}))


def analyze(experiment, out, canary):
    from harness import lib
    x = lib.read_experiment(experiment)
    out = Path(out)
    minimum = x["depth"]["min_steady_intervals"]
    results = []
    depths = x["canary"]["depths"] if canary else x["depth"]["depths"]
    for boot in sorted(p for p in out.iterdir() if p.is_dir() and p.name.split("-", 1)[0].isdigit()):
        arm = boot.name.split("-", 1)[1]
        for d in depths:
            record = json.loads((boot / f"depth-{d}.json").read_text())
            gaps = pairs(boot / "round-trace.jsonl", *record["mono_ns"], k=x["depth"]["k"])
            row = {"boot": boot.name, "arm": arm, "depth": d, "pairs_M": len(gaps), "valid": len(gaps) >= minimum}
            if gaps:
                row |= {"median_ms_M": statistics.median(gaps), "mean_ms_M": statistics.mean(gaps)}
            results.append(row)
    (out / "writer-gap.json").write_text(json.dumps(results, indent=2) + "\n")
    if not results or any(not r["valid"] for r in results):
        print("writer gap inconclusive: too few complete trace pairs", file=sys.stderr)
        return 5
    # Canaries exercise the same async latency gate; the full run also reports the paired sync control.
    if any(r["arm"] == "patch" and r["median_ms_M"] > 0.1 for r in results):
        print("writer async latency failed: median begin(V) gap exceeds 0.1 ms", file=sys.stderr)
        return 8
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", action="store_true")
    p.add_argument("--experiment", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--canary", action="store_true")
    a = p.parse_args()
    if a.fixtures:
        fixtures()
        return 0
    return analyze(a.experiment, a.out, a.canary)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print(f"writer gap parser failed: {error}", file=sys.stderr)
        sys.exit(8)
