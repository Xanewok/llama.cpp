"""Depth harness adapter: np1 acceptance equality and complete-round AllReduce gates."""
import copy
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, "/home/xanewok/repos/qwen38-eval")
from harness import depth, lib
from harness.__main__ import main

EXPERIMENT = Path(__file__).with_name("mtp-fuse-catchup.experiment.toml")
FIXTURE = Path("/home/xanewok/repos/qwen38-eval/results/2026-10-05-stack4-depth-j5q/canary-a1/1-patch")
EVALUATE = depth.evaluate_boot


def acceptance_settle(results, mode, arm, n, ident, multi=False):
    counts = ident[1]
    signature = [counts[k] for k in ("predicted_n", "draft_n", "draft_n_accepted")]
    if any(not isinstance(v, int) for v in signature) or not 0 <= signature[2] <= signature[1]:
        raise lib.Fail("invalid acceptance counters", "integrity")
    entry = results.setdefault(n, {"identity": None, "acceptance_counts": signature, "runs": []})
    if entry["acceptance_counts"] != signature:
        raise lib.Fail(f"acceptance equality failed at depth {n}: {entry['acceptance_counts']} vs {signature}", "integrity")
    return entry


def complete_rounds(path, expected, minimum, window=None):
    counts = []
    for line in Path(path).read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        phases = row.get("phases", [])
        names = [p.get("phase") for p in phases]
        shape = ["V"] + (["C"] if expected == 140 else []) + [f"D{i}" for i in range(1, 6)]
        if names != shape or phases[0].get("n_tokens") != 6 or any(p.get("result", -1) for p in phases):
            continue
        if window and not window[0] <= phases[0]["begin_ns"] <= window[1]:
            continue
        count = sum(c["count"] for p in phases for c in p["counters"] if c["kind"] == "ar_call")
        if count != expected:
            raise lib.Fail(f"forward-count proof failed: {count} calls, expected {expected}", "integrity")
        counts.append(count)
    if len(counts) < minimum:
        raise lib.Fail(f"forward-count proof inconclusive: {len(counts)} complete rounds < {minimum}", "insufficient_data")
    return {"complete_rounds_M": len(counts), "ar_calls_per_round_M": expected}


def evaluate(x, dest, arm, results, canary=False):
    local = copy.deepcopy(x)
    # The existing np1 evaluator refuses "off" before calling settle().
    local["depth"]["identity"] = "within-arm"
    failures = EVALUATE(local, dest, arm, results, canary)
    proofs = {}
    for n in depth.depths_of(x, canary):
        try:
            record = lib.read_json(dest / f"depth-{n}.json")
            proofs[n] = complete_rounds(dest / "round-trace.jsonl", 138 if arm == "patch" else 140,
                                        x["depth"]["min_steady_intervals"], record.get("mono_ns"))
        except lib.Fail as error:
            failures.append(error)
        except (OSError, ValueError, KeyError) as error:
            failures.append(lib.Fail(f"unreadable forward-count proof: {error}", "deterministic"))
    (dest / "forward-counts.json").write_text(json.dumps(proofs, indent=2))
    return failures


def fixtures():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    depth.fixture_test("/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures")
    record = lib.read_json(FIXTURE / "depth-8192.json")
    steady, ident = depth.gate({"k": 5, "min_predicted": 256, "min_steady_intervals": 40}, record, depth.first_prefill(8192))
    results = {}
    acceptance_settle(results, "off", "up", 8192, ident)
    acceptance_settle(results, "off", "patch", 8192, ident)
    wrong = copy.deepcopy(ident)
    wrong[1]["draft_n_accepted"] += 1
    try:
        acceptance_settle(results, "off", "patch", 8192, wrong)
    except lib.Fail:
        pass
    else:
        raise AssertionError("acceptance negative control escaped")
    trace = FIXTURE / "round-trace.jsonl"
    proof = complete_rounds(trace, 140, 40, record.get("mono_ns"))
    try:
        complete_rounds(trace, 138, 40, record.get("mono_ns"))
    except lib.Fail:
        pass
    else:
        raise AssertionError("eager trace escaped fused forward-count gate")
    print(json.dumps({"recorded_artifact": str(FIXTURE), "steady": steady, "proof": proof, "negative_controls": "PASS"}))
    return 0


def run(argv):
    if argv == ["--fixtures"]:
        return fixtures()
    if not EXPERIMENT.exists():
        raise lib.Fail("CPU identity failed; no frozen fusion build or queueable experiment was produced", "deterministic")
    if any(a in argv for a in ("--execute", "--canary")) and os.environ.get("GPUQ_ACTIVE") != "1":
        raise lib.Fail("GPU execution requires gpuq", "deterministic")
    depth.settle = acceptance_settle
    depth.evaluate_boot = evaluate
    return main(["depth", str(EXPERIMENT), *argv])


if __name__ == "__main__":
    sys.exit(lib.run_main(lambda: run(sys.argv[1:])))
