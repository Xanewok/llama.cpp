"""Build-window CPU validation; run only through cpugate after the measure window."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("build", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    def run(variant, mode, label, sync=False, disabled=False, expect=0):
        dest = a.out / f"{label}.jsonl"
        if dest.exists():
            raise RuntimeError(f"output exists: {dest}")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GGML_ROUND_TRACE_SYNC="1" if sync else "0")
        env.pop("LLAMA_ROUND_TRACE", None)
        if not disabled:
            env["LLAMA_ROUND_TRACE"] = str(dest.resolve())
        r = subprocess.run([str((a.build / f"trace-test-{variant}").resolve()), mode], env=env,
                           capture_output=True, text=True, timeout=120)
        (a.out / f"{label}.log").write_text(r.stdout + r.stderr)
        if r.returncode != expect:
            raise RuntimeError(f"{label}: rc {r.returncode}, expected {expect}: {r.stderr}")
        return dest, r

    old, _ = run("old", "--emit", "old")
    sync, _ = run("patched", "--emit", "sync", sync=True)
    async_, _ = run("patched", "--emit", "async")
    if not old.read_bytes() == sync.read_bytes() == async_.read_bytes():
        raise RuntimeError("byte identity failed: old / sync / async")
    rows = [json.loads(line) for line in old.read_text().splitlines()]
    if len(rows) != 17 or [r["tid"] for r in rows] != [200] * 8 + [201] * 3 + [202] * 3 + [203] * 3:
        raise RuntimeError("record order or producer IDs changed")
    counter = rows[0]["phases"][0]["counters"][0]
    if counter != {"kind": "alpha", "device": 0, "count": 19, "bytes": 409, "host_us": 0.08}:
        raise RuntimeError(f"overlap/group control failed: {counter}")
    _, mutant = run("mutant", "--emit", "mutant", expect=15)
    if "flush visibility: expected 5 records, got 4" not in mutant.stderr:
        raise RuntimeError("negative control did not fail the final-round visibility check")
    for sync_mode in (True, False):
        label = "sync" if sync_mode else "async"
        run("patched", "--ownership", f"{label}-ownership", sync=sync_mode)
        run("patched", "--stress", f"{label}-stress", sync=sync_mode)
        run("patched", "--exit-tail", f"{label}-exit", sync=sync_mode)
    if (a.out / "sync-exit.jsonl").read_bytes() != (a.out / "async-exit.jsonl").read_bytes():
        raise RuntimeError("exit drain byte identity failed")
    if len((a.out / "async-exit.jsonl").read_text().splitlines()) != 2:
        raise RuntimeError("TLS/process exit lost a tail")
    run("patched", "--disabled", "disabled", disabled=True)

    timing = {"sync": [], "async": []}
    sizes = []
    for i, arm in enumerate(("sync", "async", "async", "sync", "async", "sync", "sync", "async")):
        dest, r = run("patched", "--timing", f"timing-{i}-{arm}", sync=arm == "sync")
        timing[arm].append(json.loads(r.stdout)["begin_v_median_ns_M"])
        large = [len(line.encode()) + 1 for line in dest.read_text().splitlines() if len(json.loads(line)["events"]) == 1800]
        if len(large) != 16 or min(large) < 200000:
            raise RuntimeError("timing records are smaller than 200 KB")
        sizes.extend(large)
    ratio = statistics.median(timing["async"]) / statistics.median(timing["sync"])
    root = Path(__file__).resolve().parent.parent
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                     env=dict(os.environ, GIT_NO_LAZY_FETCH="1", CUDA_VISIBLE_DEVICES=""), text=True).strip()
    evidence = {"commit": commit,
                "trace_sha256": hashlib.sha256((root / "ggml/src/ggml-round-trace.cpp").read_bytes()).hexdigest(),
                "test_sha256": hashlib.sha256((root / "codex/test-round-trace.cpp").read_bytes()).hexdigest(),
                "byte_identity": "old == sync == async", "negative_control": "missing final enqueue fails rc 15",
                "exit_drain": "main and joined producer TLS tails retained", "stress_records_per_arm": 768,
                "record_sha256": hashlib.sha256(old.read_bytes()).hexdigest(), "timing_ns_M": timing,
                "async_over_sync_D": ratio, "record_bytes_median_M": statistics.median(sizes)}
    (a.out / "cpu-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    if ratio >= 0.2:
        print("async begin(V) >= 20 percent of sync", file=sys.stderr)
        return 22
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(21)
