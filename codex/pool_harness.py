#!/usr/bin/env python3
"""DO NOT MERGE. Parent-only AR pool A/B; one frozen binary, environment-only arms."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import statistics
import threading
import time

from attrib import analyze
from gpu_harness import Harness, INPUTS, MODEL, MODEL_SHA, ROOT, VOCAB, command, frozen, identity, sha

ORDER = (2, 8, 8, 2)
COMMON_ENV = {"LLAMA_ARG_LOG_VERBOSITY": "4"}
FALSIFICATION = (
    "If AR waits fall substantially while V-to-V time and paired-GPU kernel-boundary idle do not improve beyond boot variation, "
    "slot waits are primarily backpressure on active GPU work, not the dominant removable gap. "
    "If waits move to llama_synchronize or submission spacing stays unchanged, this experiment demonstrates no whole-loop capture saving. "
    "If waits do not fall, effective look-ahead did not change: inconclusive. "
    "Unchanged host gaps alone cannot falsify capture's separate launch/submission-overhead mechanism."
)


def check_identity(results):
    for case in ("tool", "think"):
        reference = [identity(r) for r in results[0]["records"][case]]
        for boot in results[1:]:
            if [identity(r) for r in boot["records"][case]] != reference:
                raise ValueError("completion/authoritative counter identity failed: " + case)


def trace_summary(attrib):
    waits = attrib["ar_wait_histogram"]["by_phase_device"]
    rounds = attrib["rounds_M"]
    gaps = [t for t in attrib["tables"] if t["window"] == "gpu_kernel_boundary"]
    return {"ar_waits_per_round_D": sum(w["waits_M"] for w in waits) / rounds,
            "ar_wait_host_us_per_round_D": sum(w["total_wait_us_D"] for w in waits) / rounds,
            "sync_host_us_per_round_D": sum(c["host_us"] for c in attrib["counters"] if c["kind"] == "sync") / rounds,
            "paired_gpu_boundary_idle_us_per_round_D": sum(t["device_idle_us"] for t in gaps) if gaps else None}


def summary(results):
    check_identity(results)
    rows = []
    for boot in results:
        for case, records in boot["records"].items():
            rounds = sum(r["metrics"]["drafts"] for r in records)
            if rounds <= 0 or any(v < 0 or not float(v).is_integer() for r in records for v in r["metrics"].values()):
                raise ValueError("invalid authoritative MTP counters")
            ms = sum(r["response"]["timings"]["predicted_ms"] for r in records)
            if ms <= 0:
                raise ValueError("invalid predicted_ms")
            counters = {k: sum(r["metrics"][k] for r in records) for k in ("drafts", "draft_tokens", "accepted_tokens")}
            hist = boot.get("attrib", {}).get(case)
            rows.append({"boot": boot["boot"], "pool": boot["pool"], "trace": boot["trace"], "case": case,
                         "rounds_M": rounds, "counters_M": counters, "predicted_ms_M": ms, "ms_per_round_D": ms / rounds,
                         "acceptance_D": counters["accepted_tokens"] / counters["draft_tokens"] if counters["draft_tokens"] else 0,
                         "sampled_peak_MiB_M": boot["sampled_peak_MiB_M"],
                         "steady_rounds_M": hist["rounds_M"] if hist else None,
                         "median_V_to_V_us_M": hist["median_round_us_M"] if hist else None,
                         "mechanisms": trace_summary(hist) if hist else None,
                         "ar_wait_histogram": hist["ar_wait_histogram"] if hist else None,
                         "sync": [c for c in hist["counters"] if c["kind"] == "sync"] if hist else None,
                         "kernel_boundaries": [t for t in hist["tables"] if t["window"] == "gpu_kernel_boundary"] if hist else None})
    comparisons = []
    for trace in (False, True):
        for case in ("tool", "think"):
            values = {pool: [r["ms_per_round_D"] for r in rows if r["trace"] == trace and r["case"] == case and r["pool"] == pool]
                      for pool in (2, 8)}
            if not all(len(v) == 2 for v in values.values()):
                raise ValueError("missing interleaved boots")
            comparisons.append({"trace": trace, "case": case,
                                "pool8_minus_pool2_ms_per_round_D": statistics.mean(values[8]) - statistics.mean(values[2]),
                                "within_arm_boot_range_ms_per_round_D": {str(p): max(v) - min(v) for p, v in values.items()}})
    for comparison in comparisons:
        if comparison["trace"]:
            selected = [r for r in rows if r["trace"] and r["case"] == comparison["case"]]
            if all(r["mechanisms"] is not None for r in selected):
                comparison["pool8_minus_pool2_mechanisms_D"] = {
                    key: statistics.mean(r["mechanisms"][key] for r in selected if r["pool"] == 8)
                         - statistics.mean(r["mechanisms"][key] for r in selected if r["pool"] == 2)
                    for key in selected[0]["mechanisms"] if all(r["mechanisms"][key] is not None for r in selected)}
                comparison["pool8_minus_pool2_median_V_to_V_us_D"] = (
                    statistics.mean(r["median_V_to_V_us_M"] for r in selected if r["pool"] == 8)
                    - statistics.mean(r["median_V_to_V_us_M"] for r in selected if r["pool"] == 2))
    return {"identity": "PASS: matched completion messages and authoritative drafts/draft_tokens/accepted_tokens",
            "per_boot": rows, "comparisons": comparisons, "falsification_rule": FALSIFICATION,
            "limits": "Trace-off predicted_ms/drafts is the net request period proxy, including ramp/final rounds. Trace-on attrib.py reports trimmed steady V-to-V and mechanisms. Wait time is not GPU idle. Sampled memory can miss transients. No automatic speed or capture verdict."}


class PoolHarness(Harness):
    def __init__(self, args, out):
        super().__init__(args, out)
        # Backend INFO requires trace verbosity in common_log_default_callback.
        self.env.update(COMMON_ENV)
        self.monitor_stop = threading.Event()
        self.monitor = None
        self.memory = []
        self.monitor_error = None
        # Keep other experiment overrides from leaking into either arm.
        for name in list(self.env):
            if name.startswith(("GGML_CUDA_AR_", "LLAMA_GRAPH_CACHE", "GGML_META_PLAN_CACHE")):
                self.env.pop(name)

    def sample_memory(self):
        try:
            while not self.monitor_stop.is_set():
                data = self.run(["nvidia-smi", "--id=" + self.env["CUDA_VISIBLE_DEVICES"],
                                 "--query-gpu=memory.used", "--format=csv,noheader,nounits"], limit=5, capture_output=True, text=True).stdout
                values = [int(v) for v in data.splitlines()]
                if len(values) != 2:
                    raise ValueError("memory query did not return a pair")
                self.memory.append(values)
                self.monitor_stop.wait(1)
        except Exception as error:
            self.monitor_error = error

    def idle(self):
        super().idle()
        self.memory = []
        self.monitor_error = None
        self.monitor_stop.clear()
        self.monitor = threading.Thread(target=self.sample_memory)
        self.monitor.start()

    def stop(self):
        super().stop()
        self.monitor_stop.set()
        if self.monitor:
            self.monitor.join(timeout=10)
            if self.monitor.is_alive():
                raise RuntimeError("memory monitor did not stop")
            self.monitor = None

    def pool_boot(self, label, pool, trace):
        self.env["GGML_CUDA_AR_POOL"] = str(pool)
        dest = self.boot(self.args.patch, label, trace=trace, nsys=trace)
        config = json.loads((dest / "command.json").read_text())
        config.update({name: self.env[name] for name in ("GGML_CUDA_AR_POOL", "LLAMA_ARG_LOG_VERBOSITY")})
        (dest / "command.json").write_text(json.dumps(config, indent=2))
        log = (dest / "server.log").read_text()
        pools = re.findall(r"initialized AllReduce pipeline:.*pool=(\d+)", log)
        if len(pools) != 2 or any(int(p) != pool for p in pools):
            raise ValueError(f"internal AR active-pool initialization not confirmed: expected target/draft pool={pool}, observed={pools}")
        return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--patch", type=Path, default=ROOT / "codex/frozen/ar-pool")
    p.add_argument("--out", type=Path)
    p.add_argument("--k", type=int, default=int(os.environ.get("K", "3")))
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--capture-requests", type=int, default=2)
    p.add_argument("--budget", type=int, default=2400)
    p.add_argument("--port", type=int, default=8398)
    args = p.parse_args()
    args.patch = args.patch.resolve()
    if not 1 <= args.k <= 16 or args.rounds < 40 or args.capture_requests < 1 or args.budget <= 120:
        raise ValueError("invalid K, minimum rounds, capture requests, or budget")
    manifest = frozen(args.patch)
    print(json.dumps({"trace_off_pool_order": ORDER, "separate_trace_on_pool_order": ORDER,
                      "source": (args.patch / "SOURCE").read_text().splitlines()[0], "manifest": manifest,
                      "command": command(args.patch / "bin/round-trace", args.k, args.port),
                      "arms": [{"GGML_CUDA_AR_POOL": str(p)} for p in (2, 8)],
                      "common_env": COMMON_ENV,
                      "requests": "each boot: warmup, tool, think; trace-on: fixed matched tool/think repeats under nsys",
                      "capture_requests_per_case": args.capture_requests, "minimum_steady_rounds": args.rounds,
                      "budget_s_G": args.budget, "falsification_rule": FALSIFICATION}, indent=2))
    if not args.execute:
        return
    if os.environ.get("RUN_GPU") != "1" or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("parent execute needs RUN_GPU=1 and selected CUDA_VISIBLE_DEVICES")
    out = (args.out or ROOT / "codex/pool-runs" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("OUT must stay inside this worktree")
    out.mkdir(parents=True, exist_ok=False)
    h = PoolHarness(args, out)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if sha(MODEL) != MODEL_SHA:
            raise ValueError("model SHA changed")
        pins = {str(f): sha(f) for f in [VOCAB, *INPUTS.glob("*.json"), INPUTS / "chat-template.jinja"]}
        (out / "inputs.json").write_text(json.dumps(pins, indent=2))
        results = []
        for trace in (False, True):
            for index, pool in enumerate(ORDER, 1):
                label = f"{'trace' if trace else 'off'}-{index}-pool{pool}"
                dest = h.pool_boot(label, pool, trace)
                result = {"boot": label, "pool": pool, "trace": trace, "records": {}, "attrib": {}}
                for case in ("tool", "think"):
                    result["records"][case] = [h.request(dest, case, 1)]
                if trace:
                    h.run(["nsys", "start", "--session=" + h.session, "--sample=none", "--cpuctxsw=none", "-o", str(dest / "capture")])
                    h.capturing = True
                    for case in ("tool", "think"):
                        result["records"][case] += [h.request(dest, case, n + 100) for n in range(args.capture_requests)]
                    h.run(["nsys", "stop", "--session=" + h.session], limit=120)
                    h.capturing = False
                h.stop()
                h.xid()
                frozen(dest)
                if h.monitor_error or not h.memory:
                    raise RuntimeError("memory sampling failed") from h.monitor_error
                (dest / "memory.json").write_text(json.dumps(h.memory))
                result["sampled_peak_MiB_M"] = [max(v[d] for v in h.memory) for d in (0, 1)]
                if trace:
                    h.run(["nsys", "export", "--type=sqlite", "--force-overwrite=true", "-o", str(dest / "capture.sqlite"), str(dest / "capture.nsys-rep")], limit=180)
                    rows = [json.loads(line) for line in (dest / "rounds.jsonl").read_text().splitlines()]
                    for case in ("tool", "think"):
                        capture = result["records"][case][1:]
                        selected = [r for r in rows if any(rec["begin_ns"] <= r["phases"][0]["begin_ns"] <= rec["end_ns"] for rec in capture)]
                        attrib = analyze(selected, args.k, skip=3, min_rounds=args.rounds, sqlite=dest / "capture.sqlite")
                        if attrib["nsys"]["gap_samples_M"] < args.rounds * (args.k + 2) or not attrib["ar_wait_histogram"]["by_phase_device"]:
                            raise ValueError("insufficient paired-GPU gaps or missing AR waits")
                        result["attrib"][case] = attrib
                        (dest / f"{case}-capture.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
                        (dest / f"{case}-attrib.json").write_text(json.dumps(attrib, indent=2))
                results.append(result)
                (out / "results.json").write_text(json.dumps(results, indent=2))
        # Compare all first matched requests and each fixed capture separately.
        check_identity([{**r, "records": {c: v[:1] for c, v in r["records"].items()}} for r in results])
        check_identity([{**r, "records": {c: v[1:] for c, v in r["records"].items()}} for r in results if r["trace"]])
        net = [{**r, "records": {c: v[:1] for c, v in r["records"].items()}} for r in results]
        report = summary(net)
        (out / "summary.json").write_text(json.dumps(report, indent=2))
        (out / "summary.txt").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if pins != {path: sha(Path(path)) for path in pins} or frozen(args.patch) != manifest:
            raise ValueError("frozen binary or inputs changed during run")
        if time.monotonic() >= h.deadline:
            raise TimeoutError("queue wall budget exceeded")
    finally:
        try:
            if h.capturing and h.session:
                h.run(["nsys", "stop", "--session=" + h.session], limit=15)
        finally:
            h.stop()


if __name__ == "__main__":
    main()
