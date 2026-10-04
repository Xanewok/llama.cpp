#!/usr/bin/env python3
"""Parent-only MMVQ queue items. Without --execute, read files and print a plan."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess

import gpu_harness as h

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ("base", "default", "w2r1", "w4r1", "w4r2", "w2r4", "w8r1")
TYPES = ("q5_K", "q6_K", "q4_K", "iq4_xs", "q8_0")
SHAPES = {"gate-up": (5120, 8704), "down": (8704, 5120), "qkv": (5120, 5120),
          "q": (5120, 6144), "attn-gate": (5120, 3072), "ssm-out": (3072, 5120), "head": (5120, 124160)}
MODEL = Path("/home/xanewok/models/Swift-1.5-Qwen3.8-27B-J5q-bf16.gguf")


def command(binary, k, port):
    return [str(binary), "-m", str(MODEL), "--alias", "local-benchmark", "-c", "32768", "-np", "1",
            "-ctk", "q8_0", "-ctv", "q8_0", "-ngl", "99", "-fa", "on", "--spec-type", "draft-mtp",
            "--spec-draft-n-max", str(k), "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
            "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16", "-sm", "tensor", "--tensor-split", "50,50",
            "-b", "2048", "-ub", "128", "--host", "127.0.0.1", "--port", str(port), "--metrics", "--jinja",
            "--chat-template-file", str(h.INPUTS / "chat-template.jinja"), "--temp", "1", "--top-k", "20",
            "--top-p", "0.95", "--min-p", "0.05", "--spec-draft-sampling", "probabilistic",
            "--no-spec-draft-backend-sampling", "-t", "8", "-tb", "8"]


def parse_bench(raw, repeats, trials):
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    expected = {(typ, shape, n, variant) for typ in TYPES for shape in SHAPES
                if shape != "head" or typ == "q6_K" for n in (1, 2, 4, 6, 8) for variant in VARIANTS}
    seen = set()
    for row in rows:
        key = row["type"], row["shape"], row["ncols"], row["variant"]
        if key not in expected or key in seen or (row["k"], row["m"]) != SHAPES[row["shape"]]:
            raise ValueError("unexpected or duplicate bench case")
        if row["cc"] != 1200 or row["repeats"] != repeats or row["trials"] != trials:
            raise ValueError("wrong bench architecture or repeat count")
        for name in ("us_M", "weight_GB_s_D", "nmse_D", "cpu_nmse_D"):
            if not isinstance(row[name], (float, int)) or not math.isfinite(row[name]):
                raise ValueError("non-finite bench metric")
        if row["us_M"] <= 0 or row["weight_GB_s_D"] <= 0 or not 0 <= row["nmse_D"] < 1e-8 or not 0 <= row["cpu_nmse_D"] < 2e-5:
            raise ValueError("bench timing or correctness gate failed")
        block, size = {"q5_K": (256, 176), "q6_K": (256, 210), "q4_K": (256, 144),
                       "iq4_xs": (256, 136), "q8_0": (32, 34)}[row["type"]]
        weight_bytes = row["k"] * row["m"] // block * size
        if row["weight_bytes"] != weight_bytes or not math.isclose(row["weight_GB_s_D"], weight_bytes / row["us_M"] / 1000, rel_tol=1e-5):
            raise ValueError("bench bandwidth denominator changed")
        seen.add(key)
    if seen != expected:
        raise ValueError("incomplete bench output")
    return rows


def parse_request(record, k, minimum):
    parsed = h.completion(record["chunks"])
    if parsed != record["response"]:
        raise ValueError("SSE final response disagrees with recorded response")
    final = parsed["timings"]
    if final["predicted_n"] < 500 or final["predicted_n"] >= 32000:
        raise ValueError("generated length gate failed")
    result = h.steady(record, k, minimum)
    result["acceptance_D"] = record["metrics"]["accepted_tokens"] / record["metrics"]["draft_tokens"]
    return result


def bench_item(harness, root, out, canary, repeats, trials):
    h.frozen(root)
    harness.idle()
    binary = root / "bin/bench-mmvq-shapes"
    cmd = [str(binary), "--variant", "all", "--repeats", str(repeats), "--trials", str(trials),
           "--warmup-ms", "200" if canary else "1000"]
    (out / "command.json").write_text(json.dumps(cmd, indent=2))
    harness.env["CUDA_VISIBLE_DEVICES"] = harness.env["CUDA_VISIBLE_DEVICES"].split(",")[0]
    harness.env["LD_LIBRARY_PATH"] = str(root / "bin")
    # Retain raw output even when the executable or a correctness gate fails.
    with (out / "bench.raw.jsonl").open("w") as raw, (out / "bench.stderr.log").open("w") as log:
        harness.child = subprocess.Popen(cmd, env=harness.env, stdout=raw, stderr=log)
        try:
            status = harness.child.wait(timeout=harness.timeout(limit=1200))
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("queue wall budget exhausted in bench") from error
    if status:
        raise ValueError("bench executable failed; raw output retained")
    rows = parse_bench((out / "bench.raw.jsonl").read_text(), repeats, trials)
    negative_cmd = [str(binary), "--variant", "default", "--type", "q5_K", "--shape", "qkv",
                    "--ncols", "6", "--repeats", "1", "--trials", "1", "--warmup-ms", "0", "--negative-control"]
    with (out / "negative.log").open("w") as log:
        result = subprocess.run(negative_cmd, env=harness.env, stdout=log, stderr=log, timeout=harness.timeout())
    if result.returncode != 1 or "variant disagrees with baseline" not in (out / "negative.log").read_text():
        raise ValueError("bench negative control did not fail at the comparison gate")
    for variant in VARIANTS:
        report = {"canary": canary, "variant": variant, "rows": [row for row in rows if row["variant"] == variant]}
        path = out / f"bench-{variant}.json"
        path.write_text(json.dumps(report, indent=2))
        shutil.copy2(path, ROOT / "codex" / path.name)
    # The bench process has ended. Restore the pair before the post-run memory/hazard gate.
    harness.env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    harness.xid()
    h.frozen(root)


def ab_item(harness, roots, out, canary):
    h.command = command
    order = ("patch",) if canary else ("up", "patch", "patch", "up")
    results = []
    for index, arm in enumerate(order):
        dest = harness.boot(roots[arm], f"{index + 1}-{arm}")
        records = {kind: harness.request(dest, "long-" + kind, 1, stream=True) for kind in ("tool", "think")}
        failures, metrics = [], {}
        for kind, record in records.items():
            try:
                metrics[kind] = parse_request(record, harness.args.k, harness.args.rounds)
            except (ValueError, KeyError, TypeError) as error:
                failures.append(kind + ": " + str(error))
        result = {"arm": arm, "metrics": metrics, "failures": failures}
        results.append(result)
        (out / "boots.json").write_text(json.dumps(results, indent=2))
        harness.stop()
        harness.xid()
        h.frozen(dest)
        if failures:
            raise ValueError("boot gates failed after recording every request: " + json.dumps(failures))
    summary = {}
    if not canary:
        for kind in ("tool", "think"):
            means = {arm: statistics.mean(r["metrics"][kind]["steady_ms_per_round_D"] for r in results if r["arm"] == arm)
                     for arm in ("up", "patch")}
            acceptance = {arm: statistics.mean(r["metrics"][kind]["acceptance_D"] for r in results if r["arm"] == arm)
                          for arm in ("up", "patch")}
            summary[kind] = {"mean_ms_per_round_D": means, "patch_minus_up_ms_D": means["patch"] - means["up"],
                             "acceptance_D": acceptance, "acceptance_drift": abs(acceptance["patch"] - acceptance["up"]) > 0.05}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bench", "ab", "correctness"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--pins", action="store_true")
    parser.add_argument("--up", type=Path, default=ROOT / "codex/frozen/mmvq-up")
    parser.add_argument("--patch", type=Path, default=ROOT / "codex/frozen/mmvq6")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--port", type=int, default=8406)
    parser.add_argument("--k", type=int, default=5, choices=(3, 5))
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()
    args.budget = args.budget or (600 if args.canary else 1800 if args.mode == "bench" else 1200)
    if args.canary and args.mode == "bench":
        args.repeats, args.trials = 5, 1
    estimate = (300 if args.canary else 900) if args.mode == "bench" else (180 if args.canary else 720) if args.mode == "ab" else 540
    if not 1 <= args.repeats <= 1000 or not 1 <= args.trials <= 15 or args.rounds < 40 or estimate > 0.7*args.budget or (args.canary and args.budget > 600):
        raise ValueError("invalid settings or stage estimate exceeds budget margin")
    roots = {"up": args.up.resolve(), "patch": args.patch.resolve()}
    active = (roots["up"], roots["patch"]) if args.mode == "ab" and not args.canary else (roots["patch"],)
    files = {Path(__file__).resolve(), ROOT / "codex/gpu_harness.py", ROOT / "codex/attrib.py",
             ROOT / "codex/gpu_ab.sh", ROOT / "codex/bench.sh", ROOT / "codex/census.json"}
    for root in active:
        h.frozen(root)
        files.update(path for path in root.rglob("*") if path.is_file())
    if args.mode == "ab":
        files.update((MODEL, h.INPUTS / "chat-template.jinja"))
        files.update(h.INPUTS.glob("*.json"))
    if args.pins:
        print(json.dumps([str(path) for path in sorted(files)], indent=2))
        return
    print(json.dumps({"mode": args.mode, "canary": args.canary, "estimate_s_G": estimate, "budget_s": args.budget,
                      "repeats": args.repeats, "trials": args.trials, "rounds_min": args.rounds,
                      "boots": ["patch"] if args.canary else ["up", "patch", "patch", "up"] if args.mode == "ab" else []}))
    if not args.execute:
        return
    if os.environ.get("RUN_GPU") != "1" or not os.environ.get("CUDA_VISIBLE_DEVICES") or not os.environ.get("XID_SINCE"):
        raise RuntimeError("parent gpuq environment required")
    if args.out is None or not args.out.resolve().is_relative_to(ROOT):
        raise ValueError("unique output directory inside worktree required")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    harness = h.Harness(args, out)
    def interrupted(signum, frame):
        harness.stop()
        raise KeyboardInterrupt("queue interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        pins = {str(path): h.sha(path) for path in sorted(files)}
        (out / "pins.json").write_text(json.dumps(pins, indent=2))
        if args.mode == "bench":
            bench_item(harness, roots["patch"], out, args.canary, args.repeats, args.trials)
        elif args.mode == "ab":
            ab_item(harness, roots, out, args.canary)
        else:
            harness.idle()
            with (out / "correctness.log").open("w") as log:
                harness.run([str(roots["patch"] / "bin/cuda-op-check"), "-b", "CUDA0", "-o", "MUL_MAT",
                             "-p", "q[456]_K|iq4_xs|q8_0"], stdout=log, stderr=log, limit=540)
            harness.xid()
        if pins != {path: h.sha(Path(path)) for path in pins}:
            raise ValueError("pinned files changed during run")
        harness.timeout(limit=1, reserve=0)
        verdict = {"verdict": "PASS", "canary": args.canary}
    except BaseException as error:
        verdict = {"verdict": "budget" if "budget" in str(error) or isinstance(error, subprocess.TimeoutExpired) else "hazard" if "Xid" in str(error) else "INVALID",
                   "error": str(error)}
        (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
        raise
    finally:
        harness.stop()
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
