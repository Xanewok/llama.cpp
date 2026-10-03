#!/usr/bin/env python3
"""Frozen stack3 identity gate. GPU execution requires the parent queue environment."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys

import gpu_harness as h

ROOT = Path(__file__).resolve().parent.parent
ORDER = ("stack2", "default", "staged", "staged", "default", "stack2")
KNOBS = ("LLAMA_MTP_GRAPH_REUSE", "GGML_META_PLAN_REUSE", "GGML_SCHED_INPUT_STAGING", "LLAMA_GRAPH_REUSE_DISABLE")
base_command = h.command


def command(binary, k, port):
    result = base_command(binary, k, port)
    result[result.index("-np") + 1] = "2"
    return result


h.command = command


def body(temperature, stream=None):
    label = "single" if stream is None else f"stream-{stream}"
    words = 32768 if temperature == 1 and stream is None else 1024
    return {"messages": [{"role": "system", "content": "Follow the output instruction precisely."},
            {"role": "user", "content": label + "\n" + " sequence" * words +
             "\nPrint the integers from 1 to 1000, one per line. Keep printing until the output limit."}],
            "temperature": temperature, "top_k": 20, "top_p": 0.95, "min_p": 0.05,
            "seed": 1729 if stream is None else 2718 + stream, "max_tokens": 768,
            "ignore_eos": True, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}


def output_bytes(record):
    if "error" in record:
        raise ValueError("recorded request failed: " + json.dumps(record["error"]))
    choices = h.identity(record)[0]
    return json.dumps(choices, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def same_outputs(expected, actual):
    if set(expected) != set(actual) or any(output_bytes(expected[key]) != output_bytes(actual[key]) for key in expected):
        raise ValueError("identity failed: completion bytes or finish reason changed")


def stream_bounds(record):
    arrivals = record.get("chunk_ns", [])
    if len(arrivals) != len(record["chunks"]):
        raise ValueError("missing stream arrival timestamps")
    points = [stamp for stamp, chunk in zip(arrivals, record["chunks"])
              if any(choice.get("delta", {}).get("content") or choice.get("delta", {}).get("reasoning_content") or choice.get("delta", {}).get("tool_calls")
                     for choice in chunk.get("choices", []))]
    if len(points) < 8 or points[-1] <= points[0]:
        raise ValueError("did not stream enough content")
    return points[0], points[-1]


def validate(records, pair_metrics, k, minimum=40):
    failures = []
    steady = {}
    for key, record in records.items():
        try:
            output_bytes(record)
            final = record["response"]["timings"]
            low, high = (30000, 34000) if key == "single-t1" else (768, 2048)
            if not low <= final["prompt_n"] <= high or final.get("cache_n", 0) != 0:
                raise ValueError("cold prompt size/cache gate failed")
            if final["predicted_n"] < 500 or final["predicted_n"] >= 32000:
                raise ValueError("generated length gate failed")
            if key.startswith("single-"):
                steady[key] = h.steady(record, k, minimum)
        except (KeyError, ValueError, TypeError) as error:
            failures.append(key + ": " + str(error))
    try:
        pair = [records[f"np2-{i}"] for i in (0, 1)]
        bounds = [stream_bounds(record) for record in pair]
        overlap = (min(end for _, end in bounds) - max(begin for begin, _ in bounds)) / 1e9
        if overlap <= 0.5:
            raise ValueError("generation streams did not overlap")
        rates = [r["response"]["timings"]["predicted_n"] * 1000 / r["response"]["timings"]["predicted_ms"] for r in pair]
        if any(not 5 <= rate <= 500 for rate in rates):
            raise ValueError("per-stream rate band failed")
        finals = [r["response"]["timings"] for r in pair]
        if pair_metrics["draft_tokens"] != sum(t["draft_n"] for t in finals) or pair_metrics["accepted_tokens"] != sum(t["draft_n_accepted"] for t in finals):
            raise ValueError("np2 aggregate counters disagree")
        if pair_metrics["drafts"] < 2 * minimum:
            raise ValueError("missing authoritative rounds")
    except (KeyError, ValueError, TypeError) as error:
        failures.append("np2: " + str(error))
        overlap, rates = None, None
    return {"failures": failures, "steady": steady, "generation_overlap_s_M": overlap, "stream_tok_s_D": rates}


def suite(harness, dest, canary):
    records = {}
    for temperature in (0, 1):
        key = f"single-t{temperature}"
        records[key] = harness.request(dest, key, 1, body(temperature), stream=True)
    if canary:
        records["repeat"] = harness.request(dest, "repeat", 1, body(0), stream=True)
    before = harness.metrics()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(harness.request, dest, f"np2-{i}", 1, body(1, i), True) for i in (0, 1)]
        for i, future in enumerate(futures):
            records[f"np2-{i}"] = future.result()
    after = harness.metrics()
    pair_metrics = {key: after[key] - before[key] for key in before}
    # Individual request deltas overlap. Only the pair-wide metrics are authoritative.
    (dest / "pair-metrics.json").write_text(json.dumps(pair_metrics, indent=2))
    checked = {key: value for key, value in records.items() if key != "repeat"}
    verdict = validate(checked, pair_metrics, harness.args.k)
    if canary:
        try:
            same_outputs({"repeat": records["single-t0"]}, {"repeat": records["repeat"]})
        except ValueError as error:
            verdict["failures"].append(str(error))
    (dest / "validation.json").write_text(json.dumps(verdict, indent=2))
    return checked, verdict


def pins(roots):
    files = [h.MODEL, h.VOCAB, Path(__file__).resolve(), ROOT / "codex/gpu_ab.sh", ROOT / "codex/gpu_harness.py", ROOT / "codex/attrib.py"]
    files.extend(sorted(h.INPUTS.glob("*")))
    for root in dict.fromkeys(roots.values()):
        files.extend([root / "SOURCE", root / "SHA256SUMS", root / "bin/SHA256SUMS"])
        files.extend(root / "bin" / name for name in h.frozen(root))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--pins", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--budget", type=int, default=960)
    parser.add_argument("--port", type=int, default=8403)
    args = parser.parse_args()
    args.k = 5
    estimate = 160 if args.canary else 620
    if estimate > 0.7 * args.budget or args.budget > (240 if args.canary else 960):
        raise ValueError("queue wall budget estimate exceeds margin or item limit")
    roots = {"stack2": ROOT / "codex/frozen/stack2", "default": ROOT / "codex/frozen/stack3", "staged": ROOT / "codex/frozen/stack3"}
    files = pins(roots)
    if (roots["stack2"] / "SOURCE").read_text().splitlines()[0] != "0dba27b9002b8347c6f14a9f4a1651b3478065ef":
        raise ValueError("stack2 SOURCE changed")
    if args.pins:
        print(json.dumps([str(path) for path in files]))
        return
    if not args.execute:
        print(json.dumps({"boots": ["staged"] if args.canary else ORDER, "k": 5, "np": 2, "estimated_s_G": estimate,
                          "budget_s_G": args.budget, "minimum_steady_rounds": 40, "long_prompt_tokens_G": 32768, "short_prompt_tokens_G": 1024, "pins": len(files)}))
        return
    if os.environ.get("RUN_GPU") != "1" or not os.environ.get("CUDA_VISIBLE_DEVICES") or not os.environ.get("XID_SINCE"):
        raise RuntimeError("parent gpuq environment required")
    if args.out is None:
        raise ValueError("unique attempt output required")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    harness = h.Harness(args, out)
    baseline = None
    results = []
    status = "INVALID"
    try:
        hashes = {str(path): h.sha(path) for path in files}
        if hashes[str(h.MODEL)] != h.MODEL_SHA:
            raise ValueError("model SHA changed")
        (out / "inputs.json").write_text(json.dumps(hashes, indent=2))
        boots = ("staged",) if args.canary else ORDER
        for number, arm in enumerate(boots, 1):
            for knob in KNOBS:
                harness.env.pop(knob, None)
            if arm == "staged":
                harness.env["GGML_SCHED_INPUT_STAGING"] = "1"
            dest = None
            try:
                dest = harness.boot(roots[arm], f"{number}-{arm}")
                (dest / "knobs.json").write_text(json.dumps({knob: harness.env.get(knob) for knob in KNOBS}, indent=2))
                records, verdict = suite(harness, dest, args.canary)
                if verdict["failures"]:
                    raise ValueError("boot validation failed: " + json.dumps(verdict["failures"]))
                if baseline is None:
                    baseline = records
                else:
                    same_outputs(baseline, records)
                results.append({"boot": number, "arm": arm, **verdict, "requests": {
                    key: {"prompt_ms_M": record["response"]["timings"]["prompt_ms"],
                          "predicted_ms_M": record["response"]["timings"]["predicted_ms"],
                          "wall_ms_D": (record["end_ns"] - record["begin_ns"]) / 1e6}
                    for key, record in records.items()}})
                (out / "boots.json").write_text(json.dumps(results, indent=2))
            finally:
                harness.stop()
            harness.xid()
            if dest:
                h.frozen(dest)
        if any(h.sha(Path(path)) != digest for path, digest in hashes.items()):
            raise ValueError("inputs changed during run")
        summary = {}
        for arm in boots:
            chosen = [r for r in results if r["arm"] == arm]
            summary[arm] = {
                "steady_ms_per_round_D": {key: [r["steady"][key]["steady_ms_per_round_D"] for r in chosen] for key in ("single-t0", "single-t1")},
                "long_prompt_ms_M": [r["requests"]["single-t1"]["prompt_ms_M"] for r in chosen],
                "long_request_wall_ms_D": [r["requests"]["single-t1"]["wall_ms_D"] for r in chosen]}
        paired = []
        if not args.canary:
            for row in results:
                if row["arm"] == "stack2":
                    continue
                reference = results[0] if row["boot"] <= 3 else results[-1]
                paired.append({"arm": row["arm"], "boot": row["boot"], "reference_boot": reference["boot"],
                    "steady_delta_ms_D": {key: row["steady"][key]["steady_ms_per_round_D"] - reference["steady"][key]["steady_ms_per_round_D"] for key in ("single-t0", "single-t1")},
                    "long_prompt_delta_ms_D": row["requests"]["single-t1"]["prompt_ms_M"] - reference["requests"]["single-t1"]["prompt_ms_M"]})
        summary = {"arms": summary, "paired": paired}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        status = "PASS"
    except Exception as error:
        status = "budget" if isinstance(error, TimeoutError) or "queue wall budget" in str(error) else "hazard" if "Xid" in str(error) else "INVALID"
        (out / "failure.json").write_text(json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2))
        raise
    finally:
        harness.stop()
        (out / "verdict.json").write_text(json.dumps({"verdict": status, "canary": args.canary, "valid_boots_M": len(results)}, indent=2))


if __name__ == "__main__":
    main()
