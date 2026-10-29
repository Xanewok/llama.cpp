"""Cold, overlapping requests with per-request timings and observed slot occupancy."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request

from scripts import ROOT
from scripts.instruments import bench_endpoint, bench_mtp, emit, longctx


def get_json(base_url, endpoint):
    with urllib.request.urlopen(base_url.rstrip("/").removesuffix("/v1") + endpoint,
                                timeout=10) as response:
        return json.load(response)


def read_stream(lines, started, now=time.monotonic):
    usage, timings, fragments, finish_reasons = {}, {}, [], []
    first_token = None
    for raw in lines:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            break
        event = json.loads(body)
        if event.get("error"):
            raise ValueError(f"server stream error: {event['error']}")
        usage = event.get("usage") or usage
        timings = event.get("timings") or timings
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            material = (delta.get("reasoning_content") or delta.get("reasoning")
                        or delta.get("content") or delta.get("tool_calls"))
            if material:
                if first_token is None:
                    first_token = now() - started
                fragments.append(delta)
            if choice.get("finish_reason"):
                finish_reasons.append(choice["finish_reason"])
    if not usage.get("completion_tokens") or not timings or first_token is None:
        raise ValueError("missing completion usage, server timings or streamed output")
    return {"usage": usage, "timings": timings, "ttft_seconds": first_token,
            "finish_reasons": finish_reasons, "output_deltas": fragments}


def request_one(base_url, model, prompt, index, max_tokens, timeout, barrier, stagger):
    barrier.wait()
    if stagger:
        time.sleep(index * stagger)
    started = time.monotonic()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 1.0, "seed": 1729 + index,
        "chat_template_kwargs": {"reasoning_effort": "xhigh"},
        "tools": bench_endpoint.TOOLS, "tool_choice": "auto",
        "stream": True, "stream_options": {"include_usage": True},
        "cache_prompt": False,
    }
    row = {"request_index": index, "started_monotonic": started,
           "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "error": None}
    request = urllib.request.Request(
        base_url.rstrip("/").removesuffix("/v1") + "/v1/chat/completions",
        json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            row.update(read_stream(response, started))
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    row["finished_monotonic"] = time.monotonic()
    row["wall_seconds"] = row["finished_monotonic"] - started
    return row


def summarize(requests, slot_samples):
    successful = [row for row in requests if not row["error"]]
    complete = len(successful) == len(requests) and bool(requests)
    total = sum(row["usage"]["completion_tokens"] for row in successful)
    makespan = (max(row["finished_monotonic"] for row in requests)
                - min(row["started_monotonic"] for row in requests)) if requests else 0
    first = min((row["started_monotonic"] + row["ttft_seconds"] for row in successful),
                default=None)
    last = max((row["finished_monotonic"] for row in successful), default=None)
    return {
        "complete": complete, "completed_requests": len(successful),
        "makespan_seconds": makespan, "completion_tokens": total,
        "output_tokens_per_wall_second": total / makespan if complete and makespan else None,
        # This window can contain another request's prefill and unequal decode tails.
        "output_tokens_per_first_token_to_last_finish_second":
            total / (last - first) if complete and first is not None and last > first else None,
        "max_observed_busy_slots": max((sum(bool(slot.get("is_processing"))
                                               for slot in sample.get("slots", []))
                                         for sample in slot_samples), default=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quant", required=True)
    parser.add_argument("--env-json", required=True)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--expected-slots", type=int, required=True)
    parser.add_argument("--expected-context", type=int, required=True)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--stagger-seconds", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if min(args.requests, args.expected_slots, args.expected_context, args.max_tokens) < 1:
        parser.error("requests, slots, context and output budget must be positive")
    if args.padding < 0 or args.stagger_seconds < 0:
        parser.error("padding and stagger must be nonnegative")
    props = get_json(args.base_url, "/props")
    if args.quant not in props.get("model_path", ""):
        raise SystemExit("wrong model served")
    if (props["total_slots"] != args.expected_slots
            or props["default_generation_settings"]["n_ctx"] != args.expected_context):
        raise SystemExit("server slot count or per-slot context differs from requested arm")
    model = bench_endpoint.resolve_model(args.base_url, None)
    manifest = ROOT / "subsets/polyglot.json"
    task = bench_mtp.load_tasks(manifest, 1)[0]
    prompt = bench_mtp.bench_prompt(task, manifest.parent)
    actual_padding = 0
    if args.padding:
        padding, actual_padding = longctx.build_padding(task, args.padding, args.base_url)
        prompt = longctx.wrap_prompt(prompt, padding)
    barrier = threading.Barrier(args.requests)
    stopped = threading.Event()
    samples = []

    def sample_slots():
        while not stopped.is_set():
            try:
                samples.append({"monotonic": time.monotonic(),
                                "slots": get_json(args.base_url, "/slots")})
            except Exception as error:
                samples.append({"monotonic": time.monotonic(), "error": str(error)})
            stopped.wait(1)

    if args.out.exists():
        raise SystemExit("output exists; choose a new result path")
    args.out.mkdir(parents=True)
    before = emit.scrape_metrics(args.base_url)
    sampler = threading.Thread(target=sample_slots, daemon=True)
    sampler.start()
    rows = []
    try:
        with ThreadPoolExecutor(max_workers=args.requests) as executor:
            futures = [executor.submit(
                request_one, args.base_url, model, f"Request {i}.\n{prompt}", i,
                args.max_tokens, args.timeout, barrier, args.stagger_seconds)
                for i in range(args.requests)]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                (args.out / f"request-{row['request_index']}.json").write_text(json.dumps(row, indent=2))
                print(json.dumps({key: row.get(key) for key in
                                  ("request_index", "wall_seconds", "ttft_seconds", "timings", "error")}), flush=True)
    finally:
        stopped.set()
        sampler.join(timeout=12)
    after = emit.scrape_metrics(args.base_url)
    delta = emit.metrics_delta(before, after)
    result = {
        "instrument": "cold-concurrency-v1", "timestamp": emit.utc_timestamp(),
        "quant": args.quant, "task_id": task["task_id"],
        "env": emit.load_env(args.env_json), "props": props,
        "requested_requests": args.requests, "padding_tokens": actual_padding,
        "max_output_tokens": args.max_tokens, "temperature": 1, "seed_base": 1729,
        "stagger_seconds": args.stagger_seconds, "with_tools": True, "cache_prompt": False,
        "metrics_delta": delta, "mtp": bench_mtp.acceptance(delta),
        "slot_samples": samples,
        "requests": [{key: value for key, value in row.items() if key != "output_deltas"}
                     for row in sorted(rows, key=lambda row: row["request_index"])],
        **summarize(rows, samples),
    }
    result["mtp_live"] = delta.get("draft_tokens", 0) > 0 and delta.get("draft_accepted_tokens", 0) > 0
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    if not result["complete"] or not result["mtp_live"]:
        raise SystemExit("failed requests or no accepted MTP tokens; inspect raw results")


if __name__ == "__main__":
    main()
