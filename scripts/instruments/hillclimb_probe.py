"""One disposable probe process; the coordinator owns its wall clock and safety guard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

from scripts.instruments import bench_concurrency, bench_mtp, emit, longctx


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_model(path, expected_sha, expected_bytes, gguf_path):
    if Path(path).stat().st_size != expected_bytes:
        raise ValueError("artifact size differs from pinned size")
    actual = digest_file(path)
    if actual != expected_sha:
        raise ValueError("artifact SHA256 differs from pin")
    sys.path.insert(0, str(gguf_path))
    from gguf import GGUFReader
    reader = GGUFReader(path)
    architecture = reader.fields["general.architecture"].contents()
    blocks = reader.fields[f"{architecture}.block_count"].contents()
    mtp = [{"name": t.name, "type": t.tensor_type.name, "bytes": t.n_bytes}
           for t in reader.tensors if t.name.startswith(f"blk.{blocks - 1}.")]
    if not any("nextn.eh_proj" in t["name"] for t in mtp):
        raise ValueError("embedded MTP projection is missing")
    fields = {}
    for name, field in reader.fields.items():
        if name.startswith("tokenizer."):
            digest = hashlib.sha256()
            for part in field.parts:
                digest.update(part.tobytes())
            fields[name] = digest.hexdigest()
    if "tokenizer.chat_template" not in fields:
        raise ValueError("chat template is missing")
    return {"sha256": actual, "bytes": expected_bytes, "architecture": architecture,
            "block_count": blocks, "native_context": reader.fields[
                f"{architecture}.context_length"].contents(),
            "tokenizer_field_hashes": fields, "mtp_tensors": mtp,
            "chat_template": reader.fields["tokenizer.chat_template"].contents()}


def api(url, endpoint, payload=None, timeout=15):
    request = urllib.request.Request(
        url.rstrip("/") + endpoint,
        None if payload is None else json.dumps(payload).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def prepared_prompt(url, task, target):
    base = task["prompt"]
    if target <= 0:
        return base, None
    padding, _ = longctx.build_padding(task, target, url)
    # Count the template, not just padding; all candidates receive the same budget.
    best = None
    low, high = 0, len(padding)
    for _ in range(20):
        length = (low + high) // 2
        prompt = longctx.wrap_prompt(base, padding[:length])
        formatted = api(url, "/apply-template", {
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"reasoning_effort": "xhigh"}}, timeout=120)["prompt"]
        count = longctx._tokenize(formatted, url)
        if count <= target:
            best = (prompt, count)
            low = length + 1
        else:
            high = length - 1
        if best and target - best[1] <= 32:
            break
    if best is None or best[1] < target - 512:
        raise ValueError(f"could not populate requested context: {target}, got {best and best[1]}")
    return best


def one_request(url, prompt, seed, max_tokens, index, barrier, out, cache_prompt=False):
    payload = {
        "model": "local-benchmark", "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.05,
        "seed": seed, "max_tokens": max_tokens, "cache_prompt": cache_prompt,
        "chat_template_kwargs": {"reasoning_effort": "xhigh"},
        "stream": True, "stream_options": {"include_usage": True},
    }
    barrier.wait()
    started = time.monotonic()
    row = {"request_index": index, "seed": seed, "started_monotonic": started,
           "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "error": None}
    request = urllib.request.Request(url + "/v1/chat/completions",
                                     json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            row["http_status"] = response.status
            row["response_headers"] = dict(response.headers)
            with (out / f"request-{index}.sse").open("wb") as raw_stream:
                def captured_lines():
                    for line in response:
                        raw_stream.write(line)
                        raw_stream.flush()
                        yield line
                row.update(bench_concurrency.read_stream(captured_lines(), started))
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, (urllib.error.URLError, TimeoutError)) or (
                isinstance(error, ValueError) and "missing completion usage" in str(error)):
            row["failure_class"] = "transport_failure"
    row["finished_monotonic"] = time.monotonic()
    row["wall_seconds"] = row["finished_monotonic"] - started
    usage = row.get("usage", {})
    row["actual_context_tokens"] = (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)) if usage else None
    row["early_eos"] = ("stop" in row.get("finish_reasons", []) and
                        usage.get("completion_tokens", max_tokens) < max_tokens)
    (out / f"request-{index}.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def run_probe(url, task, target, max_tokens, slots, seed, out, cache_prompt=False):
    props = api(url, "/props")
    context = props["default_generation_settings"]["n_ctx"]
    if props["total_slots"] != slots or target + max_tokens > context:
        raise ValueError("probe exceeds effective per-slot context or slot count differs")
    (out / "props.json").write_text(json.dumps(props, indent=2) + "\n")
    prompt, count = prepared_prompt(url, task, target)
    (out / "prompt.txt").write_text(prompt)
    samples, stopped = [], threading.Event()

    def sample():
        while not stopped.is_set():
            try:
                samples.append({"monotonic": time.monotonic(), "slots": api(url, "/slots", timeout=2)})
            except Exception as error:
                samples.append({"error": str(error)})
            stopped.wait(1)

    sampler = threading.Thread(target=sample, daemon=True)
    before = emit.scrape_metrics(url)
    (out / "metrics-before.json").write_text(json.dumps(before) + "\n")
    sampler.start()
    barrier = threading.Barrier(slots)
    try:
        with ThreadPoolExecutor(max_workers=slots) as pool:
            futures = [pool.submit(one_request, url, prompt, seed + i, max_tokens,
                                   i, barrier, out, cache_prompt) for i in range(slots)]
            rows = [future.result() for future in futures]
    finally:
        stopped.set()
        sampler.join(timeout=4)
    after = emit.scrape_metrics(url)
    (out / "metrics-after.json").write_text(json.dumps(after) + "\n")
    delta = emit.metrics_delta(before, after)
    summary = {
        "task_id": task["task_id"], "templated_prompt_tokens": count,
        "requested_prompt_tokens": target, "max_output_tokens": max_tokens,
        "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.05},
        "cache_prompt": cache_prompt,
        "metrics_scope": "batch", "metrics_delta": delta,
        "mtp": bench_mtp.acceptance(delta),
        "mtp_live": delta.get("draft_tokens", 0) > 0 and delta.get("draft_accepted_tokens", 0) > 0,
        "requests": [{k: v for k, v in row.items() if k != "output_deltas"} for row in rows],
        "slot_samples": samples, **bench_concurrency.summarize(rows, samples),
    }
    # Only actual usage establishes depth; cached/global high-water counters cannot.
    summary["depth_verified"] = all(
        row.get("usage", {}).get("prompt_tokens", 0) >= target - 512
        and row.get("actual_context_tokens") is not None
        and row["actual_context_tokens"] <= context for row in rows)
    summary["overlap_verified"] = target == 0 or slots == 1 or summary["max_observed_busy_slots"] >= slots
    failed = [row for row in rows if row["error"]]
    if failed and all(row.get("failure_class") == "transport_failure" for row in failed):
        summary["failure_class"] = "transport_failure"
    (out / "probe.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--path", required=True)
    inspect.add_argument("--sha256", required=True)
    inspect.add_argument("--bytes", required=True, type=int)
    inspect.add_argument("--gguf-path", required=True)
    inspect.add_argument("--out", required=True, type=Path)
    probe = sub.add_parser("probe")
    probe.add_argument("--url", required=True)
    probe.add_argument("--tasks", required=True, type=Path)
    probe.add_argument("--task", required=True)
    probe.add_argument("--target", type=int, required=True)
    probe.add_argument("--max-tokens", type=int, required=True)
    probe.add_argument("--slots", type=int, default=1)
    probe.add_argument("--seed", type=int, default=1729)
    probe.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_model(args.path, args.sha256, args.bytes, args.gguf_path)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        return 0
    task = json.loads(args.tasks.read_text())[args.task]
    result = run_probe(args.url, task, args.target, args.max_tokens, args.slots, args.seed, args.out)
    return 0 if (result["complete"] and result["mtp_live"] and result["depth_verified"]
                 and result["overlap_verified"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
