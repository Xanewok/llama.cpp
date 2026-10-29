#!/usr/bin/env python3
"""Engine-agnostic decode/prefill bench over any OpenAI-compatible endpoint.

The point is comparability: llama.cpp, vLLM and ninfer expose different metrics, so
nothing server-side is trusted here. Every number comes from the stream itself --
time to first token, wall time, and the token counts the API reports -- which is the
only instrument all three engines can be held to at once.

Rows are written as plain jsonl on purpose. The shared emit schema rejects unknown
fields at write time, which once killed two legs after their GPU time was already paid.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from scripts import ROOT
from scripts.instruments import bench_mtp, emit



def resolve_model(base_url: str, api_key: str | None, timeout: float = 20.0) -> str:
    """Ask the server what it serves. vLLM rejects a model id it does not recognise,
    and its id is the checkpoint path, which the caller has no reason to know."""

    request = urllib.request.Request(base_url.rstrip("/").removesuffix("/v1") + "/v1/models")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())["data"][0]["id"]


# A minimal but real tool schema. Attaching tools is what puts a server into constrained
# decoding, and that path is invisible to a bench that never sends any -- which is how a
# 145 tok/s engine came to serve an agent at 8.6.
TOOLS = [{
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write text to a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "contents": {"type": "string"}},
            "required": ["path", "contents"],
        },
    },
}]


def count_tokens(base_url: str, model: str, text: str, timeout: float = 30.0) -> int | None:
    """Token count from the served model's own tokenizer.

    Two wire shapes, because the engines disagree: vLLM takes {model, prompt} and answers
    {count}, llama-server takes {content} and answers {tokens: [...]} -- see
    longctx.py:_tokenize. Speaking only one of them returns None on the other engine, which
    is most of this experiment's legs.

    Returns None when neither shape works, so a missing count never reads as zero
    reasoning; callers should surface the first None rather than record it silently.
    """

    if not text.strip():
        return 0
    url = base_url.rstrip("/").removesuffix("/v1") + "/tokenize"
    for payload, read in (
        ({"model": model, "prompt": text}, lambda d: d.get("count")),
        ({"content": text}, lambda d: len(d["tokens"]) if "tokens" in d else None),
    ):
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                n = read(json.load(response))
        except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError):
            continue
        if isinstance(n, int):
            return n
    return None


def split_thinking(reasoning: str, content: str) -> tuple[str, str, bool]:
    """Return (thinking, answer, truncated).

    With no parser the thinking arrives inline: the template opens <think> and the model
    closes it, so the closing tag is the only delimiter present. When the token budget runs
    out mid-thought no closing tag ever arrives -- and treating that as pure content would
    score it as zero reasoning, the exact opposite of what the same run reports on an engine
    whose parser tags it. Unterminated output is thinking, and says so.
    """

    if reasoning:
        return reasoning, content, False
    head, sep, tail = content.partition("</think>")
    if sep:
        return head, tail, False
    return content, "", True


def stream_once(base_url: str, model: str, prompt: str, max_tokens: int,
                timeout: float, api_key: str | None = None,
                with_tools: bool = False) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if with_tools:
        payload["tools"] = TOOLS
        payload["tool_choice"] = "auto"
    request = urllib.request.Request(
        base_url.rstrip("/").removesuffix("/v1") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    started = time.monotonic()
    ttft = None
    chunks = 0
    usage = {}
    reasoning_text: list[str] = []
    answer_text: list[str] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                event = json.loads(body)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                # This vLLM streams thinking as `reasoning`; watching only
                # `reasoning_content` misses every one of them, and on a prompt whose
                # whole budget goes to thinking no `content` delta ever arrives -- so the
                # row came back with null ttft and null rates and looked like a server
                # that never streamed.
                think = delta.get("reasoning") or delta.get("reasoning_content") or ""
                said = delta.get("content") or ""
                if think:
                    reasoning_text.append(think)
                if said:
                    answer_text.append(said)
                if think or said:
                    chunks += 1
                    if ttft is None:
                        ttft = time.monotonic() - started
    total = time.monotonic() - started
    out_tokens = usage.get("completion_tokens")
    # Falling back to chunk count would quietly compare tokens against SSE events, so a
    # row without usage is reported as missing rather than estimated.
    #
    # No TTFT means no decode rate either: dividing by total time would put a
    # prefill-inclusive number in the decode column beside engines that do stream.
    # The parser does NOT withhold deltas while the model thinks -- it streams them under
    # the key `reasoning`, which is why watching only `reasoning_content` made a
    # thinking-only response look like a server that never streamed at all.
    decode_s = total - ttft if ttft is not None else None
    return {
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "total_s": round(total, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": out_tokens,
        "chunks": chunks,
        "decode_tok_s": round(out_tokens / decode_s, 2)
        if out_tokens and decode_s and decode_s > 0 else None,
        "overall_tok_s": round(out_tokens / total, 2) if out_tokens and total > 0 else None,
        "prefill_tok_s": round(usage["prompt_tokens"] / ttft, 1)
        if usage.get("prompt_tokens") and ttft else None,
        "reasoning_text": "".join(reasoning_text),
        "answer_text": "".join(answer_text),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--label", required=True, help="engine + decoding config")
    parser.add_argument("--engine", required=True, help="llama.cpp | vllm | ninfer")
    # No --gpu: hand-typing the card is how 36 of 124 rows here ended up labelled with
    # a rental label, TTL included, and the table rendered it as the card.
    parser.add_argument("--env-json", required=True,
                        help="scripts/gpuenv.py capture for the box under test")
    parser.add_argument("--experiment-id", required=True,
                        help="without it a later experiment's rows are indistinguishable "
                             "in this sink")
    parser.add_argument("--quant", required=True, help="quantization label recorded verbatim")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--manifest", type=Path, default=ROOT / "subsets" / "polyglot.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--with-tools", action="store_true",
                        help="attach a tool schema, so the row measures the path an agent uses")
    args = parser.parse_args()

    env = emit.load_env(args.env_json)
    if args.model == "auto":
        args.model = resolve_model(args.base_url, args.api_key)
        print(f"serving model: {args.model}")
    served = emit.probe_server(args.base_url).get("served_model_path")

    tasks = bench_mtp.load_tasks(args.manifest, args.tasks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    tokenize_warned = False
    for task in tasks:
        prompt = bench_mtp.bench_prompt(task, args.manifest.parent)
        row = {
            "schema_version": emit.SCHEMA_VERSION,
            "timestamp": emit.utc_timestamp(),
            "experiment_id": args.experiment_id,
            "label": args.label, "engine": args.engine, "quant": args.quant,
            "served_model_path": served,
            "task_id": task["task_id"], "max_tokens": args.max_tokens,
            "with_tools": args.with_tools,
            **env,
        }
        try:
            row.update(stream_once(args.base_url, args.model, prompt,
                                   args.max_tokens, args.timeout, args.api_key,
                                   args.with_tools))
            # Counted, not stored: the text itself is large and the transcripts already
            # hold it. The split is what the quant comparison needs -- how much of the
            # budget a quant spends thinking before it answers.
            think, said, truncated = split_thinking(
                row.pop("reasoning_text", ""), row.pop("answer_text", "")
            )
            row["reasoning_tokens"] = count_tokens(args.base_url, args.model, think)
            row["content_tokens"] = count_tokens(args.base_url, args.model, said)
            # Also truncated when the budget ran out, which the delimiter cannot show on
            # the parser path: reasoning is tagged, so no missing </think> betrays it. A
            # capped row's reasoning share is a floor, not a measurement.
            row["thinking_truncated"] = bool(
                truncated
                or (row.get("completion_tokens") or 0) >= args.max_tokens
            )
            if row["reasoning_tokens"] is None and not tokenize_warned:
                # Silently recording None makes a broken tokenizer call look like an
                # endpoint that was never there.
                print(f"WARNING: /tokenize did not answer at {args.base_url}; "
                      "the reasoning split will be null for this run", flush=True)
                tokenize_warned = True
        except Exception as error:                       # noqa: BLE001
            row["error"] = f"{type(error).__name__}: {error}"
        emit.append_row(args.out, row)
        rows += 1
        print(f"{args.label} {task['task_id']}: "
              f"{row.get('decode_tok_s')} tok/s decode, ttft {row.get('ttft_s')}s")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
