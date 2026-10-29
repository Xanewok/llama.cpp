"""smoke.py's workload against an already-running vLLM server (HyperQwen), for engine comparison.

Same corpus text, depths, n_predict, sampling and seed as smoke.py, except min_p: vLLM rejects min_p
with speculative decoding. Decode speed is measured from the stream:
(tokens - 1) / (last token time - first token time); prefill is prompt tokens / time to first token.
Draft acceptance comes from the deltas of vLLM's spec-decode counters in /metrics.
"""

import argparse
import json
from pathlib import Path
import re
import time
import urllib.request

from smoke import DEPTHS, N_PREDICT, corpus_text


def call(url, body=None, headers=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data, {"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def spec_counters(base, headers):
    text = call(base + "/metrics", headers=headers).decode()
    totals = {}
    for name in ("num_accepted_tokens", "num_draft_tokens"):
        values = re.findall(rf"^vllm:spec_decode_{name}_total\{{[^}}]*\}} ([0-9.e+]+)$", text, re.M)
        totals[name] = sum(float(v) for v in values)
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18020")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--depths", type=lambda v: [int(x) for x in v.split(",")], default=DEPTHS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    headers = {}
    if args.api_key_file and args.api_key_file.is_file():
        headers["Authorization"] = "Bearer " + args.api_key_file.read_text().strip()
    model = args.model or json.loads(call(args.url + "/v1/models", headers=headers))["data"][0]["id"]
    tokens = json.loads(call(args.url + "/tokenize", {"model": model, "prompt": corpus_text(),
                                                      "add_special_tokens": False}, headers))["tokens"]
    while len(tokens) < max(args.depths):
        tokens += tokens
    result = {"url": args.url, "model": model, "runs": []}
    for depth in args.depths:
        before = spec_counters(args.url, headers)
        body = {"model": model, "prompt": tokens[:depth], "max_tokens": N_PREDICT, "ignore_eos": True,
                "temperature": 1.0, "top_k": 20, "top_p": 0.95, "seed": 1729, "stream": True,
                "stream_options": {"include_usage": True}}
        request = urllib.request.Request(args.url + "/v1/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json", **headers})
        start, first, last, pieces, usage = time.time(), None, None, 0, {}
        with urllib.request.urlopen(request, timeout=1800) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if "error" in chunk:
                    raise SystemExit(chunk["error"]["message"])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("choices") and chunk["choices"][0].get("text"):
                    now = time.time()
                    first = first or now
                    last = now
                    pieces += 1
        after = spec_counters(args.url, headers)
        generated = usage.get("completion_tokens", pieces)
        run = {"depth": depth, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": generated,
               "ttft_s": first - start, "decode_s": last - first,
               "pp": depth / (first - start), "tg": (generated - 1) / (last - first),
               "accepted": after["num_accepted_tokens"] - before["num_accepted_tokens"],
               "drafted": after["num_draft_tokens"] - before["num_draft_tokens"]}
        result["runs"].append(run)
        print(json.dumps({"depth": depth, "pp": round(run["pp"], 1), "tg": round(run["tg"], 1),
                          "draft_acc": f'{int(run["accepted"])}/{int(run["drafted"])}'}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
