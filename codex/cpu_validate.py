#!/usr/bin/env python3
"""Run only through gate.sh and cpugate, with CUDA_VISIBLE_DEVICES empty."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


MODEL = "/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf"
ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    "Write a detailed numbered list of 100 practical steps to implement a Python JSON parser. Include examples.",
    "Implement a small key-value store in Python. Include the complete source and tests. Explain each design decision.",
    "List and explain 100 concrete edge cases for a CSV reader, with sample inputs and expected outputs.",
    "Write a long tutorial on sorting algorithms, with complete Python implementations and worked examples.",
    "Implement a command-line calculator in Python with a parser, evaluator and test suite. Show complete files.",
    "Explain how to design an HTTP server. Include a complete small Python server and tests of its behavior.",
    "Write 100 numbered notes about debugging concurrency bugs. Make each item specific and include examples.",
    "Implement a recursive descent parser for arithmetic in Python. Explain the grammar and include many tests.",
    "Design a file synchronization tool in Python. Include all source code, conflict handling and detailed tests.",
    "Give a numbered list of 100 properties to test in a binary serialization library. Include sample data.",
    "Implement a small dependency graph scheduler in Python. Include cycle detection, tests and documentation.",
    "Write a detailed explanation of AVL trees with complete Python source and tests for every rotation.",
    "Implement a streaming log aggregator in Python. Include parsing, filtering, statistics and tests.",
    "Describe 100 specific failure modes in distributed systems, with mitigation examples for each.",
    "Implement a tiny stack machine interpreter in Python. Show the complete code, bytecode and many tests.",
    "Explain database transactions with worked examples of every isolation level and test scenarios.",
]


def request(port, route, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/{route}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as res:
        return json.loads(res.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--predict", type=int, default=256)
    ap.add_argument("--out", type=Path, default=ROOT / "codex/cpu-results")
    ap.add_argument("--trace-off", action="store_true")
    ap.add_argument("--no-repack", action="store_true")
    ap.add_argument("--ctx-checkpoints", type=int)
    ap.add_argument("--verbosity", type=int, default=3)
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")))
    if available < 10 * 1024 ** 2:
        raise RuntimeError("MemAvailable below 10 GiB")
    args.out.mkdir(parents=True, exist_ok=True)
    port = 18157
    env = dict(os.environ, OMP_NUM_THREADS="8")
    if not args.trace_off:
        env["LLAMA_SPEC_TRACE"] = str(args.out / "trace.jsonl")
    flags = ["-m", MODEL, "-c", "4096", "-np", "1", "-ngl", "0", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
             "--spec-type", "draft-mtp", "--spec-draft-n-max", "5", "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
             "--spec-draft-sampling", "probabilistic", "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16",
             "--spec-draft-backend-sampling", "--backend-sampling", "--kv-pad", "4096", "-b", "2048", "-ub", "512",
             "--temp", "1", "--top-k", "20", "--top-p", "0.95", "--min-p", "0.05", "-t", "8", "-tb", "8",
             "--host", "127.0.0.1", "--port", str(port), "--cache-ram", "0"]
    flags += ["-lv", str(args.verbosity)]
    if args.ctx_checkpoints is not None:
        flags += ["--ctx-checkpoints", str(args.ctx_checkpoints)]
    if args.no_repack:
        flags.append("--no-repack")
    records, streams = [], []
    with (args.out / "server.log").open("w") as log:
        proc = subprocess.Popen([str(ROOT / "build-cpu/bin/spec-tf-cpu-server"), *flags], env=env, stdout=log, stderr=log)
        (args.out / "server.pid").write_text(str(proc.pid) + "\n")
        try:
            for _ in range(300):
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited {proc.returncode}")
                try:
                    request(port, "health")
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            else:
                raise RuntimeError("server did not become ready")
            for i, prompt in enumerate(PROMPTS[:args.prompts]):
                prompt_ids = request(port, "tokenize", {"content": prompt, "add_special": True})["tokens"]
                for seed in (1729, 2718):
                    subprocess.run([str(ROOT / "codex/gate.sh"), "true"], check=True)
                    body = {"prompt": prompt_ids, "n_predict": args.predict, "seed": seed, "temperature": 1,
                            "top_k": 20, "top_p": 0.95, "min_p": 0.05, "ignore_eos": True,
                            "stream": False, "return_tokens": True, "cache_prompt": False}
                    index = len(records)
                    (args.out / f"request-{index}.json").write_text(json.dumps(body))
                    started = time.monotonic()
                    response = request(port, "completion", body)
                    elapsed = time.monotonic() - started
                    (args.out / f"response-{index}.json").write_text(json.dumps(response))
                    tokens = response.get("tokens", [])
                    if not tokens or not isinstance(tokens[0], int):
                        raise RuntimeError("completion omitted token ids")
                    streams.append({"id": f"p{i:02d}-s{seed}", "prompt": prompt_ids, "continuation": tokens})
                    records.append({"stream": index, "elapsed_s_M": elapsed, "request": body, "response": response})
                    (args.out / "streams.json").write_text(json.dumps(streams, indent=2))
                    (args.out / "records.json").write_text(json.dumps(records, indent=2))
                    print(f"stream {index} tokens={len(tokens)} elapsed_s_M={elapsed:.3f}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()
