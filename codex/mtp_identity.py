"""CPU identity matrix. Run with CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/mtp_identity.py OUT BASE PATCH."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.request

PROMPTS = [
    "Write an iterative Fibonacci function in Python.",
    "Explain why the sky is blue.",
    "List prime numbers between 100 and 150.",
    "Translate into Polish: The quick brown fox jumps over the lazy dog.",
    "Implement merge sort in Python.",
    "Explain binary search and its complexity.",
    "Write a haiku about rain.",
    "Describe the water cycle.",
    "Give a recipe for pancakes.",
    "Explain how a compiler differs from an interpreter.",
    "Compute 37 times 29 and show your work.",
    "Write a SQL query grouping sales by customer.",
    "Explain the difference between a stack and a queue.",
    "Describe how to test a sorting function.",
    "List the planets in order from the Sun.",
    "Explain why seasons occur.",
    "Write a JavaScript function that removes duplicate numbers.",
    "Describe the basic rules of chess.",
    "Explain recursion using factorial.",
    "Give an example of a race condition and a way to prevent it.",
]
MODEL = "/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf"


def post(port, endpoint, body):
    request = urllib.request.Request(f"http://127.0.0.1:{port}/{endpoint}",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def run(binary, arm, mode, np, out, prompts, seeds, predict, fuse):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dest = out / f"{mode}-np{np}-{arm}"
    dest.mkdir(parents=True, exist_ok=True)
    threads = 4 if mode == "meta" else 8
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LLAMA_ARG_ALLOW_CPU_DEVICE="1",
               LLAMA_MTP_FUSE_CATCHUP=str(fuse), LLAMA_ROUND_TRACE=str(dest / "trace.jsonl"), LLAMA_TRACE="1",
               OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               LD_LIBRARY_PATH=str(binary.parent), GGML_SCHED_INPUT_STAGING="1")
    command = ["nice", "-n", "19", str(binary), "-m", MODEL, "-c", str(4096 * np), "-np", str(np),
               "-ngl", "0", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-b", "512", "-ub", "128",
               "--spec-type", "draft-mtp", "--spec-draft-n-max", "5", "--spec-draft-n-min", "0",
               "--spec-draft-p-min", "0", "--spec-draft-sampling", "probabilistic", "--cache-ram", "0",
               "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16", "--spec-draft-backend-sampling",
               "--backend-sampling", "-t", str(threads), "-tb", str(threads), "--host", "127.0.0.1", "--port", str(port)]
    if mode == "meta":
        command += ["-sm", "tensor", "--device", "CPU,CPU", "--tensor-split", "50,50"]
    (dest / "command.json").write_text(json.dumps(command))
    records = []
    with (dest / "server.log").open("w") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(300):
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited: {dest}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                        break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError(f"health timeout: {dest}")
            tokens = [post(port, "tokenize", {"content": p, "add_special": True})["tokens"] for p in prompts]
            for seed in seeds:
                for offset in range(0, len(tokens), np):
                    group = list(range(offset, min(offset + np, len(tokens))))
                    barrier = threading.Barrier(len(group))

                    def request(i):
                        body = {"prompt": tokens[i], "id_slot": i - offset, "n_predict": predict,
                                "seed": seed, "temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0.05,
                                "ignore_eos": True, "cache_prompt": False, "return_tokens": True}
                        barrier.wait()
                        started = time.monotonic()
                        response = post(port, "completion", body)
                        ended = time.monotonic()
                        return {"prompt": i, "seed": seed, "started": started, "ended": ended, "response": response}

                    with concurrent.futures.ThreadPoolExecutor(max_workers=np) as pool:
                        records += list(pool.map(request, group))
                    (dest / "records.json").write_text(json.dumps(records))
                    print(f"M: {dest.name}: {len(records)} requests", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    return records


def identity(record):
    r = record["response"]
    return r["tokens"], r["timings"]["draft_n"], r["timings"]["draft_n_accepted"]


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("patch", type=Path)
    parser.add_argument("--prompts", type=int, default=20)
    parser.add_argument("--predict", type=int, default=32)
    parser.add_argument("--modes", nargs="+", default=["cpu", "meta"])
    parser.add_argument("--parallel", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7])
    parser.add_argument("--patch-fuse", type=int, default=1)
    parser.add_argument("--continue-on-difference", action="store_true")
    args = parser.parse_args()
    summary = []
    for mode in args.modes:
        for np in args.parallel:
            base = run(args.base.resolve(), "base", mode, np, args.out, PROMPTS[:args.prompts], args.seeds, args.predict, 0)
            patch = run(args.patch.resolve(), "patch", mode, np, args.out, PROMPTS[:args.prompts], args.seeds, args.predict, args.patch_fuse)
            differences = [(a["prompt"], a["seed"]) for a, b in zip(base, patch) if identity(a) != identity(b)]
            summary.append({"mode": mode, "np": np, "requests_M": len(base), "differences": differences})
            (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
            if differences and not args.continue_on_difference:
                raise RuntimeError(f"identity failed: {summary[-1]}")
    print(json.dumps(summary, indent=2))
    if any(s["differences"] for s in summary):
        raise RuntimeError("identity matrix contains differences")


if __name__ == "__main__":
    main()
