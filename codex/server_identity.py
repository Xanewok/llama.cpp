"""CPU llama-server identity, rebased tree vs topic/stack3-mask, serving flags minus the GPU-only ones.

Usage: server_identity.py OUT_DIR [ARM=BINARY ...]
Each arm boots once and answers the same pre-tokenized /completion requests in the same order.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
MODEL = os.environ.get("MODEL", str(Path.home() / "models/Qwen3.5-4B-UD-Q4_K_XL.gguf"))
VOCAB = Path.home() / "models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json"
N_PREDICT = int(os.environ.get("N_PREDICT", "64"))
EXTRA = json.loads(os.environ.get("SERVER_EXTRA", "[]"))
PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain in two sentences why the sky is blue.",
    "List five prime numbers greater than 100 and explain how you checked them.",
    "Translate to Polish: The quick brown fox jumps over the lazy dog.",
    "def merge_sort(xs):\n    \"\"\"Sort a list with merge sort.\"\"\"\n",
]
SEEDS = [1, 7, 1729, 2718, 31337]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port, path, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def command(binary, port):
    return [str(binary), "-m", MODEL, "-c", "102400", "-np", "1", "-ctk", "q8_0", "-ctv", "q8_0", "-ngl", "99", "-fa", "on",
            "--spec-type", "draft-mtp", "--spec-draft-n-max", "5", "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
            "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16", "-b", "2048", "-ub", "512", "--jinja",
            "--temp", "1", "--top-k", "20", "--top-p", "0.95", "--min-p", "0.05", "--metrics",
            "--spec-draft-sampling", "probabilistic", "--spec-draft-backend-sampling", "--backend-sampling",
            "-t", "8", "-tb", "8", "--spec-draft-vocab", str(VOCAB), "--host", "127.0.0.1", "--port", str(port), *EXTRA]


def run_arm(name, binary, out, prompts_tok):
    port = free_port()
    env = dict(os.environ, GGML_SCHED_INPUT_STAGING="1", OMP_NUM_THREADS="8", LD_LIBRARY_PATH=str(Path(binary).parent))
    log = (out / f"{name}-server.log").open("w")
    proc = subprocess.Popen(["nice", "-n", "19", *command(binary, port)], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            if proc.poll() is not None:
                raise RuntimeError(f"{name}: server exited rc={proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError(f"{name}: server did not become healthy")
        if prompts_tok is None:
            prompts_tok = [post(port, "/tokenize", {"content": p, "add_special": True})["tokens"] for p in PROMPTS]
        records = []
        cases = [("greedy", i, None) for i in range(len(PROMPTS))] + [("seeded", i, s) for i, s in zip(range(len(PROMPTS)), SEEDS)]
        for kind, i, seed in cases:
            body = {"prompt": prompts_tok[i], "n_predict": N_PREDICT, "cache_prompt": True, "return_tokens": True,
                    "temperature": 0.0 if kind == "greedy" else 1.0}
            if seed is not None:
                body["seed"] = seed
            r = post(port, "/completion", body)
            t = r.get("timings", {})
            records.append({"kind": kind, "prompt": i, "seed": seed, "tokens": r["tokens"], "content": r["content"],
                            "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted")})
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as r:
            (out / f"{name}-metrics.txt").write_bytes(r.read())
        (out / f"{name}.json").write_text(json.dumps(records, indent=1))
        return prompts_tok, records
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    arms = [a.split("=", 1) for a in sys.argv[2:]]
    prompts_tok = None
    results = {}
    for name, binary in arms:
        prompts_tok, results[name] = run_arm(name, binary, out, prompts_tok)
    (out / "prompts.json").write_text(json.dumps(prompts_tok))
    ref_name = arms[0][0]
    for name, _ in arms[1:]:
        for a, b in zip(results[ref_name], results[name]):
            same = a["tokens"] == b["tokens"]
            print(f"{a['kind']:6s} prompt {a['prompt']} seed {a['seed']}: {ref_name} vs {name}: {'IDENTICAL' if same else 'DIFFER'}"
                  f" ({len(a['tokens'])}/{len(b['tokens'])} tokens, sha {hashlib.sha256(json.dumps(a['tokens']).encode()).hexdigest()[:12]}"
                  f"/{hashlib.sha256(json.dumps(b['tokens']).encode()).hexdigest()[:12]}, drafts {a['draft_n']}/{a['draft_n_accepted']} vs {b['draft_n']}/{b['draft_n_accepted']})")


main()
