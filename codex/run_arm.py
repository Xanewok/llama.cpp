"""One CPU server arm: python3 run_arm.py OUT_DIR BINARY [--env K=V ...] [--prompts N] [--seeds ...] [--predict N]."""
import argparse, json, os, socket, subprocess, time, urllib.request
from pathlib import Path

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

def post(port, ep, body):
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/{ep}", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=900) as r:
        return json.load(r)

def mem_available_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path); ap.add_argument("binary", type=Path)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--prompts", type=int, default=2); ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--predict", type=int, default=24); ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--np", type=int, default=1)
    ap.add_argument("--turns", type=int, default=1, help="continue each completion as a cached second turn")
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert mem_available_gib() >= 10, "MemAvailable < 10 GiB"
    a.out.mkdir(parents=True, exist_ok=True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", LD_LIBRARY_PATH=str(a.binary.parent),
               GGML_SCHED_INPUT_STAGING="1")
    for kv in a.env:
        k, v = kv.split("=", 1); env[k] = v
    cmd = ["nice", "-n", "19", str(a.binary), "-m", MODEL, "-c", str(4096 * a.np), "-np", str(a.np), "-ngl", "0", "-fa", "on",
           "-ctk", "q8_0", "-ctv", "q8_0", "-b", "512", "-ub", "128", "--spec-type", "draft-mtp",
           "--spec-draft-n-max", "5", "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
           "--spec-draft-sampling", "probabilistic", "--cache-ram", "0", "--spec-draft-type-k", "f16",
           "--spec-draft-type-v", "f16", "--spec-draft-backend-sampling", "--backend-sampling",
           "-t", "8", "-tb", "8", "--host", "127.0.0.1", "--port", str(port)] + a.extra
    (a.out / "command.json").write_text(json.dumps({"cmd": cmd, "env": a.env}))
    recs = []
    with (a.out / "server.log").open("w") as log:
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(300):
                if p.poll() is not None: raise RuntimeError("server exited")
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); break
                except Exception: time.sleep(1)
            toks = [post(port, "tokenize", {"content": q, "add_special": True})["tokens"] for q in PROMPTS[:a.prompts]]
            import concurrent.futures, threading
            follow = post(port, "tokenize", {"content": " Now explain it again in one sentence.", "add_special": False})["tokens"]
            def one(i, seed, slot):
                prompt = toks[i]
                out = []
                for turn in range(a.turns):
                    body = {"prompt": prompt, "id_slot": slot, "n_predict": a.predict, "seed": seed, "temperature": 1,
                            "top_k": 20, "top_p": 0.95, "min_p": 0.05, "ignore_eos": True,
                            "cache_prompt": a.turns > 1, "return_tokens": True}
                    r = post(port, "completion", body)
                    out.append({"prompt": i, "seed": seed, "turn": turn, "tokens": r["tokens"],
                                "draft_n": r["timings"].get("draft_n"), "draft_n_accepted": r["timings"].get("draft_n_accepted"),
                                "prompt_n": r["timings"].get("prompt_n")})
                    prompt = prompt + r["tokens"] + follow
                return out
            for seed in a.seeds:
                for off in range(0, len(toks), a.np):
                    group = list(range(off, min(off + a.np, len(toks))))
                    with concurrent.futures.ThreadPoolExecutor(max_workers=a.np) as pool:
                        for res in pool.map(lambda i: one(i, seed, i - off), group):
                            recs += res
                    (a.out / "records.json").write_text(json.dumps(recs))
        finally:
            p.terminate()
            try: p.wait(timeout=30)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
    print(json.dumps([{k: v for k, v in r.items() if k != "tokens"} for r in recs]))

main()
