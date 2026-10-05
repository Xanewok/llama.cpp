"""CLI/env plumbing of --kv-pad on the CPU server copy (named cpu-serving, never llama-server): the value reaches the target and
the MTP draft context, n_ctx is padded to it, a non-multiple of 256 is rounded up with a warning, the default stays 256.
Usage: kvpad_cli_check.py [BIN_DIR]; one short boot per case, nice 19, 4 threads, CUDA hidden."""
import os, re, socket, subprocess, sys, time, urllib.request, json
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
ROOT = Path(__file__).resolve().parent.parent
BIN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build-cpu/bin"
MODEL = os.environ.get("MODEL", "/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf")
# (label, extra argv, env, -c, expected n_kv_pad lines, expected n_ctx, warning expected)
CASES = [("default", [], {}, 5000, "256", 5120, False),
         ("flag4096", ["--kv-pad", "4096"], {}, 5000, "4096", 8192, False),
         ("env1024", [], {"LLAMA_ARG_KV_PAD": "1024"}, 5000, "1024", 5120, False),
         ("flag300", ["--kv-pad", "300"], {}, 5000, "512", 5120, True)]


def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


ok = True
for label, extra, env, ctx, pad, n_ctx, warn in CASES:
    p = port(); log = ROOT / f"codex/kvpad-results/cli-{label}.log"
    cmd = ["nice", "-n", "19", str(BIN / "cpu-serving"), "-m", MODEL, "-c", str(ctx), "-np", "1", "-ngl", "0", "-fa", "on",
           "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "-t", "4", "-tb", "4", "--host", "127.0.0.1", "--port", str(p), *extra]
    proc = subprocess.Popen(cmd, env=dict(os.environ, LD_LIBRARY_PATH=str(BIN), LLAMA_ARG_LOG_VERBOSITY="4", **env), stdout=log.open("w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{p}/health", timeout=2).status == 200: break
            except Exception: time.sleep(1)
        req = urllib.request.Request(f"http://127.0.0.1:{p}/completion", data=json.dumps({"prompt": "Hello", "n_predict": 8, "temperature": 0}).encode(), headers={"Content-Type": "application/json"})
        served = len(json.loads(urllib.request.urlopen(req, timeout=300).read())["content"]) > 0
    finally:
        proc.terminate(); proc.wait(timeout=60)
    text = log.read_text()
    pads = re.findall(r"llama_context: n_kv_pad += (\d+)\s*$", text, re.M)
    ctxs = set(re.findall(r"llama_context: n_ctx += (\d+)", text))
    good = len(pads) >= 2 and set(pads) == {pad} and ctxs == {str(n_ctx)} and served and (("not a multiple of 256" in text) == warn)
    ok &= good
    print(("PASS" if good else "FAIL"), label, "n_kv_pad lines", pads, "n_ctx", sorted(ctxs), "warn", "not a multiple of 256" in text, "served", served, flush=True)
sys.exit(0 if ok else 1)
