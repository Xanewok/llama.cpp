"""CPU-only, two concurrent MTP requests per temperature against frozen baseline bytes."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build-tmp/np2-identity"
MODEL = os.environ.get("LLAMACPP_TEST_MODELFILE", "/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf")


def post(port, path, body):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read())


def run(arm, binary, temperature, prompts):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [str(binary), "-m", MODEL, "-ngl", "0", "-np", "2", "-c", "8192", "-b", "512", "-ub", "128",
               "-t", "8", "-tb", "8", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--cache-ram", "0",
               "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
               "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16", "--spec-draft-sampling", "probabilistic",
               "--no-spec-draft-backend-sampling", "--host", "127.0.0.1", "--port", str(port)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
               GGML_SCHED_INPUT_STAGING="1", LD_LIBRARY_PATH=str(binary.parent))
    label = f"{arm}-t{temperature}"
    begin = time.monotonic()
    with (OUT / f"{label}.log").open("w") as log:
        proc = subprocess.Popen(["nice", "-n", "19", *command], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"{label}: server exited {proc.returncode}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("CPU server readiness")
                    time.sleep(0.2)
            if prompts is None:
                prompts = [post(port, "/tokenize", {"content": text, "add_special": True})["tokens"] for text in (
                    "List the first fifty prime numbers, one per line, with a brief explanation for each.",
                    "Write an iterative Python merge sort and explain each step in detail.")]
            barrier = threading.Barrier(2)

            def request(slot):
                barrier.wait(timeout=10)
                response = post(port, "/completion", {"prompt": prompts[slot], "id_slot": slot,
                    "n_predict": 48, "seed": 1729 + slot, "temperature": temperature,
                    "top_k": 20, "top_p": 0.95, "min_p": 0.05, "cache_prompt": False,
                    "return_tokens": True, "ignore_eos": True})
                (OUT / f"{label}-s{slot}.json").write_text(json.dumps(response, indent=2) + "\n")
                timings = response["timings"]
                assert len(response["tokens"]) == timings["predicted_n"] == 48
                assert timings["draft_n"] > 0
                return {"slot": slot, "tokens": response["tokens"], "draft_n": timings["draft_n"],
                        "draft_n_accepted": timings["draft_n_accepted"]}

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                records = list(pool.map(request, range(2)))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    return prompts, records, time.monotonic() - begin


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = None
    results = []
    for temperature in (0, 1):
        expected = None
        for arm, binary in (("up", ROOT / "build-cpu-up/bin/cpu-serving"), ("patch", ROOT / "build-cpu/bin/cpu-serving")):
            prompts, records, elapsed = run(arm, binary, temperature, prompts)
            results.append({"arm": arm, "temperature": temperature, "wall_s_M": elapsed, "records": records})
            (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            if expected is None:
                expected = records
            else:
                assert records == expected, f"t{temperature}: token/draft identity failed; see {OUT}/results.json"
                for record in records:
                    digest = hashlib.sha256(json.dumps(record["tokens"]).encode()).hexdigest()
                    print(f"M PASS t{temperature} s{record['slot']}: token sha256={digest}, drafts={record['draft_n']}, accepted={record['draft_n_accepted']}", flush=True)


if __name__ == "__main__":
    main()
