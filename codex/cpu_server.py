#!/usr/bin/env python3
import json
import http.client
import socket
import os
import pathlib
import subprocess
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
root = pathlib.Path(__file__).resolve().parent.parent
model = sys.argv[1]
slots = int(os.environ.get("SLOTS", "1"))
assert slots in (1, 2)
out = root / "codex/cpu-server-results" / f"np{slots}"
out.mkdir(parents=True, exist_ok=True)
vocab = out / "vocab.json"
vocab.write_text(json.dumps(list(range(248299, 0, -10))))
sock_path = out / "server.sock"
assert not sock_path.exists(), f"refuse existing socket: {sock_path}"


class LocalConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(sock_path))


def call(path, body=None):
    connection = LocalConnection("localhost", timeout=120)
    try:
        connection.request("POST" if body is not None else "GET", path,
                           body=json.dumps(body) if body is not None else None,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise OSError(f"HTTP {response.status}: {response.read()}")
        return json.loads(response.read())
    finally:
        connection.close()


runs = {}
for k in (3, 4, 5):
    for arm in ("plain", "chain"):
        tag = f"{arm}{k}"
        cmd = [str(root / "build-cpu/bin/chain-cpu"), "-m", model, "-ngl", "0", "-fit", "off", "-t", "8", "-tb", "8",
               "-c", "2048", "-np", str(slots), "-b", "128", "-ub", "32", "-fa", "on", "--host", str(sock_path),
               "--spec-type", "draft-mtp", "--spec-draft-n-max", str(k), "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
               "--spec-draft-vocab", str(vocab), "--spec-draft-sampling", "probabilistic", "--spec-draft-backend-sampling",
               "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16", "--backend-sampling"]
        if arm == "chain":
            cmd += ["--spec-mtp-chain"]
        with (out / f"{tag}.server.log").open("w") as log:
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            try:
                for _ in range(120):
                    if child.poll() is not None:
                        raise RuntimeError(f"{tag} exited {child.returncode}")
                    try:
                        if call("/health").get("status") == "ok":
                            break
                    except OSError:
                        pass
                    time.sleep(0.5)
                else:
                    raise RuntimeError("health timeout")
                for seed in (1, 7):
                    r = call("/completion", {"prompt": "The capital city of Poland is", "seed": seed, "n_predict": 32,
                             "temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0.05, "cache_prompt": False,
                             "return_tokens": True, "id_slot": seed % slots})
                    (out / f"{tag}-{seed}.json").write_text(json.dumps(r))
                    assert r["timings"]["draft_n"] > 0, "drafts did not run"
                    runs[(tag, seed)] = r
            finally:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                sock_path.unlink(missing_ok=True)
        log_text = (out / f"{tag}.server.log").read_text()
        assert "MTP chain unavailable" not in log_text
    for seed in (1, 7):
        a, b = runs[(f"plain{k}", seed)], runs[(f"chain{k}", seed)]
        assert a["content"] == b["content"] and a.get("tokens") == b.get("tokens"), f"k={k} seed={seed}: completion differs"
        assert a["timings"]["draft_n"] == b["timings"]["draft_n"]
        assert a["timings"]["draft_n_accepted"] == b["timings"]["draft_n_accepted"]
        print(f"M: k={k} seed={seed}: CPU server completion, draft count and acceptance identical", flush=True)
