"""Load/VRAM/throughput smoke for Swift IQ4_XS on the 5060 Ti x2 pair at one tensor split.

Server flags mirror skill profile swift-daily (ctx 90112) except devices and --tensor-split.
Prompts are repo source text cut to exact token counts; cache_prompt=false so every depth is a
cold prefill. Speeds come from llama-server's per-request `timings`.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.request

ROOT = Path("/home/xanewok/repos/qwen38-eval")
LLAMA = Path("/home/xanewok/repos/llama.cpp/build/bin/llama-server")
MODEL = ROOT / ".cache/hillclimb-models/swift-iq4-xs/Swift-Qwen3.8-27B-IQ4_XS.gguf"
TEMPLATE = ROOT / ("results/2026-09-12-mixed-pair/campaign-swift-fullsuite-20260915/"
                   "swift-iq4-xs/attempts/66ebeeaec6a7043b-01/chat-template.jinja")
DEVICES = ["GPU-ee9a3956-297c-604b-b4c1-f07930d27e46", "GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f"]
DEPTHS = [512, 8192, 32768, 65536, 86000]
N_PREDICT = 512


def call(url, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], threading.Event()

    def run(self):
        fields = "uuid,memory.used,memory.free,power.draw,utilization.gpu,temperature.gpu,pcie.link.gen.current"
        while not self.stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout
            row = {"t": time.time()}
            for line in out.strip().splitlines():
                uuid, *vals = [v.strip() for v in line.split(",")]
                if uuid in DEVICES:
                    row[str(DEVICES.index(uuid))] = dict(zip(fields.split(",")[1:], map(float, vals)))
            self.samples.append(row)
            time.sleep(0.5)

    def window(self, start, end):
        rows = [s for s in self.samples if start <= s["t"] <= end]
        return {gpu: {"min_free_mib": min(r[gpu]["memory.free"] for r in rows),
                      "max_used_mib": max(r[gpu]["memory.used"] for r in rows),
                      "max_power_w": max(r[gpu]["power.draw"] for r in rows),
                      "mean_util": sum(r[gpu]["utilization.gpu"] for r in rows) / len(rows),
                      "max_temp_c": max(r[gpu]["temperature.gpu"] for r in rows)}
                for gpu in ("0", "1")} if rows else {}


def corpus_text():
    parts = [p.read_text(errors="ignore") for pattern in ("scripts/**/*.py", "tests/*.py", "*.md")
             for p in sorted(ROOT.glob(pattern))]
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--ctx", type=int, default=90112)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=LLAMA)
    parser.add_argument("--draft-max", default="3")
    parser.add_argument("--spec-type", default="draft-mtp")
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--engine", choices=["mainline", "ik"], default="mainline",
                        help="ik: ik_llama.cpp flag surface (--spec-type mtp:n_max=K,p_min=0.0, no --spec-draft-*)")
    parser.add_argument("--sm", default="tensor", help="--split-mode value (ik: graph)")
    parser.add_argument("--extra", default="", help="extra llama-server args, space-separated")
    parser.add_argument("--depths", type=lambda v: [int(x) for x in v.split(",")], default=DEPTHS)
    parser.add_argument("--devices", default=",".join(DEVICES), help="comma-separated GPU UUIDs (default: the local pair)")
    parser.add_argument("--template", type=Path, default=TEMPLATE, help="chat template file")
    parser.add_argument("--corpus", type=Path, help="text file to cut prompts from (default: this repo's sources)")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    DEVICES[:] = args.devices.split(",")
    url = f"http://127.0.0.1:{args.port}"
    command = [str(args.binary), "-m", str(args.model), "-c", str(args.ctx), "-np", "1", "-ctk", "q8_0", "-ctv", "q8_0",
               "-ngl", "99", "-fa", "on"]
    if args.engine == "ik":
        command += ["--spec-type", f"mtp:n_max={args.draft_max},p_min=0.0"]
    else:
        command += ["--spec-type", args.spec_type, "--spec-draft-n-max", args.draft_max,
                    "--spec-draft-n-min", "0", "--spec-draft-p-min", "0", "--spec-draft-type-k", "f16",
                    "--spec-draft-type-v", "f16"]
    command += ["-sm", args.sm, "--tensor-split", args.split,
               "-b", "2048", "-ub", "128", "--jinja", "--chat-template-file", str(args.template),
               "--temp", "1", "--top-k", "20", "--top-p", "0.95", "--min-p", "0.05",
               "--host", "127.0.0.1", "--port", str(args.port), "--metrics"]
    if args.draft_model:
        command += ["-md", str(args.draft_model), "-ngld", "99"]
    command += args.extra.split()
    extra_env = dict(item.split("=", 1) for item in args.env)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(DEVICES), **extra_env)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log = open(args.out.with_suffix(".server.log"), "w")
    sampler = Sampler()
    sampler.start()
    time.sleep(1.5)
    idle = sampler.samples[-1]
    t0 = time.time()
    server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    result = {"split": args.split, "ctx": args.ctx, "command": command, "env": extra_env, "idle_before": idle, "runs": []}
    try:
        while True:
            if server.poll() is not None:
                raise SystemExit(f"server exited {server.returncode} during load; see {log.name}")
            try:
                if call(url + "/health", timeout=3).get("status") == "ok":
                    break
            except Exception:
                pass
            if time.time() - t0 > 600:
                raise SystemExit("load timeout")
            time.sleep(1)
        result["load_seconds"] = time.time() - t0
        time.sleep(3)
        result["after_load"] = sampler.samples[-1]
        tokens = call(url + "/tokenize", {"content": args.corpus.read_text(errors="ignore") if args.corpus else corpus_text()})["tokens"]
        while len(tokens) < max(args.depths):
            tokens += tokens
        for depth in args.depths:
            if depth + N_PREDICT > args.ctx:
                continue
            prompt = call(url + "/detokenize", {"tokens": tokens[:depth]})["content"]
            start = time.time()
            response = call(url + "/completion", {"prompt": prompt, "n_predict": N_PREDICT, "cache_prompt": False,
                                                  "seed": 1729, "ignore_eos": True})
            end = time.time()
            timings = response.get("timings", {})
            run = {"depth_target": depth, "wall_s": end - start, "timings": timings,
                   "gpu": sampler.window(start, end)}
            result["runs"].append(run)
            print(json.dumps({"depth": timings.get("prompt_n"), "pp": round(timings.get("prompt_per_second", 0), 1),
                              "tg": round(timings.get("predicted_per_second", 0), 1),
                              "draft_acc": f'{timings.get("draft_n_accepted")}/{timings.get("draft_n")}',
                              "min_free": {g: v["min_free_mib"] for g, v in run["gpu"].items()}}), flush=True)
    finally:
        server.terminate()
        try:
            server.wait(60)
        except subprocess.TimeoutExpired:
            server.kill()
        sampler.stop.set()
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
