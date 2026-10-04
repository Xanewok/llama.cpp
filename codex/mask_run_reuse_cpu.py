"""Sequential CPU A/B. Each inference process is nice 19 and uses at most eight CPU threads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
parser = argparse.ArgumentParser()
parser.add_argument("--real", action="store_true")
parser.add_argument("--seeds", type=int, default=16)
parser.add_argument("--rounds", type=int, default=12)
args = parser.parse_args()
out = ROOT / "codex/mask-reuse-results" / ("real" if args.real else "synthetic")
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
           TMPDIR=str(ROOT / "build-tmp"), PYTHONDONTWRITEBYTECODE="1")
env.pop("LLAMA_GRAPH_REUSE_DISABLE", None)
env.pop("DEBUG_FIXTURE", None)
if args.real:
    env["CPU_TARGET_MODEL"] = str(ROOT / "build-tmp/reuse-untied-target.gguf")
model = os.environ.get("MODEL", str(Path.home() / "models/Qwen3.5-4B-UD-Q4_K_XL.gguf")) if args.real else "-"


def preparation_us(row):
    intervals = sorted((e["begin_ns"], e["end_ns"]) for e in row["events"]
                       if row["phases"][e["phase"]]["phase"] in ("C", "D1") and e["kind"] in ("llama_build", "meta_build"))
    total = 0
    last = 0
    for start, end in intervals:
        total += max(0, end - max(start, last))
        last = max(last, end)
    return total / 1000


results = []
for meta in (0, 1):
    for k in (3, 5):
        for seed in range(1, args.seeds + 1):
            expected = None
            for boot, arm in enumerate(("up", "patch", "patch", "up"), 1):
                label = f"m{meta}-k{k}-s{seed}-{boot}-{arm}"
                binary = ROOT / ("/home/xanewok/repos/stack3-mask/build-cpu" if arm == "up" else "build-cpu") / "bin/reuse-cpu"
                path = out / (label + ".bin")
                trace = out / (label + ".jsonl")
                if trace.exists():
                    trace.unlink()
                child_env = dict(env, LLAMA_ROUND_TRACE=str(trace))
                command = ["nice", "-n", "19", str(binary), model, str(meta), str(k), str(seed), str(args.rounds), str(path), "1"]
                with (out / (label + ".log")).open("w") as log:
                    subprocess.run(command, env=child_env, stdout=log, stderr=subprocess.STDOUT, check=True)
                data = path.read_bytes()
                if expected is None:
                    expected = data
                assert data == expected, (label, "token/id/q/logit byte mismatch")
                rows = [json.loads(line) for line in trace.read_text().splitlines()][3:]
                counts = {phase: {"hit_M": 0, "miss_M": 0, "meta_build_M": 0} for phase in ("C", "D1")}
                for row in rows:
                    for phase in row["phases"]:
                        if phase["phase"] not in counts:
                            continue
                        for event in phase["counters"]:
                            key = {"graph_hit": "hit_M", "graph_miss": "miss_M", "meta_build": "meta_build_M"}.get(event["kind"])
                            if key:
                                counts[phase["phase"]][key] += event["count"]
                results.append({"fixture": "real" if args.real else "synthetic", "meta": meta, "k": k, "seed": seed, "boot": boot,
                                "arm": arm, "bytes_M": len(data), "sha256": hashlib.sha256(data).hexdigest(), "steady_rounds_M": len(rows),
                                "C_D1_build_union_us_per_round_D": statistics.mean(map(preparation_us, rows)),
                                "decode_wall_us_per_round_D": statistics.mean(sum(p["end_ns"] - p["begin_ns"] for p in row["phases"]) / 1000 for row in rows), "counts": counts})
            print(f"M: {'real' if args.real else 'synthetic'} meta={meta} K={k} seed={seed} token/id/q/logit byte identity", flush=True)
            (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
summary = []
for meta in (0, 1):
    for k in (3, 5):
        for arm in ("up", "patch"):
            selected = [r for r in results if (r["meta"], r["k"], r["arm"]) == (meta, k, arm)]
            counts = {p: {name: sum(r["counts"][p][name] for r in selected) for name in ("hit_M", "miss_M", "meta_build_M")} for p in ("C", "D1")}
            summary.append({"meta": meta, "k": k, "arm": arm, "runs_M": len(selected), "steady_rounds_M": sum(r["steady_rounds_M"] for r in selected),
                            "C_D1_build_union_us_per_round_median_D": statistics.median(r["C_D1_build_union_us_per_round_D"] for r in selected), "decode_wall_us_per_round_median_D": statistics.median(r["decode_wall_us_per_round_D"] for r in selected), "counts": counts})
result = {"fixture": "real" if args.real else "synthetic", "seeds_M": args.seeds, "rounds_per_run_M": args.rounds,
          "identity": "token/id/q/logit bytes: up/patch/patch/up for every seed, K, backend", "summary": summary}
(ROOT / "codex" / ("mask-real-reuse-results.json" if args.real else "mask-synthetic-reuse-results.json")).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
