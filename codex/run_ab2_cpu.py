"""CPU-only cached-request A/B against the frozen source base; no HTTP listener."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

from gpu_harness import steady
from attrib import union_ns

ROOT = Path(__file__).resolve().parent.parent
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
campaign = "ab2-candidate" if os.environ.get("CPU_TARGET_MODEL") else "ab2"
out = ROOT / "codex/reuse-results" / campaign
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
           TMPDIR=str(ROOT / "build-tmp"), PYTHONDONTWRITEBYTECODE="1")
env.pop("LLAMA_GRAPH_REUSE_DISABLE", None)
model = os.environ.get("CPU_TARGET_MODEL", os.environ.get("MODEL", str(Path.home() / "models/Qwen3.5-4B-UD-Q4_K_XL.gguf")))
results = []


def measure(label, arm, k, disabled=False):
    binary = ROOT / ("build-base-cpu" if arm == "up" else "build-cpu") / "bin/reuse-server-cpu"
    trace = out / (label + ".jsonl")
    if trace.exists():
        trace.unlink()
    path = out / (label + ".bin")
    child = dict(env, LLAMA_ROUND_TRACE=str(trace))
    if disabled:
        child["LLAMA_GRAPH_REUSE_DISABLE"] = "1"
    with (out / (label + ".log")).open("w") as log:
        subprocess.run(["nice", "-n", "19", str(binary), model, str(path), "1", str(k), "1729", "64", "2", "1"],
                       cwd=ROOT, env=child, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    records = json.loads(Path(str(path) + ".stats.json").read_text())
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    summaries = []
    for record in records:
        sample = steady(record, k, 40)
        full = [row for row in rows if record["begin_ns"] <= row["phases"][0]["begin_ns"] < record["end_ns"]
                and 0 < row["next_begin_ns"] < record["end_ns"]
                and [p["phase"] for p in row["phases"]] == ["V", "C", *[f"D{i}" for i in range(1, k + 1)]]
                and row["phases"][0]["n_tokens"] == k + 1]
        assert len(full) >= 40, (label, len(full))
        def counts(selected):
            result = {p: {c: 0 for c in ("graph_hit", "graph_miss", "meta_build")} for p in ("C", "D1")}
            build = 0
            for row in selected:
                build += union_ns([(e["begin_ns"], e["end_ns"]) for e in row["events"]
                                   if row["phases"][e["phase"]]["phase"] in result and e["kind"] in ("llama_build", "meta_build")]) / 1e6
                for phase in row["phases"]:
                    if phase["phase"] in result:
                        for count in phase["counters"]:
                            if count["kind"] in result[phase["phase"]]:
                                result[phase["phase"]][count["kind"]] += count["count"]
            return {"counts_M": result, "build_union_ms_D": build, "rounds_M": len(selected)}
        sample.pop("intervals")
        timings = record["response"]["timings"]
        assert record["request"] == 0 or timings["cache_n"] > 0, "cached request lost its prefix"
        summaries.append({"request": record["request"], "metrics_M": record["metrics"], "cached_tokens_M": timings["cache_n"],
                          "prompt_ms_M": timings["prompt_ms"], "predicted_ms_M": timings["predicted_ms"], **sample,
                          "first_three_full_rounds": counts(full[:3]), "later_full_rounds": counts(full[3:])})
    return path.read_bytes(), {"label": label, "arm": arm, "k": k, "disabled": disabled,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "requests": summaries}


for k in ((5,) if os.environ.get("CPU_TARGET_MODEL") else (3, 5)):
    reference = None
    for boot, arm in enumerate(("up", "patch", "patch", "up"), 1):
        data, result = measure(f"k{k}-{boot}-{arm}", arm, k)
        if reference is None:
            reference = data
        assert data == reference, "cached-request token/final-logit byte identity failed"
        results.append(result)
        (ROOT / f"codex/{campaign}-cpu-results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"M: K={k} boot={boot} arm={arm}: both requests have >=40 SSE intervals, cached prefix and byte identity", flush=True)
    if k == 5:
        data, result = measure("k5-negative", "patch", k, disabled=True)
        assert data == reference, "reuse-disable changed tokens/logits"
        assert all(r["later_full_rounds"]["counts_M"][p]["graph_hit"] == 0 for r in result["requests"] for p in ("C", "D1"))
        assert any(r["later_full_rounds"]["counts_M"][p]["graph_hit"] > 0 for r in results[-2]["requests"] for p in ("C", "D1")), "patch never reused a graph"
        results.append(result)
        (ROOT / f"codex/{campaign}-cpu-results.json").write_text(json.dumps(results, indent=2) + "\n")
        print("M: disabling reuse fails the C/D1-hit contract while preserving cached-request bytes", flush=True)
