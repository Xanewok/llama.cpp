"""CPU prefill host attribution using the existing in-process server fixture."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

from attrib import union_ns

ROOT = Path(__file__).resolve().parent.parent
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
campaign = os.environ.get("CPU_CAMPAIGN", "prefill")
assert campaign.replace("-", "").isalnum(), "unsafe CPU campaign name"
OUT = ROOT / "codex/mask-reuse-results" / campaign
OUT.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
           TMPDIR=str(ROOT / "build-tmp"), PYTHONDONTWRITEBYTECODE="1", CPU_PREFILL_WORDS=os.environ.get("CPU_PREFILL_WORDS", "512"), CPU_UBATCH=os.environ.get("CPU_UBATCH", "32"))
model = os.environ.get("CPU_TARGET_MODEL", str(ROOT / "build-tmp/reuse-untied-target.gguf"))
results = []
reference = None
arms = os.environ.get("CPU_ARMS", "patch,unpooled,disabled").split(",")
for boot, arm in enumerate(arms, 1):
    label = f"{boot}-{arm}"
    binary = ROOT / ("/home/xanewok/repos/stack3/build-cpu" if arm == "up" else "build-cpu") / "bin/reuse-server-cpu"
    trace = OUT / (label + ".jsonl")
    if trace.exists():
        trace.unlink()
    path = OUT / (label + ".bin")
    child = dict(env, LLAMA_ROUND_TRACE=str(trace), LD_LIBRARY_PATH=str(binary.parent))
    child.pop("LLAMA_GRAPH_REUSE_DISABLE", None)
    child.pop("GGML_META_PLAN_REUSE", None)
    if arm == "unpooled":
        child["GGML_META_PLAN_REUSE"] = "0"
    if arm == "disabled":
        child["LLAMA_GRAPH_REUSE_DISABLE"] = "1"
    with (OUT / (label + ".log")).open("w") as log:
        subprocess.run(["nice", "-n", "19", str(binary), model, str(path), "1", "3", "1729", "16", "2", "1"],
                       cwd=ROOT, env=child, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    data = path.read_bytes()
    if reference is None:
        reference = data
    assert data == reference, "prefill tokens/final-logit bytes differ"
    records = json.loads(Path(str(path) + ".stats.json").read_text())
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    summaries = []
    for record in records:
        phases = []
        for row in rows:
            for index, phase in enumerate(row["phases"]):
                if record["begin_ns"] <= phase["begin_ns"] < record["end_ns"]:
                    events = [e for e in row["events"] if e["phase"] == index]
                    ubatches = [e for e in events if e["kind"] == "ubatch"]
                    phases.append(dict(phase, ubatches=ubatches, events=events))
        groups = {}
        for label, selected in (("target_prefill", [p for p in phases if p["phase"] == "V" and p["n_tokens"] > 4]),
                                ("draft_prefill", [p for p in phases if p["phase"] == "C" and p["n_tokens"] > 4]),
                                ("decode", [p for p in phases if p["n_tokens"] <= 4])):
            counts, host_us, byte_counts = {}, {}, {}
            ubatches = [e for p in selected for e in p["ubatches"]]
            for phase in selected:
                for counter in phase["counters"]:
                    kind = counter["kind"]
                    counts[kind] = counts.get(kind, 0) + counter["count"]
                    host_us[kind] = host_us.get(kind, 0) + counter["host_us"]
                    byte_counts[kind] = byte_counts.get(kind, 0) + counter["bytes"]
            groups[label] = {"ubatches_M": len(ubatches), "tokens_M": sum(e["count"] for e in ubatches),
                             "counts_M": counts, "host_us_M": host_us, "bytes_M": byte_counts,
                             "per_ubatch": [{"tokens_M": e["count"], "outputs_M": e["bytes"],
                                             "host_us_M": (e["end_ns"]-e["begin_ns"])/1000,
                                             "events": [x for p in selected for x in p["events"]
                                                        if e["begin_ns"] <= x["begin_ns"] and x["end_ns"] <= e["end_ns"]]}
                                            for e in ubatches]}
        summaries.append({"request": record["request"], "timings_M": record["response"]["timings"], "groups": groups})
    results.append({"arm": arm, "sha256": hashlib.sha256(data).hexdigest(), "requests": summaries})
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("M:", arm, [{key: {"ubatches": group["ubatches_M"], "miss": group["counts_M"].get("graph_miss", 0),
                             "meta_build_us": group["host_us_M"].get("meta_build", 0),
                             "arena_count": group["counts_M"].get("meta_arena", 0),
                             "arena_bytes": group["bytes_M"].get("meta_arena", 0)}
                       for key, group in r["groups"].items()} for r in summaries], flush=True)


def allocation_contract(result):
    target = result["requests"][0]["groups"]["target_prefill"]
    assert target["counts_M"].get("meta_arena", 0) == 0, "prefill allocated fresh meta arenas"


patches = [r for r in results if r["arm"] == "patch"]
for result in patches:
    allocation_contract(result)
for result in results:
    if result["arm"] == "unpooled":
        assert patches, "negative control needs a candidate"
        try:
            allocation_contract(result)
        except AssertionError:
            print("M: runtime unpooled control fails the arena-allocation contract; bytes remain identical", flush=True)
        else:
            raise AssertionError("unpooled negative control passed")
    if result["arm"] == "disabled":
        assert all(g["counts_M"].get("graph_hit", 0) == 0 for r in result["requests"] for g in r["groups"].values())
        assert any(g["counts_M"].get("graph_hit", 0) > 0 for p in patches for r in p["requests"] for g in r["groups"].values())
        print("M: reuse-disabled control removes graph hits; bytes remain identical", flush=True)

summary = []
for row in results:
    requests = []
    for record in row["requests"]:
        groups = {}
        for name, group in record["groups"].items():
            build_ms = sum(union_ns([(e["begin_ns"], e["end_ns"]) for e in u["events"]
                                    if e["kind"] in ("llama_build", "meta_build")]) for u in group["per_ubatch"]) / 1e6
            n = group["ubatches_M"]
            groups[name] = {"ubatches_M": n, "tokens_M": group["tokens_M"],
                           "misses_M": group["counts_M"].get("graph_miss", 0),
                           "hits_M": group["counts_M"].get("graph_hit", 0),
                           "meta_builds_M": group["counts_M"].get("meta_build", 0),
                           "meta_executes_D": group["counts_M"].get("meta_compute", 0) - group["counts_M"].get("meta_plan_create", 0),
                           "arena_creates_M": group["counts_M"].get("meta_arena", 0),
                           "arena_bytes_M": group["bytes_M"].get("meta_arena", 0),
                           "build_union_ms_D": build_ms, "build_us_per_ubatch_D": build_ms * 1000 / n if n else None,
                           "retention_ms_M": group["host_us_M"].get("meta_retention", 0) / 1000,
                           "arena_init_ms_M": group["host_us_M"].get("meta_arena", 0) / 1000,
                           "input_ms_M": group["host_us_M"].get("set_inputs", 0) / 1000}
        requests.append({"request": record["request"], "prompt_ms_M": record["timings_M"]["prompt_ms"], "groups": groups})
    summary.append({"arm": row["arm"], "sha256": row["sha256"], "requests": requests})
(ROOT / f"codex/{campaign}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
