#!/usr/bin/env python3
"""Read checkpoint headers and existing traces without importing GPU libraries."""
import argparse
import collections
import csv
import hashlib
import json
import pathlib
import sqlite3
import struct

MODELS = {
    "g128": "/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast",
    "g32i8": "/mnt/scratch/work/models/hq/Swift-1.5-Qwen3.8-27B-W4A16-g32-attn-int8",
    "codexh": "/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-g32-attn-int8-codexh",
}
SP = pathlib.Path("/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad")
SHAPES = [
    {"name": "mlp.gate_up_proj", "k": 5120, "n": 17408, "count": 64},
    {"name": "mlp.down_proj", "k": 8704, "n": 5120, "count": 64},
    {"name": "linear_attn.in_proj_qkvz", "k": 5120, "n": 8192, "count": 48},
    {"name": "linear_attn.out_proj", "k": 3072, "n": 5120, "count": 48},
    {"name": "self_attn.qkv_proj", "k": 5120, "n": 7168, "count": 16},
    {"name": "self_attn.o_proj", "k": 3072, "n": 5120, "count": 16},
]


def headers(root):
    result = {}
    sources = []
    for path in sorted(root.glob("*.safetensors")):
        with path.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            data = f.read(size)
        header = json.loads(data)
        overlap = result.keys() & (header.keys() - {"__metadata__"})
        if overlap:
            raise ValueError(f"duplicate tensors: {overlap}")
        result.update({k: v for k, v in header.items() if k != "__metadata__"})
        sources.append({"path": str(path), "header_sha256": hashlib.sha256(data).hexdigest(), "file_bytes": path.stat().st_size})
    if not sources:
        raise ValueError(f"no safetensors at {root}")
    return result, sources


def inventory():
    models = {}
    modules = []
    for arm, folder in MODELS.items():
        root = pathlib.Path(folder)
        config_bytes = (root / "config.json").read_bytes()
        config = json.loads(config_bytes)["quantization_config"]
        tensors, sources = headers(root)
        totals = collections.Counter()
        for name, tensor in sorted(tensors.items()):
            if not name.endswith(".weight_packed"):
                continue
            module = name.removesuffix(".weight_packed")
            scale = tensors[module + ".weight_scale"]
            packed_bytes = tensor["data_offsets"][1] - tensor["data_offsets"][0]
            scale_bytes = scale["data_offsets"][1] - scale["data_offsets"][0]
            category = "body" if module.startswith("model.language_model.layers.") else "other"
            if category == "body" and ".self_attn." in module:
                category = "attention"
            totals[category + "_weight_bytes"] += packed_bytes
            totals[category + "_scale_bytes"] += scale_bytes
            modules.append(dict(arm=arm, module=module, category=category,
                                weight_shape=tensor["shape"], scale_shape=scale["shape"],
                                scale_dtype=scale["dtype"], weight_bytes=packed_bytes, scale_bytes=scale_bytes))
        models[arm] = dict(path=folder, config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                           groups=config["config_groups"], headers=sources, totals=dict(totals))
    return models, modules


def trace(path):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
        rows = con.execute("""
            SELECT k.deviceId, s.value, count(*), sum(k.end-k.start)/1e6,
                   avg(k.end-k.start)/1e3, k.blockX, k.registersPerThread,
                   k.dynamicSharedMemory, k.localMemoryPerThread
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
            WHERE s.value LIKE '%marlin::Marlin<%'
            GROUP BY k.deviceId, s.value, k.blockX, k.registersPerThread,
                     k.dynamicSharedMemory, k.localMemoryPerThread
        """).fetchall()
    return {"path": str(path), "scope": "whole capture, includes warmup; not a matched checkpoint A/B",
            "kernels": [dict(zip(["device", "name", "calls_M", "total_ms_M", "mean_us_M",
                                  "threads", "registers", "dynamic_shared_bytes", "local_bytes"], row)) for row in rows]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path(__file__).parent / "evidence")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    models, modules = inventory()
    baseline = models["g128"]["totals"]
    mixed = models["g32i8"]["totals"]
    if mixed != models["codexh"]["totals"]:
        raise ValueError("codexh byte layout differs")
    expected_weight_bytes = sum(s["k"] * s["n"] * s["count"] for s in SHAPES)
    if expected_weight_bytes != baseline["body_weight_bytes"] + baseline["attention_weight_bytes"]:
        raise ValueError("TP2 fused shapes do not match checkpoint headers")
    if mixed["body_scale_bytes"] != 4 * baseline["body_scale_bytes"]:
        raise ValueError("g32 body scales are not four times g128")
    body_scale_delta = mixed["body_scale_bytes"] - baseline["body_scale_bytes"]
    attn_delta = sum(mixed[k] - baseline[k] for k in ["attention_weight_bytes", "attention_scale_bytes"])
    bw = 448e9
    roofline = {"label": "D, one target body pass per step, TP2 equal sharding; no cache/re-read correction",
                "bandwidth_bytes_s_assumed": bw,
                "body_scale_extra_bytes_rank": body_scale_delta / 2,
                "attention_extra_bytes_rank": attn_delta / 2,
                "body_scale_extra_ms": body_scale_delta / 2 / bw * 1000,
                "attention_extra_ms": attn_delta / 2 / bw * 1000,
                "total_extra_ms": (body_scale_delta + attn_delta) / 2 / bw * 1000}
    traces = [trace(SP / p) for p in ["engine-plan/step0/out/profile/decode8k.sqlite", "vast-hq-d/ab/prof/nsys-n3-1.sqlite"]]
    (args.out / "inventory.json").write_text(json.dumps(dict(models=models, roofline=roofline, traces=traces), indent=2) + "\n")
    with (args.out / "modules.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(modules[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(modules)
    (args.out / "shapes.json").write_text(json.dumps(SHAPES, indent=2) + "\n")
    print(json.dumps({"totals": {a: m["totals"] for a, m in models.items()}, "roofline": roofline}, indent=2))


if __name__ == "__main__":
    main()
