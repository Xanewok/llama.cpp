"""Sequential, resumable full 78-task campaigns for the three Swift variants."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time

from scripts import ROOT
from scripts.instruments import agentic_campaign as a
from scripts.instruments import confirmed_agentic as c
from scripts.instruments import hillclimb as h
from scripts.instruments.chat_templates import BARTOWSKI_PREFIX


# Separate child campaigns preserve each model's measured context and VRAM policy.
MODELS = [
    {"id": "swift-iq4-xs", "repo": "ukisai/Swift-Qwen3.8-27B-GGUF", "revision": "b12fc8f44af7bdf31a3c3309891acb345d3106a2",
     "file": "Swift-Qwen3.8-27B-IQ4_XS.gguf", "bytes": 15688288352,
     "sha256": "27f5531a012a4bb3c3e379318dc62216f057131d34347498fabfc23186a4ffda",
     "local": str(ROOT / ".cache/hillclimb-models/swift-iq4-xs/Swift-Qwen3.8-27B-IQ4_XS.gguf")},
    {"id": "swift-q4-m", "repo": "ukisai/Swift-Qwen3.8-27B-GGUF", "revision": "b12fc8f44af7bdf31a3c3309891acb345d3106a2",
     "file": "Swift-Qwen3.8-27B-Q4_K_M.gguf", "bytes": 18024380576,
     "sha256": "ad5811e291431bd0de1cec0c4004a5eac98daee9850882edac69a823209e88ab",
     "local": str(ROOT / ".cache/hillclimb-models/swift-q4-m/Swift-Qwen3.8-27B-Q4_K_M.gguf")},
    {"id": "swift-iq3-xxs", "repo": "ukisai/Swift-Qwen3.8-27B-GGUF", "revision": "b12fc8f44af7bdf31a3c3309891acb345d3106a2",
     "file": "Swift-Qwen3.8-27B-IQ3_XXS.gguf", "bytes": 12532504672,
     "sha256": "1d16444b9164595fb59cdfc1b11c24b1e167c68238d33a0e4d261495a5178f60",
     "local": str(ROOT / ".cache/hillclimb-models/swift-iq3-xxs/Swift-Qwen3.8-27B-IQ3_XXS.gguf")},
]
PROFILE = {
    "swift-iq4-xs": {"context": 102400, "split": "69,31", "ubatch": 128, "reserves": [512, 256], "recovery_extra_headroom_mib": 128},
    "swift-q4-m": {"context": 92160, "split": "68.75,31.25", "ubatch": 128, "reserves": [128, 256], "recovery_extra_headroom_mib": 0},
    "swift-iq3-xxs": {"context": 102400, "split": "69,31", "ubatch": 128, "reserves": [512, 256], "recovery_extra_headroom_mib": 128},
}


def configure(model_id, tasks):
    model = next(model for model in MODELS if model["id"] == model_id)
    h.MODELS[:] = [json.loads(json.dumps(model))]
    h.TASKS[:] = list(tasks)


def task_ids(assets):
    rows = json.loads((assets / "tasks.json").read_text())
    ids = [row["task_id"] for row in rows]
    if len(ids) != 78 or len(set(ids)) != 78:
        raise ValueError("trusted assets must contain 78 unique tasks")
    return a.TASK_ORDER + [task for task in ids if task not in a.TASK_ORDER]


def queue(model_id, tasks):
    p = PROFILE[model_id]
    config = {key: p[key] for key in ("context", "split", "ubatch")}
    config.update(slots=1, mode="tensor", maximum=3, chat_template=BARTOWSKI_PREFIX)
    return [{"id": "solve-" + model_id + "-" + task.replace("/", "-"), "kind": "profile",
             "model": model_id, "task": task, "workload": "agentic", "targets": [0],
             "configs": [config]} for task in tasks]


def child_policy(model_id):
    p = PROFILE[model_id]
    return {"models": [model_id], "profile": {key: p[key] for key in ("context", "split", "ubatch")},
            "queue_order": "model-first", "production_promotion": False,
            "watchdog_recovery": c.WATCHDOG_RECOVERY,
            "recovery_extra_headroom_mib": p["recovery_extra_headroom_mib"]}


def summarize_child(campaign):
    manifest = h.read_sealed(campaign / "manifest.json")
    rows = []
    for job in manifest["queue"]:
        path = campaign / ("job-" + job["id"] + ".json")
        if path.exists():
            rows.append(h.read_sealed(path))
    measured = [row for row in rows if row.get("status") == "passed" and row.get("probe")]
    probes = [row["probe"] for row in measured]
    metrics = [probe.get("metrics_delta", {}) for probe in probes]
    output = sum(metric.get("decode_tokens", 0) for metric in metrics)
    decode = sum(probe.get("decode_seconds", 0) for probe in probes)
    model = manifest["models"][0]["id"]
    result = {"updated_epoch": time.time(), "state": json.loads((campaign / "state.json").read_text())["status"],
              "model": model, "processed": len(rows), "measured": len(measured),
              "solved": sum(bool(probe.get("task_success")) for probe in probes),
              "timeouts": sum(bool(probe.get("timed_out")) for probe in probes),
              "wall_seconds": sum(probe.get("verified_wall_seconds", 0) for probe in probes),
              "prefill_seconds": sum(probe.get("prefill_seconds", 0) for probe in probes),
              "decode_seconds": decode, "output_tokens": output,
              "decode_tps": output / decode if decode else None,
              "peak_context_tokens": max((probe.get("peak_context_tokens", 0) for probe in probes), default=0),
              "draft_acceptance": (sum(metric.get("draft_accepted_tokens", 0) for metric in metrics) /
                                   sum(metric.get("draft_tokens", 0) for metric in metrics)
                                   if sum(metric.get("draft_tokens", 0) for metric in metrics) else None),
              "config": PROFILE[model]}
    h.atomic_json(campaign / "SUMMARY.json", result)
    return result


def write_parent_summary(campaign, children, state):
    result = {"updated_epoch": time.time(), "state": state, "scope": "three sequential Swift full 78-task campaigns; no production promotion",
              "children": {model_id: summarize_child(child) for model_id, child in children.items()}}
    h.atomic_json(campaign / "SUMMARY.json", result)
    lines = ["# Swift full-suite comparison", "", json.dumps(result, indent=2), "",
             "| Model | State | Processed | Solved | Wall h | Decode tok/s | Output tokens |", "|---|---|---:|---:|---:|---:|---:|"]
    for model_id, row in result["children"].items():
        lines.append("| %s | %s | %s | %s | %.2f | %s | %s |" %
                     (model_id, row["state"], row["processed"], row["solved"], row["wall_seconds"] / 3600,
                      "" if row["decode_tps"] is None else f"{row['decode_tps']:.2f}", row["output_tokens"]))
    (campaign / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def prepare(campaign, assets, tasks):
    campaign.mkdir(parents=True, exist_ok=False)
    children = {}
    for model in MODELS:
        model_id = model["id"]
        child = campaign / model_id
        configure(model_id, tasks)
        p = PROFILE[model_id]
        h.initialize(child, ROOT / ".cache/hillclimb-models", profiles=queue(model_id, tasks), max_attempts=3,
                     agentic_assets=assets, continuation=child_policy(model_id),
                     vram_reserves=tuple(p["reserves"]))
        children[model_id] = child
    h.atomic_json(campaign / "manifest.json", {"schema_version": 1, "models": [m["id"] for m in MODELS],
                                                 "children": {k: str(v) for k, v in children.items()},
                                                 "tasks": tasks, "assets": str(assets), "created": time.time()})
    return children


def run_child(model_id, child, deadline):
    state = json.loads((child / "state.json").read_text())
    if state["status"] == "complete" and not state.get("owned"):
        return summarize_child(child)
    if state["status"] in {"stopped", "error"} and not state.get("owned"):
        reason = state.get("reason", "")
        if reason.startswith("memory_") or reason.startswith("memory_pressure"):
            return summarize_child(child)
        raise RuntimeError(f"child {model_id} stopped: {reason}")
    manifest = h.read_sealed(child / "manifest.json")
    configure(model_id, [job["task"] for job in manifest["queue"]])
    selection = {"models": {model_id: {"ubatch": PROFILE[model_id]["ubatch"]}}}
    c.ConfirmedRunner(child, deadline, selection).run()
    result = summarize_child(child)
    if result["state"] == "stopped":
        reason = json.loads((child / "state.json").read_text()).get("reason", "")
        if not (reason.startswith("memory_") or reason.startswith("memory_pressure")):
            raise RuntimeError(f"child {model_id} stopped: {reason}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "resume", "status", "stop"])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--deadline-epoch", type=float)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.command == "status":
        for name in ("SUMMARY.json", "REPORT.md"):
            if (campaign / name).exists():
                print((campaign / name).read_text())
        return
    if args.command == "stop":
        (campaign / "STOP").touch()
        for model in MODELS:
            child = campaign / model["id"]
            if child.exists():
                (child / "STOP").touch()
        return
    if args.deadline_epoch is None or not math.isfinite(args.deadline_epoch) or args.deadline_epoch <= time.time():
        parser.error("run/resume requires a future absolute deadline")
    if args.command == "run":
        if args.assets is None:
            parser.error("run requires --assets")
        assets = args.assets.resolve()
        tasks = task_ids(assets)
        if campaign.exists():
            manifest = json.loads((campaign / "manifest.json").read_text())
            if manifest["tasks"] != tasks:
                raise ValueError("existing parent campaign has different task order")
            children = {model["id"]: Path(manifest["children"][model["id"]]) for model in MODELS}
        else:
            children = prepare(campaign, assets, tasks)
    else:
        manifest = json.loads((campaign / "manifest.json").read_text())
        children = {model["id"]: Path(manifest["children"][model["id"]]) for model in MODELS}
    if (campaign / "STOP").exists():
        raise RuntimeError("parent STOP marker exists")
    for model in MODELS:
        result = run_child(model["id"], children[model["id"]], args.deadline_epoch)
        write_parent_summary(campaign, children, "running")
        if result["state"] not in {"complete", "stopped"}:
            raise RuntimeError(f"child {model['id']} did not reach a terminal state")
    write_parent_summary(campaign, children, "complete")


if __name__ == "__main__":
    main()
