"""Run a separate, resumable Swift-Qwen IQ4_XS long-tail coding campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts import ROOT
from scripts.instruments import agentic_campaign as agentic
from scripts.instruments.chat_templates import BARTOWSKI_PREFIX
from scripts.instruments import hillclimb as h


LONG_TAIL_TASKS = [
    "rust/doubly-linked-list",
    "rust/book-store",
    "rust/react",
    "rust/alphametics",
    "rust/forth",
    "rust/xorcism",
    "rust/nucleotide-codons",
    "javascript/complex-numbers",
]
FIT_TASK = "javascript/affine-cipher"

MODEL_PATH = ROOT / ".cache/hillclimb-models/swift-iq4-xs/Swift-Qwen3.8-27B-IQ4_XS.gguf"
MODEL = {
    "id": "swift-iq4-xs",
    "repo": "ukisai/Swift-Qwen3.8-27B-GGUF",
    "revision": "b12fc8f44af7bdf31a3c3309891acb345d3106a2",
    "file": MODEL_PATH.name,
    "bytes": 15688288352,
    # HF's X-Linked-Etag for the pinned blob is used as the artifact digest.
    "sha256": "27f5531a012a4bb3c3e379318dc62216f057131d34347498fabfc23186a4ffda",
    "local": str(MODEL_PATH),
}

TUNING_TASKS = ["javascript/affine-cipher", "rust/forth"]
TUNING_SEEDS = [1729, 1901, 2027]
TUNING_BASELINE = {
    "context": 102400, "slots": 1, "mode": "tensor", "split": "69,31",
    "batch": 2048, "ubatch": 128, "maximum": 3, "minimum": 0, "p_min": 0,
    "draft_k": "f16", "draft_v": "f16", "chat_template": BARTOWSKI_PREFIX,
}
TUNING_STAGES = ("baseline", "split", "microbatch", "batch", "interaction", "layer",
                 "mtp", "probability", "minimum", "draft-cache", "swarm-allocation")


def tuning_contract():
    return {
        "name": "swift-iq4-single-agent-and-swarm-v1", "model_sha256": MODEL["sha256"],
        "template_adapter": BARTOWSKI_PREFIX, "target_kv": "q8_0/q8_0",
        "sampling": {"temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0.05},
        "seeds": TUNING_SEEDS, "throughput_tasks": TUNING_TASKS,
        "input_targets": [8192, 49152, 98304], "throughput_output_budget": 2048,
        "admission_output_budget": 32, "reasoning_effort": "xhigh",
        "cache_prompt": False, "context_unit": "tokens_per_slot; server_total=context*slots",
        "metrics_scope": "server counters belong to the batch, never individual requests",
        "production_promotion": False, "minimum_gain_fraction": 0.05,
        "single_agent_min_context": 95000, "deep_decode_target_tps": 35,
        "required_before_run": "idle interactive service explicitly stopped; no unowned compute workloads",
        "required_after_run": "release benchmark GPU memory, restore selected skill serving profile",
        "agentic_output_budget": "full client budget; throughput cap does not apply",
    }


def tuning_queue(stage, selected=None, *, repeat=False, splits=None, microbatches=None):
    """Generate one stage; selection of later-stage inputs requires measured evidence."""
    if stage not in TUNING_STAGES:
        raise ValueError("unknown tuning stage")
    if stage not in ("baseline", "split") and selected is None:
        raise ValueError("this stage requires an explicitly selected configuration")
    config = {**TUNING_BASELINE, **(selected or {})}
    if config["context"] != 102400 or config["slots"] != 1:
        raise ValueError("single-agent tuning retains context102400 and one slot")
    if config["chat_template"] != BARTOWSKI_PREFIX:
        raise ValueError("tuning must retain the selected template")
    candidates = []
    if stage == "baseline":
        candidates = [("single-agent-baseline", dict(TUNING_BASELINE))]
    elif stage == "split":
        candidates = [(f"split-{left}", {**TUNING_BASELINE, "split": f"{left},{100-left}"})
                      for left in (65, 67, 71, 73, 75)]
    elif stage == "interaction":
        if not splits or not microbatches or len(set(splits)) != 2 or len(set(microbatches)) != 2:
            raise ValueError("interaction requires two distinct selected splits and microbatches")
        candidates = [(f"cross-{i}-{batch}", {**config, "split": split, "ubatch": batch})
                      for i, split in enumerate(splits) for batch in microbatches]
    elif stage == "layer":
        candidates = [("layer-control", {**config, "mode": "layer"})]
    elif stage == "draft-cache":
        candidates = [("draft-q8-experimental", {**config, "draft_k": "q8_0", "draft_v": "q8_0"})]
    elif stage == "swarm-allocation":
        candidates = [(f"swarm-{slots}-context{context}", {**config, "slots": slots, "context": context})
                      for slots in (2, 3, 4) for context in (32768, 65536, 102400)]
    else:
        key, values = {
            "microbatch": ("ubatch", (64, 256, 512)), "batch": ("batch", (1024, 4096)),
            "mtp": ("maximum", (2, 4, 6)), "probability": ("p_min", (0.5, 0.75)),
            "minimum": ("minimum", (2,)),
        }[stage]
        candidates = [(f"{key}-{str(value).replace('.', '_')}", {**config, key: value})
                      for value in values]
    seeds = TUNING_SEEDS if repeat or stage == "baseline" else TUNING_SEEDS[:1]
    targets = [8192, 49152, 98304] if stage == "baseline" else [49152, 98304]

    def jobs(label, settings, depths, run_seeds):
        return [{"id": f"{stage}-{label}-{task.replace('/', '-')}-{target}-{seed}",
                 "kind": "profile", "model": MODEL["id"], "task": task, "seed": seed,
                 "targets": [target], "configs": [dict(settings)]}
                for target in depths for task in TUNING_TASKS for seed in run_seeds]

    result = []
    if stage != "baseline":
        result += jobs("baseline-before", TUNING_BASELINE, [49152, 98304], seeds)
        if config != TUNING_BASELINE and stage != "swarm-allocation":
            result += jobs("matched-control", config, targets, seeds)
    for label, settings in candidates:
        if stage == "swarm-allocation":
            result.append({"id": label, "kind": "profile", "model": MODEL["id"],
                           "task": TUNING_TASKS[0], "seed": TUNING_SEEDS[0],
                           "targets": [0], "configs": [settings]})
        else:
            result += jobs(label, settings, targets, seeds)
    if stage != "baseline":
        result += jobs("baseline-after", TUNING_BASELINE, [49152, 98304], seeds)
    return result


def tuning_main(argv):
    parser = argparse.ArgumentParser(description="Prepare Swift tuning stages without launching inference")
    parser.add_argument("stage", choices=TUNING_STAGES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path)
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--splits", nargs=2)
    parser.add_argument("--microbatches", nargs=2, type=int)
    args = parser.parse_args(argv)
    selected = json.loads(args.selected_config.read_text()) if args.selected_config else None
    jobs = tuning_queue(args.stage, selected, repeat=args.repeat,
                        splits=args.splits, microbatches=args.microbatches)
    h.MODELS[:] = [MODEL]
    h.TASKS[:] = TUNING_TASKS
    h.validate_profiles(jobs)
    # Queue files are planning artifacts, never mutable running manifests.
    with args.out.open("x") as stream:
        json.dump(jobs, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.out)


def queue():
    return [{
        "id": "solve-swift-iq4-xs-" + task.replace("/", "-"),
        "kind": "profile",
        "model": MODEL["id"],
        "task": task,
        "workload": "agentic",
        "targets": [0],
        "configs": [{"context": 102400, "slots": 1, "mode": "tensor",
                     "split": "69,31", "ubatch": 128, "maximum": 3,
                     "chat_template": BARTOWSKI_PREFIX}],
    } for task in LONG_TAIL_TASKS]


def configure_globals():
    # h.initialize validates profile model IDs against this process-local registry.
    h.MODELS[:] = [MODEL]
    # The generic fit probe uses this small task before the long-tail queue starts.
    h.TASKS[:] = list(dict.fromkeys([FIT_TASK, *LONG_TAIL_TASKS]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(campaign: Path):
    rows = []
    for job in queue():
        path = campaign / ("job-" + job["id"] + ".json")
        if path.exists():
            rows.append(h.read_sealed(path))
    passed = [row for row in rows if row.get("status") == "passed" and row.get("probe")]
    probes = [row["probe"] for row in passed]
    metrics = [probe.get("metrics_delta", {}) for probe in probes]
    result = {
        "updated_epoch": time.time(),
        "state": json.loads((campaign / "state.json").read_text())["status"],
        "model": MODEL["id"],
        "tasks": len(rows),
        "solved": sum(bool(probe.get("task_success")) for probe in probes),
        "timeouts": sum(bool(probe.get("timed_out")) for probe in probes),
        "wall_seconds": sum(probe.get("verified_wall_seconds", 0) for probe in probes),
        "output_tokens": sum(metric.get("decode_tokens", 0) for metric in metrics),
        "prefill_seconds": sum(probe.get("prefill_seconds", 0) for probe in probes),
        "decode_seconds": sum(probe.get("decode_seconds", 0) for probe in probes),
        "peak_context_tokens": max((probe.get("peak_context_tokens", 0) for probe in probes), default=0),
        "draft_acceptance": (
            sum(metric.get("draft_accepted_tokens", 0) for metric in metrics)
            / sum(metric.get("draft_tokens", 0) for metric in metrics)
            if sum(metric.get("draft_tokens", 0) for metric in metrics) else None
        ),
    }
    h.atomic_json(campaign / "SUMMARY.json", result)
    lines = ["# Swift-Qwen long-tail campaign", "", json.dumps(result, indent=2), "", "| Task | Result | Wall s | Output tokens | Peak context |", "|---|---|---:|---:|---:|"]
    for row in rows:
        probe = row.get("probe", {})
        lines.append("| %s | %s | %.2f | %s | %s |" % (
            row.get("task", ""), "solved" if probe.get("task_success") else row.get("status", ""),
            probe.get("verified_wall_seconds", 0), probe.get("metrics_delta", {}).get("decode_tokens", ""),
            probe.get("peak_context_tokens", "")))
    (campaign / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, default=time.time() + 12 * 3600)
    args = parser.parse_args()
    configure_globals()
    if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size != MODEL["bytes"]:
        raise SystemExit("Swift artifact is missing or incomplete")
    if sha256(MODEL_PATH) != MODEL["sha256"]:
        raise SystemExit("Swift artifact checksum mismatch")
    campaign = args.campaign.resolve()
    h.initialize(campaign, ROOT / ".cache/hillclimb-models", profiles=queue(),
                 max_attempts=3, agentic_assets=args.assets.resolve())
    runner = agentic.AgenticRunner(campaign, 12, args.deadline_epoch)
    runner.run()
    result = summary(campaign)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "tuning-plan":
        tuning_main(sys.argv[2:])
    else:
        main()
