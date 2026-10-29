"""Run a separate, resumable Swift Q4_K_M long-tail coding campaign."""

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
MODEL_PATH = ROOT / ".cache/hillclimb-models/swift-q4-m/Swift-Qwen3.8-27B-Q4_K_M.gguf"
MODEL = {
    "id": "swift-q4-m",
    "repo": "ukisai/Swift-Qwen3.8-27B-GGUF",
    "revision": "b12fc8f44af7bdf31a3c3309891acb345d3106a2",
    "file": MODEL_PATH.name,
    "bytes": 18024380576,
    "sha256": "ad5811e291431bd0de1cec0c4004a5eac98daee9850882edac69a823209e88ab",
    "local": str(MODEL_PATH),
}


def queue(context=102400, split="69,31", ubatch=64):
    return [{
        "id": "solve-swift-q4-m-" + task.replace("/", "-"),
        "kind": "profile",
        "model": MODEL["id"],
        "task": task,
        "workload": "agentic",
        "targets": [0],
        "configs": [{"context": context, "slots": 1, "mode": "tensor",
                     "split": split, "ubatch": ubatch, "maximum": 3,
                     "chat_template": BARTOWSKI_PREFIX}],
    } for task in LONG_TAIL_TASKS]


def normalized_context(value: int) -> int:
    # llama-server rounds slot contexts down to a 2048-token graph multiple.
    context = value - value % 2048
    if context < 2048:
        raise ValueError("context rounds below one graph block")
    return context


def configure_globals():
    h.MODELS[:] = [MODEL]
    h.TASKS[:] = list(dict.fromkeys([FIT_TASK, *LONG_TAIL_TASKS]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(campaign: Path, context=102400, split="69,31", ubatch=64):
    rows = []
    for job in queue(context, split, ubatch):
        path = campaign / ("job-" + job["id"] + ".json")
        if path.exists():
            rows.append(h.read_sealed(path))
    passed = [row for row in rows if row.get("status") == "passed" and row.get("probe")]
    probes = [row["probe"] for row in passed]
    metrics = [probe.get("metrics_delta", {}) for probe in probes]
    draft_tokens = sum(metric.get("draft_tokens", 0) for metric in metrics)
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
        "draft_acceptance": (sum(metric.get("draft_accepted_tokens", 0) for metric in metrics) / draft_tokens
                             if draft_tokens else None),
    }
    h.atomic_json(campaign / "SUMMARY.json", result)
    lines = ["# Swift-Qwen Q4_K_M long-tail campaign", "", json.dumps(result, indent=2), "",
             "| Task | Result | Wall s | Output tokens | Peak context |", "|---|---|---:|---:|---:|"]
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
    parser.add_argument("--context", type=int, default=102400)
    parser.add_argument("--split", default="69,31")
    parser.add_argument("--ubatch", type=int, default=64)
    parser.add_argument("--reserve", type=int, default=256)
    parser.add_argument("--deadline-epoch", type=float, default=time.time() + 12 * 3600)
    args = parser.parse_args()
    if args.context < 1 or args.context > 262144:
        raise SystemExit("context must be between 1 and 262144")
    if args.ubatch < 1 or args.ubatch > 4096:
        raise SystemExit("ubatch must be between 1 and 4096")
    if args.reserve < 0:
        raise SystemExit("reserve must be non-negative")
    context = normalized_context(args.context)
    try:
        split_values = [float(value) for value in args.split.split(",")]
    except ValueError:
        raise SystemExit("split must be two comma-separated numbers")
    if len(split_values) != 2 or any(value <= 0 for value in split_values):
        raise SystemExit("split must be two positive comma-separated numbers")
    configure_globals()
    if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size != MODEL["bytes"]:
        raise SystemExit("Swift Q4_K_M artifact is missing or incomplete")
    if sha256(MODEL_PATH) != MODEL["sha256"]:
        raise SystemExit("Swift Q4_K_M artifact checksum mismatch")
    campaign = args.campaign.resolve()
    # Reserve is campaign-scoped; ordinary campaigns retain the 512/256 MiB defaults.
    h.initialize(campaign, ROOT / ".cache/hillclimb-models", profiles=queue(context, args.split, args.ubatch),
                 max_attempts=3, agentic_assets=args.assets.resolve(), vram_reserves=(args.reserve, args.reserve))
    agentic.AgenticRunner(campaign, 12, args.deadline_epoch).run()
    print(json.dumps(summary(campaign, context, args.split, args.ubatch), indent=2), flush=True)


if __name__ == "__main__":
    main()
