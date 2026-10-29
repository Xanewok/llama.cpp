"""Run a Swift IQ3_XXS two-GPU long-tail comparison campaign."""

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
MODEL_PATH = ROOT / ".cache/hillclimb-models/swift-iq3-xxs/Swift-Qwen3.8-27B-IQ3_XXS.gguf"
MODEL = {
    "id": "swift-iq3-xxs",
    "repo": "ukisai/Swift-Qwen3.8-27B-GGUF",
    "revision": "b12fc8f44af7bdf31a3c3309891acb345d3106a2",
    "file": MODEL_PATH.name,
    "bytes": 12532504672,
    "sha256": "1d16444b9164595fb59cdfc1b11c24b1e167c68238d33a0e4d261495a5178f60",
    "local": str(MODEL_PATH),
}


def queue():
    return [{
        "id": "solve-swift-iq3-xxs-" + task.replace("/", "-"),
        "kind": "profile", "model": MODEL["id"], "task": task,
        "workload": "agentic", "targets": [0],
        "configs": [{"context": 102400, "slots": 1, "mode": "tensor",
                     "split": "69,31", "ubatch": 128, "maximum": 3,
                     "chat_template": BARTOWSKI_PREFIX}],
    } for task in LONG_TAIL_TASKS]


def configure_globals():
    h.MODELS[:] = [MODEL]
    h.TASKS[:] = list(dict.fromkeys([FIT_TASK, *LONG_TAIL_TASKS]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SwiftRunner(agentic.AgenticRunner):
    def execute_job(self, job):
        config = {"devices": h.DEVICES, **job["configs"][0]}
        model = self.ensure_model(job["model"])
        fit = self.attempt("agentic-fit-" + job["model"], model, config, 0)
        if fit["status"] != "passed":
            return {"status": "skipped", "model": job["model"], "task": job["task"],
                    "reason": "no_safe_model_fit", "fit_reason": fit.get("reason")}
        config["workload"] = "agentic"
        return {**self.attempt(job["id"], model, config, 0, task=job["task"]), "workload": "agentic"}


def summary(campaign: Path):
    rows = []
    for job in queue():
        path = campaign / ("job-" + job["id"] + ".json")
        if path.exists(): rows.append(h.read_sealed(path))
    passed = [row for row in rows if row.get("status") == "passed" and row.get("probe")]
    probes = [row["probe"] for row in passed]
    metrics = [probe.get("metrics_delta", {}) for probe in probes]
    drafts = sum(metric.get("draft_tokens", 0) for metric in metrics)
    result = {
        "updated_epoch": time.time(), "state": json.loads((campaign / "state.json").read_text())["status"],
        "model": MODEL["id"], "tasks": len(rows),
        "solved": sum(bool(probe.get("task_success")) for probe in probes),
        "timeouts": sum(bool(probe.get("timed_out")) for probe in probes),
        "wall_seconds": sum(probe.get("verified_wall_seconds", 0) for probe in probes),
        "output_tokens": sum(metric.get("decode_tokens", 0) for metric in metrics),
        "prefill_seconds": sum(probe.get("prefill_seconds", 0) for probe in probes),
        "decode_seconds": sum(probe.get("decode_seconds", 0) for probe in probes),
        "peak_context_tokens": max((probe.get("peak_context_tokens", 0) for probe in probes), default=0),
        "draft_acceptance": (sum(metric.get("draft_accepted_tokens", 0) for metric in metrics) / drafts if drafts else None),
    }
    h.atomic_json(campaign / "SUMMARY.json", result)
    lines = ["# Swift IQ3_XXS long-tail campaign", "", json.dumps(result, indent=2), "",
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
    parser.add_argument("--deadline-epoch", type=float, default=time.time() + 12 * 3600)
    args = parser.parse_args()
    configure_globals()
    if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size != MODEL["bytes"]:
        raise SystemExit("Swift IQ3_XXS artifact is missing or incomplete")
    if sha256(MODEL_PATH) != MODEL["sha256"]:
        raise SystemExit("Swift IQ3_XXS artifact checksum mismatch")
    campaign = args.campaign.resolve()
    h.initialize(campaign, ROOT / ".cache/hillclimb-models", profiles=queue(),
                 max_attempts=3, agentic_assets=args.assets.resolve())
    SwiftRunner(campaign, 12, args.deadline_epoch).run()
    print(json.dumps(summary(campaign), indent=2), flush=True)


if __name__ == "__main__":
    main()
