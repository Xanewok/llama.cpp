"""Gated, resumable real-task stage behind the overnight throughput queue."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from scripts import ROOT
from scripts.instruments import hillclimb as h
from scripts.instruments.chat_templates import BARTOWSKI_PREFIX


MODEL_ORDER = ["iq3-control", "bartowski-iq4-xs", "unsloth-q4-m", "unsloth-iq4-xs",
               "jpetrina-iq4-pure", "unsloth-q4-xl"]
TASK_ORDER = ["javascript/affine-cipher", "rust/word-count", "javascript/bowling", "rust/forth",
              "javascript/alphametics", "rust/robot-name", "javascript/connect", "rust/doubly-linked-list"]


def task_queue(task_ids=None, model_ids=None):
    return [{"id": "solve-" + model + "-" + task.replace("/", "-"), "kind": "profile",
             "model": model, "task": task, "workload": "agentic", "targets": [0],
             "configs": [{"context": 65536 if model == "iq3-control" else 102400,
                          "slots": 1, "mode": "tensor", "split": "70,30" if model == "unsloth-q4-xl" else "69,31",
                          "ubatch": 512 if model == "iq3-control" else 128, "maximum": 3,
                          **({"chat_template": BARTOWSKI_PREFIX} if model == "bartowski-iq4-xs" else {})}]}
            for task in (TASK_ORDER if task_ids is None else task_ids)
            for model in (MODEL_ORDER if model_ids is None else model_ids)]


class AgenticRunner(h.Runner):
    def execute_job(self, job):
        config = {"devices": h.DEVICES, **job["configs"][0]}
        if job["model"] == "iq3-control":
            config.update(devices=[h.GPU0], mode="layer", split="1")
        model = self.ensure_model(job["model"])
        fit = self.attempt("agentic-fit-" + job["model"], model, config, 0)
        if fit["status"] != "passed":
            return {"status": "skipped", "model": job["model"], "task": job["task"],
                    "reason": "no_safe_model_fit", "fit_reason": fit.get("reason")}
        config["workload"] = "agentic"
        result = self.attempt(job["id"], model, config, 0, task=job["task"])
        return {**result, "workload": "agentic"}

    def run_probe(self, config, target, task, seed, out):
        if config.get("workload") != "agentic":
            return super().run_probe(config, target, task, seed, out)
        command = [sys.executable, "-m", "scripts.instruments.agentic_probe", "probe",
                   "--assets", str(self.snapshot / "agentic-assets"), "--task", task,
                   "--url", self.url, "--context", str(config["context"]),
                   "--timeout", "7200", "--out", str(out)]
        child = self.spawn(command, out / "agentic.log")
        return self.wait_child(child, 7250)

    def report(self):
        super().report()
        full_suite = bool(self.manifest.get("continuation"))
        scope = ("All 78 declared tasks; natural task history, not padded-100k throughput probes."
                 if full_suite else "Natural task history; these are not padded-100k throughput probes or the full 78-task suite.")
        lines = ["# Real coding-task " + ("full suite" if full_suite else "pilot"), "", f"Status: {self.state['status']}",
                 f"Processed: {len(self.state['completed'])}/{len(self.manifest['queue'])}",
                 f"Active: {self.state.get('active')}", "",
                 "Full Codex tool loops, isolated reference-free workspace and pristine grading.",
                 scope,
                 "A valid measurement may contain a failed/timed-out task. No throughput generation cap applies.", "",
                 "| Model | Task | Result | Verified wall s | Prefill s | Decode s | Other s | Output tokens |",
                 "|---|---|---|---:|---:|---:|---:|---:|"]
        for path in self.state["completed"].values():
            row = h.read_sealed(self.path / path)
            probe = row.get("probe", {})
            outcome = ("solved" if probe.get("task_success") else
                       "timeout" if probe.get("timed_out") else "failed tests") if row["status"] == "passed" else row.get("reason", row["status"])
            values = [probe.get(key) for key in ("verified_wall_seconds", "prefill_seconds", "decode_seconds", "other_seconds")]
            formatted = ["" if value is None else f"{value:.2f}" for value in values]
            tokens = probe.get("metrics_delta", {}).get("decode_tokens", "")
            lines.append("| " + " | ".join([row.get("model", ""), row.get("task", ""), outcome, *formatted, str(tokens)]) + " |")
        temp = self.path / "REPORT.md.tmp"
        temp.write_text("\n".join(lines) + "\n")
        os.replace(temp, self.path / "REPORT.md")


def wait_for_predecessor(campaign, predecessor, deadline):
    error_since = None
    while True:
        if (campaign / "STOP").exists() or time.time() >= deadline:
            h.atomic_json(campaign / "stage-status.json", {"phase": "stopped", "reason": "stop_or_deadline", "updated": time.time()})
            return False
        state = json.loads((predecessor / "state.json").read_text())
        if state["status"] == "complete":
            return True
        error_since = (error_since if error_since is not None else time.monotonic()) if state["status"] == "error" else None
        if state["status"] == "stopped" or (error_since is not None and time.monotonic() - error_since >= 180):
            h.atomic_json(campaign / "stage-status.json", {"phase": "blocked", "reason": "predecessor_not_complete",
                                                         "predecessor_reason": state.get("reason"), "updated": time.time()})
            return False
        h.atomic_json(campaign / "stage-status.json", {"phase": "waiting_for_throughput", "updated": time.time(),
                                                     "predecessor_completed": len(state["completed"]), "deadline_epoch": deadline})
        time.sleep(min(15, max(0, deadline - time.time())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "resume", "status", "stop"])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--after-campaign", type=Path)
    parser.add_argument("--deadline-epoch", type=float)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.command == "prepare":
        if args.assets is None:
            parser.error("--assets is required")
        assets = args.assets.resolve()
        if not json.loads((assets / "selftest.json").read_text())["harness_verified"]:
            parser.error("trusted grader has not passed its self-test")
        h.initialize(campaign, ROOT / ".cache/hillclimb-models", profiles=task_queue(),
                     max_attempts=3, agentic_assets=assets)
        print(campaign)
        return
    if args.command == "stop":
        (campaign / "STOP").touch()
        return
    if args.command == "status":
        for name in ("stage-status.json", "REPORT.md"):
            if (campaign / name).exists():
                print((campaign / name).read_text())
        return
    if args.deadline_epoch is None or args.deadline_epoch <= 0:
        parser.error("--deadline-epoch is required")
    snapshot = campaign / "snapshot"
    if ROOT.resolve() != snapshot.resolve():
        command = [sys.executable, "-m", "scripts.instruments.agentic_campaign", "resume",
                   "--campaign", str(campaign), "--deadline-epoch", str(args.deadline_epoch)]
        if args.after_campaign:
            command += ["--after-campaign", str(args.after_campaign.resolve())]
        env = dict(os.environ, PYTHONPATH=str(snapshot), PYTHONDONTWRITEBYTECODE="1")
        os.chdir(snapshot)
        os.execve(sys.executable, command, env)
    if args.after_campaign and not wait_for_predecessor(campaign, args.after_campaign, args.deadline_epoch):
        return
    if (campaign / "STOP").exists():
        return
    h.atomic_json(campaign / "stage-status.json", {"phase": "running_tasks", "updated": time.time(),
                                                 "deadline_epoch": args.deadline_epoch})
    AgenticRunner(campaign, 24, args.deadline_epoch).run()


if __name__ == "__main__":
    main()
