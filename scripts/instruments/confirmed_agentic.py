"""Frozen, confirmation-gated full-suite continuation; no automatic production promotion."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

from scripts import ROOT
from scripts.instruments import agentic_campaign as a, hillclimb as h
from scripts.instruments.hillclimb_probe import digest_file


MODELS = ["iq3-control", "unsloth-q4-m", "bartowski-iq4-xs"]
CONFIRM_TASKS = ["javascript/affine-cipher", "javascript/bowling", "rust/word-count", "rust/forth"]
SEEDS = [1911, 1912, 1913]
MEMORY_RECOVERY = {"wait_seconds": 900, "stable_seconds": 10, "extra_headroom_mib": 128}
WATCHDOG_RECOVERY = {"max_retries": 3, "backoff_seconds": [20, 40, 60],
                     "wait_seconds": 120, "stable_seconds": 10}


def watchdog_timeout(log):
    lower = log.lower()
    return ("the launch timed out and was terminated" in lower
            and not any(word in lower for word in ("illegal memory access", "device-side assert",
                        "unspecified launch failure", "fallen off the bus"))
            and all(code == "8" for code in re.findall(r"(?i)\bxid\s*(?:\([^)]*\))?\s*:\s*(\d+)", log)))


def watchdog_kernel_check(since):
    try:
        result = subprocess.run(["journalctl", "-k", "--since", "@" + str(int(since)),
                                 "--no-pager", "-o", "cat", "--grep", r"NVRM: Xid"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise h.StopCampaign("watchdog_kernel_health_unconfirmed") from error
    codes = re.findall(r"(?i)\bxid\s*(?:\([^)]*\))?\s*:\s*(\d+)", result.stdout)
    if result.returncode not in (0, 1) or not codes or any(code != "8" for code in codes):
        raise h.StopCampaign("watchdog_kernel_health_unconfirmed")
    return result.stdout


def memory_pressure(reason):
    return reason == "oom" or str(reason).startswith("vram_reserve: ")


def recovery_requirements(result, before, reserves=(512, 256), extra_headroom_mib=128):
    required = {}
    for gpu, reserve in zip(result["config"]["devices"], reserves):
        allocated = max(0, before["gpus"][gpu]["memory.free"] - result["minimum_free_mib"][gpu])
        required[gpu] = allocated + reserve + extra_headroom_mib
    return required


def choose_microbatches(rows):
    expected = {(m, t, s, u) for m in MODELS[1:] for t in CONFIRM_TASKS
                for s in SEEDS for u in (128, 512)}
    indexed = {}
    for row in rows:
        config = row.get("config", {})
        key = (row.get("model"), row.get("task"), row.get("seed"), config.get("ubatch"))
        if key not in expected or key in indexed:
            raise ValueError("confirmation must contain each declared case exactly once")
        if any(config.get(k) != v for k, v in
               {"context": 102400, "slots": 1, "maximum": 3, "split": "69,31", "mode": "tensor"}.items()):
            raise ValueError("confirmation settings differ from declared single-agent control")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("confirmation cases are incomplete")

    def valid(row):
        probe = row.get("probe", {})
        return (row["status"] == "passed" and all(probe.get(k) for k in
                ("complete", "mtp_live", "depth_verified", "overlap_verified")) and
                all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) and row[k] > 0
                    for k in ("prefill_tps", "decode_tps")))

    selection = {}
    for model in MODELS[1:]:
        arms = {u: [indexed[model, t, s, u] for t in CONFIRM_TASKS for s in SEEDS] for u in (128, 512)}
        good = {u: [r for r in arms[u] if valid(r)] for u in arms}
        choice, reason, ratios = 128, "retain baseline: improvement not established", {}
        if all(len(good[u]) == 12 for u in arms):
            ratios = {metric: math.exp(statistics.mean(math.log(c[metric] / b[metric])
                       for b, c in zip(arms[128], arms[512]))) for metric in ("prefill_tps", "decode_tps")}
            if ((ratios["prefill_tps"] >= 1.05 and ratios["decode_tps"] >= .95) or
                    (ratios["decode_tps"] >= 1.05 and ratios["prefill_tps"] >= .95)):
                choice, reason = 512, "confirmed useful throughput improvement; coding validation pending"
        elif len(good[512]) == 12:
            choice, reason = 512, "candidate has complete safe confirmation; baseline does not"
        elif not good[128]:
            choice, reason = None, "no safe baseline and no complete candidate confirmation"
        else:
            reason = "retain baseline provisionally: incomplete safe confirmation"
        chosen = good.get(choice, [])
        median = statistics.median(r["decode_tps"] for r in chosen) if chosen else None
        selection[model] = {"ubatch": choice, "reason": reason,
                            "valid_repeats": {str(u): len(rs) for u, rs in good.items()},
                            "paired_geometric_ratios": ratios, "median_decode_tps": median,
                            "near_100k_speed_gate": len(chosen) == 12 and median >= 35,
                            "production_qualified": False}
    return selection


def confirmation_observation(job, row):
    if row.get("model") != job["model"]:
        raise ValueError("confirmation model differs from declared job")
    if row.get("job") == job["id"] + "-fit" and row.get("target") == 0 and row.get("status") == "failed":
        # Fit probes use placeholder prompt/seed; attribute rejection to the declared request.
        return {**row, "task": job["task"], "seed": job["seed"]}
    if (row.get("job") != job["id"] or row.get("task") != job["task"]
            or row.get("seed") != job["seed"] or row.get("target") != 98304):
        raise ValueError("confirmation observation differs from declared request")
    return row


def resolve_selection(source, campaign):
    manifest = h.read_sealed(source / "manifest.json")
    rows, evidence = [], {}
    for job in manifest["queue"]:
        path = source / ("job-" + job["id"] + ".json")
        rows.append(confirmation_observation(job, h.read_sealed(path)))
        evidence[str(path)] = digest_file(path)
    result = {"source": str(source), "source_manifest_sha256": digest_file(source / "manifest.json"),
              "evidence_pins": evidence, "models": choose_microbatches(rows)}
    path = campaign / "selection.json"
    if path.exists():
        if h.read_sealed(path) != result:
            raise ValueError("confirmation evidence or selection changed after freezing")
    else:
        h.sealed_json(path, result)
    return result


def tree_fingerprint(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            value = "link:" + os.readlink(path)
        elif path.is_file():
            value = digest_file(path)
        else:
            continue
        digest.update(json.dumps([str(path.relative_to(directory)), value]).encode())
    return digest.hexdigest()


def revalidate_measurement(source, campaign, job, row, fingerprint):
    from scripts.instruments import agentic_probe as probe

    assets_dir = campaign / "snapshot/agentic-assets"
    checks = json.loads((assets_dir / "selftest.json").read_text())
    if checks.get("test_scope") != probe.TEST_SCOPE or not checks["harness_verified"]:
        raise ValueError("grader migration requires current full-suite self-test")
    expected = next(t["expected_passed_tests"] for t in checks["tasks"] if t["task_id"] == job["task"])
    recorded = row["probe"]
    # A false repair changes model history and elapsed time; a later passing solution cannot salvage it.
    eligible = (recorded.get("complete") and recorded.get("task_success")
                and not recorded.get("timed_out") and not recorded.get("error")
                and recorded.get("grading_rounds") == 1 and recorded.get("expected_passed_tests") == expected)
    request = {"source_attempt_sha256": fingerprint, "expected_passed_tests": expected,
               "source_result_sha256": digest_file(source / ("job-" + job["id"] + ".json")),
               "destination_manifest_sha256": digest_file(campaign / "manifest.json")}
    path = campaign / "revalidation" / (job["id"] + ".json")
    if path.exists():
        report = h.read_sealed(path)
        if report["request"] != request:
            raise ValueError("grader revalidation inputs changed")
        return report
    report = {"request": request, "eligible": bool(eligible), "test_scope": probe.TEST_SCOPE}
    if eligible:
        tasks = json.loads((assets_dir / "tasks.json").read_text())
        task = next(t for t in tasks if t["task_id"] == job["task"])
        assets = json.loads((assets_dir / "tools.json").read_text())
        original = json.loads((source / row["attempt"] / "grade-0.json").read_text())
        if (original["returncode"] != 0 or original["passed"] != expected
                or original["failed"] or original["ignored"]):
            raise ValueError("recorded first-round success contradicts saved grading evidence")
        with tempfile.TemporaryDirectory(prefix="qwen-regrade-") as scratch:
            work = Path(scratch) / "grade"
            probe.materialize(task, assets_dir, work)
            probe.transfer_solution(task, source / row["attempt"] / "grade-0", work)
            grade = probe.test_suite(task, work, assets, 300, asset_dir=assets_dir)
        report["grade"] = grade
        report["eligible"] = (grade["returncode"] == 0 and grade["passed"] == expected
                              and grade["failed"] == grade["ignored"] == 0)
    path.parent.mkdir(exist_ok=True)
    h.sealed_json(path, report)
    print(json.dumps({"revalidated": job["id"], "reuse": report["eligible"]}), flush=True)
    return report


def reuse_completed(source, campaign, *, allow_queue_reorder=False, revalidate_grader=False,
                    allow_watchdog_recovery=False):
    source, campaign = source.resolve(), campaign.resolve()
    if source == campaign:
        raise ValueError("recovery requires a new campaign")
    old, new = (h.read_sealed(p / "manifest.json") for p in (source, campaign))
    for key in ("engine_commit", "binary", "binary_pins", "devices", "port", "models", "safety", "max_attempts"):
        if old[key] != new[key]:
            raise ValueError("recovery changed comparison contract: " + key)
    old_jobs, new_jobs = ({job["id"]: job for job in m["queue"]} for m in (old, new))
    if (len(old_jobs) != len(old["queue"]) or len(new_jobs) != len(new["queue"])
            or old_jobs != new_jobs or (not allow_queue_reorder and old["queue"] != new["queue"])):
        raise ValueError("recovery changed comparison contract: queue")
    allowed = {"scripts/instruments/hillclimb.py", "scripts/instruments/confirmed_agentic.py"}
    if revalidate_grader:
        allowed |= {"scripts/instruments/agentic_probe.py", "agentic-assets/selftest.json"}
    changed = {key for key in old["snapshot_pins"].keys() | new["snapshot_pins"].keys()
               if old["snapshot_pins"].get(key) != new["snapshot_pins"].get(key)}
    if changed - allowed:
        raise ValueError("recovery changed inference/grader assets: " + str(sorted(changed - allowed)))
    excluded_policy = {"memory_recovery"} | ({"queue_order"} if allow_queue_reorder else set())
    if allow_watchdog_recovery:
        if new["continuation"].get("watchdog_recovery") != WATCHDOG_RECOVERY:
            raise ValueError("recovery requires the bounded watchdog policy")
        excluded_policy.add("watchdog_recovery")
    old_policy, new_policy = ({k: v for k, v in m["continuation"].items() if k not in excluded_policy}
                              for m in (old, new))
    if new_policy != old_policy:
        raise ValueError("recovery changed selection policy")
    old_state, state = (json.loads((p / "state.json").read_text()) for p in (source, campaign))
    if old_state["status"] not in {"stopped", "complete"} or old_state.get("owned"):
        raise ValueError("source campaign must be inactive and have no owned processes")
    if state["status"] != "prepared" or state.get("owned"):
        raise ValueError("destination campaign must not have started")
    imported, requeued = [], []
    for job in new["queue"]:
        name = old_state["completed"].get(job["id"])
        if not name:
            continue
        row = h.read_sealed(source / name)
        if row["status"] != "passed":
            requeued.append({"job": job["id"], "reason": row.get("reason"), "source_record": str(source / name)})
            continue
        if row.get("model") != job["model"] or row.get("task") != job["task"]:
            raise ValueError("source result does not match queued task")
        relative = Path(row["attempt"])
        if relative.is_absolute() or relative.parts[0] != "attempts" or ".." in relative.parts:
            raise ValueError("invalid source attempt path")
        origin, target = source / relative, campaign / relative
        fingerprint = tree_fingerprint(origin)
        if revalidate_grader:
            report = revalidate_measurement(source, campaign, job, row, fingerprint)
            if not report["eligible"]:
                requeued.append({"job": job["id"], "reason": "grader_revalidation_required_rerun",
                                 "source_record": str(source / name)})
                continue
        if not target.exists():
            scratch = Path(tempfile.mkdtemp(prefix=".import-", dir=campaign / "attempts"))
            shutil.copytree(origin, scratch, symlinks=True, dirs_exist_ok=True)
            if tree_fingerprint(scratch) != fingerprint:
                raise ValueError("copied attempt verification failed")
            os.rename(scratch, target)
        elif tree_fingerprint(target) != fingerprint:
            raise ValueError("existing imported attempt differs")
        destination = campaign / ("job-" + job["id"] + ".json")
        if destination.exists() and h.read_sealed(destination) != row:
            raise ValueError("existing imported result differs")
        h.sealed_json(destination, row)
        state["completed"][job["id"]] = destination.name
        imported.append({"job": job["id"], "source_result_sha256": digest_file(source / name),
                         "attempt_tree_sha256": fingerprint})
        if revalidate_grader:
            imported[-1]["revalidation_sha256"] = digest_file(campaign / "revalidation" / (job["id"] + ".json"))
    audit = {"source": str(source), "source_manifest_sha256": digest_file(source / "manifest.json"),
             "changed_coordinator_files": sorted(changed & {
                 "scripts/instruments/hillclimb.py", "scripts/instruments/confirmed_agentic.py"}),
             "changed_snapshot_files": sorted(changed), "queue_order_changed": old["queue"] != new["queue"],
             "grader_revalidated": revalidate_grader,
             "watchdog_recovery_changed": old["continuation"].get("watchdog_recovery") != new["continuation"].get("watchdog_recovery"),
             "reused": imported, "requeued": requeued}
    h.sealed_json(campaign / "RECOVERY.json", audit)
    h.atomic_json(campaign / "state.json", state)
    return audit


def suite_queue(task_ids, order="task-first"):
    jobs = a.task_queue(task_ids, MODELS)
    for job in jobs:
        if job["model"] != "iq3-control":
            alternative = dict(job["configs"][0], ubatch=512)
            job["configs"].append(alternative)
    if order == "model-first":
        jobs.sort(key=lambda job: MODELS.index(job["model"]))
    elif order != "task-first":
        raise ValueError("unknown queue order")
    return jobs


def handoff_at_boundary(source, campaign, job_id, deadline):
    source, campaign = source.resolve(), campaign.resolve()
    manifest = h.read_sealed(source / "manifest.json")
    if source == campaign or job_id not in {j["id"] for j in manifest["queue"]}:
        raise ValueError("handoff requires a distinct source and a declared boundary job")
    request = {"source": str(source), "job": job_id,
               "source_manifest_sha256": digest_file(source / "manifest.json"), "deadline_epoch": deadline}
    request_path = campaign / "HANDOFF.json"
    if request_path.exists() and h.read_sealed(request_path) != request:
        raise ValueError("handoff request changed")
    h.sealed_json(request_path, request)
    done_path, stop_path = campaign / "HANDOFF-DONE.json", campaign / "HANDOFF-STOP.json"
    if done_path.exists():
        if h.read_sealed(done_path)["recovery_sha256"] != digest_file(campaign / "RECOVERY.json"):
            raise ValueError("handoff recovery audit changed")
        return True

    def status(phase, **extra):
        h.atomic_json(campaign / "stage-status.json", {"phase": phase, "source": str(source),
                                                     "boundary_job": job_id, "updated": time.time(), **extra})
    status("waiting_for_task_boundary")
    while True:
        if (campaign / "STOP").exists() or time.time() >= deadline:
            status("stopped", reason="stop_or_deadline")
            return False
        state = json.loads((source / "state.json").read_text())
        if (state["status"] in {"stopped", "error"} and not
                (state["status"] == "stopped" and state.get("reason") == "requested_stop" and stop_path.exists())):
            status("blocked", reason="source_stopped", source_reason=state.get("reason"))
            return False
        if (source / "STOP").exists() and not stop_path.exists():
            status("blocked", reason="source_stop_not_owned_by_handoff")
            return False
        boundary = source / ("job-" + job_id + ".json")
        if job_id in state.get("completed", {}):
            if h.read_sealed(boundary)["status"] != "passed":
                status("blocked", reason="boundary_not_valid_measurement")
                return False
            # STOP is first requested after the full logical result is sealed, never mid-task.
            if state["status"] == "running" and not stop_path.exists():
                h.sealed_json(stop_path, {"requested_at": time.time()})
                status("stopping_at_task_boundary")
            if stop_path.exists():
                (source / "STOP").touch()
            coordinator = state.get("coordinator", {})
            alive = (coordinator.get("start") is not None and
                     h.proc_identity(coordinator["pid"]) == coordinator["start"])
            if state["status"] in {"stopped", "complete"} and not state.get("owned") and not alive:
                reuse_completed(source, campaign, allow_queue_reorder=True)
                h.sealed_json(done_path, {"recovery_sha256": digest_file(campaign / "RECOVERY.json")})
                status("handoff_complete")
                return True
        if stop_path.exists() and time.time() >= h.read_sealed(stop_path)["requested_at"] + 180:
            status("blocked", reason="source_shutdown_timeout")
            return False
        time.sleep(.2)


class ConfirmedRunner(a.AgenticRunner):
    def __init__(self, campaign, deadline, selection):
        super().__init__(campaign, max(0, deadline - time.time()) / 3600, deadline)
        self.selection = selection
        self.watchdog_policy = self.manifest.get("continuation", {}).get("watchdog_recovery")
        self.recovery_extra_headroom_mib = self.manifest.get("continuation", {}).get(
            "recovery_extra_headroom_mib", MEMORY_RECOVERY["extra_headroom_mib"])
        if self.watchdog_policy is not None and self.watchdog_policy != WATCHDOG_RECOVERY:
            raise ValueError("unrecognized watchdog recovery policy")

    def retryable_failure(self, reason):
        return (reason == "cuda_watchdog" and self.watchdog_policy is not None
                or memory_pressure(reason) or super().retryable_failure(reason))

    def retryable_gpu_fault(self, log):
        return self.watchdog_policy is not None and watchdog_timeout(log)

    def retry_limit(self, reason):
        if reason == "cuda_watchdog" and self.watchdog_policy is not None:
            return 1 + self.watchdog_policy["max_retries"]
        return super().retry_limit(reason)

    def retry_count(self, failures, started_attempts):
        if failures and failures[-1].get("reason") == "cuda_watchdog" and self.watchdog_policy is not None:
            # A coordinator crash during a retry consumes that attempt, even without result.json.
            return started_attempts
        return super().retry_count(failures, started_attempts)

    def retry_backoff(self, failures, count):
        if failures[-1].get("reason") == "cuda_watchdog" and self.watchdog_policy is not None:
            return self.watchdog_policy["backoff_seconds"][count - 1]
        return super().retry_backoff(failures, count)

    def wait_for_watchdog(self, job_id, result, seconds):
        if self.children or self.state.get("owned"):
            raise h.StopCampaign("watchdog_recovery_requires_stopped_children")
        attempt = self.path / result["attempt"]
        with (attempt / "telemetry.jsonl").open() as stream:
            before = json.loads(stream.readline())
        required = recovery_requirements(result, before, self.vram_reserves, self.recovery_extra_headroom_mib)
        since = result["finished_at"] - result["elapsed_seconds"]
        kernel = watchdog_kernel_check(since)
        h.atomic_json(attempt / "watchdog-kernel.json", {"since_epoch": since, "log": kernel})
        until = result["finished_at"] + self.watchdog_policy["wait_seconds"]
        retry_at = time.monotonic() + seconds
        stable_since = None
        self.state["resource_wait"] = {"job": job_id, "reason": "cuda_watchdog",
                                       "required_free_mib": required, "until_epoch": until}
        self.save()
        self.event("watchdog_wait", **self.state["resource_wait"])
        self.report()
        while True:
            self.poll(reserves=False)
            if time.time() >= until:
                raise h.StopCampaign("watchdog_health_unresolved: " + job_id)
            enough = all(self.samples[-1]["gpus"][gpu]["memory.free"] >= value
                         for gpu, value in required.items())
            stable_since = (stable_since if stable_since is not None else time.monotonic()) if enough else None
            if (stable_since is not None and time.monotonic() - stable_since >= self.watchdog_policy["stable_seconds"]
                    and time.monotonic() >= retry_at):
                watchdog_kernel_check(since)
                h.endpoint_free(self.manifest["port"])
                self.state.pop("resource_wait", None)
                self.save()
                self.event("watchdog_recovered", job=job_id, required_free_mib=required)
                return
            time.sleep(.5)

    def wait_before_retry(self, job_id, result, seconds):
        if result.get("reason") == "cuda_watchdog" and self.watchdog_policy is not None:
            return self.wait_for_watchdog(job_id, result, seconds)
        if not memory_pressure(result.get("reason")):
            return super().wait_before_retry(job_id, result, seconds)
        if self.children:
            raise h.StopCampaign("memory_recovery_requires_stopped_children")
        with (self.path / result["attempt"] / "telemetry.jsonl").open() as stream:
            before = json.loads(stream.readline())
        required = recovery_requirements(result, before, self.vram_reserves, self.recovery_extra_headroom_mib)
        until = result["finished_at"] + MEMORY_RECOVERY["wait_seconds"]
        retry_at = time.monotonic() + seconds
        stable_since = None
        self.state["resource_wait"] = {"job": job_id, "reason": result["reason"],
                                       "required_free_mib": required, "until_epoch": until}
        self.save()
        self.event("memory_wait", **self.state["resource_wait"])
        self.report()
        while True:
            # No inference process exists while waiting; thermal/RAM/power/STOP guards still apply.
            self.poll(reserves=False)
            if time.time() >= until:
                raise h.StopCampaign("memory_pressure_unresolved: " + job_id)
            sample = self.samples[-1]
            enough = all(sample["gpus"][gpu]["memory.free"] >= value for gpu, value in required.items())
            stable_since = (stable_since if stable_since is not None else time.monotonic()) if enough else None
            if (stable_since is not None and time.monotonic() - stable_since >= MEMORY_RECOVERY["stable_seconds"]
                    and time.monotonic() >= retry_at):
                self.state.pop("resource_wait", None)
                self.save()
                self.event("memory_recovered", job=job_id, required_free_mib=required)
                return
            time.sleep(.5)

    def acquire(self):
        until = min(self.deadline, time.monotonic() + 180)
        while True:
            if (self.path / "STOP").exists() or time.monotonic() >= self.deadline:
                raise h.StopCampaign("stop_or_deadline")
            try:
                return super().acquire()
            except h.CampaignBusy:
                for handle in self.lockfiles:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    handle.close()
                self.lockfiles.clear()
                if time.monotonic() >= until:
                    raise h.StopCampaign("gpu_lock_timeout")
                time.sleep(min(5, max(0, until - time.monotonic())))

    def execute_job(self, job):
        if job["model"] == "iq3-control":
            return self.checked_execute(job)
        choice = self.selection["models"][job["model"]]["ubatch"]
        if choice is None:
            return {"status": "skipped", "model": job["model"], "task": job["task"],
                    "reason": "no_safe_confirmed_profile"}
        configs = [config for config in job["configs"] if config["ubatch"] == choice]
        if len(configs) != 1:
            raise ValueError("selection must resolve one predeclared configuration")
        resolved = copy.deepcopy(job)
        resolved["configs"] = configs
        return self.checked_execute(resolved)

    def checked_execute(self, job):
        result = super().execute_job(job)
        if result.get("fit_reason", result.get("reason")) in {"cuda_watchdog", "gpu_fault"}:
            raise h.StopCampaign("watchdog_retries_exhausted: " + job["id"])
        if memory_pressure(result.get("fit_reason", result.get("reason"))):
            raise h.StopCampaign("memory_retries_exhausted: " + job["id"])
        if result.get("reason") in {"probe_failed", "probe_incomplete"}:
            raise h.StopCampaign("agentic_instrument_fault: " + result["reason"])
        return result


def summarize(campaign):
    groups = defaultdict(list)
    manifest = h.read_sealed(campaign / "manifest.json")
    for job in manifest["queue"]:
        path = campaign / ("job-" + job["id"] + ".json")
        if path.exists():
            groups[job["model"]].append(h.read_sealed(path))
    summary = {"updated_epoch": time.time(), "state": json.loads((campaign / "state.json").read_text())["status"],
               "models": {}, "scope": "full declared task comparison; configurations are not production-promoted"}
    for model, rows in groups.items():
        measured = [r for r in rows if r["status"] == "passed"]
        probes = [r["probe"] for r in measured]
        summary["models"][model] = {"processed": len(rows), "measured": len(probes),
            "solved": sum(p["task_success"] for p in probes), "timeouts": sum(p["timed_out"] for p in probes),
            "wall_seconds": sum(p["verified_wall_seconds"] for p in probes),
            "prefill_seconds": sum(p["prefill_seconds"] for p in probes),
            "decode_seconds": sum(p["decode_seconds"] for p in probes),
            "output_tokens": sum(p["metrics_delta"].get("decode_tokens", 0) for p in probes),
            "nonpilot_solved": sum(r["probe"]["task_success"] for r in measured if r["task"] not in a.TASK_ORDER)}
    h.atomic_json(campaign / "SUMMARY.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "resume", "handoff", "status", "stop"])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--reuse-successes-from", type=Path)
    parser.add_argument("--revalidate-grader", action="store_true")
    parser.add_argument("--retry-watchdog", action="store_true")
    parser.add_argument("--queue-order", choices=["task-first", "model-first"], default="task-first")
    parser.add_argument("--after-campaign", type=Path)
    parser.add_argument("--after-job")
    parser.add_argument("--deadline-epoch", type=float)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.revalidate_grader and (args.command != "prepare" or not args.reuse_successes_from):
        parser.error("grader revalidation requires prepare with a stopped source campaign")
    if args.retry_watchdog and args.command != "prepare":
        parser.error("watchdog recovery must be frozen during prepare")
    if args.command == "prepare":
        if args.assets is None or args.confirmation is None:
            parser.error("prepare requires assets and confirmation")
        assets, source = args.assets.resolve(), args.confirmation.resolve()
        tasks = json.loads((assets / "tasks.json").read_text())
        task_ids = [t["task_id"] for t in tasks]
        if len(task_ids) != 78 or len(set(task_ids)) != 78 or not set(a.TASK_ORDER) <= set(task_ids):
            parser.error("full suite requires all 78 unique tasks including the pilot")
        ordered = a.TASK_ORDER + [t for t in task_ids if t not in a.TASK_ORDER]
        h.read_sealed(source / "manifest.json")
        policy = {"confirmation": str(source), "manifest_sha256": digest_file(source / "manifest.json"),
                  "selection": "ub128-vs512-v1", "models": MODELS, "production_promotion": False,
                  "memory_recovery": MEMORY_RECOVERY}
        if args.retry_watchdog:
            policy["watchdog_recovery"] = WATCHDOG_RECOVERY
        if args.queue_order != "task-first":
            policy["queue_order"] = args.queue_order
        h.initialize(campaign, ROOT / ".cache/hillclimb-models", profiles=suite_queue(ordered, args.queue_order),
                     max_attempts=3, agentic_assets=assets, continuation=policy)
        if args.reuse_successes_from:
            audit = reuse_completed(args.reuse_successes_from, campaign,
                                    allow_queue_reorder=args.queue_order == "model-first",
                                    revalidate_grader=args.revalidate_grader,
                                    allow_watchdog_recovery=args.retry_watchdog)
            print(f"Reused {len(audit['reused'])} valid measurements; requeued {len(audit['requeued'])} invalid jobs")
        print(campaign)
        return
    if args.command == "stop":
        (campaign / "STOP").touch()
        return
    if args.command == "status":
        for name in ("stage-status.json", "selection.json", "SUMMARY.json", "REPORT.md"):
            if (campaign / name).exists():
                print((campaign / name).read_text())
        return
    if args.deadline_epoch is None or not math.isfinite(args.deadline_epoch) or args.deadline_epoch <= time.time():
        parser.error("resume requires a future absolute deadline")
    if args.command == "handoff" and (args.after_campaign is None or args.after_job is None):
        parser.error("handoff requires --after-campaign and --after-job")
    snapshot = campaign / "snapshot"
    if ROOT.resolve() != snapshot.resolve():
        command = [sys.executable, "-m", "scripts.instruments.confirmed_agentic", args.command,
                   "--campaign", str(campaign), "--deadline-epoch", str(args.deadline_epoch)]
        if args.command == "handoff":
            command += ["--after-campaign", str(args.after_campaign.resolve()), "--after-job", args.after_job]
        env = dict(os.environ, PYTHONPATH=str(snapshot), PYTHONDONTWRITEBYTECODE="1")
        os.chdir(snapshot)
        os.execve(sys.executable, command, env)
    manifest = h.read_sealed(campaign / "manifest.json")
    if args.command == "handoff" and not handoff_at_boundary(
            args.after_campaign, campaign, args.after_job, args.deadline_epoch):
        return
    policy = manifest["continuation"]
    source = Path(policy["confirmation"])
    if digest_file(source / "manifest.json") != policy["manifest_sha256"]:
        raise ValueError("confirmation manifest changed")
    if not a.wait_for_predecessor(campaign, source, args.deadline_epoch):
        return
    selection = resolve_selection(source, campaign)
    h.atomic_json(campaign / "stage-status.json", {"phase": "running_full_suite", "updated": time.time()})
    ConfirmedRunner(campaign, args.deadline_epoch, selection).run()
    result = summarize(campaign)
    h.atomic_json(campaign / "stage-status.json", {"phase": result["state"], "updated": time.time()})
    if result["state"] == "complete":
        from scripts.analysis import session_signatures
        signatures = session_signatures.analyze_campaign(campaign)
        h.atomic_json(campaign / "SESSION-SIGNATURES.json", signatures)
        (campaign / "SESSION-SIGNATURES.md").write_text(session_signatures.render(signatures))


if __name__ == "__main__":
    main()
