"""Time-to-solution harness: Harbor + Codex in a task container against a local Responses server.

Per task: a usage relay on docker0 (container-reachable; the server itself stays on loopback)
sits between the container's Codex and the server. `harbor run` does image start, in-container
Codex install, the agent run and the task verifier. Everything here is CPU-only parsing of what
Harbor and the relay wrote; the parent owns the server lifecycle (see make_manifest.py).

Scoring rule (pre-registered, fable-bench REPORT 2.2): an unsolved or timed-out task scores the
cap. `wall_seconds` in SUMMARY.json is the sum of scored agent walls (image pull, container
start and Codex install are excluded: they are recorded separately).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import egress_scan  # noqa: E402
import relay  # noqa: E402

HERE = Path(__file__).resolve().parent
CODEX_VERSION = "0.160.0"
HARBOR_VERSION = "0.23.0"
DATASET = "swe-rebench/swe-rebench-leaderboard"
CONTEXT = 98304
COMPACT = 88000
CAP_S = 1200
RELAY_BIND = "172.17.0.1"
RUNAWAY_TOKENS = 32768
MAX_TOKEN_REASONS = {"max_output_tokens", "max_tokens", "max_new_tokens", "length"}
CLIENT_DISCONNECTS = ("BrokenPipeError", "ConnectionResetError", "ConnectionAbortedError")
CAP_EPS_S = 30.0  # a call that began this close to the cap, or ended in a disconnect, was cut by Harbor's kill
DOCKER_FS = "/mnt/scratch"  # Docker data root: the image layers Harbor pulls land here
INTERRUPT_GRACE_S = 20  # gpu_swe_fresh.sh gives this process 60 s after SIGTERM before SIGKILL
EXIT_DISK_LOW = 5
EXIT_INTERRUPTED = 130
DEPTH_BUCKETS = ((0, 8192), (8192, 32768), (32768, 65536), (65536, 10**9))

METRIC_NAMES = {
    "llamacpp:prompt_tokens_total": "prompt_tokens_total",
    "llamacpp:prompt_seconds_total": "prompt_seconds_total",
    "llamacpp:tokens_predicted_total": "tokens_predicted_total",
    "llamacpp:tokens_predicted_seconds_total": "tokens_predicted_seconds_total",
    "llamacpp:spec_decode_num_draft_tokens_total": "draft_tokens_total",
    "llamacpp:spec_decode_num_accepted_tokens_total": "draft_accepted_total",
    "vllm:request_prefill_kv_computed_tokens_sum": "prompt_tokens_total",
    "vllm:request_prefill_time_seconds_sum": "prompt_seconds_total",
    "vllm:generation_tokens_total": "tokens_predicted_total",
    "vllm:request_decode_time_seconds_sum": "tokens_predicted_seconds_total",
    "vllm:spec_decode_num_draft_tokens_total": "draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total": "draft_accepted_total",
}


def number(value):
    return relay.number(value)


# ---------------------------------------------------------------- engine counters

def scrape_metrics(url, timeout=5.0):
    """Counters by canonical name; {} if the endpoint is absent. Never raises."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError, UnicodeError):
        return {}
    found = {}
    for line in body.splitlines():
        parts = line.strip().split()
        if len(parts) < 2 or line.startswith("#"):
            continue
        name = METRIC_NAMES.get(parts[0].split("{", 1)[0])
        if name:
            try:
                found[name] = found.get(name, 0.0) + float(parts[1])
            except ValueError:
                pass
    return found


def metrics_delta(before, after):
    """{} unless every shared counter is monotone: a server restart makes the pair meaningless."""
    if not before or not after:
        return {}
    delta = {}
    for key in before.keys() & after.keys():
        if after[key] < before[key]:
            return {}
        delta[key] = after[key] - before[key]
    return delta


# ---------------------------------------------------------------- relay call statistics

def depth_bucket(tokens):
    for low, high in DEPTH_BUCKETS:
        if low <= tokens < high:
            return f"{low}-{high if high < 10**9 else 'inf'}"
    return None


def truncated(call):
    """A generation the server cut at its token limit: scored by the task's outcome, flagged `runaway`."""
    if call.get("event") != "response.incomplete" or call.get("status") != 200 or call.get("error"):
        return False
    reason = call.get("incomplete_reason")
    if reason is None:
        return bool(number(call.get("output_tokens")) and call["output_tokens"] >= RUNAWAY_TOKENS)
    return reason in MAX_TOKEN_REASONS


def inflight_cutoff(facts, cap_s):
    """Epoch after which a failed call was in flight when Harbor killed the agent; None if unknowable."""
    if number(facts.get("agent_started_epoch")):
        return facts["agent_started_epoch"] + cap_s - CAP_EPS_S
    return None


def cap_end_epoch(facts, cap_s):
    """Exact end of the scored window: Harbor's agent start plus the cap; None if unknowable."""
    if number(facts.get("agent_started_epoch")):
        return facts["agent_started_epoch"] + cap_s
    return None


def call_statistics(calls, cutoff_epoch=None, cap_epoch=None):
    """Aggregate relay rows (relay.normalize_call output) for one task.

    Exact decode rate and acceptance exist only where the server reports `timings` (llama);
    otherwise decode_tps_delivery is the client-side estimate and acceptance stays None.
    `cutoff_epoch` names the failed calls the cap kill aborted (`inflight_at_cap_calls`).
    A call started after `cap_epoch`, or rejected by the relay for it, is outside the scored window: it never
    counts as completed or failed, and is listed in `post_cap_calls` (`post_cap_served_calls`: those the model served).
    """
    responses = [c for c in calls if relay.response_call(c)]
    late = [c for c in responses if c.get("post_cap")
            or (cap_epoch is not None and number(c.get("started_epoch")) and c["started_epoch"] > cap_epoch)]
    rows = [c for c in responses if c not in late]
    completed = [c for c in rows if (c.get("status") == 200 and not c.get("error")
                                     and c.get("event") == "response.completed") or truncated(c)]
    timed = [c for c in completed if isinstance(c.get("timings"), dict)]
    predicted_n = sum(c["timings"].get("predicted_n", 0) for c in timed if number(c["timings"].get("predicted_n")))
    predicted_s = sum(c["timings"]["predicted_ms"] / 1000 for c in timed if number(c["timings"].get("predicted_ms")))
    prompt_n = sum(c["timings"].get("prompt_n", 0) for c in timed if number(c["timings"].get("prompt_n")))
    prompt_s = sum(c["timings"]["prompt_ms"] / 1000 for c in timed if number(c["timings"].get("prompt_ms")))
    drafted = sum(c["timings"].get("draft_n", 0) for c in timed if number(c["timings"].get("draft_n")))
    accepted = sum(c["timings"].get("draft_n_accepted", 0) for c in timed if number(c["timings"].get("draft_n_accepted")))
    drafts_complete = bool(timed) and all(number(c["timings"].get("draft_n")) and
                                         number(c["timings"].get("draft_n_accepted")) for c in timed)
    delivery_tokens = sum(c["output_tokens"] - 1 for c in completed
                          if number(c.get("output_tokens")) and c["output_tokens"] > 1
                          and number(c.get("delivery_decode_seconds")) and c["delivery_decode_seconds"] > 0)
    delivery_s = sum(c["delivery_decode_seconds"] for c in completed
                     if number(c.get("output_tokens")) and c["output_tokens"] > 1
                     and number(c.get("delivery_decode_seconds")) and c["delivery_decode_seconds"] > 0)
    waits = sorted(c["prompt_wait_seconds"] for c in completed if number(c.get("prompt_wait_seconds")))
    contexts = [(c["input_tokens"], c["output_tokens"]) for c in completed
                if number(c.get("input_tokens")) and number(c.get("output_tokens"))]
    inputs = [c["input_tokens"] for c in completed if number(c.get("input_tokens"))]
    cached = sum(c["cached_tokens"] for c in completed if number(c.get("cached_tokens")))
    total_in = sum(inputs)
    buckets = {}
    for c in timed:
        if not number(c.get("input_tokens")):
            continue
        b = buckets.setdefault(depth_bucket(c["input_tokens"]), {"calls": 0, "predicted_n": 0, "predicted_s": 0.0,
                                                                 "draft_n": 0, "draft_accepted": 0})
        b["calls"] += 1
        t = c["timings"]
        b["predicted_n"] += t.get("predicted_n", 0) if number(t.get("predicted_n")) else 0
        b["predicted_s"] += t["predicted_ms"] / 1000 if number(t.get("predicted_ms")) else 0.0
        b["draft_n"] += t.get("draft_n", 0) if number(t.get("draft_n")) else 0
        b["draft_accepted"] += t.get("draft_n_accepted", 0) if number(t.get("draft_n_accepted")) else 0
    for b in buckets.values():
        b["decode_tps"] = b["predicted_n"] / b["predicted_s"] if b["predicted_s"] else None
        b["draft_acceptance"] = b["draft_accepted"] / b["draft_n"] if b["draft_n"] else None
    drops = sum(1 for a, b in zip(inputs, inputs[1:]) if b < 0.7 * a)
    failed = [c for c in rows if c not in completed]
    inflight = [] if cutoff_epoch is None else [
        c.get("call") for c in failed
        if (number(c.get("started_epoch")) and c["started_epoch"] >= cutoff_epoch)
        or (str(c.get("error") or "").startswith(CLIENT_DISCONNECTS)
            and number(c.get("finished_epoch")) and c["finished_epoch"] >= cutoff_epoch)]
    return {
        "calls": len(rows), "completed_calls": len(completed),
        "failed_calls": [c.get("call") for c in failed], "inflight_at_cap_calls": inflight,
        "post_cap_calls": [c.get("call") for c in late],
        "post_cap_served_calls": [c.get("call") for c in late if not c.get("post_cap")],
        "truncated_calls": [c.get("call") for c in completed if truncated(c)],
        "http_500_calls": [c.get("call") for c in rows if c.get("status") == 500],
        "input_tokens": total_in, "cached_tokens": cached,
        "output_tokens": sum(c["output_tokens"] for c in completed if number(c.get("output_tokens"))),
        "cache_hit_fraction": cached / total_in if total_in else None,
        "peak_context_tokens": max((i + o for i, o in contexts), default=0),
        "peak_input_tokens": max(inputs, default=0),
        "runaway_calls": [c.get("call") for c in completed if number(c.get("output_tokens"))
                          and c["output_tokens"] >= RUNAWAY_TOKENS],
        "prefill_seconds": prompt_s if timed else None,
        "prefill_tokens": prompt_n if timed else None,
        "prefill_tps": prompt_n / prompt_s if prompt_s else None,
        "decode_seconds": predicted_s if timed else None,
        "decode_tokens": predicted_n if timed else None,
        "decode_tps": predicted_n / predicted_s if predicted_s else None,
        "decode_tps_delivery": delivery_tokens / delivery_s if delivery_s else None,
        "draft_acceptance": accepted / drafted if drafted and drafts_complete else None,
        "ttft_median_s": statistics.median(waits) if waits else None,
        "ttft_p90_s": waits[min(len(waits) - 1, math.ceil(0.9 * len(waits)) - 1)] if waits else None,
        "input_drops_over_30pct": drops,
        "by_depth": buckets,
        "timing_scope": "decode_tps/acceptance exact only where the server reports timings; "
                        "decode_tps_delivery is a client estimate",
    }


def read_calls(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------- Harbor trial parsing

def epoch(stamp):
    """Harbor timestamps are UTC with a Z suffix; relay rows use time.time(), so the two compare directly."""
    if not stamp:
        return None
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


def seconds(timing):
    if not isinstance(timing, dict) or not timing.get("started_at") or not timing.get("finished_at"):
        return None
    return epoch(timing["finished_at"]) - epoch(timing["started_at"])


def find_trial(job_dir):
    """The single trial directory of a one-task job (result.json with a `trial_name`)."""
    found = [p.parent for p in Path(job_dir).glob("*/result.json")
             if "trial_name" in json.loads(p.read_text())]
    if len(found) != 1:
        raise ValueError(f"expected one trial under {job_dir}, found {len(found)}")
    return found[0]


def stamped_after(stamp, limit):
    try:
        return epoch(stamp) > limit
    except (AttributeError, TypeError, ValueError):
        return False


def rollout_stats(trial, cap_epoch=None):
    """Tool calls, assistant turns and compaction records from the Codex rollout(s) Harbor copied out.

    `post_cap_tool_calls`: tool calls stamped after `cap_epoch`, i.e. possible edits made after the cap
    while Harbor's verifier ran (Harbor kills only the exec client, not Codex).
    """
    files = sorted((Path(trial) / "agent" / "sessions").rglob("rollout-*.jsonl"))
    stats = {"rollout_files": len(files), "tool_calls": 0, "assistant_messages": 0, "compactions": 0,
             "post_cap_tool_calls": 0}
    for path in files:
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            kind, payload = row.get("type"), row.get("payload") or {}
            if kind == "compacted":
                stats["compactions"] += 1
            elif kind == "response_item" and isinstance(payload, dict):
                item = payload.get("type")
                if item in ("function_call", "custom_tool_call", "local_shell_call"):
                    stats["tool_calls"] += 1
                    if cap_epoch is not None and stamped_after(row.get("timestamp"), cap_epoch):
                        stats["post_cap_tool_calls"] += 1
                elif item == "message" and payload.get("role") == "assistant":
                    stats["assistant_messages"] += 1
    return stats


def parse_trial(trial, cap_s=None):
    """Harbor result.json -> facts. `solved` requires a verifier reward of 1 and no exception.

    `cap_s` dates the rollout's post-cap tool calls; without it none are counted."""
    trial = Path(trial)
    result = json.loads((trial / "result.json").read_text())
    verifier = result.get("verifier_result") or {}
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    reward = None
    if isinstance(rewards, dict):
        values = [v for v in rewards.values() if number(v)]
        reward = rewards.get("reward") if number(rewards.get("reward")) else (values[0] if values else None)
    job_config = trial.parent / "config.json"
    datasets = json.loads(job_config.read_text()).get("datasets") if job_config.exists() else None
    exception = result.get("exception_info") or None
    kind = exception.get("exception_type") if exception else None
    reward_file = trial / "verifier" / "reward.txt"
    started = epoch((result.get("agent_execution") or {}).get("started_at"))
    return {
        "trial_name": result.get("trial_name"), "task_name": result.get("task_name"),
        "task_checksum": result.get("task_checksum"),
        "dataset_ref": datasets[0].get("ref") if datasets else None,
        "reward": reward,
        "reward_txt": reward_file.read_text().strip() if reward_file.exists() else None,
        "solved": bool(number(reward) and reward >= 1 and not (exception and kind != "AgentTimeoutError")),
        "exception_type": kind, "exception_message": (exception or {}).get("exception_message"),
        "timed_out": kind == "AgentTimeoutError",
        "environment_setup_seconds": seconds(result.get("environment_setup")),
        "agent_setup_seconds": seconds(result.get("agent_setup")),
        "agent_wall_seconds": seconds(result.get("agent_execution")),
        "agent_started_epoch": started,
        "verifier_seconds": seconds(result.get("verifier")),
        "agent_version": (result.get("agent_info") or {}).get("version"),
        "harbor_agent_tokens": {k: (result.get("agent_result") or {}).get(k)
                                for k in ("n_input_tokens", "n_cache_tokens", "n_output_tokens")},
        **rollout_stats(trial, cap_end_epoch({"agent_started_epoch": started}, cap_s) if cap_s else None),
        "egress_scan": egress_scan.scan_trial(trial),
    }


def post_cap_activity(facts, stats):
    """The agent acted after the cap: a model call the model served, or a tool call (a possible edit).

    Harbor still runs the verifier then, so a reward earned this way is not credited as a solve."""
    return bool(stats.get("post_cap_served_calls") or facts.get("post_cap_tool_calls"))


def classify(facts, stats, cap_s):
    """Return (status, gates). Unsolved and timeout are valid outcomes; a run that never
    reached the model, or whose calls failed, is not a measurement."""
    gates = []
    if facts["exception_type"] and facts["exception_type"] != "AgentTimeoutError":
        gates.append(f"harbor_exception: {facts['exception_type']}")
    if facts["reward"] is None and not facts["timed_out"] and not gates:
        gates.append("no_verifier_reward")
    if not stats["calls"]:
        gates.append("no_model_calls")
    failed = stats["failed_calls"]
    if facts["timed_out"]:
        # Harbor's kill at the cap aborts the call being served; that is the timeout, not a model-call failure,
        # whatever the verifier then says (it runs after the kill, on a tree Codex may still be changing).
        cut = set(stats.get("inflight_at_cap_calls", ()))
        failed = [c for c in failed if c not in cut]
    if failed:
        gates.append(f"call_failed: {failed[:5]}")
    if stats["calls"] and any(not number(c) for c in (stats["input_tokens"], stats["output_tokens"])):
        gates.append("missing_call_usage")
    wall = facts["agent_wall_seconds"]
    if wall is None and not gates:
        gates.append("missing_agent_timing")
    if number(wall) and wall > cap_s * 1.1 + 60:
        gates.append(f"cap_not_enforced: {wall:.0f}s")
    if gates:
        return "invalid", gates
    if facts["solved"] and not (facts["timed_out"] and post_cap_activity(facts, stats)):
        return "passed", gates
    return ("timeout" if facts["timed_out"] else "unsolved"), gates


def build_job(task, run, agent, facts, stats, delta, args, started_epoch, finished_epoch):
    status, gates = classify(facts, stats, args.cap_s)
    wall = facts["agent_wall_seconds"]
    scored = min(wall, float(args.cap_s)) if status == "passed" and number(wall) else float(args.cap_s)
    mtp = None
    if stats["draft_acceptance"] is not None:
        mtp = {"draft_accept_rate": stats["draft_acceptance"], "source": "per_call_timings"}
    elif delta.get("draft_tokens_total"):
        mtp = {"draft_accept_rate": delta.get("draft_accepted_total", 0) / delta["draft_tokens_total"],
               "source": "engine_counter_delta_exclusive_server"}
    if stats["decode_tps"] is not None:
        decode_tps, decode_source = stats["decode_tps"], "per_call_timings"
    elif delta.get("tokens_predicted_seconds_total"):
        decode_tps = delta["tokens_predicted_total"] / delta["tokens_predicted_seconds_total"]
        decode_source = "engine_counter_delta_exclusive_server"
    else:
        decode_tps, decode_source = None, None
    return {
        "job": f"swe-fresh-{task}-r{run}", "model": args.model, "task": task, "run": run, "agent": agent,
        "config": {"engine": args.engine, "context": args.context, "slots": args.agents,
                   "workload": "swe-fresh", "codex": args.codex_version, "compact": args.compact,
                   "cap_s": args.cap_s, "provider": args.provider},
        "status": status, "gates": gates,
        "probe": {
            "workload": "swe-fresh", "task_id": task, "complete": status in ("passed", "unsolved", "timeout"),
            "task_success": status == "passed", "timed_out": facts["timed_out"],
            "error": facts["exception_message"] if status == "invalid" else None,
            "verified_wall_seconds": scored, "agent_wall_seconds": wall,
            "censored": status != "passed" or facts["timed_out"],
            "verifier_seconds": facts["verifier_seconds"],
            "environment_setup_seconds": facts["environment_setup_seconds"],
            "agent_setup_seconds": facts["agent_setup_seconds"],
            "peak_context_tokens": stats["peak_context_tokens"],
            "context_peak_source": "relay: max input+output tokens of one /v1/responses call",
            "client_context": args.context, "client_compaction": args.compact,
            "metrics_scope": "exclusive_server_task" if args.agents == 1 else "unavailable_concurrent_agents",
            "metrics_delta": delta, "mtp": mtp, "mtp_live": bool(mtp),
            "agent_stats": {"tool_calls": facts["tool_calls"], "agent_turns": stats["calls"],
                            "assistant_messages": facts["assistant_messages"],
                            "prompt_tokens": stats["input_tokens"], "completion_tokens": stats["output_tokens"],
                            "cached_input_tokens": stats["cached_tokens"]},
            "compactions": facts["compactions"], "runaway_calls": stats["runaway_calls"],
            "runaway": bool(stats["runaway_calls"] or stats["truncated_calls"]),
            "ignored_inflight_calls": stats["inflight_at_cap_calls"] if facts["timed_out"] else [],
            "post_cap": {"calls": stats["post_cap_calls"], "served_calls": stats["post_cap_served_calls"],
                         "tool_calls": facts.get("post_cap_tool_calls", 0),
                         "reward_not_credited": bool(facts["timed_out"] and facts["solved"] and status != "passed")},
            "egress_scan": facts.get("egress_scan") or {"flagged": False, "hits": [], "calls_scanned": 0},
            "reward": facts["reward"], "task_checksum": facts["task_checksum"],
            "dataset_ref": facts.get("dataset_ref"),
            "pristine_grader": True, "external_network": None,
        },
        "call_stats": stats,
        "decode_tps": decode_tps, "decode_tps_source": decode_source,
        "decode_tps_delivery": stats["decode_tps_delivery"], "prefill_tps": stats["prefill_tps"],
        "started_at": started_epoch, "finished_at": finished_epoch, "elapsed_seconds": finished_epoch - started_epoch,
    }


def campaign_summary(jobs, items, wall, delta, args, gates=()):
    problems = list(gates)
    if sorted((j["task"], j["run"]) for j in jobs) != sorted(map(tuple, items)):
        problems.append("incomplete_task_queue")
    for j in jobs:
        problems += [f"{j['task']}#r{j['run']}: {g}" for g in j["gates"]]
    done = [j for j in jobs if j["status"] != "invalid"]
    decode_s = sum(j["call_stats"]["decode_seconds"] or 0 for j in done)
    decode_n = sum(j["call_stats"]["decode_tokens"] or 0 for j in done)
    prefill_s = sum(j["call_stats"]["prefill_seconds"] or 0 for j in done)
    if not decode_s and delta.get("tokens_predicted_seconds_total"):
        decode_s, decode_n = delta["tokens_predicted_seconds_total"], delta["tokens_predicted_total"]
    if not prefill_s and delta.get("prompt_seconds_total"):
        prefill_s = delta["prompt_seconds_total"]
    drafted = delta.get("draft_tokens_total")
    per_call_accept = [j["probe"]["mtp"]["draft_accept_rate"] for j in done
                       if j["probe"].get("mtp") and j["probe"]["mtp"]["source"] == "per_call_timings"]
    if drafted:
        accept = delta.get("draft_accepted_total", 0) / drafted
    else:
        accept = statistics.fmean(per_call_accept) if per_call_accept else None
    output = sum(j["call_stats"]["output_tokens"] for j in done)
    scored = sum(j["probe"]["verified_wall_seconds"] for j in jobs)
    return {
        "updated_epoch": time.time(), "state": "complete" if not problems else "invalid",
        "valid": not problems, "gates": problems, "model": args.model, "tasks": len(jobs),
        "solved": sum(j["status"] == "passed" for j in jobs),
        "unsolved": sum(j["status"] == "unsolved" for j in jobs),
        "timeouts": sum(j["status"] == "timeout" for j in jobs),
        "invalid_tasks": sum(j["status"] == "invalid" for j in jobs),
        "wall_seconds": scored, "wall_scope": "sum of scored agent walls; unsolved/timeout score the cap",
        "solved_wall_seconds": sum(j["probe"]["verified_wall_seconds"] for j in jobs if j["status"] == "passed"),
        "campaign_elapsed_seconds": wall, "cap_s": args.cap_s,
        "output_tokens": float(output),
        "prefill_seconds": prefill_s or None, "decode_seconds": decode_s or None,
        "decode_tps": decode_n / decode_s if decode_s else None,
        "decode_tps_delivery": statistics.fmean([j["decode_tps_delivery"] for j in done if j["decode_tps_delivery"]])
        if any(j["decode_tps_delivery"] for j in done) else None,
        "peak_context_tokens": max((j["probe"]["peak_context_tokens"] for j in done), default=0),
        "draft_acceptance": accept,
        "acceptance_source": "engine_counter_delta" if drafted else ("per_call_timings" if per_call_accept else None),
        "compactions": sum(j["probe"]["compactions"] for j in done),
        "runaway_calls": sum(len(j["probe"]["runaway_calls"]) for j in done),
        "runaway_tasks": [f'{j["task"]}#r{j["run"]}' for j in jobs if j["probe"].get("runaway")],
        "post_cap_activity_tasks": [f'{j["task"]}#r{j["run"]}' for j in jobs
                                    if j["probe"].get("post_cap", {}).get("served_calls") or j["probe"].get("post_cap", {}).get("tool_calls")],
        "egress_flagged_tasks": [f'{j["task"]}#r{j["run"]}' for j in jobs if j["probe"].get("egress_scan", {}).get("flagged")],
        "http_500_calls": sum(len(j["call_stats"]["http_500_calls"]) for j in done),
        "tool_calls": sum(j["probe"]["agent_stats"]["tool_calls"] for j in done),
        "metrics_scope": "exclusive_server_whole_campaign", "metrics_delta": delta,
        "per_task": [{"task": j["task"], "run": j["run"], "status": j["status"], "scored_wall_s": j["probe"]["verified_wall_seconds"],
                      "peak_context_tokens": j["probe"]["peak_context_tokens"],
                      "calls": j["call_stats"]["calls"], "output_tokens": j["call_stats"]["output_tokens"],
                      "decode_tps": j["decode_tps"], "draft_acceptance": j["call_stats"]["draft_acceptance"],
                      "compactions": j["probe"]["compactions"]} for j in sorted(jobs, key=lambda j: (j["run"], j["task"]))],
    }


def canary_failures(jobs):
    """The canary proves plumbing, not capability: a 27B model may legitimately fail the task."""
    return [f'{j["task"]}#r{j["run"]}' for j in jobs if j["status"] == "invalid" or j["call_stats"]["completed_calls"] < 2
            or not j["probe"]["agent_stats"]["tool_calls"]]


def campaign_verdict(jobs, items, wall, delta, args, canary=False, over_budget=False, expect_counters=False, stops=()):
    gates = []
    if expect_counters and not delta:
        gates.append("engine_counters_unavailable")
    gates += stops
    summary = campaign_summary(jobs, items, wall, delta, args, gates)
    bad = canary_failures(jobs) if canary else []
    if bad:
        summary["gates"].append(f"canary_plumbing_failed: {bad}")
        summary.update(valid=False, state="invalid")
    if over_budget:
        summary["gates"].append("budget")
        summary.update(valid=False, state="invalid")
    return summary


def exit_code(summary):
    if summary["valid"]:
        return 0
    return EXIT_DISK_LOW if any(g.startswith("disk_low") for g in summary["gates"]) else 2


# ---------------------------------------------------------------- running Harbor

def codex_config(args):
    config = {"model_context_window": args.context, "model_auto_compact_token_limit": args.compact,
              "web_search": "disabled", "project_doc_max_bytes": 0}
    if args.provider == "local":
        config.update(model_provider="local", model_providers={"local": {
            "name": "Local relay", "base_url": "RELAY_BASE_URL", "wire_api": "responses",
            "requires_openai_auth": False, "supports_websockets": False,
            "request_max_retries": 2, "stream_max_retries": 2, "stream_idle_timeout_ms": 900000}})
    return config


def harbor_requirement(args):
    """Never `latest`: a floating Harbor changes task handling between the canary and the measurement."""
    return f"harbor=={args.harbor_version or HARBOR_VERSION}"


def harbor_version_command(args):
    return ["uvx", harbor_requirement(args), "--version"]


def harbor_command(task, base_url, jobs_dir, job_name, multiplier, args):
    config = codex_config(args)
    if args.provider == "local":
        config["model_providers"]["local"]["base_url"] = base_url
    command = ["uvx", harbor_requirement(args), "run"]
    command += ["-a", "codex", "-m", f"openai/{args.model}", "-i", task if args.dataset_path else f"swe-rebench/{task}", "-n", "1", "-k", "1", "--delete",
                "--job-name", job_name, "-o", str(jobs_dir), "--yes", "--quiet",
                "--agent-timeout-multiplier", f"{multiplier:.6f}",
                "--ae", "OPENAI_API_KEY=sk-swefresh-dummy", "--ae", f"OPENAI_BASE_URL={base_url}",
                "--allow-agent-host", args.relay_bind,
                "--ak", f"version={args.codex_version}", "--ak", f"reasoning_effort={args.reasoning}",
                "--ak", "config=" + json.dumps(config, separators=(",", ":"))]
    dataset = DATASET + (f"@{args.dataset_ref}" if args.dataset_ref else "")
    command += ["-p", str(args.dataset_path)] if args.dataset_path else ["-d", dataset]
    return command


def agent_timeout(task, args):
    """Task-declared agent timeout, to turn the wall cap into Harbor's multiplier."""
    if args.dataset_path:
        text = (Path(args.dataset_path) / task / "task.toml").read_text()
        import tomllib
        return float(tomllib.loads(text).get("agent", {}).get("timeout_sec", 3000))
    return 3000.0


def stop_group(process, grace=90, pgid=None):
    pgid = pgid or process.pid
    for sig, wait in ((signal.SIGINT, grace), (signal.SIGKILL, 10)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(wait)
            return
        except subprocess.TimeoutExpired:
            continue


STOP = threading.Event()  # set by SIGINT/SIGTERM; workers forward it to the Harbor groups they started
ACTIVE = {}
ACTIVE_LOCK = threading.RLock()


def install_stop_handlers():
    def handler(signum, frame):
        STOP.set()

    return {sig: signal.signal(sig, handler) for sig in (signal.SIGINT, signal.SIGTERM)}


def restore_stop_handlers(previous):
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def disk_free_gb(path=DOCKER_FS):
    return shutil.disk_usage(path).free // 2**30


def proc_start_tick(pid):
    """/proc starttime: tells a live recorded process from a recycled pid."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def compose_project(trial_name):
    """Harbor's compose project for a trial's environment (docker.py _sanitize_docker_compose_project_name)."""
    name = f"{trial_name}__env".lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    return re.sub(r"[^a-z0-9_-]", "-", name)


class ProjectWatcher:
    """Records the compose projects of this job's trials as their directories appear: the names exist only at run time."""

    def __init__(self, job_dir, record):
        self.job_dir, self.record, self.names = Path(job_dir), Path(record), []
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def scan(self):
        if not self.job_dir.is_dir():
            return
        for path in sorted(self.job_dir.iterdir()):
            name = compose_project(path.name)
            if path.is_dir() and name not in self.names:
                self.names.append(name)
                relay.write_json(self.record, self.names)

    def poll(self):
        while not self.done.wait(0.5):
            self.scan()

    def start(self):
        self.thread.start()

    def stop(self):
        self.done.set()
        self.thread.join()
        self.scan()
        return list(self.names)


def compose_down(names, runner=subprocess.run):
    """Only the named projects; never a prune, never a filter wider than the recorded names."""
    for name in names:
        try:
            runner(["docker", "compose", "--project-name", name, "down", "--volumes", "--remove-orphans"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45, check=False)
        except (OSError, subprocess.SubprocessError) as error:
            print(f"compose down {name} failed: {error}", file=sys.stderr)


def wait_harbor(process, pgid, hard_timeout_s, stop, grace):
    end = time.monotonic() + hard_timeout_s
    while True:
        try:
            process.wait(0.5)
            return "interrupted" if stop.is_set() else None
        except subprocess.TimeoutExpired:
            pass
        if stop.is_set():
            stop_group(process, grace, pgid)
            return "interrupted"
        if time.monotonic() >= end:
            stop_group(process, 90, pgid)
            return "hard_timeout"


def run_harbor(command, out, job_dir, hard_timeout_s, stop=None, grace=INTERRUPT_GRACE_S):
    """Run Harbor in its own session; None if it exited by itself, else "hard_timeout" or "interrupted".

    Contract: Harbor's own cleanup runs only if it gets the signal, so an abnormal end signals the
    process group recorded here (and no other), then downs the compose projects recorded under this job.
    """
    stop = STOP if stop is None else stop
    watcher = ProjectWatcher(job_dir, Path(out) / "compose-projects.json")
    abnormal = None
    with (Path(out) / "harbor.log").open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    pgid = os.getpgid(process.pid)
    relay.write_json(Path(out) / "harbor.process.json",
                     {"pid": process.pid, "pgid": pgid, "start_tick": proc_start_tick(process.pid)})
    with ACTIVE_LOCK:
        ACTIVE[process.pid] = pgid
    watcher.start()
    try:
        abnormal = wait_harbor(process, pgid, hard_timeout_s, stop, grace)
    except BaseException:
        abnormal = "interrupted"
        stop_group(process, grace, pgid)
        raise
    finally:
        with ACTIVE_LOCK:
            ACTIVE.pop(process.pid, None)
        names = watcher.stop()
        if abnormal:
            compose_down(names)
    return abnormal


def cleanup_campaign(campaign, runner=subprocess.run):
    """Backstop after this runner was SIGKILLed: unfinished attempts' recorded group and compose projects only."""
    for attempt in sorted(Path(campaign).glob("attempts/*")):
        if (attempt / "job.json").exists():
            continue
        record = attempt / "harbor.process.json"
        if record.exists():
            info = json.loads(record.read_text())
            if proc_start_tick(info["pid"]) == info["start_tick"]:
                try:
                    os.killpg(info["pgid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        projects = attempt / "compose-projects.json"
        if projects.exists():
            compose_down(json.loads(projects.read_text()), runner=runner)


def run_task(agent, item, args, campaign):
    task, run = item
    out = campaign / "attempts" / f"agent-{agent}-{task}-r{run}"
    out.mkdir(parents=True)
    started_epoch = time.time()
    deadline = time.monotonic() + args.hard_timeout_s
    proxy = relay.UsageProxy(args.upstream_port, out / "relay", agent, f"{task}#r{run}", deadline,
                             bind=args.relay_bind, upstream_host=args.upstream_host, cap_s=args.cap_s)
    (out / "relay").mkdir()
    proxy.start()
    before = scrape_metrics(args.metrics_url) if args.metrics_url else {}
    base_url = f"http://{args.relay_bind}:{proxy.port}/v1"
    job_name = f"{campaign.name}-{agent}-{task}-r{run}"[:120]
    command = harbor_command(task, base_url, out / "harbor", job_name,
                             args.cap_s / agent_timeout(task, args), args)
    (out / "harbor.command.json").write_text(json.dumps(command, indent=1))
    facts, error = None, None
    try:
        error = run_harbor(command, out, out / "harbor" / job_name, args.hard_timeout_s)
        if error != "interrupted":
            facts = parse_trial(find_trial(out / "harbor" / job_name), args.cap_s)
    except Exception as exc:
        error = error or f"{type(exc).__name__}: {exc}"
    finally:
        calls = proxy.stop()
    after = scrape_metrics(args.metrics_url) if args.metrics_url else {}
    delta = metrics_delta(before, after) if args.agents == 1 else {}
    if facts is None:
        facts = {"trial_name": None, "task_name": task, "task_checksum": None, "reward": None, "reward_txt": None,
                 "solved": False, "exception_type": "HarnessError", "exception_message": error, "timed_out": False,
                 "environment_setup_seconds": None, "agent_setup_seconds": None, "agent_wall_seconds": None,
                 "agent_started_epoch": None, "verifier_seconds": None, "agent_version": None, "harbor_agent_tokens": {},
                 "rollout_files": 0, "tool_calls": 0, "assistant_messages": 0, "compactions": 0, "post_cap_tool_calls": 0}
    stats = call_statistics(calls, inflight_cutoff(facts, args.cap_s), cap_end_epoch(facts, args.cap_s))
    job = build_job(task, run, agent, facts, stats, delta, args, started_epoch, time.time())
    relay.write_json(out / "job.json", job)
    relay.write_json(campaign / f"job-agent-{agent}-{task}-r{run}.json", job)
    return job


def run_queue(tasks, agents, execute, deadline, min_free_gb=0, free_gb=None, stops=None, stop=None):
    """Workers stop before starting a task past `deadline`, on `stop`, or with < min_free_gb GiB free (named in `stops`)."""
    stop = STOP if stop is None else stop
    stops = [] if stops is None else stops
    free_gb = free_gb or disk_free_gb
    pending = queue.Queue()
    for task in tasks:
        pending.put(task)

    def worker(agent):
        rows = []
        while time.monotonic() < deadline and not stop.is_set() and not stops:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                break
            if min_free_gb:
                free = free_gb()
                if free < min_free_gb:
                    label = f"{task[0]}#r{task[1]}" if isinstance(task, tuple) else str(task)
                    stops.append(f"disk_low: {free} GiB < {min_free_gb} GiB before {label}")
                    break
            rows.append(execute(agent, task))
        return rows

    with ThreadPoolExecutor(max_workers=agents) as pool:
        futures = [pool.submit(worker, agent) for agent in range(agents)]
        return [row for future in futures for row in future.result()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_tasks(args):
    if args.tasks_file:
        document = json.loads(Path(args.tasks_file).read_text())
        return [t["instance_id"] if isinstance(t, dict) else t for t in document["tasks"]]
    return args.tasks.split(",")


def add_common(parser):
    parser.add_argument("--model", default="local-benchmark", help="served model alias; Codex sends this name")
    parser.add_argument("--engine", default="unknown")
    parser.add_argument("--context", type=int, default=CONTEXT)
    parser.add_argument("--compact", type=int, default=COMPACT)
    parser.add_argument("--cap-s", type=float, default=CAP_S)
    parser.add_argument("--agents", type=int, default=1, choices=[1, 2])
    parser.add_argument("--codex-version", default=CODEX_VERSION)
    parser.add_argument("--harbor-version", default=HARBOR_VERSION, help="uvx harbor==X; there is no unpinned mode")
    parser.add_argument("--reasoning", default="xhigh")
    parser.add_argument("--provider", choices=["builtin", "local"], default="builtin",
                        help="builtin: openai provider + openai_base_url; local: model_provider=local fallback")
    parser.add_argument("--dataset-ref", help="registry digest/version to pin, e.g. sha256:... (job config.json records the resolved one)")
    parser.add_argument("--dataset-path", type=Path, help="local Harbor dataset snapshot (-p) instead of -d")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    add_common(run)
    run.add_argument("--campaign", type=Path, required=True)
    group = run.add_mutually_exclusive_group(required=True)
    group.add_argument("--tasks")
    group.add_argument("--tasks-file")
    run.add_argument("--upstream-port", type=int, required=True, help="server port on --upstream-host")
    run.add_argument("--upstream-host", default="127.0.0.1")
    run.add_argument("--relay-bind", default=RELAY_BIND)
    run.add_argument("--metrics-url", help="server base URL whose /metrics holds engine counters")
    run.add_argument("--budget-s", type=float, default=3600)
    run.add_argument("--hard-timeout-s", type=float, default=3300,
                     help="per task, covers pull + build + Codex install + agent + verifier")
    run.add_argument("--repeat", type=int, default=1, help="runs per task; round-major order")
    run.add_argument("--canary", action="store_true")
    run.add_argument("--min-free-gb", type=int, default=0, help="stop before a task when Docker's filesystem has less free")
    summ = sub.add_parser("summarize", help="rebuild SUMMARY.json from a campaign directory")
    add_common(summ)
    summ.add_argument("--campaign", type=Path, required=True)
    summ.add_argument("--rescan-egress", action="store_true", help="re-scan saved rollouts into the job files first")
    proof = sub.add_parser("canary-proof", help="measure stage: locate (and copy) this OUT's passed, pinned canary")
    proof.add_argument("--out", type=Path, required=True)
    proof.add_argument("--pins-sha", required=True)
    clean = sub.add_parser("cleanup", help="backstop: kill recorded Harbor groups and down recorded compose projects")
    clean.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        return summarize_dir(args)
    if args.command == "canary-proof":
        source = adopt_canary_proof(args.out.resolve(), args.pins_sha)
        if source is None:
            print("no valid canary with these pins under " + str(args.out), file=sys.stderr)
            return 1
        print(source)
        return 0
    if args.command == "cleanup":
        cleanup_campaign(args.campaign.resolve())
        return 0
    if args.context - args.compact < 8000 or args.cap_s <= 0:
        parser.error("require headroom between window and compaction, positive cap")
    args.campaign = args.campaign.resolve()
    args.campaign.mkdir(parents=True, exist_ok=False)
    tasks = load_tasks(args)
    items = [(t, k) for k in range(1, args.repeat + 1) for t in tasks]  # round-major: drift hits every task alike
    started = time.monotonic()
    manifest = {"tasks": tasks, "items": items, "args": {k: str(v) for k, v in vars(args).items()},
                "runner_sha256": sha256(__file__), "relay_sha256": sha256(HERE / "relay.py"),
                "harbor_version_output": subprocess.run(harbor_version_command(args), capture_output=True,
                                                        text=True).stdout.strip(),
                "started_epoch": time.time()}
    relay.write_json(args.campaign / "manifest.json", manifest)
    summary = {"valid": False, "state": "invalid", "gates": ["startup_incomplete"]}
    previous = install_stop_handlers()
    try:
        before = scrape_metrics(args.metrics_url) if args.metrics_url else {}
        relay.write_json(args.campaign / "metrics-before.json", before)
        begin = time.monotonic()
        stops = []
        jobs = run_queue(items, args.agents, lambda a, t: run_task(a, t, args, args.campaign), started + args.budget_s,
                         min_free_gb=args.min_free_gb, stops=stops)
        wall = time.monotonic() - begin
        after = scrape_metrics(args.metrics_url) if args.metrics_url else {}
        relay.write_json(args.campaign / "metrics-after.json", after)
        summary = campaign_verdict(jobs, items, wall, metrics_delta(before, after), args, canary=args.canary,
                                   over_budget=time.monotonic() >= started + args.budget_s,
                                   expect_counters=bool(args.metrics_url),
                                   stops=stops + (["interrupted"] if STOP.is_set() else []))
    except Exception as error:
        summary = {"valid": False, "state": "invalid", "gates": [f"{type(error).__name__}: {error}"]}
    finally:
        relay.write_json(args.campaign / "SUMMARY.json", summary)
        restore_stop_handlers(previous)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_task", "metrics_delta")}, indent=1))
    return EXIT_INTERRUPTED if STOP.is_set() else exit_code(summary)


def canary_proof(out, pins_sha):
    """Directory of a passed canary whose pins match, for a measure stage of the same OUT.

    gpuq's fresh retry renames OUT to `OUT.void-a<N>` before the stage starts, which takes the canary with it;
    a retried measure therefore finds its proof there (newest attempt first)."""
    out = Path(out)

    def attempt(path):
        tail = path.name.rsplit(".void-a", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    for directory in [out, *sorted(out.parent.glob(out.name + ".void-a*"), key=attempt, reverse=True)]:
        try:
            summary = json.loads((directory / "canary" / "SUMMARY.json").read_text())
            pins = (directory / "canary.pins").read_text().strip()
        except (OSError, ValueError):
            continue
        if isinstance(summary, dict) and summary.get("valid") is True and pins == pins_sha:
            return directory
    return None


def adopt_canary_proof(out, pins_sha):
    """Find the canary proof and keep a copy in `out/canary-proof`; None if there is no valid, pinned canary."""
    source = canary_proof(out, pins_sha)
    if source is None:
        return None
    if source != Path(out):
        target = Path(out) / "canary-proof"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "canary" / "SUMMARY.json", target / "SUMMARY.json")
        shutil.copy2(source / "canary.pins", target / "canary.pins")
        (target / "source.txt").write_text(f"{source}\n")
    return source


def summarize_dir(args):
    """Recompute SUMMARY.json from the job files already on disk (no Harbor, no server)."""
    campaign = args.campaign.resolve()
    if args.rescan_egress:
        egress_scan.rescan_campaign(campaign)
    jobs = [json.loads(p.read_text()) for p in sorted(campaign.glob("job-agent-*.json"))]
    manifest = json.loads((campaign / "manifest.json").read_text())
    before = json.loads((campaign / "metrics-before.json").read_text()) if (campaign / "metrics-before.json").exists() else {}
    after = json.loads((campaign / "metrics-after.json").read_text()) if (campaign / "metrics-after.json").exists() else {}
    summary = campaign_summary(jobs, manifest["items"], 0.0, metrics_delta(before, after), args)
    relay.write_json(campaign / "SUMMARY.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_task", "metrics_delta")}, indent=1))
    return exit_code(summary)


if __name__ == "__main__":
    raise SystemExit(main())
