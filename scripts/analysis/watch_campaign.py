"""Read-only, change-only campaign status for a second tmux pane."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


WHY = {
    "baseline": "qualify normal-desktop headroom and deep-context decode",
    "split": "compare placement while retaining the 102400-token window",
    "microbatch": "measure prefill workspace at the selected placement",
    "batch": "isolate logical prefill batch size",
    "interaction": "recheck the two best splits with the two best microbatches",
    "layer": "matched layer-split control",
    "mtp": "compare maximum draft lengths by useful throughput",
    "probability": "compare probability floors at the selected draft length",
    "minimum": "isolate minimum draft length",
    "draft-cache": "screen experimental Q8 draft-cache capacity and kernels",
    "swarm": "screen simultaneous slots at the explicitly named per-agent context",
    "solve": "verify a coding task with the trusted grader",
}
EVENTS = {"attempt_start", "attempt_end", "transient_retry", "campaign_stop",
          "campaign_error", "job_end", "job_recovered", "verify_model", "download"}


class EventLog:
    def __init__(self, path):
        self.path, self.offset, self.identity = path, 0, None
        self.rows = []

    def read(self):
        try:
            with self.path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if identity != self.identity or stat.st_size < self.offset:
                    self.rows, self.offset, self.identity = [], 0, identity
                stream.seek(self.offset)
                while line := stream.readline():
                    if not line.endswith(b"\n"):
                        break
                    self.offset = stream.tell()
                    try:
                        row = json.loads(line)
                        if row.get("kind") in EVENTS:
                            self.rows.append(row)
                    except (ValueError, AttributeError):
                        continue
        except FileNotFoundError:
            pass
        return self.rows


def job_for_event(event, queue):
    name = event.get("job", "")
    return next((job for job in queue if name in (job["id"], job["id"] + "-fit")), None)


def signature(job):
    config = job.get("configs", [{}])[0]
    return (tuple(job.get("targets", [])), config.get("context"), config.get("slots"),
            job.get("workload", "throughput"))


def estimate(queue, completed, events, now):
    starts, durations, groups = {}, [], {}
    for event in events:
        job = job_for_event(event, queue)
        if not job:
            continue
        name = job["id"]
        if event["kind"] == "attempt_start":
            starts.setdefault(name, event["time"])
        elif event["kind"] == "job_end" and name in starts and event.get("status") == "passed":
            duration = event["time"] - starts.pop(name)
            if duration > 0:
                durations.append(duration)
                groups.setdefault(signature(job), []).append(duration)
    remaining = [job for job in queue if job["id"] not in completed]
    if not remaining:
        return "ETA: complete"
    if not durations:
        return "ETA: unknown; no completed timed measurements"
    seconds, unmatched = 0, 0
    for job in remaining:
        samples = groups.get(signature(job))
        unmatched += not bool(samples)
        duration = statistics.median(samples or durations)
        if job["id"] in starts:
            elapsed = now - starts[job["id"]]
            if elapsed > duration:
                return "ETA: unknown; active job exceeds its measured estimate"
            duration -= elapsed
        seconds += duration
    end = datetime.fromtimestamp(now) + timedelta(seconds=seconds)
    return f"ETA: {end:%H:%M:%S} (~{seconds / 60:.0f} min)"


def owner_status(state):
    owner = state.get("coordinator", {})
    if state.get("status") != "running" or not owner:
        return None
    try:
        if owner.get("pid_namespace") != os.readlink("/proc/self/ns/pid"):
            return "coordinator liveness unknown across PID namespaces"
        fields = Path(f"/proc/{int(owner['pid'])}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[19] == owner.get("start") and fields[0] != "Z":
            return None
    except (OSError, KeyError, ValueError, IndexError):
        pass
    return "coordinator absent; saved running state is stale"


def resolve_campaign(campaign):
    """Follow a newer numbered preparation when the requested path is stale."""
    try:
        state = json.loads((campaign / "state.json").read_text())
    except (OSError, ValueError):
        return campaign
    if state.get("status") != "prepared":
        return campaign
    candidates = []
    prefix = campaign.name + "-"
    for sibling in campaign.parent.glob(prefix + "*"):
        if not sibling.is_dir():
            continue
        try:
            sibling_state = json.loads((sibling / "state.json").read_text())
        except (OSError, ValueError):
            continue
        if sibling_state.get("status") in {"running", "complete", "stopped", "error"}:
            candidates.append((sibling_state.get("status") == "running", sibling.stat().st_mtime, sibling))
    if not candidates:
        return campaign
    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return candidates[0][2]


def render(message, redraw=False, stream=None):
    stream = sys.stdout if stream is None else stream
    if redraw:
        stream.write("\x1b[2J\x1b[H")
    stream.write(message.rstrip() + "\n")
    stream.flush()


def snapshot(campaign, events, now=None, redraw=False, elapsed_since_update=None, display_time=None):
    now = time.time() if now is None else now
    display_time = now if display_time is None else display_time
    manifest = json.loads((campaign / "manifest.json").read_text())
    state = json.loads((campaign / "state.json").read_text())
    queue, completed = manifest["queue"], state.get("completed", {})
    active = state.get("active") or {}
    live = owner_status(state)
    latest = events[-1] if events else {}
    attempt = next((e for e in reversed(events) if e["kind"] == "attempt_start"
                    and active and job_for_event(e, [active])), {})
    config = attempt.get("config") or active.get("configs", [{}])[0]
    status = state["status"]
    # Time and the moving ETA never trigger a new line by themselves.
    key = json.dumps([status, active, completed, state.get("reason"), state.get("resource_wait"),
                      latest, live], sort_keys=True)
    counter = f"{len(completed)}/{len(queue)}"
    if redraw:
        counter = "\x1b[1m" + counter + "\x1b[22m"
    lines = [f"[{datetime.fromtimestamp(display_time):%H:%M:%S}] {campaign.name}: {status}; {counter} jobs"]
    if active and status == "running":
        name = active["id"]
        why = next((text for prefix, text in WHY.items() if name.startswith(prefix + "-")),
                   "execute the next arm in the frozen manifest")
        if "baseline-before" in name or "baseline-after" in name:
            why = "baseline bracket to detect desktop or thermal drift"
        lines.append(f"Arm: {name}. Why: {why}.")
        context, slots = config.get("context"), config.get("slots", 1)
        total = context * slots if context is not None else "unknown"
        lines.append(f"Model: {attempt.get('model', active.get('model', 'unknown'))}; "
                     f"context {context}/agent × {slots} slots = {total} total; "
                     f"{config.get('mode', 'tensor')} split {config.get('split', '69,31')}; "
                     f"batch/ub {config.get('batch', 2048)}/{config.get('ubatch', 128)}; "
                     f"MTP max/min/pmin {config.get('maximum', 3)}/{config.get('minimum', 0)}/{config.get('p_min', 0)}; "
                     f"draft KV {config.get('draft_k', 'f16')}/{config.get('draft_v', 'f16')}.")
        targets = active.get("targets")
        if isinstance(targets, list) and len(targets) == 1:
            targets = targets[0]
        lines.append(f"Task: {active.get('task', 'unknown')}; input target {targets}; "
                     f"seed {active.get('seed', 1729)}.")
    if latest:
        detail = latest.get("reason") or latest.get("status") or latest.get("model") or latest.get("job", "")
        lines.append(f"Update: {latest['kind']} {detail}".rstrip())
    if live or state.get("reason") or state.get("resource_wait"):
        lines.append(f"State detail: {live or state.get('reason') or state.get('resource_wait')}")
    if status == "complete":
        eta = "ETA: complete"
    elif status != "running" or live:
        eta = "ETA: unavailable while the campaign is not running"
    else:
        eta = estimate(queue, completed, events, now)
    if elapsed_since_update is not None:
        lines.append(f"Since update: {max(0, int(elapsed_since_update))}s | {eta}")
    else:
        lines.append(eta)
    return key, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--current-file", type=Path,
                        help="follow a dispatcher-published campaign path")
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--redraw", action="store_true",
                        help="clear the terminal and redraw the latest changed snapshot")
    parser.add_argument("--timer", action="store_true",
                        help="with --redraw, show seconds since the last state update")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("interval must be positive and finite")
    if args.timer and not args.redraw:
        parser.error("--timer requires --redraw")
    log, previous = EventLog(args.campaign / "events.jsonl"), None
    last_change = time.monotonic()
    last_change_wall = time.time()
    last_draw = 0.0
    if args.redraw:
        sys.stdout.write("\x1b[?25l")
        sys.stdout.flush()
    try:
        while True:
            requested_campaign = args.campaign
            if args.current_file:
                try:
                    published = args.current_file.read_text().strip()
                    if published:
                        requested_campaign = Path(published)
                except OSError:
                    pass
            effective_campaign = resolve_campaign(requested_campaign)
            now_monotonic = time.monotonic()
            try:
                if log.path != effective_campaign / "events.jsonl":
                    log = EventLog(effective_campaign / "events.jsonl")
                key, message = snapshot(
                    effective_campaign, log.read(), redraw=args.redraw,
                    elapsed_since_update=(now_monotonic - last_change if args.timer else None),
                    display_time=last_change_wall)
                key = f"{effective_campaign}\0{key}"
                if effective_campaign != requested_campaign:
                    message = f"Following active campaign: {effective_campaign}\n" + message
            except (OSError, ValueError, KeyError, TypeError) as error:
                key = message = f"Waiting for readable campaign state: {error}"
            changed = key != previous
            if changed:
                last_change = now_monotonic
                last_change_wall = time.time()
                if args.timer:
                    key, message = snapshot(effective_campaign, log.rows, redraw=args.redraw,
                                            elapsed_since_update=0, display_time=last_change_wall)
                    if effective_campaign != requested_campaign:
                        message = f"Following active campaign: {effective_campaign}\n" + message
                render(message, args.redraw)
                previous = key
                last_draw = now_monotonic
            elif args.timer and now_monotonic - last_draw >= 1:
                _, message = snapshot(effective_campaign, log.rows, now=time.time(), redraw=args.redraw,
                                       elapsed_since_update=now_monotonic - last_change,
                                       display_time=last_change_wall)
                if effective_campaign != requested_campaign:
                    message = f"Following active campaign: {effective_campaign}\n" + message
                render(message, args.redraw)
                last_draw = now_monotonic
            if args.once:
                return
            time.sleep(min(args.interval, 1) if args.timer else args.interval)
    except KeyboardInterrupt:
        return
    finally:
        if args.redraw:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
