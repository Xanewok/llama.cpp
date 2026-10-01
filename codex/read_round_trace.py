#!/usr/bin/env python3
"""Offline full-round activity unions and idle-only host attribution."""
import argparse
import bisect
import collections
import json
from pathlib import Path
import sqlite3
import statistics

from attrib import CLASS, PRIORITY, nsys, partition, validate


def merge(intervals):
    result = []
    for a, b in sorted(intervals):
        if b < a:
            raise ValueError("reversed activity")
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


class Activity:
    def __init__(self, intervals):
        self.intervals = merge(intervals)
        self.ends = [b for _, b in self.intervals]

    def clipped(self, a, b):
        result = []
        for index in range(bisect.bisect_right(self.ends, a), len(self.intervals)):
            s, t = self.intervals[index]
            if s >= b:
                break
            result.append((max(a, s), min(b, t)))
        return result

    def busy(self, a, b):
        return sum(t - s for s, t in self.clipped(a, b))

    def idle(self, a, b):
        cursor = a
        for s, t in self.clipped(a, b):
            if s > cursor:
                yield cursor, s
            cursor = t
        if cursor < b:
            yield cursor, b


def retained(rows):
    validate(rows)
    candidates = []
    for r, nxt in zip(rows, rows[1:]):
        if (r["pid"], r["tid"]) != (nxt["pid"], nxt["tid"]):
            raise ValueError("mixed submission threads")
        if [p["phase"] for p in r["phases"]] != ["V", "C", "D1", "D2", "D3"] or r["phases"][0]["n_tokens"] != 4:
            continue
        duration = nxt["phases"][0]["begin_ns"] - r["phases"][0]["begin_ns"]
        if 0 < duration <= 1_000_000_000:
            candidates.append((r, nxt, duration))
    candidates = candidates[3:]
    median = statistics.median(c[2] for c in candidates)
    return [c for c in candidates if abs(c[2] - median) <= median * 0.03]


def analyze(path, case):
    rows = [json.loads(line) for line in (path / f"capture-{case}.jsonl").read_text().splitlines()]
    kept = retained(rows)
    ns = nsys(path / "capture.sqlite", rows)
    host_retained = len(kept)
    kernel_periods = []
    for r, nxt, _ in kept:
        first = ns["kernels"][r["pid"], r["tid"], r["round"], 0]
        following = ns["kernels"][nxt["pid"], nxt["tid"], nxt["round"], 0]
        kernel_periods.append(min(v[0] for v in following) - min(v[0] for v in first))
    kernel_median = statistics.median(kernel_periods)
    excluded = [{"round": r["round"], "next_round": nxt["round"], "kernel_period_ms_M": period / 1e6}
                for (r, nxt, _), period in zip(kept, kernel_periods) if abs(period - kernel_median) > kernel_median * 0.03]
    kept = [value for value, period in zip(kept, kernel_periods) if abs(period - kernel_median) <= kernel_median * 0.03]
    pid = kept[0][0]["pid"]
    db = sqlite3.connect((path / "capture.sqlite").resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    activity = collections.defaultdict(list)
    ar_kernels = collections.defaultdict(list)
    all_kernels = []
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    for table in ("KERNEL", "MEMCPY", "MEMSET"):
        name = "CUPTI_ACTIVITY_KIND_" + table
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone():
            continue
        columns = "start,end,deviceId,globalPid" + (",demangledName" if table == "KERNEL" else "")
        for row in db.execute(f"SELECT {columns} FROM {name}"):
            a, b, device, global_pid = row[:4]
            if (global_pid >> 24) & 0xFFFFFF != pid:
                continue
            activity[device].append((a, b))
            if table == "KERNEL":
                all_kernels.append((a, b, device))
                if "ggml_cuda_ar_kernel<" in strings[row[4]]:
                    ar_kernels[device].append((a, b))
    db.close()
    paired = Activity([v for values in activity.values() for v in values])
    devices = {d: Activity(v) for d, v in activity.items()}
    ar = {d: Activity(v) for d, v in ar_kernels.items()}
    kernel_starts = sorted(a for a, _, _ in all_kernels)
    runtime = sorted(ns["runtime"][pid, kept[0][0]["tid"]])
    runtime_starts = [r[0] for r in runtime]
    samples = collections.defaultdict(list)
    rounds = []
    for r, nxt, duration in kept:
        shift = ns["offset"][r["pid"], r["tid"]]
        events = [(e["begin_ns"] + shift, e["end_ns"] + shift, CLASS.get(e["kind"], "host_unclassified"))
                  for row in (r, nxt) for e in row["events"]]
        start = bisect.bisect_left(runtime_starts, r["phases"][0]["begin_ns"] + shift)
        end = bisect.bisect_right(runtime_starts, nxt["phases"][0]["end_ns"] + shift)
        events += runtime[start:end]
        # The tracer timestamps next_begin before synchronous JSON serialization.
        flush_a = r["next_begin_ns"] + shift
        flush_b = nxt["phases"][0]["begin_ns"] + shift
        events.append((flush_a, flush_b, "host_unclassified"))

        def window(kind, label, a, b):
            if b < a:
                raise ValueError("overlapping phase envelopes")
            idle_parts = collections.Counter({c: 0.0 for c in (*PRIORITY, "host_unclassified")})
            idle_ns = 0
            for s, t in paired.idle(a, b):
                idle_ns += t - s
                idle_parts.update(partition(s, t, events))
            busy_ns = paired.busy(a, b)
            assert busy_ns + idle_ns == b - a
            assert abs(sum(idle_parts.values()) * 1000 - idle_ns) < 0.1
            flush_idle_ns = sum(t - s for s, t in paired.idle(max(a, flush_a), min(b, flush_b))) if max(a, flush_a) < min(b, flush_b) else 0
            value = {"span_ms_D": (b - a) / 1e6, "any_gpu_active_ms_D": busy_ns / 1e6,
                     "both_idle_ms_D": idle_ns / 1e6, "idle_host_ms_D": {k: v / 1000 for k, v in idle_parts.items()},
                     "trace_flush_idle_ms_D": flush_idle_ns / 1e6,
                     "device_active_ms_D": {str(d): v.busy(a, b) / 1e6 for d, v in devices.items()},
                     "ar_kernel_active_ms_D": {str(d): v.busy(a, b) / 1e6 for d, v in ar.items()}}
            samples[kind, label].append(value)
            return value

        envelopes = []
        for i, phase in enumerate(r["phases"]):
            key = pid, r["tid"], r["round"], i
            kernels = ns["kernels"][key]
            if {v[2] for v in kernels} != {0, 1}:
                raise ValueError("missing device in phase")
            envelopes.append((min(v[0] for v in kernels), max(v[1] for v in kernels)))
        next_kernels = ns["kernels"][pid, nxt["tid"], nxt["round"], 0]
        next_start = min(v[0] for v in next_kernels)
        phase_values = []
        boundary_values = []
        for i, p in enumerate(r["phases"]):
            a, b = envelopes[i]
            phase_values.append(window("phase", p["phase"], a, b))
            end = envelopes[i + 1][0] if i + 1 < len(envelopes) else next_start
            label = p["phase"] + "->" + (r["phases"][i + 1]["phase"] if i + 1 < len(envelopes) else "V")
            boundary_values.append(window("boundary", label, b, end))
        total = window("round", "first_V_kernel->next_first_V_kernel", envelopes[0][0], next_start)
        assert abs(sum(v["span_ms_D"] for v in phase_values + boundary_values) - total["span_ms_D"]) < 1e-8
        # Check phase ownership against all executed kernels, including graph nodes.
        executed = bisect.bisect_left(kernel_starts, envelopes[-1][1]) - bisect.bisect_left(kernel_starts, envelopes[0][0])
        matched = sum(len(ns["kernels"][pid, r["tid"], r["round"], i]) for i in range(5))
        if executed != matched:
            raise ValueError(f"kernel ownership incomplete: {executed} executed, {matched} matched")
        rounds.append({"round": r["round"], "next_round": nxt["round"], "next_V_tokens": nxt["phases"][0]["n_tokens"],
                       "host_V_to_V_ms_M": duration / 1e6, "trace_flush_ms_M": (flush_b - flush_a) / 1e6,
                       "matched_kernels_M": matched, **total})
    def average(values):
        return {k: average([v[k] for v in values]) if isinstance(values[0][k], dict)
                else statistics.mean(v[k] for v in values) for k in values[0]}
    return {"case": case, "host_retained_rounds_M": host_retained, "rounds_M": len(kept),
            "median_kernel_period_ms_M": kernel_median / 1e6, "excluded_gpu_rounds_M": excluded, "mean_host_V_to_V_ms_D": statistics.mean(r["host_V_to_V_ms_M"] for r in rounds),
            "mean_trace_flush_ms_D": statistics.mean(r["trace_flush_ms_M"] for r in rounds),
            "next_V_token_counts_M": dict(collections.Counter(r["next_V_tokens"] for r in rounds)),
            "windows": [{"window": k[0], "phase": k[1], **average(v)} for k, v in samples.items()], "rounds": rounds}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps([analyze(args.capture, case) for case in ("think", "tool")], indent=2) + "\n")
