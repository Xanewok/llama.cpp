#!/usr/bin/env python3
"""DO NOT MERGE. Host intervals are not GPU idle; use NVTX/CUPTI for kernel gaps."""
import argparse
import bisect
import collections
import json
import re
import sqlite3
import statistics
from pathlib import Path

KINDS = {"graph_hit", "graph_miss", "llama_build", "meta_build", "cuda_launch",
         "cuda_instantiate", "cuda_capture", "ar_call", "ar_wait", "input",
         "readback", "sampling", "sync", "input_api", "cuda_submit"}
# Nested sampling and graph-build scopes must not be summed twice.
PRIORITY = ("ar_wait", "inputs", "readback", "sync", "graph_builds", "sampling", "cuda_api")
CLASS = {"input": "inputs", "readback": "readback", "ar_wait": "ar_wait", "sync": "sync",
         "llama_build": "graph_builds", "meta_build": "graph_builds",
         "cuda_instantiate": "graph_builds", "cuda_capture": "graph_builds", "sampling": "sampling",
         "cuda_launch": "cuda_api"}


def union_ns(intervals):
    total = end = 0
    for a, b in sorted(intervals):
        if b < a:
            raise ValueError("reversed interval")
        total += max(0, b - max(end, a))
        end = max(end, b)
    return total


def partition(a, b, events):
    if b < a:
        raise ValueError("reversed window")
    clipped = [(max(a, e[0]), min(b, e[1]), e[2]) for e in events if e[0] < b and e[1] > a]
    points = sorted({a, b, *(x for e in clipped for x in e[:2])})
    result = collections.Counter()
    for x, y in zip(points, points[1:]):
        active = {c for s, t, c in clipped if s < y and t > x}
        bucket = next((c for c in PRIORITY if c in active), "host_unclassified")
        result[bucket] += y - x
    if sum(result.values()) != b - a:
        raise ValueError("accounting does not cover window")
    return {c: result[c] / 1000 for c in (*PRIORITY, "host_unclassified")}


def validate(rows):
    previous = {}
    for row in rows:
        if row["schema"] != 1 or not row["phases"]:
            raise ValueError("invalid schema or empty round")
        key = row["pid"], row["tid"]
        phases = row["phases"]
        if phases[0]["phase"] != "V":
            raise ValueError("round does not begin with V")
        if key in previous and row["round"] <= previous[key]["round"]:
            raise ValueError("duplicate or reversed round")
        previous[key] = row
        last = 0
        for p in phases:
            if not re.fullmatch(r"V|C|D[1-9][0-9]*", p["phase"]):
                raise ValueError("invalid phase")
            if p["begin_ns"] < last or p["end_ns"] < p["begin_ns"]:
                raise ValueError("reversed or overlapping decodes")
            if p["result"] != 0 or p["n_tokens"] < 1:
                raise ValueError("failed or empty decode")
            last = p["end_ns"]
        for e in row["events"]:
            if e["kind"] not in KINDS or not 0 <= e["phase"] < len(phases):
                raise ValueError("unknown event or phase")
            a, b = e["begin_ns"], e["end_ns"]
            if a < phases[e["phase"]]["begin_ns"] or b < a:
                raise ValueError("invalid event interval")
            if e["phase"] + 1 < len(phases) and b > phases[e["phase"] + 1]["begin_ns"]:
                raise ValueError("event crosses phase ownership")
            if row["next_begin_ns"] and b > row["next_begin_ns"]:
                raise ValueError("event crosses round boundary")
            if any(type(e[c]) is not int or e[c] < 0 for c in ("count", "bytes")):
                raise ValueError("invalid counter")
        for i, p in enumerate(phases):
            expected = collections.defaultdict(list)
            for e in row["events"]:
                if e["phase"] == i:
                    expected[e["kind"], e["device"]].append(e)
            actual = {(c["kind"], c["device"]): c for c in p["counters"]}
            if set(actual) != set(expected) or len(actual) != len(p["counters"]):
                raise ValueError("missing or duplicate phase counters")
            for key, es in expected.items():
                c = actual[key]
                if c["count"] != sum(e["count"] for e in es) or c["bytes"] != sum(e["bytes"] for e in es):
                    raise ValueError("counter does not match events")
                duration = union_ns([(e["begin_ns"], e["end_ns"]) for e in es]) / 1000
                if abs(c["host_us"] - duration) > 0.001:
                    raise ValueError("duration does not match interval union")
    return rows


def nsys(path, rows):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    strings = {r["id"]: r["value"] for r in db.execute("SELECT id,value FROM StringIds")}
    markers = {}
    for r in db.execute("SELECT * FROM NVTX_EVENTS"):
        label = r["text"] if "text" in r.keys() else None
        if not label and "textId" in r.keys():
            label = strings.get(r["textId"], "")
        m = re.fullmatch(r"round_trace:(\d+):(\d+):(V|C|D\d+):(\d+)", label or "")
        if not m:
            continue
        tid, rnd, phase, index = m.groups()
        pid = (r["globalTid"] >> 24) & 0xFFFFFF
        key = pid, int(tid), int(rnd), int(index)
        if (r["globalTid"] & 0xFFFFFF) != int(tid) or key in markers or r["end"] is None:
            raise ValueError("NVTX identity collision or incomplete range")
        markers[key] = (r["start"], r["end"], phase, r["globalTid"])
    phase_refs = {}
    offsets = collections.defaultdict(list)
    for row in rows:
        for i, p in enumerate(row["phases"]):
            key = row["pid"], row["tid"], row["round"], i
            if key in markers:
                mark = markers[key]
                if mark[2] != p["phase"]:
                    raise ValueError("NVTX phase disagrees with JSONL")
                offsets[row["pid"], row["tid"]].append(mark[0] - p["begin_ns"])
                phase_refs[key] = mark
    if not phase_refs:
        raise ValueError("no matching JSONL/NVTX phases")
    offset = {key: statistics.median(values) for key, values in offsets.items()}
    for values in offsets.values():
        if max(values) - min(values) > 100_000:
            raise ValueError("NVTX clock alignment spread exceeds diagnostic bound")
    by_tid = collections.defaultdict(list)
    for key, mark in phase_refs.items():
        by_tid[mark[3]].append((mark[0], mark[1], key))
    for ranges in by_tid.values():
        ranges.sort()
    starts = {tid: [r[0] for r in ranges] for tid, ranges in by_tid.items()}
    correlation = {}
    runtime = collections.defaultdict(list)
    for r in db.execute("SELECT start,end,nameId,correlationId,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        tid = r["globalTid"]
        runtime[((tid >> 24) & 0xFFFFFF, tid & 0xFFFFFF)].append((r["start"], r["end"], "cuda_api"))
        ranges = by_tid.get(tid, [])
        index = bisect.bisect_right(starts.get(tid, []), r["start"]) - 1
        if index >= 0 and r["end"] <= ranges[index][1]:
            key = (tid >> 24) << 24, r["correlationId"]
            if key in correlation:
                raise ValueError("duplicate process-scoped correlation")
            correlation[key] = ranges[index][2]
    kernels = collections.defaultdict(list)
    busy = collections.defaultdict(list)
    unmatched = collections.Counter()
    for table in ("KERNEL", "MEMCPY", "MEMSET"):
        name = "CUPTI_ACTIVITY_KIND_" + table
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone():
            continue
        for r in db.execute(f"SELECT start,end,deviceId,correlationId,globalPid FROM {name}"):
            pid = (r["globalPid"] >> 24) & 0xFFFFFF
            busy[pid].append((r["start"], r["end"]))
            if table == "KERNEL":
                key = correlation.get((r["globalPid"], r["correlationId"]))
                if key:
                    kernels[key].append((r["start"], r["end"], r["deviceId"]))
                elif any(k[0] == pid for k in phase_refs):
                    unmatched[pid] += 1
    db.close()
    return {"offset": offset, "kernels": kernels, "runtime": runtime, "busy": busy,
            "unmatched_kernels": dict(unmatched), "matched_phases": len(phase_refs)}


def wait_histograms(rows):
    waits = collections.defaultdict(list)
    for row in rows:
        for e in row["events"]:
            if e["kind"] == "ar_wait":
                if e["count"] != 1:
                    raise ValueError("AR wait histogram requires one event per API call")
                phase = row["phases"][e["phase"]]["phase"]
                waits[phase, e["device"]].append((e["end_ns"] - e["begin_ns"]) / 1000)
    result = []
    for (phase, device), values in sorted(waits.items()):
        bins = [0] * 5
        for value in values:
            bins[bisect.bisect_right((1, 10, 100, 1000), value)] += 1
        result.append({"phase": phase, "device": device, "waits_M": len(values),
                       "waits_per_round_D": len(values) / len(rows),
                       "total_wait_us_D": sum(values), "mean_wait_us_D": statistics.mean(values),
                       "p50_wait_us_D": statistics.median(values),
                       "p95_wait_us_D": sorted(values)[max(0, (95 * len(values) + 99) // 100 - 1)],
                       "max_wait_us_M": max(values), "bins_M": bins})
    return {"bins_us": ["[0,1)", "[1,10)", "[10,100)", "[100,1000)", "[1000,inf)"],
            "by_phase_device": result}


def analyze(rows, k=3, trim=0.03, skip=3, min_rounds=0, sqlite=None):
    validate(rows)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["pid"], r["tid"]].append(r)
    ns = nsys(sqlite, rows) if sqlite else None
    candidates = []
    expected = ["V", "C", *(f"D{i}" for i in range(1, k + 1))]
    for stream in groups.values():
        for r, nxt in zip(stream, stream[1:]):
            if [p["phase"] for p in r["phases"]] != expected or r["phases"][0]["n_tokens"] != k + 1:
                continue
            duration = nxt["phases"][0]["begin_ns"] - r["phases"][0]["begin_ns"]
            if duration <= 0 or duration > 1_000_000_000:
                continue
            candidates.append((r, nxt, duration))
    candidates = candidates[skip:]
    median = statistics.median(c[2] for c in candidates) if candidates else 0
    kept = [c for c in candidates if abs(c[2] - median) <= median * trim]
    if len(kept) < min_rounds:
        raise ValueError(f"steady rounds {len(kept)} below required {min_rounds}")
    samples = collections.defaultdict(list)
    counters = collections.defaultdict(lambda: collections.Counter())
    gpu_used = 0
    for r, nxt, duration in kept:
        phases = r["phases"]
        events = [(e["begin_ns"], e["end_ns"], CLASS.get(e["kind"], "host_unclassified")) for e in r["events"]]
        for i, p in enumerate(phases):
            for c in p["counters"]:
                counters[p["phase"], c["kind"], c["device"]].update(count=c["count"], bytes=c["bytes"], host_us=c["host_us"])
            following = phases[i + 1] if i + 1 < len(phases) else nxt["phases"][0]
            label = p["phase"] + "->" + following["phase"]
            for window, a, b in (("decode_call", p["begin_ns"], p["end_ns"]),
                                 ("host_boundary", p["end_ns"], following["begin_ns"])):
                parts = partition(a, b, events)
                parts["total_us"] = (b - a) / 1000
                samples[window, p["phase"] if window == "decode_call" else label].append(parts)
            if ns:
                key = r["pid"], r["tid"], r["round"], i
                key_next = (r["pid"], r["tid"], r["round"], i + 1) if i + 1 < len(phases) else (nxt["pid"], nxt["tid"], nxt["round"], 0)
                first, second = ns["kernels"].get(key, []), ns["kernels"].get(key_next, [])
                if {e[2] for e in first} != {0, 1} or {e[2] for e in second} != {0, 1}:
                    continue
                a, b = max(e[1] for e in first), min(e[0] for e in second)
                if b < a:
                    continue
                shift = ns["offset"][r["pid"], r["tid"]]
                next_events = [(e["begin_ns"], e["end_ns"], CLASS.get(e["kind"], "host_unclassified")) for e in nxt["events"]]
                aligned = [(s + shift, t + shift, c) for s, t, c in events + next_events]
                aligned += ns["runtime"].get((r["pid"], r["tid"]), [])
                parts = partition(a, b, aligned)
                parts["total_us"] = (b - a) / 1000
                active = union_ns([(max(a, s), min(b, t)) for s, t in ns["busy"][r["pid"]] if s < b and t > a])
                parts["device_idle_us"] = (b - a - active) / 1000
                samples["gpu_kernel_boundary", label].append(parts)
                gpu_used += 1
    tables = [{"window": window, "phase": phase, "samples_M": len(values),
               **{name: statistics.mean(v[name] for v in values) for name in values[0]}}
              for (window, phase), values in sorted(samples.items())]
    return {"labels": "counts/durations M; partition/idle/averages D; no speed claim",
            "rounds_M": len(kept), "candidate_rounds_M": len(candidates), "median_round_us_M": median / 1000,
            "tables": tables, "counters": [{"phase": key[0], "kind": key[1], "device": key[2], **value}
                                             for key, value in sorted(counters.items())],
            "ar_wait_histogram": wait_histograms([r for r, _, _ in kept]),
            "nsys": {"matched_phases_M": ns["matched_phases"], "gap_samples_M": gpu_used,
                     "unmatched_kernels_M": ns["unmatched_kernels"]} if ns else None,
            "limits": "Host residual includes scheduler stalls and uninstrumented work. GPU kernel boundaries include copy activity; device_idle subtracts its union. Without SQLite, GPU gaps are unknown."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", type=Path)
    p.add_argument("--sqlite", type=Path)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--skip", type=int, default=3)
    p.add_argument("--trim", type=float, default=0.03)
    p.add_argument("--min-rounds", type=int, default=0)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    result = analyze(rows, args.k, args.trim, args.skip, args.min_rounds, args.sqlite)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result["labels"])
        print(f"M retained rounds: {result['rounds_M']}; M median round: {result['median_round_us_M']:.3f} us")
        print("window\tphase\tn_M\ttotal_us_D\thost_unclassified_us_D\tinputs_us_D\treadback_us_D\tar_wait_us_D\tgraph_builds_us_D\tsampling_us_D\tsync_us_D\tcuda_api_us_D")
        for r in result["tables"]:
            names = ("total_us", "host_unclassified", "inputs", "readback", "ar_wait", "graph_builds", "sampling", "sync", "cuda_api")
            print(f"{r['window']}\t{r['phase']}\t{r['samples_M']}\t" + "\t".join(f"{r[n]:.3f}" for n in names))
        print(result["limits"])


if __name__ == "__main__":
    main()
