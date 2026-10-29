"""Two-agent concurrent runs (two-agent-*-vN/measure): SUMMARY.json + manifest.json.

Only the `measure` leg is ingested; `canary` is a smoke gate, not a measurement.
Calls are stored per agent as one call table so wall_est can replay both streams.
"""
import csv
from pathlib import Path

from .. import configs
from ..store import Reject, sha256_file
from .base import Batch, load_json, measurement, need
from .campaign import EIGHT_TASKS, RUNAWAY_TOKENS


def _calls(agent):
    out, prev = [], None
    for c in sorted(agent["calls"], key=lambda r: r["started"]):
        if c.get("status") != 200 or c.get("event") != "response.completed":
            continue
        cached = c.get("cached_tokens") or 0
        out.append([c["input_tokens"], c["input_tokens"] - cached, c["output_tokens"], 1 if cached else 0,
                    round(c["started"] - prev, 3) if prev is not None else 0.0, round(c["finished"] - c["started"], 3)])
        prev = c["finished"]
    return out


def _runaway_count(v):
    if isinstance(v, bool) or v is None:
        return 0
    return v if isinstance(v, int) else len(v)


def _idle_used(path):
    """nvidia-smi `index, uuid, memory.used [MiB], memory.total [MiB]` taken before the server starts."""
    used = {}
    for i, row in enumerate(csv.reader(open(path))):
        if i == 0:
            continue
        if len(row) < 4:
            raise Reject(f"{path}:{i + 1}: malformed memory.csv row")
        try:
            used[int(row[0])] = float(row[2].split()[0])
        except (ValueError, IndexError):
            raise Reject(f"{path}:{i + 1}: non-numeric memory.used")
    return used


def parse(path):
    root = Path(path)
    mdir = root / "measure" if (root / "measure" / "SUMMARY.json").exists() else root
    sf = mdir / "SUMMARY.json"
    if not sf.exists():
        raise Reject(f"{mdir}: no SUMMARY.json (canary-only or failed two-agent run)")
    s = load_json(sf)
    need(s, ["valid", "verdict", "tasks", "solved", "wall_seconds", "agents", "metrics_delta"], str(sf))
    man = load_json(mdir / "manifest.json")
    need(man, ["engine", "launch", "tasks"], str(mdir / "manifest.json"))
    if not s["agents"] or any("calls" not in a for a in s["agents"]):
        raise Reject(f"{sf}: agents without call tables")
    if man.get("canary"):
        raise Reject(f"{mdir}: canary leg is not a measurement")
    b = Batch()
    cmd = man["launch"]["command"]
    binary = cmd[0] if man["engine"] == "llama" else None
    cfg = configs.from_command(cmd, man["launch"].get("env"), stack=configs.build_family(binary) or None)
    cid = b.add_config(cfg)
    reasons = []
    if not s["valid"] or s["verdict"] != "VALID":
        reasons.append(f"verdict={s['verdict']}")
    if s.get("gates"):
        reasons.append(f"gates={s['gates']}")
    if sum(a["tasks"] for a in s["agents"]) != s["tasks"] or s["solved"] > s["tasks"]:
        reasons.append("agent task counts disagree with SUMMARY.tasks")
    status, reason = ("invalid", "; ".join(reasons)) if reasons else ("measured", None)
    if reasons:
        b.flag(mdir, "invalid: " + reason)
    wl = "two-agent-8task" if set(man["tasks"]) == EIGHT_TASKS else f"two-agent-{len(man['tasks'])}task"
    sha = sha256_file(sf)
    name = f"{root.name if mdir != root else root.parent.name}@{sha[:8]}"
    prov = {"out_dir": str(mdir), "summary_path": str(sf), "summary_sha256": sha,
            "harness_sha": man.get("runner_sha256"), "build_sha": None, "model_sha256": None}
    common = dict(source="two-agent", config_id=cid, workload_id=wl, prov=prov, status=status, reason=reason,
                  item_id=name)

    def add(metric, value, unit, **kw):
        if value is not None:
            b.measurements.append(measurement(metric=metric, value=value, unit=unit, **{**common, **kw}))

    md = s["metrics_delta"]
    add("wall_s", s["wall_seconds"], "s")
    add("solve_rate", s["solved"] / s["tasks"], "ratio", n=s["tasks"])
    add("output_tokens", s.get("output_tokens"), "tokens")
    add("agg_tok_s", s.get("aggregate_delivery_tps"), "tok/s", dims={"streams": 2, "scope": "client-delivery"})
    add("decode_tok_s", md.get("decode_tps"), "tok/s", dims={"scope": "per-stream-server"})
    add("prefill_tok_s", md.get("prefill_tps"), "tok/s")
    add("draft_acceptance", s.get("draft_acceptance"), "ratio")
    add("agent_overlap_s", s.get("agent_overlap_seconds"), "s")
    add("prefill_new_tokens", md.get("prefill_tokens"), "tokens")
    add("prefill_s", md.get("prefill_seconds"), "s")
    add("decode_s", md.get("decode_seconds"), "s")
    for a in s["agents"]:
        add("agent_wall_s", a.get("wall_seconds"), "s", dims={"agent": a["agent"]})
        add("agent_peak_context_tokens", a.get("peak_context_tokens"), "tokens", dims={"agent": a["agent"]})
    runaway = sum(_runaway_count(a.get("runaway_calls")) for a in s["agents"])
    add("runaway_calls", runaway, "count")
    jobs = s.get("jobs") or []
    if jobs:
        hit = sum(_runaway_count(j.get("runaway_calls")) > 0 or any((c.get("output_tokens") or 0) >= RUNAWAY_TOKENS
                                                                      for c in (j.get("calls") if isinstance(j.get("calls"), list) else []) if isinstance(c, dict))
                  for j in jobs)
        add("runaway_rate", hit / len(jobs), "ratio", n=len(jobs))
    mem = (root if mdir != root else root.parent) / "memory.csv"
    if mem.exists():
        for gpu, used in _idle_used(mem).items():
            add("gpu_idle_used_mib", used, "MiB", dims={"gpu": gpu, "scope": "preflight snapshot, not minimum free"})
    base = (s.get("one_agent_comparison") or {}).get("summary") or {}
    add("one_agent_baseline_wall_s", base.get("wall_seconds"), "s", status="derived",
        reason=f"baseline campaign {(s.get('one_agent_comparison') or {}).get('path')}")
    b.add_workload({"id": f"calls:{sha[:16]}", "kind": "two-agent-calls", "campaign": name, "config_id": cid,
                    "summary_sha256": sha, "workload_id": wl, "valid": status == "measured", "out_dir": str(mdir),
                    "fields": ["depth", "new_tokens", "out_tokens", "prefix_kept", "gap_s", "dur_s"],
                    "agents": {str(a["agent"]): _calls(a) for a in s["agents"]}, "wall_s": s["wall_seconds"]})
    b.add_workload({"id": wl, "kind": "task-set", "tasks": sorted(man["tasks"]), "agents": 2})
    return b
