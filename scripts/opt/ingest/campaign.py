"""Eight-task / lost4 / single-task agentic campaign directories (SUMMARY.json + job-*.json).

A campaign is one engine boot per attempt running tasks sequentially, so wall_seconds is
the sum of the tasks' verified walls; that identity is checked and a mismatch marks the
campaign invalid instead of letting a half-written summary through.
"""
import glob
import json
import os
import re
from pathlib import Path

from .. import configs
from ..store import Reject, sha256_file
from .base import Batch, load_json, measurement, need, pair_name, verify_sha_sidecar

EIGHT_TASKS = {"rust/doubly-linked-list", "rust/book-store", "rust/react", "rust/alphametics", "rust/forth",
               "rust/xorcism", "rust/nucleotide-codons", "javascript/complex-numbers"}
RUNAWAY_TOKENS = 32000


def workload_id(tasks, agents=1):
    s = set(tasks)
    if s == EIGHT_TASKS:
        return "codex-8task-1agent"
    return f"codex-{len(s)}task-{agents}agent" + ("" if len(s) > 1 else ":" + next(iter(s)))


def _sum(jobs, key):
    vals = [j["probe"].get("metrics_delta", {}).get(key) for j in jobs]
    return sum(v for v in vals if v is not None) if any(v is not None for v in vals) else None


def _launch_for(cdir, job):
    if not job.get("attempt"):
        return None
    p = Path(cdir) / job["attempt"] / "launch.json"
    return load_json(p) if p.exists() else None


def campaign_config(cdir, manifest, jobs):
    """Config from the first job's recorded launch; returns (config_record, notes)."""
    notes = []
    if manifest.get("engine") == "vllm" or "launch" in manifest:
        la = manifest["launch"]
        rec = configs.from_command(la["command"], la.get("env"), model_path=manifest["model"]["local"],
                                   template_sha256=la.get("chat_template_sha256"), vllm_version=la.get("vllm_version"),
                                   stack=configs.quant_label(manifest["model"]["local"]))
        return rec, notes
    launch = None
    for j in jobs:
        launch = _launch_for(cdir, j)
        if launch:
            break
    if not launch:
        raise Reject(f"{cdir}: no attempts/*/launch.json; cannot identify the engine config")
    binary = manifest.get("binary")
    pins = manifest.get("binary_pins", {})
    sha = manifest.get("engine_commit")
    tmpl = (launch.get("chat_template_override") or {}).get("effective_sha256")
    rec = configs.from_command(launch["command"], launch.get("env"),
                               build_sha=sha, binary_sha256=pins.get(binary), model_sha256=launch.get("model_sha256"),
                               template_sha256=tmpl, stack=configs.build_family(binary))
    for j in jobs[1:]:
        other = _launch_for(cdir, j)
        if other and [c for c in other["command"] if "/attempts/" not in c] != [c for c in launch["command"] if "/attempts/" not in c]:
            notes.append(f"attempt launch differs for {j['job']}")
    return rec, notes


def mine_calls(cdir, jobs):
    """Per-task call tables from attempts/*/model-calls.jsonl, or None if any job lacks one."""
    tasks = {}
    for j in jobs:
        if not j.get("attempt"):
            return None
        p = Path(cdir) / j["attempt"] / "model-calls.jsonl"
        if not p.exists():
            return None
        rows = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise Reject(f"{p}: corrupt model-calls row ({e})")
        rows = [r for r in rows if r.get("status") == 200 and "input_tokens" in r and "finished" in r]
        rows.sort(key=lambda r: r["started"])
        calls, prev = [], None
        for r in rows:
            cached = r.get("cached_tokens") or 0
            calls.append([r["input_tokens"], r["input_tokens"] - cached, r["output_tokens"], 1 if cached else 0,
                          round(r["started"] - prev, 3) if prev is not None else 0.0,
                          round(r["finished"] - r["started"], 3)])
            prev = r["finished"]
        tasks[j["task"]] = calls
    return tasks


def parse(path):
    cdir = Path(path)
    if not (cdir / "SUMMARY.json").exists():
        raise Reject(f"{cdir}: no SUMMARY.json (aborted/partial campaign)")
    summ = load_json(cdir / "SUMMARY.json")
    need(summ, ["state", "tasks", "solved", "wall_seconds", "model"], f"{cdir}/SUMMARY.json")
    summary_sha = sha256_file(cdir / "SUMMARY.json")
    mf = cdir / "manifest.json"
    manifest = load_json(mf)
    verify_sha_sidecar(mf)
    jobs = []
    for jf in sorted(glob.glob(str(cdir / "job-*.json"))):
        verify_sha_sidecar(jf)
        j = load_json(jf)
        need(j, ["task", "status"], jf)
        if "probe" not in j:
            if j["status"] == "passed":
                raise Reject(f"{jf}: passed job without a probe record")
            # failed/skipped before the task ran (e.g. no safe model fit): kept as a not-run task, never a pass
            j = {**j, "probe": {}, "_not_run": True, "job": j.get("job") or Path(jf).stem.removeprefix("job-"),
                 "attempt": j.get("attempt")}
        need(j, ["job", "probe", "attempt"], jf)
        jobs.append(j)
    if not jobs:
        raise Reject(f"{cdir}: SUMMARY.json present but no job-*.json")

    b = Batch()
    cfg, notes = campaign_config(cdir, manifest, jobs)
    cid = b.add_config(cfg)
    for n in notes:
        b.flag(cdir, n)
    tasks = [j["task"] for j in jobs]
    wl = workload_id(tasks)
    reasons = []
    if summ["state"] != "complete":
        reasons.append(f"campaign state={summ['state']}")
    if len(jobs) != summ["tasks"]:
        reasons.append(f"{len(jobs)} job files for {summ['tasks']} tasks")
    not_run = [j for j in jobs if j.get("_not_run")]
    if not_run:
        reasons.append(f"{len(not_run)} task(s) did not run: " + "; ".join(
            f"{j['task']} ({j.get('reason') or j['status']})" for j in not_run))
    walls = [j["probe"].get("verified_wall_seconds") for j in jobs if not j.get("_not_run")]
    if None in walls:
        reasons.append("job without verified_wall_seconds")
    elif abs(sum(walls) - summ["wall_seconds"]) > 0.01 * summ["wall_seconds"] + 1:
        reasons.append(f"sum(task wall)={sum(walls):.0f}s != SUMMARY wall={summ['wall_seconds']:.0f}s")
    if sum(j["status"] == "passed" for j in jobs) != summ["solved"]:
        reasons.append("SUMMARY.solved disagrees with job statuses")
    if any(not j["probe"].get("harness_verified", True) or j["probe"].get("timed_out") for j in jobs):
        reasons.append("harness_verified false or timed_out task")
    status, reason = ("invalid", "; ".join(reasons)) if reasons else ("measured", None)
    if reasons:
        b.flag(cdir, "invalid: " + reason)

    name = f"{cdir.name}@{summary_sha[:8]}"  # same-named directories with different content must not share an id
    prov = {"out_dir": str(cdir), "summary_path": str(cdir / "SUMMARY.json"), "summary_sha256": summary_sha,
            "harness_sha": (manifest.get("sources") or {}).get("scripts/instruments/agentic_probe.py") or manifest.get("repo_head"),
            "build_sha": cfg.get("build_sha"), "model_sha256": cfg.get("model_sha256")}
    g0 = jobs[0].get("gpu_before") or []
    host = {"power_limit_w": [g.get("power_limit") for g in g0], "gpus": len(g0)} if g0 else {}
    common = dict(source="campaign", config_id=cid, workload_id=wl, prov=prov, status=status, reason=reason,
                  ts=summ.get("updated_epoch"), host=host, item_id=name, pair_group=pair_name(name))

    def add(metric, value, unit, **kw):
        if value is not None:
            b.measurements.append(measurement(metric=metric, value=value, unit=unit, **{**common, **kw}))

    n_tasks = summ["tasks"]
    add("wall_s", summ["wall_seconds"], "s")
    add("solve_rate", summ["solved"] / n_tasks if n_tasks else None, "ratio", n=n_tasks)
    add("output_tokens", summ.get("output_tokens"), "tokens")
    add("timeouts", summ.get("timeouts"), "count")
    add("draft_acceptance", summ.get("draft_acceptance"), "ratio")
    add("peak_context_tokens", summ.get("peak_context_tokens"), "tokens")
    pf_t, pf_s, dc_t, dc_s = (_sum(jobs, k) for k in ("prefill_tokens", "prefill_seconds", "decode_tokens", "decode_seconds"))
    add("prefill_tok_s", pf_t / pf_s if pf_t and pf_s else None, "tok/s", n=len(jobs))
    add("decode_tok_s", dc_t / dc_s if dc_t and dc_s else None, "tok/s", n=len(jobs))
    add("prefill_new_tokens", pf_t, "tokens")
    add("prefill_s", pf_s, "s")
    add("decode_s", dc_s, "s")
    add("cached_prompt_tokens", _sum(jobs, "cached_prompt_tokens"), "tokens")
    other = [j["probe"].get("other_seconds") for j in jobs]
    add("other_s", sum(o for o in other if o is not None) if any(o is not None for o in other) else None, "s")
    calls_total = sum((j["probe"].get("agent_stats") or {}).get("tool_calls", 0) for j in jobs)
    add("tool_calls", calls_total or None, "count")
    for j in jobs:
        p = j["probe"]
        d = {"task": j["task"]}
        add("task_wall_s", p.get("verified_wall_seconds"), "s", dims=d, attempt=j["attempt"])
        add("task_output_tokens", (p.get("agent_stats") or {}).get("completion_tokens"), "tokens", dims=d, attempt=j["attempt"])
        fail = {} if j["status"] == "passed" else {"reason": "; ".join(x for x in (
            f"task {j['status']}", j.get("reason"), j.get("fit_reason"), common["reason"]) if x)}
        add("task_solved", 1 if j["status"] == "passed" else 0, "bool",
            dims={**d, "ran": not j.get("_not_run"), "outcome": j["status"]}, attempt=j["attempt"], **fail)
        add("task_peak_context_tokens", p.get("peak_context_tokens"), "tokens", dims=d, attempt=j["attempt"])
    mins, seen = {}, {}
    devs = manifest.get("devices")
    for j in jobs:
        for dev, v in (j.get("minimum_free_mib") or {}).items():
            idx = devs.index(dev) if devs and dev in devs else seen.setdefault(dev, len(seen))
            mins[idx] = min(mins.get(idx, v), v)
    for idx, v in sorted(mins.items()):
        add("vram_min_free_mib", v, "MiB", dims={"gpu": idx, "np": cfg.get("np"), "ctx": cfg.get("ctx")})
    fit = cdir / "fit.json"
    if fit.exists():
        f = load_json(fit)
        need(f, ["kv_pool_tokens"], str(fit))
        add("kv_pool_tokens", f["kv_pool_tokens"], "tokens")
    calls = mine_calls(cdir, jobs)
    if calls is not None:
        mx = [max((c[2] for c in cl), default=0) for cl in calls.values()]
        add("runaway_rate", sum(m >= RUNAWAY_TOKENS for m in mx) / len(mx), "ratio", n=len(mx))
        add("model_calls", sum(len(cl) for cl in calls.values()), "count")
        b.add_workload({"id": f"calls:{summary_sha[:16]}", "kind": "campaign-calls", "campaign": name, "config_id": cid,
                        "summary_sha256": summary_sha, "workload_id": wl, "valid": status == "measured", "out_dir": str(cdir),
                        "fields": ["depth", "new_tokens", "out_tokens", "prefix_kept", "gap_s", "dur_s"],
                        "tasks": calls, "wall_s": summ["wall_seconds"]})
    b.add_workload({"id": wl, "kind": "task-set", "tasks": sorted(set(tasks)), "agents": 1})
    return b
