#!/usr/bin/env python3
"""Per-campaign metrics and an arm comparison that keeps every campaign value (no pooling across campaigns).

Contract: never raise on a partial/capped campaign, a failed job, a missing probe or a missing log; unknown is None/"n/a".
"""
import json
import re
import statistics
import sys
from pathlib import Path

from ab_common import ARMS, RUNAWAY_TOKENS, derive, ratio, read_json

REQUEST_EVAL = re.compile(r"\|\s+eval time =\s*[\d.]+ ms /\s*(\d+) tokens")
METRICS = [  # key, label, format, higher_is_better
    ("decode_tps", "decode tok/s", "{:.1f}"), ("decode_tps_excl", "decode tok/s excl. runaway tasks", "{:.1f}"),
    ("round_ms", "ms/round (verify + plain rounds)", "{:.2f}"), ("tokens_per_round", "tokens/round", "{:.3f}"),
    ("acceptance", "acceptance (accepted/drafted)", "{:.3f}"), ("mean_draft_len", "mean draft length", "{:.3f}"),
    ("plain_frac", "plain-round fraction", "{:.3f}"), ("output_tokens", "output tokens", "{:.0f}"),
    ("wall_s", "task wall s", "{:.0f}"), ("solved", "solved /8", "{:.0f}"), ("runaways", "runaway tasks", "{:.0f}")]


def response_tokens(log):
    try:
        return [int(n) for n in REQUEST_EVAL.findall(Path(log).read_text(errors="replace"))]
    except OSError:
        return []


def job_sample(campaign, job):
    probe = job.get("probe") or {}
    metrics = probe.get("metrics_delta") or {}
    attempt = job.get("attempt")
    responses = response_tokens(Path(campaign) / attempt / "server.log") if attempt else []
    peak = max(responses, default=None)
    sample = {"task": job.get("task"), "status": job.get("status"), "reason": job.get("reason"),
              "solved": bool(probe.get("task_success")), "timed_out": bool(probe.get("timed_out")),
              "wall_s": probe.get("verified_wall_seconds"), "responses": len(responses), "max_response": peak,
              "runaway": peak is not None and peak >= RUNAWAY_TOKENS}
    return sample | derive(metrics)


def totals(samples):
    t = {k: sum(s[k] for s in samples) for k in ("tokens", "decode_s", "drafts", "draft_tokens", "accepted", "plain", "rounds")}
    t["wall_s"] = sum(s["wall_s"] or 0 for s in samples)
    t["solved"] = sum(1 for s in samples if s["solved"])
    return t


def metrics_of(samples):
    t = totals(samples)
    return {"decode_tps": ratio(t["tokens"], t["decode_s"]),
            "round_ms": ratio(1000 * t["decode_s"], t["rounds"]) if t["rounds"] else None,
            "tokens_per_round": ratio(t["tokens"], t["rounds"]), "acceptance": ratio(t["accepted"], t["draft_tokens"]),
            "mean_draft_len": ratio(t["draft_tokens"], t["drafts"]), "plain_frac": ratio(t["plain"], t["rounds"]),
            "output_tokens": t["tokens"] if samples else None, "wall_s": t["wall_s"] if samples else None,
            "solved": t["solved"] if samples else None}


def campaign_row(path, meta=None):
    path = Path(path)
    meta = meta or read_json(path / "ab.json", {}) or {}
    summary = read_json(path / "SUMMARY.json", {}) or {}
    state = read_json(path / "state.json", {}) or {}
    samples, errors = [], []
    for file in sorted(path.glob("job-*.json")):
        job = read_json(file)
        if job is None:
            errors.append(f"unreadable {file.name}")
            continue
        try:
            samples.append(job_sample(path, job))
        except (TypeError, ValueError, KeyError) as error:
            errors.append(f"{file.name}: {type(error).__name__}")
    ok = [s for s in samples if s["status"] == "passed" and None not in (s["tokens"], s["decode_s"], s["rounds"]) and s["decode_s"]]
    runaway = [s for s in ok if s["runaway"]]
    row = {"path": str(path), **{k: meta.get(k) for k in ("idx", "arm", "k", "p_min", "elapsed_s", "capped", "rc", "note")},
           "state": summary.get("state") or state.get("status") or "missing", "jobs": len(samples), "jobs_ok": len(ok),
           "failed": [{"task": s["task"], "reason": s["reason"], "max_response": s["max_response"]} for s in samples if s["status"] != "passed"],
           "runaway_tasks": [{"task": s["task"], "max_response": s["max_response"]} for s in samples if s["runaway"]],
           "errors": errors, "tasks": samples}
    row |= metrics_of(ok)
    excl = metrics_of([s for s in ok if not s["runaway"]])
    row["decode_tps_excl"] = excl["decode_tps"] if runaway else row["decode_tps"]
    row["runaways"] = len([s for s in samples if s["runaway"]])
    row["eligible"] = row["state"] == "complete" and len(samples) == 8 and row["jobs_ok"] == 8 and not errors
    return row


def fmt(spec, value):
    return "n/a" if value is None else spec.format(value)


def arm_label(arm):
    s = ARMS.get(arm)
    return f"{arm} ({s['label']})" if s else str(arm)


def compare(rows):
    by_arm = {arm: [r for r in rows if r.get("arm") == arm and r["eligible"]] for arm in ARMS}
    result = {}
    for key, _, _ in METRICS:
        per = {arm: [(r.get("idx"), r.get(key)) for r in group if r.get(key) is not None] for arm, group in by_arm.items()}
        stats = {}
        for arm, values in per.items():
            nums = [v for _, v in values]
            stats[arm] = {"values": values, "mean": statistics.mean(nums) if nums else None,
                          "min": min(nums) if nums else None, "max": max(nums) if nums else None}
        base = stats["A"]
        for arm in ("B",):
            s = stats[arm]
            if base["mean"] is None or s["mean"] is None:
                s["vs_A"] = None
                continue
            separated = s["min"] > base["max"] or s["max"] < base["min"]
            s["vs_A"] = {"delta": s["mean"] - base["mean"], "pct": 100 * (s["mean"] - base["mean"]) / base["mean"] if base["mean"] else None,
                         "separated_from_A_range": separated}
        result[key] = stats
    return result


def render(rows, plan=None, notes=()):
    lines = ["# Real-traffic A/B (A = stack2-metafix, B = stack3-mask + input staging; both k5 p_min 0)", ""]
    lines += list(notes) + [""] if notes else []
    if plan:
        lines += [f"Plan: {plan}", ""]
    lines += ["Sample unit: one eight-task Codex campaign. Values are per campaign, never pooled. decode tok/s = sum(decode tokens) / "
              "sum(decode seconds) over the campaign's passed tasks; a round is a verify step plus a plain decode step (the server counts "
              "only verify steps; plain steps are recovered as tokens - accepted - verify steps). Acceptance rises mechanically when p_min "
              "shortens drafts, so compare tokens/round and decode tok/s, not acceptance (both arms here draft k5, so acceptance and tokens/round "
              "must agree up to sampling noise; a gap there is a correctness signal, not a speedup). Runaway = any single response >= "
              f"{RUNAWAY_TOKENS} tokens.", "",
              "| # | arm | state | tasks ok | solved | runaway | decode tok/s | excl. runaway | ms/round | tok/round | accept | draft len | plain | out tok | wall s | elapsed s |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        state = r["state"] + (" (capped)" if r.get("capped") else "") + ("" if r["eligible"] else " [excluded]")
        lines.append("| " + " | ".join([str(r.get("idx")), str(r.get("arm")), state, f"{r['jobs_ok']}/8", fmt("{:.0f}", r.get("solved")),
                                        str(r["runaways"])] + [fmt(f, r.get(k)) for k, f in (
            ("decode_tps", "{:.1f}"), ("decode_tps_excl", "{:.1f}"), ("round_ms", "{:.2f}"), ("tokens_per_round", "{:.3f}"),
            ("acceptance", "{:.3f}"), ("mean_draft_len", "{:.2f}"), ("plain_frac", "{:.3f}"), ("output_tokens", "{:.0f}"),
            ("wall_s", "{:.0f}"), ("elapsed_s", "{:.0f}"))]) + " |")
    flagged = [r for r in rows if r["runaway_tasks"] or r["failed"]]
    if flagged:
        lines += ["", "Runaway or failed tasks:"]
        for r in flagged:
            for t in r["runaway_tasks"]:
                lines.append(f"- #{r.get('idx')} {r.get('arm')}: {t['task']} response of {t['max_response']} tokens")
            for t in r["failed"]:
                lines.append(f"- #{r.get('idx')} {r.get('arm')}: {t['task']} not passed ({t['reason']}), longest response {t['max_response']}")
    comparison = compare(rows)
    lines += ["", "## Arms side by side (campaign values in run order; delta = arm mean - A mean)", "",
              "| metric | A | B | B - A |", "|---|---|---|---|"]
    for key, label, spec in METRICS:
        cells = []
        for arm in ARMS:
            s = comparison[key][arm]
            cells.append(", ".join(f"#{i}: {fmt(spec, v)}" for i, v in s["values"]) or "n/a")
        deltas = []
        for arm in ("B",):
            v = comparison[key][arm].get("vs_A")
            pct = "" if not v or v["pct"] is None else f" ({v['pct']:+.1f}%)"
            deltas.append("n/a" if not v else f"{v['delta']:+.4g}{pct} " + ("separated" if v["separated_from_A_range"] else "within spread"))
        lines.append(f"| {label} | " + " | ".join(cells + deltas) + " |")
    lines += ["", "separated = the arm's campaign values lie entirely outside A's min-max range; with n=2 per arm that is a weak screen, not a test. Plan ABBA cancels linear drift only; campaigns are not paired by task output.", ""]
    return "\n".join(lines) + "\n"


def campaign_meta(path):
    match = re.fullmatch(r"campaign-(\d+)-([A-Z])", Path(path).name)
    if not match:
        return None
    arm = match[2]
    spec = ARMS.get(arm, {})
    return {"idx": int(match[1]), "arm": arm, "k": spec.get("k"), "p_min": spec.get("p_min")} | (read_json(Path(path) / "ab.json", {}) or {})


def summarize(run_dir, plan=None, notes=()):
    run_dir = Path(run_dir)
    rows = []
    for path in sorted(p for p in run_dir.glob("campaign-*") if p.is_dir()):
        meta = campaign_meta(path)
        if meta is None:
            continue
        try:
            rows.append(campaign_row(path, meta))
        except Exception as error:  # a damaged campaign dir must not hide the others
            rows.append({"path": str(path), "state": "unreadable", "eligible": False, "jobs_ok": 0, "runaways": 0, "runaway_tasks": [],
                         "failed": [], "errors": [repr(error)], "tasks": [], **{k: None for k in ("elapsed_s", "capped", "rc", "note")}, **meta})
    rows.sort(key=lambda r: r["idx"])
    text = render(rows, plan, notes)
    (run_dir / "summary.md").write_text(text)
    (run_dir / "summary.json").write_text(json.dumps({"plan": plan, "notes": list(notes), "campaigns": rows, "comparison": compare(rows)}, indent=2, default=str) + "\n")
    return rows, text


if __name__ == "__main__":
    print(summarize(sys.argv[1])[1], end="")
