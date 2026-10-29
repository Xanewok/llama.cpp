"""Workload profiles mined from recorded call tables, the wall_est cost model, and its calibration.

wall_est(cfg, W) = sum over calls in W of
    fixed_call_s(engine) + new_tokens / prefill_tps(depth) + out_tokens / decode_tps(depth) + gap_s
  + task_overhead_s per task.
A profile pools every valid campaign of one task set; the estimate is the mean campaign.
Two agents: max over the two streams of their own time plus RHO times the other stream's prefill
time (rho=1 is the pessimistic bound if the other slot's prefill stalls this stream's decode).
"""
import statistics
from collections import defaultdict

from .store import Store

FIELDS = ["depth", "new_tokens", "out_tokens", "prefix_kept", "gap_s", "dur_s"]
LLAMA_FIXED_CALL_S = 0.54  # ledger-prior: REPORT §1.1, hybrid-checkpoint followup cost on llama (F)
RHO = 0.0
RHO_SOURCE = "fitted on n=2 two-agent runs (one HQ, one llama; rho=0 within 1%, rho=1 over-predicts by 6-13%); not validated out of sample"


def _median(xs):
    xs = [x for x in (xs or []) if x is not None]
    return statistics.median(xs) if xs else None


def tps_at(spec, depth):
    """spec: float (flat) or sorted [(depth, tps)] curve, piecewise linear, clamped at the ends."""
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return float(spec)
    pts = sorted(spec)
    if depth <= pts[0][0]:
        return pts[0][1]
    for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
        if depth <= d1:
            return v0 + (v1 - v0) * (depth - d0) / (d1 - d0)
    return pts[-1][1]


# ---- speeds ---------------------------------------------------------------------------------

def campaign_speeds(store):
    """{config_id: {prefill, decode, n, basis}} from measured campaign / two-agent aggregates."""
    acc = defaultdict(lambda: defaultdict(list))
    for m in store.rows("measurements"):
        if m["status"] != "measured" or m["source"] not in ("campaign", "two-agent"):
            continue
        if m["metric"] in ("prefill_tok_s", "decode_tok_s") and m["source"] == "campaign":
            acc[(m["config_id"], "campaign-agg")][m["metric"]].append(m["value"])
        elif m["source"] == "two-agent" and m["metric"] in ("prefill_tok_s", "decode_tok_s", "agg_tok_s"):
            acc[(m["config_id"], "two-agent")][m["metric"]].append(m["value"])
    out = {}
    for (cid, basis), d in acc.items():
        out[(cid, basis)] = {"prefill": _median(d.get("prefill_tok_s")), "decode": _median(d.get("decode_tok_s")),
                             "agg": _median(d.get("agg_tok_s")), "n": len(d.get("decode_tok_s", [])), "basis": basis}
    return out


def run_speeds(store):
    """{summary_sha256: {prefill, decode}} per campaign run (campaign_speeds pools replicates of one config)."""
    acc = defaultdict(dict)
    for m in store.rows("measurements"):
        if m["source"] == "campaign" and m["status"] == "measured" and m["metric"] in ("prefill_tok_s", "decode_tok_s"):
            acc[m["provenance"].get("summary_sha256")][m["metric"].removesuffix("_tok_s")] = m["value"]
    return acc


def f1_speeds(store):
    """{config_id: {"curves": {np: {"decode": [(depth, tps)], "prefill": [...]}}, basis: "F1"}} (queue-np, hq-conc)."""
    acc = defaultdict(lambda: defaultdict(list))
    for m in store.rows("measurements"):
        if m["status"] != "measured" or m["source"] not in ("queue-np", "hq-conc"):
            continue
        if m["metric"] in ("decode_tok_s", "prefill_tok_s", "stream_tok_s") and "depth" in m["dims"]:
            np_ = m["dims"].get("np") or m["dims"].get("streams") or 1
            acc[m["config_id"]][(np_, m["metric"], m["dims"]["depth"])].append(m["value"])
    out = {}
    for cid, d in acc.items():
        curves = defaultdict(lambda: {"decode": [], "prefill": []})
        for (np_, metric, depth), v in d.items():
            curves[np_]["prefill" if metric == "prefill_tok_s" else "decode"].append((depth, statistics.mean(v)))
        for c in curves.values():
            c["decode"].sort()
            c["prefill"].sort()
        out[cid] = {"curves": dict(curves), "basis": "F1"}
    return out


def f1_for(entry, np_wanted):
    """(np used, {"decode", "prefill"} curves, note). Decode comes from the nearest recorded np; a missing
    prefill curve is borrowed from another np and says so."""
    cs = {n: c for n, c in entry["curves"].items() if c["decode"]}
    if not cs:
        return None
    n = np_wanted if np_wanted in cs else min(cs, key=lambda x: abs(x - np_wanted))
    pre, note = cs[n]["prefill"], ""
    if not pre:
        src = [(m, c["prefill"]) for m, c in entry["curves"].items() if c["prefill"]]
        if not src:
            return None
        m, pre = min(src, key=lambda t: abs(t[0] - n))
        note = f"prefill borrowed from np={m}"
    return n, {"decode": cs[n]["decode"], "prefill": pre}, note


# ---- profiles --------------------------------------------------------------------------------

def _valid_calls(store, kind):
    return sorted((w for w in store.rows("workloads") if w.get("kind") == kind and w.get("valid")), key=lambda w: w["id"])


def _camp_stats(w):
    """Per-campaign task overhead and call count (wall is the sum of task walls)."""
    calls = [c for t in w["tasks"].values() for c in t]
    return len(calls), sum(c[5] + c[4] for c in calls)


def fit_constants(camps, spd, tw):
    """(fixed_call_s, task_overhead_s): medians of per-campaign residuals over the given call-table campaigns.
    Calibration refits these without the held-out run, so the constants never see the run they predict."""
    fixed, over = [], []
    for w in camps:
        s = spd.get((w["config_id"], "campaign-agg"))
        if not s or not s["prefill"] or not s["decode"]:
            continue
        calls = [c for t in w["tasks"].values() for c in t]
        rest_calls = sum(c[5] for c in calls) - sum(c[1] / s["prefill"] + c[2] / s["decode"] for c in calls)
        fixed.append(rest_calls / len(calls))
        for task, cl in w["tasks"].items():
            wall = tw.get((w.get("summary_sha256"), task))
            if wall is not None:
                over.append(wall - sum(c[5] + c[4] for c in cl))
    return _median(fixed), _median(over)


def build_profiles(store, write=True):
    """Pool valid 8-task campaigns into `profile:codex-8task-1agent` and two-agent v3 runs into `profile:two-agent-8task`.
    Fitted constants are stored in the profile: fixed_call_s (hq) and task_overhead_s."""
    out = []
    allc = _valid_calls(store, "campaign-calls")
    camps = [w for w in allc if w["workload_id"] == "codex-8task-1agent"]
    spd = campaign_speeds(store)
    tw = _task_walls(store)
    fixed_c, over_c = fit_constants(allc, spd, tw)
    prof = {"id": "profile:codex-8task-1agent", "kind": "profile", "workload_id": "codex-8task-1agent", "agents": 1,
            "fields": FIELDS, "sources": [w["campaign"] for w in camps], "n_campaigns": len(camps),
            "tasks_per_campaign": 8, "calls": [c for w in camps for t in w["tasks"].values() for c in t],
            "task_overhead_s": over_c, "fixed_call_s": {"hq": fixed_c},
            "fit_note": "fixed_call_s and task_overhead_s are medians over all valid call-table campaigns' own residuals; llama fixed is a ledger-prior"}
    prof["calls_per_campaign"] = len(prof["calls"]) / max(len(camps), 1)
    out.append(prof)
    runs = _valid_calls(store, "two-agent-calls")
    if runs:
        out.append({"id": "profile:two-agent-8task", "kind": "profile", "workload_id": "two-agent-8task", "agents": 2,
                    "fields": FIELDS, "sources": [r["campaign"] for r in runs], "n_campaigns": len(runs),
                    "runs": [{"name": r["campaign"], "config_id": r["config_id"], "wall_s": r["wall_s"], "agents": r["agents"]} for r in runs],
                    "task_overhead_s": 0.0, "fixed_call_s": prof["fixed_call_s"],
                    "fit_note": "per-agent call tables include inter-task gaps, so no separate task overhead"})
    if write:
        for p in out:
            store.upsert("workloads", p)
    return out


def _task_walls(store):
    tw = {}
    for m in store.rows("measurements"):
        if m["metric"] == "task_wall_s" and m["status"] == "measured":
            tw[(m["provenance"].get("summary_sha256"), m["dims"]["task"])] = m["value"]
    return tw


def get_profile(store, workload_id):
    p = store.tables["workloads"].get(f"profile:{workload_id}")
    if not p:
        raise KeyError(f"no profile for workload {workload_id!r}; run `opt profile build`")
    return p


# ---- wall_est --------------------------------------------------------------------------------

def _call_cost(c, prefill, decode, fixed, kept_fn, scale=(1.0, 1.0)):
    depth, new, out, kept, gap, _dur = c
    if kept and kept_fn is not None and not kept_fn(depth):
        new = depth
    return {"fixed": fixed, "prefill": scale[1] * new / tps_at(prefill, depth),
            "decode": scale[0] * out / tps_at(decode, depth), "gap": gap}


def _sum_parts(calls, prefill, decode, fixed, kept_fn, scale=(1.0, 1.0)):
    parts = defaultdict(float)
    for c in calls:
        for k, v in _call_cost(c, prefill, decode, fixed, kept_fn, scale).items():
            parts[k] += v
    return parts


def volume_scale(profile, volume):
    """(out, new) multipliers that make the pooled profile carry `volume` = {"out": tokens, "new": tokens}
    per campaign. The call shape (depths, per-call split) is kept; only the token volume is re-based."""
    n = profile["n_campaigns"]
    out = sum(c[2] for c in profile["calls"]) / n
    new = sum(c[1] for c in profile["calls"]) / n
    return (volume["out"] / out, volume["new"] / new)


def engine_volume(store, engine, exclude_items=()):
    """Median output / new-prompt tokens per 8-task campaign over valid measured campaigns of one engine."""
    cfgs = store.tables["configs"]
    per = defaultdict(dict)
    for m in store.rows("measurements"):
        if (m["source"] == "campaign" and m["status"] == "measured" and m["workload_id"] == "codex-8task-1agent"
                and m["metric"] in ("output_tokens", "prefill_new_tokens") and m["item_id"] not in exclude_items
                and cfgs.get(m["config_id"], {}).get("engine") == engine):
            per[m["item_id"]][m["metric"]] = m["value"]
    both = [v for v in per.values() if len(v) == 2]
    if not both:
        return None
    return {"out": _median([v["output_tokens"] for v in both]), "new": _median([v["prefill_new_tokens"] for v in both]),
            "n": len(both)}


def fixed_for(profile, engine):
    f = profile["fixed_call_s"].get(engine)
    return f if f is not None else LLAMA_FIXED_CALL_S


def wall_est_single(profile, prefill, decode, engine, kept_fn=None, fixed=None, calls=None, volume=None):
    """Mean-campaign wall (s) with its parts; `calls` overrides the pooled calls (own-call oracle);
    `volume` re-bases the token volume (see volume_scale)."""
    fixed = fixed_for(profile, engine) if fixed is None else fixed
    n = profile["n_campaigns"] if calls is None else 1
    scale = volume_scale(profile, volume) if volume else (1.0, 1.0)
    parts = _sum_parts(profile["calls"] if calls is None else calls, prefill, decode, fixed, kept_fn, scale)
    parts = {k: v / n for k, v in parts.items()}
    parts["overhead"] = (profile.get("task_overhead_s") or 0.0) * profile["tasks_per_campaign"]
    return sum(parts.values()), parts


def wall_est_two(profile, prefill, decode, engine, kept_fn=None, rho=RHO, fixed=None, runs=None):
    """Mean over recorded runs of the slower stream; parts are those of the critical stream, averaged."""
    fixed = fixed_for(profile, engine) if fixed is None else fixed
    walls, crit = [], []
    for run in (runs if runs is not None else profile["runs"]):
        ag = [_sum_parts(c, prefill, decode, fixed, kept_fn) for c in run["agents"].values()]
        times = []
        for i, p in enumerate(ag):
            other = sum(q["prefill"] for j, q in enumerate(ag) if j != i)
            times.append((sum(p.values()) + rho * other, {**p, "blocked_prefill": rho * other}))
        w, parts = max(times, key=lambda t: t[0])
        walls.append(w)
        crit.append(parts)
    keys = {k for c in crit for k in c}
    return statistics.mean(walls), {k: statistics.mean(c.get(k, 0.0) for c in crit) for k in keys}


# ---- calibration -----------------------------------------------------------------------------

def _err(est, real):
    return 100.0 * (est - real) / real


def _stats(errs):
    if not errs:
        return None
    a = [abs(e) for e in errs]
    return {"n": len(errs), "median_abs_pct": statistics.median(a), "max_abs_pct": max(a), "median_signed_pct": statistics.median(errs)}


def _rank_corr(xs, ys):
    """Spearman with average ranks for ties; None below 3 points or without variance."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def _pool(profile, camps):
    p = dict(profile)
    p["calls"] = [c for w in camps for t in w["tasks"].values() for c in t]
    p["n_campaigns"] = len(camps)
    return p


def cfg_key(c):
    """Replicate identity of a run: same engine, model, KV, draft length and non-volatile variant."""
    return (c["engine"], c.get("model_id"), c.get("kv_k"), c.get("kv_v"), c.get("k"), c.get("variant_id"))


DEFAULT_TIE_PCT = 25.0


def calibrate(store, profiles=None):
    """Error of wall_est against the recorded campaign wall (percent, signed = est - real).
    Tiers (A and B0 are IN-SAMPLE structure checks and are not prediction errors):
      A   own calls + own speeds + constants fitted on all runs
      B0  leave-one-config-out call pool, but constants fitted on all runs and the run's own speeds
      B   held out: call pool and constants from other configs only, speeds from replicate runs of the same
          config (never the run's own); needs a replicate
      C0/C  HQ-mined profile on llama campaigns (token volume re-based for C), the run's own speeds (oracle speed)
      D   C with F1 depth-curve speeds (a true prediction from synthetic measurements)
    `held_out_pct` is the median |error| per engine/basis; candidates closer than that are a tie."""
    profiles = profiles or build_profiles(store, write=False)
    prof = next(p for p in profiles if p["id"] == "profile:codex-8task-1agent")
    cfgs = store.tables["configs"]
    spd = campaign_speeds(store)
    f1 = f1_speeds(store)
    tw = _task_walls(store)
    res = {"tiers": {}, "rows": defaultdict(list), "rank_corr": {}, "constants": {
        "hq_fixed_call_s": prof["fixed_call_s"]["hq"], "task_overhead_s": prof["task_overhead_s"],
        "llama_fixed_call_s_prior": LLAMA_FIXED_CALL_S, "rho": RHO, "rho_source": RHO_SOURCE}}
    camps = _valid_calls(store, "campaign-calls")
    eight = [w for w in camps if w["workload_id"] == "codex-8task-1agent"]

    rs = run_speeds(store)

    def own_speed(w):
        s = rs.get(w.get("summary_sha256")) or spd.get((w["config_id"], "campaign-agg"))
        return s if s and s.get("prefill") and s.get("decode") else None

    def tier(name, key, pairs):
        res["tiers"][name] = _stats([_err(e, r) for e, r in pairs])
        res["rank_corr"][key] = _rank_corr([e for e, _ in pairs], [r for _, r in pairs])

    A = []
    for w in camps:
        s = own_speed(w)
        if not s:
            continue
        est, _ = wall_est_single({**prof, "tasks_per_campaign": len(w["tasks"])}, s["prefill"], s["decode"], "hq",
                                 calls=[c for t in w["tasks"].values() for c in t])
        A.append((est, w["wall_s"]))
        res["rows"]["A"].append((w["campaign"], est, w["wall_s"], _err(est, w["wall_s"])))
    tier("A (in-sample): own calls + own speeds + constants fitted on all runs (all call-table campaigns)", "A", A)

    B0, B = [], []
    for w in eight:
        s = own_speed(w)
        others = [x for x in eight if x["config_id"] != w["config_id"]]
        if not s or len(others) < 3:
            continue
        est, _ = wall_est_single(_pool(prof, others), s["prefill"], s["decode"], "hq")
        B0.append((est, w["wall_s"]))
        res["rows"]["B0"].append((w["campaign"], est, w["wall_s"], _err(est, w["wall_s"])))
        peers = [own_speed(x) for x in eight if x is not w and
                 cfg_key(cfgs[x["config_id"]]) == cfg_key(cfgs[w["config_id"]]) and own_speed(x)]
        if not peers:
            continue
        held = [x for x in camps if x["config_id"] != w["config_id"]]
        fx, ov = fit_constants(held, spd, tw)
        if fx is None or ov is None:
            continue
        p2 = {**_pool(prof, others), "task_overhead_s": ov, "fixed_call_s": {"hq": fx}}
        est, _ = wall_est_single(p2, _median(p["prefill"] for p in peers), _median(p["decode"] for p in peers), "hq")
        B.append((est, w["wall_s"]))
        res["rows"]["B"].append((w["campaign"], est, w["wall_s"], _err(est, w["wall_s"])))
    tier("B0 (in-sample): leave-one-config-out call pool, constants fitted on all runs, own speeds (8-task HQ)", "B0", B0)
    tier("B (held out): other configs' calls and constants, speeds from replicate runs of the same config (8-task HQ)", "B", B)

    llama = []
    for m in store.rows("measurements"):
        if m["source"] == "campaign" and m["metric"] == "wall_s" and m["status"] == "measured":
            c = cfgs.get(m["config_id"])
            if c and c["engine"] == "llama" and m["workload_id"] == "codex-8task-1agent":
                llama.append((m, c))
    C0, C, D, DC = [], [], [], []
    for m, c in llama:
        s = spd.get((m["config_id"], "campaign-agg"))
        if not s or not s["prefill"] or not s["decode"]:
            continue
        vol = engine_volume(store, "llama", exclude_items=pair_siblings(m["item_id"]))
        raw, _ = wall_est_single(prof, s["prefill"], s["decode"], "llama")
        C0.append((raw, m["value"]))
        est, _ = wall_est_single(prof, s["prefill"], s["decode"], "llama", volume=vol)
        C.append((est, m["value"]))
        res["rows"]["C"].append((m["item_id"], est, m["value"], _err(est, m["value"])))
        fcid = _f1_match(f1, cfgs, c)
        if fcid:
            n, curve, _ = f1_for(f1[fcid], 1)
            estd, _ = wall_est_single(prof, curve["prefill"], curve["decode"], "llama", volume=vol)
            D.append((estd, m["value"]))
            DC.append(_err(estd, est))
            res["rows"]["D"].append((m["item_id"], estd, m["value"], _err(estd, m["value"])))
    tier("C0: HQ-mined profile as is (HQ token volume) on llama campaigns, own aggregate speeds (oracle speed)", "C0", C0)
    tier("C: C0 with token volume re-based to the other llama campaigns' median (llama fixed = prior)", "C", C)
    tier("D: C with F1 depth-curve speeds (cross-build F1 match; a true prediction)", "D", D)
    res["tiers"]["DvC: F1-curve estimate against aggregate-speed estimate (no campaign wall)"] = _stats(DC)
    res["rank_corr_B"], res["rank_corr_C"], res["rank_corr_D"] = (res["rank_corr"].get(k) for k in "BCD")

    two = _calibrate_two(store, profiles, spd)
    res["two_agent"] = two
    named = {k.split(" ")[0].rstrip(":"): v for k, v in res["tiers"].items()}
    held = {"hq": (named.get("B") or {}).get("median_abs_pct"), "llama": (named.get("C") or {}).get("median_abs_pct"),
            "F1": (named.get("D") or {}).get("median_abs_pct")}
    if two:
        # in-sample at rho=0, so the usable width is how far rho in [0,1] moves the estimate
        held["two-agent"] = max(max(abs(r["err_rho0_pct"]), abs(r["err_rho1_pct"])) for r in two)
    res["held_out_pct"] = held
    return res


def pair_siblings(item_id):
    """The campaign itself plus nothing else: volume medians exclude only the campaign being predicted."""
    return {item_id}


def _f1_match(f1, cfgs, c):
    """F1 config with the same (engine, quant, kv, k) and the richest np=1 curve (cross-build; indicative)."""
    best, best_n = None, 0
    for cid, e in f1.items():
        o = cfgs.get(cid)
        if not o or (o["engine"], o.get("model_id"), o.get("kv_k"), o.get("kv_v"), o.get("k")) != \
                (c["engine"], c.get("model_id"), c.get("kv_k"), c.get("kv_v"), c.get("k")):
            continue
        cur = e["curves"].get(1)
        if cur and len(cur["decode"]) >= 3 and len(cur["prefill"]) >= 1 and len(cur["decode"]) > best_n:
            best, best_n = cid, len(cur["decode"])
    return best


def two_agent_speeds(store, config_id):
    """Per-stream decode, shared prefill and the run's measured numbers for a two-agent run."""
    d = {}
    for m in store.rows("measurements"):
        if m["source"] == "two-agent" and m["config_id"] == config_id and m["status"] == "measured":
            d[m["metric"]] = m["value"]
    return d


def _calibrate_two(store, profiles, spd):
    prof = next((p for p in profiles if p["id"] == "profile:two-agent-8task"), None)
    if not prof:
        return None
    cfgs = store.tables["configs"]
    rows = []
    for run in prof["runs"]:
        s = spd.get((run["config_id"], "two-agent"))
        if not s or not s["prefill"] or not s["decode"]:
            continue
        eng = cfgs[run["config_id"]]["engine"]
        row = {"run": run["name"], "real_s": run["wall_s"], "engine": eng}
        for rho in (0.0, 1.0):
            est, _ = wall_est_two(prof, s["prefill"], s["decode"], eng, rho=rho, runs=[run])
            row[f"est_rho{int(rho)}"] = est
            row[f"err_rho{int(rho)}_pct"] = _err(est, run["wall_s"])
        rows.append(row)
    return rows


def report_text(res):
    L = ["## wall_est calibration (error = est - real, percent of the real campaign wall)", "",
         "A and B0 are in-sample structure checks, not prediction errors; B, D and the two-agent rows are out of sample "
         "(D vs C has no real wall).", "",
         "| tier | n | median abs % | max abs % | median signed % | Spearman |", "|---|---|---|---|---|---|"]
    for name, s in res["tiers"].items():
        key = name.split(" ")[0].rstrip(":")
        rc = res["rank_corr"].get(key)
        rcs = "n/a" if rc is None else f"{rc:.2f}"
        if s:
            L.append(f"| {name} | {s['n']} | {s['median_abs_pct']:.1f} | {s['max_abs_pct']:.1f} | {s['median_signed_pct']:+.1f} | {rcs} |")
        else:
            L.append(f"| {name} | 0 | n/a | n/a | n/a | n/a |")
    c = res["constants"]
    L += ["", f"constants: hq fixed {c['hq_fixed_call_s']:.3f} s/call and task overhead {c['task_overhead_s']:.1f} s/task are fitted on all runs; "
          f"llama fixed {c['llama_fixed_call_s_prior']} s/call is a ledger-prior; RHO = {c['rho']} is {c['rho_source']}", ""]
    ho = res.get("held_out_pct") or {}
    L.append("error used for tie bands (hq/llama/F1: median held-out |error| %; two-agent: rho 0..1 span, in-sample): " + ", ".join(
        f"{k} {v:.0f}" if v is not None else f"{k} n/a (default {DEFAULT_TIE_PCT:.0f})" for k, v in ho.items()))
    if res.get("two_agent"):
        L += ["", "two-agent v3 runs, own speeds (rho = fraction of the other stream's prefill that stalls this stream; n=2 runs, the same runs rho was fitted on)", "",
              "| run | engine | real s | est rho=0 | err % | est rho=1 | err % |", "|---|---|---|---|---|---|---|"]
        for r in res["two_agent"]:
            L.append(f"| {r['run']} | {r['engine']} | {r['real_s']:.0f} | {r['est_rho0']:.0f} | {r['err_rho0_pct']:+.1f} | "
                     f"{r['est_rho1']:.0f} | {r['err_rho1_pct']:+.1f} |")
    return "\n".join(L) + "\n"
