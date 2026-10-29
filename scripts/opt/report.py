"""`opt report goal.toml`: candidates ranked by wall_est with constraint verdicts, Pareto wall_est x KLD.

Candidate = engine configs that agree on (engine, stack, model id, model sha, kv, k, p_min, ub, np, ctx/slot, util,
variant); the variant is every other non-volatile flag and GGML_/VLLM_/... env var, so A/B arms and model recipes
are never pooled. Speeds are pooled (median) over the member configs. Each cell carries a status tag:
m = measured, d = derived (model over measured inputs), p = ledger-prior. Missing data is UNKNOWN, never a pass.
A candidate whose KLD is unverified ranks below every candidate whose KLD is verified; rows with F1-basis speeds
and rows with measured-basis speeds are ranked in separate lists. Candidates inside the held-out error of the
estimate are a tie band, not an order.
"""
import fnmatch
import statistics
from collections import Counter, defaultdict

from . import fit, goal as goalmod, workload as W
from .store import Store

CAND_KEYS = ("engine", "stack_name", "model_id", "model_sha256", "kv_k", "kv_v", "k", "p_min", "ub", "np", "ctx_per_slot",
             "gpu_util", "variant_id")
KNOWN_WORKLOADS = {"codex-8task-1agent", "two-agent-8task"}
LEDGER_METRIC = "kld@ledger-q6k-8k"


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _pool_curve(curves):
    by = defaultdict(list)
    for c in curves:
        for d, v in c:
            by[d].append(v)
    return sorted((d, statistics.median(v)) for d, v in by.items())


def _measured(store, metric, workload=None):
    out = defaultdict(list)
    for m in store.rows("measurements"):
        if m["metric"] == metric and m["status"] == "measured" and (workload is None or m["workload_id"] == workload):
            out[m["config_id"]].append(m)
    return out


def kld_table(store):
    """{(model_id, metric): {kv: (median, n)}} over measured, non-control all_prefill rows of the default numerics.
    n counts distinct (arm label, value to 5 decimals): the same dump rendered in a .txt and a .json is one."""
    cfgs = store.tables["configs"]
    acc = defaultdict(lambda: defaultdict(dict))
    for m in store.rows("measurements"):
        if m["source"] != "kld" or not m["metric"].startswith("kld@") or m["status"] != "measured":
            continue
        d = m["dims"]
        if d.get("bucket") != "all_prefill" or d.get("control"):
            continue
        c = cfgs[m["config_id"]]
        if not c.get("model_id") or c.get("variant_id"):
            continue
        acc[(c["model_id"], m["metric"])][c["kv"]][(d.get("arm"), round(m["value"], 5))] = m["value"]
    return {k: {kv: (statistics.median(v.values()), len(v)) for kv, v in d.items()} for k, d in acc.items()}


def kld_for(table, cand, metric):
    """Exact model id and exact KV only. A KLD measured at another KV is not evidence for this one."""
    d = table.get((cand["model_id"], metric))
    kv = f"{cand['kv_k']}/{cand['kv_v']}"
    if not d or kv not in d:
        return None
    return d[kv][0], d[kv][1], ""


def kld_other_kv(table, cand, metric):
    d = table.get((cand["model_id"], metric)) or {}
    kv = f"{cand['kv_k']}/{cand['kv_v']}"
    return sorted((k, v[0], v[1]) for k, v in d.items() if k != kv)


def ledger_for(store, cand):
    """ledger-prior KLD context rows of this exact model id (any KV); never evidence for a constraint."""
    cfgs = store.tables["configs"]
    return [(m["value"], m.get("reason")) for m in store.rows("measurements")
            if m["source"] == "ledger" and m["metric"] == LEDGER_METRIC and cfgs[m["config_id"]].get("model_id") == cand["model_id"]]


def _group(cfgs):
    groups = defaultdict(list)
    for c in cfgs:
        if c["engine"] in ("llama", "hq"):
            groups[tuple(c.get(k) for k in CAND_KEYS)].append(c)
    # a member without a recorded model sha joins the group with the same other fields when that sha is unambiguous
    i = CAND_KEYS.index("model_sha256")
    rest = lambda k: k[:i] + k[i + 1:]
    shas = defaultdict(list)
    for k in groups:
        if k[i]:
            shas[rest(k)].append(k)
    for k in [k for k in groups if not k[i]]:
        target = shas.get(rest(k), [])
        if len(target) == 1:
            groups[target[0]] += groups.pop(k)
    return groups


def candidates(store, agents, workload=None):
    workload = workload or ("two-agent-8task" if agents == 2 else "codex-8task-1agent")
    groups = _group(store.tables["configs"].values())
    camp = W.campaign_speeds(store)
    f1 = W.f1_speeds(store)
    vram = _measured(store, "vram_min_free_mib")
    runaway = _measured(store, "runaway_rate")
    walls = _measured(store, "wall_s", workload)
    solve = _measured(store, "solve_rate", workload)
    pools = _measured(store, "kv_pool_tokens")
    out = []
    for key, members in groups.items():
        cand = dict(zip(CAND_KEYS, key))
        cand["members"] = [m["id"] for m in members]
        cand["np"] = cand.get("np") or 1
        want = "two-agent" if agents == 2 else "campaign-agg"
        sp = [camp[(m["id"], want)] for m in members if (m["id"], want) in camp and camp[(m["id"], want)]["prefill"]
              and camp[(m["id"], want)]["decode"]]
        speeds = None
        if sp:
            speeds = {"prefill": _med(s["prefill"] for s in sp), "decode": _med(s["decode"] for s in sp), "basis": want,
                      "note": ""}
        else:
            fm = [(m, f1[m["id"]]) for m in members if m["id"] in f1]
            picks = [(m["id"], W.f1_for(e, 2 if agents == 2 else 1)) for m, e in fm]
            picks = [(i, p) for i, p in picks if p]
            if picks:
                nps = {p[0] for _, p in picks}
                wanted = 2 if agents == 2 else 1
                best = wanted if wanted in nps else min(nps, key=lambda n: abs(n - wanted))
                use = [p for _, p in picks if p[0] == best]
                notes = sorted({u[2] for u in use if u[2]})
                speeds = {"prefill": _pool_curve([u[1]["prefill"] for u in use]), "decode": _pool_curve([u[1]["decode"] for u in use]),
                          "basis": "F1", "note": "; ".join(([f"F1 decode measured at np={best}, goal needs np={wanted}"] if best != wanted else []) + notes)}
        if speeds is None:
            continue
        if speeds["basis"] == "F1":
            n_pts = min(len(speeds["decode"]), len(speeds["prefill"]))
            if n_pts < 3:
                speeds["note"] = "; ".join(x for x in (speeds["note"], f"F1 curve has {n_pts} depth point(s): flat beyond it") if x)
        cand["speeds"] = speeds
        cand["klass"] = "F1" if speeds["basis"] == "F1" else "measured"
        mins = defaultdict(list)
        for m in members:
            for r in vram.get(m["id"], []):
                mins[r["dims"].get("gpu")].append(r["value"])
        cand["vram"] = ("measured", min(_med(v) for v in mins.values())) if mins else None
        cand["runaway"] = _med(r["value"] for m in members for r in runaway.get(m["id"], []))
        w = [r["value"] for m in members for r in walls.get(m["id"], [])]
        cand["measured_wall"] = (statistics.median(w), len(w)) if w else None
        sr = [r["value"] for m in members for r in solve.get(m["id"], [])]
        cand["solve_rate"] = _med(sr)
        pl = [r["value"] for m in members for r in pools.get(m["id"], [])]
        cand["pool"] = _med(pl)
        cand["rep"] = members[0]
        out.append(cand)
    return out


def _unit(cand, cfg):
    return ((cfg.get("variant") or {}).get("flags") or {}).get("prefix-match-unit") or 1


def _kept_fn(store, cand, agents, hqfit):
    """HQ with two agents keeps a prefix only below ~0.30 of the KV pool (IN-SAMPLE rule); llama checkpoints always keep it."""
    if cand["engine"] != "hq" or agents < 2:
        return None, "n/a"
    ck, k, util = cand["model_id"], cand.get("k"), cand.get("gpu_util") or 0.86
    if ck not in hqfit["pool0"] or not k:
        return None, "pool unknown"
    pool = fit.hq_pool(hqfit["pool0"], hqfit["per_k"], ck, k, util)
    unit = _unit(cand, cand["rep"])
    return (lambda d: fit.hq_prefix_state(pool, d, unit) == "kept"), f"pool {pool / 1000:.0f}k (rule in-sample)"


def _pool_per_slot(cand, hqfit):
    """(tokens per slot, 'measured'|'predicted') for an HQ candidate, or None when neither exists."""
    np_ = cand["np"]
    if cand.get("pool"):
        return cand["pool"] / np_, "measured"
    k, ck = cand.get("k"), cand["model_id"]
    if ck in hqfit["pool0"] and k:
        return fit.hq_pool(hqfit["pool0"], hqfit["per_k"], ck, k, cand.get("gpu_util") or 0.86) / np_, "predicted"
    return None


def _fmt_var(v, base):
    if not v:
        return ""
    diff = []
    for part in ("flags", "env"):
        have, ref = v.get(part, {}), (base or {}).get(part, {})
        for k in sorted(set(have) | set(ref)):
            if have.get(k) != ref.get(k):
                diff.append(f"{k}={have[k]}" if k in have else f"-{k}")
    return ",".join(diff[:5]) + ("..." if len(diff) > 5 else "")


def evaluate(store, goal, kld_cache=None):
    if goal["workload"] not in KNOWN_WORKLOADS:
        raise goalmod.GoalError(f"unknown workload {goal['workload']!r}; known: {sorted(KNOWN_WORKLOADS)}")
    agents = 2 if goal["workload"].startswith("two-agent") else 1
    try:
        prof = W.get_profile(store, goal["workload"])
        prof_note = f"profile {prof['id']} from the DB"
    except KeyError:
        prof = next((p for p in W.build_profiles(store, write=False) if p["id"] == f"profile:{goal['workload']}"), None)
        if prof is None:
            raise ValueError(f"no profile for workload {goal['workload']!r}; ingest campaigns with model-calls first")
        prof_note = f"profile {prof['id']} built in memory (not stored)"
    if not prof["n_campaigns"]:
        raise ValueError(f"{prof['id']} has no valid source runs in the DB; ingest campaigns with model-calls first")
    model, hq = fit.fit_llama(store), fit.fit_hq(store)
    band = fit.loo_band(model)
    kt = kld_table(store)
    vols = {e: W.engine_volume(store, e) for e in ("llama", "hq")}
    cons = goal["constraints"]
    cal = W.calibrate(store)
    held = cal["held_out_pct"]
    kld_metrics = list(dict.fromkeys(m for m in list(cons["bounds"]) + goal["tiebreak"] if m.startswith("kld@")))
    pareto_metric = kld_metrics[0] if kld_metrics else "kld@REFBF"
    if pareto_metric not in kld_metrics:
        kld_metrics.append(pareto_metric)
    cands = candidates(store, agents, goal["workload"])
    base_var = {}
    for e in ("llama", "hq"):
        vs = Counter(c["variant_id"] for c in cands if c["engine"] == e)
        if vs:
            vid = vs.most_common(1)[0][0]
            base_var[e] = next(c["rep"].get("variant") for c in cands if c["engine"] == e and c["variant_id"] == vid)
    rows, excluded = [], Counter()
    for cand in cands:
        r = {"cand": cand, "checks": {}, "notes": [], "metrics": {}}
        c = cand["rep"]
        if cons["engine"] and cand["engine"] not in cons["engine"]:
            continue
        if cons.get("model") and not any(fnmatch.fnmatchcase(cand["model_id"] or "", pat) for pat in cons["model"]):
            excluded[cand["model_id"]] += 1
            continue
        r["variant_diff"] = _fmt_var(c.get("variant"), base_var.get(cand["engine"]))
        sp = cand["speeds"]
        if sp["note"]:
            r["notes"].append(sp["note"])
        kept, kept_note = _kept_fn(store, cand, agents, hq)
        r["kept"] = kept_note
        engine = cand["engine"]
        if agents == 1:
            vol = vols["llama"] if engine == "llama" else None
            est, parts = W.wall_est_single(prof, sp["prefill"], sp["decode"], engine, kept_fn=kept, volume=vol)
        else:
            est, parts = W.wall_est_two(prof, sp["prefill"], sp["decode"], engine, kept_fn=kept)
        r["wall_est"], r["parts"] = est, parts
        r["metrics"]["wall_est"] = est
        pct = held.get("F1" if sp["basis"] == "F1" else "two-agent" if agents == 2 else engine if engine in held else "hq")
        r["tie_pct"] = pct if pct is not None else W.DEFAULT_TIE_PCT
        # constraints: True pass, False fail, None unknown
        f = cons["fits"]
        if f:
            ok = []
            if "np" in f:
                ok.append(cand["np"] >= f["np"])
            if "ctx_per_slot" in f:
                ok.append(None if cand.get("ctx_per_slot") is None else cand["ctx_per_slot"] >= f["ctx_per_slot"])
                if engine == "hq":
                    pp = _pool_per_slot(cand, hq)
                    ok.append(None if pp is None else pp[0] >= f["ctx_per_slot"])
                    if pp:
                        r["notes"].append(f"KV pool per slot {pp[0] / 1000:.0f}k ({pp[1]})")
                    else:
                        r["notes"].append("KV pool unknown")
            if "engine" in f:
                ok.append(cand["engine"] == f["engine"])
            r["checks"]["fits"] = False if False in ok else None if None in ok else True
        v = cand["vram"]
        if v is None and engine == "llama":
            p = fit.predict_free(model, c)
            if p:
                v = ("derived", p[0])
                r["notes"].append(f"VRAM predicted ({p[1]})")
        r["vram"] = v
        if v:
            r["vram_class"] = fit.classify(v[1], band if v[0] == "derived" else 0.0)
            r["metrics"]["vram_min_free_mib"] = v[1]
        r["runaway"] = cand["runaway"]
        if cand["runaway"] is not None:
            r["metrics"]["runaway_rate"] = cand["runaway"]
        if cand["solve_rate"] is not None:
            r["metrics"]["solve_rate"] = cand["solve_rate"]
        for metric in kld_metrics:
            k = kld_for(kt, cand, metric)
            r.setdefault("kld", {})[metric] = k
            r.setdefault("kld_other", {})[metric] = kld_other_kv(kt, cand, metric)
            if k:
                r["metrics"][metric] = k[0]
        r["ledger"] = ledger_for(store, cand)
        for metric, b in cons["bounds"].items():
            val = r["metrics"].get(metric)
            if metric == "vram_min_free_mib" and r["vram"] and r["vram"][0] == "derived" and r["vram_class"] == "probe-needed":
                val = None
            if val is None:
                r["checks"][metric] = None
                continue
            r["checks"][metric] = (("min" not in b or val >= b["min"]) and ("max" not in b or val <= b["max"]))
        if cons["headless_only"]:
            r["checks"]["headless_only"] = None
        if v and v[0] == "derived" and r["vram_class"] == "probe-needed":
            r["notes"].append(f"VRAM prediction within the probe band (+-{band:.0f} MiB leave-one-out p90)")
        r["verdict"] = "FAIL" if False in r["checks"].values() else "UNKNOWN" if None in r["checks"].values() else "PASS"
        r["fail_nonkld"] = any(v is False and not k.startswith("kld@") for k, v in r["checks"].items())
        r["kld_unverified"] = any(v is None for k, v in r["checks"].items() if k.startswith("kld@"))
        rows.append(r)

    def key(r):
        ks = [r["wall_est"]]
        for t in goal["tiebreak"]:
            val = r["metrics"].get(t)
            ks.append(float("inf") if val is None else val if goalmod.direction(t) == "min" else -val)
        return tuple(ks)

    for r in rows:
        r["key"] = key(r)
    order = {"PASS": 0, "UNKNOWN": 1, "FAIL": 2}
    # unverified KLD sorts after every verified-KLD row of the same verdict; basis classes never interleave
    rows.sort(key=lambda r: (r["cand"]["klass"] != "measured", order[r["verdict"]], r["kld_unverified"], r["key"]))
    _tie_bands(rows)
    return {"rows": rows, "profile": prof, "prof_note": prof_note, "agents": agents, "pareto_metric": pareto_metric,
            "kld_metrics": kld_metrics, "model": model, "hq": hq, "calibration": cal, "excluded": excluded, "vram_band": band}


def _tie_bands(rows):
    """Number the rows of each (basis class, verdict, KLD-verified) run: a row within its tie_pct of the
    run's best wall_est joins that band; the next row beyond it starts a new one."""
    prev, anchor, band = None, None, 0
    for r in rows:
        k = (r["cand"]["klass"], r["verdict"], r["kld_unverified"])
        if k != prev or r["wall_est"] > anchor * (1 + max(r["tie_pct"], tp) / 100):
            band = band + 1 if k == prev else 1
            anchor, tp, prev = r["wall_est"], r["tie_pct"], k
        r["band"] = band


def pareto(rows, metric):
    pts = [r for r in rows if not r["fail_nonkld"] and r["metrics"].get(metric) is not None]
    front = []
    for r in pts:
        a, b = r["wall_est"], r["metrics"][metric]
        dominated = any((o["wall_est"] <= a and o["metrics"][metric] <= b) and (o["wall_est"] < a or o["metrics"][metric] < b)
                        for o in pts)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r["wall_est"])


def _label(r):
    c = r["cand"]
    kv = f"{c['kv_k']}/{c['kv_v']}"
    parts = [c["engine"], c.get("stack_name") or "?", c["model_id"] or "?", kv, f"k{c.get('k')}"]
    if c.get("model_sha256"):
        parts.append("sha:" + c["model_sha256"][:6])
    if c.get("p_min"):
        parts.append(f"p{c['p_min']}")
    if c.get("ub"):
        parts.append(f"ub{c['ub']}")
    if c.get("gpu_util"):
        parts.append(f"u{c['gpu_util']}")
    if r.get("variant_diff"):
        parts.append(f"[{r['variant_diff']}]")
    return " ".join(str(p) for p in parts)


def _kld_cell(r, m):
    k = r.get("kld", {}).get(m)
    if k:
        return f"{k[0]:.4f} m (n={k[1]})"
    other = r.get("kld_other", {}).get(m) or []
    if other:
        return "unverified (other KV: " + ", ".join(f"{kv} {v:.4f} n={n}" for kv, v, n in other) + ")"
    return "missing"


def _row_md(i, r, metrics, tie=True):
    c = r["cand"]
    mw = c.get("measured_wall")
    cells = [str(i), _label(r), f"np{c['np']}/{(c.get('ctx_per_slot') or 0) // 1000}k",
             f"{r['wall_est']:.0f} d (+-{r['tie_pct']:.0f}%)", f"T{r['band']}" if tie else "-",
             f"{mw[0]:.0f} m (n={mw[1]})" if mw else "-", c["speeds"]["basis"]]
    for m in metrics:
        cells.append(_kld_cell(r, m))
    cells.append("; ".join(f"{v:.4f} p" for v, _ in r["ledger"]) + (" (ledger, 8k Q6_K scale; not REFBF)" if r["ledger"] else "") if r["ledger"] else "-")
    v = r["vram"]
    cells.append(f"{v[1]:.0f} {'m' if v[0] == 'measured' else 'd'} {r['vram_class']}" if v else "unknown")
    cells.append("-" if r["runaway"] is None else f"{r['runaway']:.2f} m")
    cells.append("-" if c.get("solve_rate") is None else f"{c['solve_rate']:.2f} m")
    cells.append(r["kept"])
    cells.append(", ".join(k for k, v in r["checks"].items() if v is None) or "-")
    cells.append("; ".join(r["notes"]) or "")
    return "| " + " | ".join(cells) + " |"


def render(store, goal, show_fail=False, limit=None):
    ev = evaluate(store, goal)
    rows = ev["rows"]
    limit = limit or goal["top"]
    kl = ev["kld_metrics"]
    head = ["#", "candidate", "np/ctx per slot", "wall_est s (held-out error)", "tie band", "measured wall s", "basis"] + kl + [
        "ledger KLD (context)", "min free MiB", "runaway", "solve_rate", "prefix", "unverified", "notes"]
    sep = "|" + "---|" * len(head)
    cons = goal["constraints"]
    n = {v: sum(r["verdict"] == v for r in rows) for v in ("PASS", "UNKNOWN", "FAIL")}
    L = [f"# opt report: {goal['name']}", "",
         f"- workload `{goal['workload']}`; primary `{goal['primary']}` (minimize); tie-break {goal['tiebreak'] or 'none'} (lexicographic)",
         f"- constraints: fits {cons['fits'] or '-'}, bounds {cons['bounds'] or '-'}, engine {cons['engine'] or 'any'}, "
         f"model {cons.get('model') or 'any'}, headless_only {cons['headless_only']}",
         f"- {ev['prof_note']}: {ev['profile']['n_campaigns']} source runs; cell tags m = measured, d = derived, p = ledger-prior",
         f"- candidates evaluated {len(rows)}: all constraints satisfied {n['PASS']}, none violated but some unverifiable {n['UNKNOWN']}, violated {n['FAIL']}"]
    if ev["excluded"]:
        L.append(f"- excluded by the model filter: {sum(ev['excluded'].values())} candidates ({', '.join(sorted(map(str, ev['excluded'])))})")
    if ev["agents"] == 1:
        v = W.engine_volume(store, "llama")
        if v:
            L.append(f"- llama candidates: token volume re-based to the median llama 8-task campaign "
                     f"({v['out'] / 1000:.0f}k output, {v['new'] / 1000:.0f}k new prompt tokens, n={v['n']}); call shape is HQ-mined")
    else:
        L.append(f"- two agents: RHO = {W.RHO}, {W.RHO_SOURCE}")
    L.append("")
    for klass, title in (("measured", "measured-basis speeds (campaign-agg / two-agent)"),
                         ("F1", "F1-basis speeds (steady-round depth curves; indicative, never comparable to the list above)")):
        part = [r for r in rows if r["cand"]["klass"] == klass]
        live = [r for r in part if r["verdict"] != "FAIL"]
        L += [f"## Ranked, {title}: {len(live)} not failing", ""]
        if live:
            L += ["| " + " | ".join(head) + " |", sep]
            for i, r in enumerate(live[:limit], 1):
                L.append(_row_md(i, r, kl))
            if len(live) > limit:
                L.append(f"\n{len(live) - limit} more not shown (--limit).")
            L += ["", "Rows sharing a tie band are within the held-out error of the band's best estimate: not ordered. "
                  "`unverified` lists constraints with no data (not a pass); a row with unverified KLD ranks below every verified-KLD row."]
        else:
            L.append("none")
        L.append("")
        if show_fail:
            sel = [r for r in part if r["verdict"] == "FAIL"]
            L += [f"### Violating a constraint, {klass} basis ({len(sel)})", "", "| " + " | ".join(head) + " |", sep]
            for i, r in enumerate(sel[:limit], 1):
                L.append(_row_md(i, r, kl, tie=False))
            L.append("")
    pm = ev["pareto_metric"]
    for klass in ("measured", "F1"):
        front = pareto([r for r in rows if r["cand"]["klass"] == klass], pm)
        L += [f"## Pareto front: wall_est x {pm}, {klass} basis", "",
              "Candidates not failing a non-KLD constraint; KLD-constraint failures stay on the front (exploratory axis).", "",
              f"| wall_est s | {pm} | candidate | verdict | basis |", "|---|---|---|---|---|"]
        for r in front:
            k = r["kld"][pm]
            L.append(f"| {r['wall_est']:.0f} d | {k[0]:.4f} m | {_label(r)} np{r['cand']['np']} | {r['verdict']} | {r['cand']['speeds']['basis']} |")
        if not front:
            L.append("| - | - | no candidate has both axes | - | - |")
        L.append("")
    cal = ev["calibration"]
    named = {k.split(" ")[0].rstrip(":"): v for k, v in cal["tiers"].items()}
    fmt = lambda v: "n/a" if not v else f"median {v['median_abs_pct']:.0f}%, max {v['max_abs_pct']:.0f}% (n={v['n']})"
    rc = lambda k: "n/a" if cal["rank_corr"].get(k) is None else f"{cal['rank_corr'][k]:.2f}"
    L += ["## Caveats", "",
          f"- wall_est error against recorded campaign walls (`opt calibrate`). In-sample structure checks: A {fmt(named.get('A'))}, B0 {fmt(named.get('B0'))}. "
          f"Out of sample: B held-out HQ {fmt(named.get('B'))} (Spearman {rc('B')}); llama C with the run's own speeds {fmt(named.get('C'))} (Spearman {rc('C')}); "
          f"llama D with F1 curves {fmt(named.get('D'))} (Spearman {rc('D')}). Tie bands use the held-out medians: "
          + ", ".join(f"{k} {v:.0f}%" if v is not None else f"{k} default {W.DEFAULT_TIE_PCT:.0f}%" for k, v in cal["held_out_pct"].items()) + ".",
          "- basis campaign-agg / two-agent = measured aggregate speeds of recorded runs; basis F1 = steady-round depth curves (smoke gain is not agent gain; the F1 basis overestimates wall). The two bases are ranked in separate lists.",
          "- np2 F1 speeds are per-stream under concurrent load: pessimistic for a single agent.",
          f"- RHO (two-agent stall fraction) is {W.RHO_SOURCE}.",
          "- KLD joins on model id (version + recipe) + KV type + reference; a KLD measured at another KV is shown as unverified context, never used. "
          "ledger KLD is the PLAN.md number on the 8k Q6_K scale (vLLM lane for HQ), shown as context only.",
          f"- derived VRAM carries the leave-one-out p90 error band (+-{ev['vram_band']:.0f} MiB): a prediction inside it needs a probe.",
          "- the HQ prefix kept/lost rule (0.302/0.315 of the pool) is in-sample."]
    return "\n".join(L) + "\n", ev
