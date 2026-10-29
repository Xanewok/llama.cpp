"""F0 feasibility models fitted from recorded boots: llama VRAM headroom and the HQ KV pool.

llama:  free_MiB(gpu) = a[group, gpu] - b[group] * ctx_total - c * k
        group = (engine, quant, kv, build family, ub). ctx_total = ctx * 1 (llama -c is already
        the total over slots). np and ctx are confounded in the data (~100k tokens per slot), so
        b is an effective slope over total context, not a per-slot cost.
HQ:     pool(ckpt, k, util) = pool0[ckpt] - POOL_PER_K * (k - 5) + POOL_PER_UTIL * (util - 0.86)
        POOL_PER_K is fitted from recorded boots; POOL_PER_UTIL is a ledger-prior (PLAN.md:642),
        every recorded boot is at util 0.86, so it is not testable from the DB.
"""
import statistics
from collections import defaultdict

from .store import Store

REJECT_MIB = 200
PROBE_MIB = 500
RESID_FLOOR_MIB = 100
RESID_FRAC = 0.05
POOL_PER_UTIL = 860000
POOL_PER_UTIL_SRC = "ledger-prior: PLAN.md:642 (8.6k tokens per 0.01 GPU_UTIL)"
PREFIX_KEPT_FRAC = 0.302  # IN-SAMPLE: thresholds bracket the same probes they are checked against; observed brackets at pool 199033: depth 60000 kept, 62747 lost (PLAN: kept <~0.30, lost >~0.315)
PREFIX_LOST_FRAC = 0.315


def group_key(cfg):
    """Tensor split moves free memory by GB between GPUs, so it identifies a group like ub does."""
    fl = cfg.get("canon", {}).get("flags", {})
    return (cfg["engine"], cfg.get("model_id") or cfg.get("model_label"), f"{cfg.get('kv_k')}/{cfg.get('kv_v')}",
            cfg.get("stack_name"), cfg.get("ub"), f"{fl.get('sm', '-')}:{fl.get('tensor-split', '-')}")


def _median(xs):
    return statistics.median(xs) if xs else None


def collect(store):
    """{(group, gpu, ctx, np, k, config_id): [min_free per boot]} for measured llama boots."""
    cfgs = store.tables["configs"]
    pts = defaultdict(list)
    for m in store.rows("measurements"):
        if m["metric"] != "vram_min_free_mib" or m["status"] != "measured":
            continue
        c = cfgs.get(m["config_id"])
        if not c or c["engine"] != "llama" or not c.get("k"):
            continue
        d = m["dims"]
        if not d.get("ctx") or d.get("gpu") is None:
            continue
        pts[(group_key(c), d["gpu"], d["ctx"], d.get("np"), c["k"], m["config_id"])].append(m["value"])
    return pts


MIN_CTX_SPAN = 0.2  # a slope from contexts closer than this is noise (e.g. 198144 vs 204800)
KV_BYTES = {"f16": 2.0, "bf16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625, "fp8": 1.0}


def _kv_bytes(kv):
    k, _, v = kv.partition("/")
    return (KV_BYTES.get(k, 2.0) + KV_BYTES.get(v or k, 2.0)) / 2


def _ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if n < 2 or sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


def _cells(store):
    cells = defaultdict(list)
    for (g, gpu, ctx, np_, k, cid), vals in collect(store).items():
        cells[(g, gpu, ctx, np_, k)].append(_median(vals))
    return {kk: _median(v) for kk, v in cells.items()}


def _fit_cells(cells):
    by_k = defaultdict(dict)
    for (g, gpu, ctx, np_, k), v in cells.items():
        by_k[(g, gpu, ctx, np_)][k] = v
    slopes = []
    for ks in by_k.values():
        if len(ks) >= 2:
            o = _ols(list(ks), list(ks.values()))
            if o:
                slopes.append(-o[0])
    c = _median(slopes) or 0.0

    work = defaultdict(list)
    for (g, gpu, ctx, np_, k), v in cells.items():
        work[(g, gpu)].append((ctx, v + c * k, np_, k, v))
    b_by_group = defaultdict(list)
    for (g, gpu), p in work.items():
        ctxs = [x[0] for x in p]
        o = _ols(ctxs, [x[1] for x in p]) if max(ctxs) >= (1 + MIN_CTX_SPAN) * min(ctxs) else None
        if o:
            b_by_group[g].append(-o[0])
    b_group = {g: _median(v) for g, v in b_by_group.items()}
    by_kv = defaultdict(list)
    for g, b in b_group.items():
        by_kv[(g[0], g[2])].append(b)
    b_kv = {k: _median(v) for k, v in by_kv.items()}
    b_unit = _median([b / _kv_bytes(g[2]) for g, b in b_group.items()])

    groups, points = {}, []
    for (g, gpu), p in work.items():
        if g in b_group:
            b, bsrc = b_group[g], "fitted"
        elif (g[0], g[2]) in b_kv:
            b, bsrc = b_kv[(g[0], g[2])], "borrowed from same kv"
        elif b_unit is not None:
            b, bsrc = b_unit * _kv_bytes(g[2]), "scaled by KV bytes"
        else:
            b = None
        if b is None:
            continue
        a = _median([y + b * x for x, y, *_ in p])
        gr = groups.setdefault(g, {"b": b, "b_source": bsrc, "a": {}, "n": 0})
        gr["a"][gpu] = a
        gr["n"] += len(p)
        for ctx, y, np_, k, free in p:
            pred = a - b * ctx - c * k
            points.append({"group": g, "gpu": gpu, "ctx": ctx, "np": np_, "k": k, "free": free, "pred": pred,
                           "resid": free - pred})
    return {"c": c, "c_n": len(slopes), "groups": groups, "points": points, "b_unit": b_unit}


def _predict_cell(model, g, gpu, ctx, k):
    """(pred, basis): own group when it has this GPU, else mean over other build families of the same quant+kv+ub."""
    gr = model["groups"].get(g)
    if gr and gpu in gr["a"]:
        return gr["a"][gpu] - gr["b"] * ctx - model["c"] * k, "group"
    near = [v for kk, v in model["groups"].items() if kk[:3] == g[:3] and kk[4:] == g[4:] and gpu in v["a"]]
    if not near:
        return None, None
    a = statistics.mean(v["a"][gpu] for v in near)
    b = statistics.mean(v["b"] for v in near)
    return a - b * ctx - model["c"] * k, "other build family"


def fit_llama(store):
    """Fit on all recorded boots; `loo` holds leave-one-cell-out residuals (the out-of-sample error)."""
    cells = _cells(store)
    model = _fit_cells(cells)
    loo = []
    for cell, free in cells.items():
        m = _fit_cells({k: v for k, v in cells.items() if k != cell})
        g, gpu, ctx, np_, k = cell
        pred, basis = _predict_cell(m, g, gpu, ctx, k)
        if pred is not None:
            loo.append({"group": g, "gpu": gpu, "ctx": ctx, "np": np_, "k": k, "free": free, "pred": pred,
                        "resid": free - pred, "basis": basis})
    model["loo"] = loo
    model["cells"] = len(cells)
    return model


def stale(point):
    return abs(point["resid"]) > max(RESID_FRAC * abs(point["free"]), RESID_FLOOR_MIB)


def classify(free_mib, band=0.0):
    """A measured value classifies as is. A predicted one carries its leave-one-out error band: it is
    `reject` only if even the optimistic end is below REJECT_MIB and `fits` only if the pessimistic end clears
    PROBE_MIB, so a prediction inside the error band always needs a probe."""
    if free_mib + band < REJECT_MIB:
        return "reject"
    if free_mib - band < PROBE_MIB:
        return "probe-needed"
    return "fits"


def loo_band(model):
    """p90 of |leave-one-cell-out residual| in MiB (floor RESID_FLOOR_MIB); the width of the derived-VRAM band."""
    ar = sorted(abs(p["resid"]) for p in model.get("loo", []))
    return max(ar[int(0.9 * (len(ar) - 1))], RESID_FLOOR_MIB) if ar else float(PROBE_MIB)


def predict_free(model, cfg, ctx=None, np_=None):
    """Worst-GPU predicted min-free (MiB) and basis for a config record, or None without a model."""
    g = group_key(cfg)
    gr = model["groups"].get(g)
    basis = "group"
    if gr is None:
        near = [(k, v) for k, v in model["groups"].items() if k[:3] == g[:3] and k[4:] == g[4:]]
        if not near:
            return None
        gr, basis = near[0][1], f"nearest stack ({near[0][0][3]})"
    ctx = ctx or cfg.get("ctx")
    if not ctx or not cfg.get("k"):
        return None
    free = min(a - gr["b"] * ctx - model["c"] * cfg["k"] for a in gr["a"].values())
    return free, basis + ("" if gr["b_source"] == "fitted" else f"; b {gr['b_source']}")


def fit_hq(store):
    cfgs = store.tables["configs"]
    pools, seen = [], set()
    for m in store.rows("measurements"):
        if m["metric"] != "kv_pool_tokens" or m["status"] != "measured":
            continue
        c = cfgs.get(m["config_id"])
        if not c or c["engine"] != "hq" or not c.get("k") or not c.get("gpu_util"):
            continue
        # fit.json and prefix.json of one boot both report its pool: one boot is one observation
        dedupe = (m["item_id"], m["config_id"], m["value"])
        if dedupe in seen:
            continue
        seen.add(dedupe)
        pools.append({"item": m["item_id"], "ckpt": c.get("model_id") or c["model_label"], "k": c["k"], "util": c["gpu_util"],
                      "np": c.get("np"), "maxlen": c.get("ctx"), "pool": m["value"], "config_id": m["config_id"]})
    by_ck_k = defaultdict(lambda: defaultdict(list))
    for p in pools:
        if p["util"] == 0.86 and p["np"] == 1:
            by_ck_k[p["ckpt"]][p["k"]].append(p["pool"])
    slopes = []
    for ks in by_ck_k.values():
        m = {k: _median(v) for k, v in ks.items()}
        if len(m) >= 2:
            o = _ols(list(m), list(m.values()))
            slopes.append(-o[0])
    per_k = _median(slopes)
    prior = False
    if per_k is None:
        per_k, prior = 4355.0, True
    pool0 = {ck: _median([p["pool"] + per_k * (p["k"] - 5) for p in pools if p["ckpt"] == ck and p["np"] == 1 and p["util"] == 0.86])
             for ck in by_ck_k}
    rows = []
    for p in pools:
        if p["ckpt"] in pool0:
            pred = hq_pool(pool0, per_k, p["ckpt"], p["k"], p["util"])
            rows.append({**p, "pred": pred, "resid": p["pool"] - pred})
    return {"per_k": per_k, "per_k_prior": prior, "per_k_n": len(slopes), "pool0": pool0, "rows": rows, "pools": pools}


def hq_pool(pool0, per_k, ckpt, k, util):
    return pool0[ckpt] - per_k * (k - 5) + POOL_PER_UTIL * (util - 0.86)


def hq_prefix_state(pool, depth, unit=1):
    """Depth is rounded up to whole prefix-cache blocks (`prefix-match-unit`) before comparing with the pool."""
    r = -(-depth // unit) * unit / pool
    return "kept" if r <= PREFIX_KEPT_FRAC else "lost" if r >= PREFIX_LOST_FRAC else "boundary"


def check_prefix_rule(store, hq):
    """Compare the kept/lost rule with recorded prefix_kept_frac rows joined to the pool of the same boot."""
    pool_of = {r["item"]: r["pool"] for r in hq["pools"]}
    out = []
    for m in store.rows("measurements"):
        if m["metric"] != "prefix_kept_frac" or m["status"] != "measured" or m["item_id"] not in pool_of:
            continue
        depth = m["dims"].get("depth")
        if not depth:
            continue
        obs = "kept" if m["value"] >= 1.0 else "lost" if m["value"] < 1.0 else "?"
        out.append({"item": m["item_id"], "depth": depth, "pool": pool_of[m["item_id"]],
                    "ratio": depth / pool_of[m["item_id"]], "obs_frac": m["value"], "rule": hq_prefix_state(pool_of[m["item_id"]], depth),
                    "obs": obs})
    return out


def report(store=None):
    store = store or Store()
    lm, hq = fit_llama(store), fit_hq(store)
    lines = ["## llama VRAM fit: free = a[group,gpu] - b*ctx - c*k", ""]
    lines.append(f"c = {lm['c']:.1f} MiB per draft token (median of {lm['c_n']} within-cell slopes); "
                 f"b per KV byte-factor = {lm['b_unit'] or 0:.4f} MiB/token")
    lines.append("")
    lines.append("| engine | quant | kv | stack | ub | split | n | b MiB/tok | b source | a(gpu0) | a(gpu1) | max abs resid MiB | rms MiB | stale pts |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    stale_pts = []
    by_g = defaultdict(list)
    for p in lm["points"]:
        by_g[p["group"]].append(p)
    for g, gr in sorted(lm["groups"].items(), key=lambda x: str(x[0])):
        ps = by_g[g]
        mx = max(abs(p["resid"]) for p in ps)
        rms = (sum(p["resid"] ** 2 for p in ps) / len(ps)) ** 0.5
        st_n = sum(stale(p) for p in ps)
        stale_pts += [p for p in ps if stale(p)]
        a = gr["a"]
        lines.append(f"| {g[0]} | {g[1]} | {g[2]} | {g[3]} | {g[4]} | {g[5]} | {gr['n']} | {gr['b']:.4f} | {gr['b_source']} | "
                     f"{a.get(0, float('nan')):.0f} | {a.get(1, float('nan')):.0f} | {mx:.0f} | {rms:.0f} | {st_n} |")
    loo = lm["loo"]
    lines += ["", f"in-sample: {len(lm['points'])} points, stale {len(stale_pts)} (> max(5% of free, {RESID_FLOOR_MIB} MiB))"]
    if loo:
        ar = sorted(abs(p["resid"]) for p in loo)
        lstale = [p for p in loo if stale(p)]
        lines += ["", f"leave-one-boot-cell-out: {len(loo)}/{lm['cells']} cells predictable; median |resid| {statistics.median(ar):.0f} MiB, "
                  f"p90 {ar[int(0.9 * (len(ar) - 1))]:.0f}, max {ar[-1]:.0f}; stale (> max(5% of free, {RESID_FLOOR_MIB})): {len(lstale)}",
                  *[f"- basis {bs}: {len(x)} cells, median |resid| {statistics.median(abs(p['resid']) for p in x):.0f} MiB, "
                    f"max {max(abs(p['resid']) for p in x):.0f} MiB"
                    for bs in sorted({p['basis'] for p in loo}) for x in [[p for p in loo if p['basis'] == bs]]],
                  "", "| stale LOO point | gpu | ctx | k | measured | predicted | resid | basis |", "|---|---|---|---|---|---|---|---|"]
        for p in sorted(lstale, key=lambda p: -abs(p["resid"]))[:12]:
            lines.append(f"| {p['group'][1]}/{p['group'][2]}/{p['group'][3]}/ub{p['group'][4]}/{p['group'][5]} | {p['gpu']} | {p['ctx']} | {p['k']} | "
                         f"{p['free']:.0f} | {p['pred']:.0f} | {p['resid']:+.0f} | {p['basis']} |")
        stale_pts = lstale
    lines += ["", "## HQ pool", "",
              f"pool = pool0[ckpt] - {hq['per_k']:.0f}*(k-5) + {POOL_PER_UTIL}*(util-0.86); per-k slope "
              f"{'ledger-prior' if hq['per_k_prior'] else 'fitted from ' + str(hq['per_k_n']) + ' checkpoint(s)'}; "
              f"util slope {POOL_PER_UTIL_SRC}", "",
              "| ckpt | pool0 (k=5, util 0.86) | n boots | max abs resid tokens |", "|---|---|---|---|"]
    for ck, v in sorted(hq["pool0"].items()):
        rs = [r for r in hq["rows"] if r["ckpt"] == ck and r["np"] == 1]
        lines.append(f"| {ck} | {v:.0f} | {len(rs)} | {max(abs(r['resid']) for r in rs):.0f} |")
    npr = [r for r in hq["rows"] if r["np"] != 1]
    if npr:
        seen = sorted({(r["ckpt"], r["maxlen"], r["pool"], round(r["pred"]), round(r["resid"])) for r in npr})
        lines += ["", "np>1 boots (pool depends on max-model-len; not in the fit): " +
                  ", ".join(f"{a} maxlen {b} pool {c} (rule {d}, {e:+d})" for a, b, c, d, e in seen)]
    chk = check_prefix_rule(store, hq)
    if chk:
        ok = sum((c["rule"] == "kept") == (c["obs"] == "kept") for c in chk if c["rule"] != "boundary")
        tot = sum(c["rule"] != "boundary" for c in chk)
        lines += ["", f"prefix rule (kept <= {PREFIX_KEPT_FRAC} pool, lost >= {PREFIX_LOST_FRAC} pool) is IN-SAMPLE: the thresholds "
                  f"were set from these same probes, so {ok}/{tot} agreeing is not a validation ({len(chk) - tot} on the boundary)"]
    return "\n".join(lines) + "\n", {"llama": lm, "hq": hq, "stale": stale_pts, "prefix": chk}
