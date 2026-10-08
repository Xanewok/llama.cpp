"""Report boot records; --record writes gated notes and a ledger draft. Trace bodies and summary.json are never inputs."""
import argparse
import json
import math
import re
import statistics
import sys
import tomllib
from collections import Counter
from pathlib import Path

from . import depth, lib

AA_REFERENCE = Path(__file__).resolve().parents[1] / ".cache/harness-report-aa.json"


def load(path):
    return json.loads(path.read_text())


def classify(delta, se, floor):
    if se is None or abs(delta) <= max(2 * se, floor):
        return "null"
    return "faster" if delta < 0 else "slower"


def finish_report(result):
    result["exit_code"] = 6 if result["hard_gates"] else 5 if result["insufficient"] else 0
    if result["exit_code"]:
        result["class"] = "provisional"
    return result


def counter_match(by_arm, metric="steady_ms_per_round_D"):
    """np2: ms/round is comparable only between boots whose per-stream (accepted, drafted) counters are equal; a stream
    that collapses to another outcome changes the round mix. Delta = boot-weighted mean over shared signatures of the
    later arm minus the first, each side a mean within the signature (streams averaged per boot)."""
    arms = list(by_arm)
    groups = {}
    for a, runs in by_arm.items():
        for r in runs:
            sig = " | ".join(f"{t['draft_n_accepted']}/{t['draft_n']}" for t in (s["timings"] for s in r["streams"]))
            groups.setdefault(sig, {x: [] for x in arms})[a].append(statistics.mean(s[metric] for s in r["streams"]))
    shared = {sig: g for sig, g in groups.items() if all(g.values())}
    weight = {sig: sum(map(len, g.values())) for sig, g in shared.items()}
    delta = sum(w * (statistics.mean(shared[sig][arms[1]]) - statistics.mean(shared[sig][arms[0]])) for sig, w in weight.items()) / sum(weight.values()) if shared else None
    return {"groups": {sig: {a: len(v) for a, v in g.items()} for sig, g in groups.items()},
            "matched": {"delta_D": delta, "boots": {a: sum(len(g[a]) for g in shared.values()) for a in arms}, "signatures": list(shared)}}


def depth_report(root, aa=False):
    pins = load(root / "pins.json")
    out = {"shape": "depth", "run_dir": str(root), "plan": pins["order"], "depths": {},
           "hard_gates": [], "warnings": [], "insufficient": [], "aa": aa}
    hard, warn, missing = out["hard_gates"], out["warnings"], out["insufficient"]
    experiment = Path(pins.get("experiment", ""))
    if not experiment.is_file():
        missing.append("experiment unavailable: identity mode cannot be established")
        d = {}
    else:
        d = tomllib.loads(experiment.read_text())["depth"]
        digest = pins.get("pins", {}).get(str(experiment))
        if digest and lib.file_sha(experiment) != digest:
            hard.append("experiment differs from pins.json")
    n = pins.get("streams", d.get("streams", 1))
    out["streams"] = n
    mode = d.get("identity", "unknown")
    out["identity"] = mode
    if mode == "off":
        reason = d.get("identity_off_reason")
        if n == 1 or not reason:
            hard.append("identity off requires np2 and identity_off_reason")
        warn.append(f"identity off: {reason or 'no reason'}")
    elif mode not in ("cross-arm", "within-arm", "unknown"):
        hard.append(f"unknown identity mode {mode}")
    if pins.get("texts", d.get("texts", 1)) > 1:
        missing.append("texts > 1: depth records hold N texts per boot; this report reads one request per boot")
        return finish_report(out)
    order = pins["order"]
    arms = list(dict.fromkeys(order))
    if len(arms) != 2:
        missing.append("exactly two arms required for deltas")
        return finish_report(out)
    out["arms"] = arms
    counts = Counter(order)
    if len(set(counts.values())) != 1 or any(set(order[i:i + 2]) != set(arms) for i in range(0, len(order), 2)):
        hard.append("plan is not balanced/interleaved (each adjacent pair must contain both arms)")
    if min(counts.values()) < 2:
        hard.append("fewer than two boots per arm; uncertainty G")
    boots = sorted((p for p in root.iterdir() if p.is_dir() and p.name.split("-", 1)[0].isdigit()),
                   key=lambda p: int(p.name.split("-", 1)[0]))
    if [p.name for p in boots] != [f"{i}-{a}" for i, a in enumerate(order)]:
        hard.append("stored boots do not match pins.json order")
    results = {v: {a: [] for a in arms} for v in pins["depths"]}
    identities, trace, vram, envs = {}, {a: [] for a in arms}, {}, {a: [] for a in arms}
    for boot in boots:
        arm = boot.name.split("-", 1)[1]
        if arm not in arms:
            continue
        verdict = boot / "verdict.json"
        if not verdict.exists():
            missing.append(f"{boot.name}: verdict.json missing")
        else:
            v = load(verdict)
            if v.get("failures") or any(v[k] for k in v if k != "failures"):
                hard.append(f"{boot.name}: verdict non-empty: {v}")
        env = boot / "env.json"
        if env.exists():
            envs[arm].append(load(env))
        else:
            missing.append(f"{boot.name}: env.json missing")
        vr = boot / "vram.json"
        if vr.exists() and load(vr):
            vram[boot.name] = load(vr)
        else:
            warn.append(f"{boot.name}: vram.json missing or empty")
        total_rounds = 0
        for dep in pins["depths"]:
            try:
                record = load(boot / f"depth-{dep}.json")
                records = record["streams"] if n > 1 else [record]
                if len(records) != n:
                    raise lib.Fail("wrong stream count", "integrity")
                k = pins["k"][arm] if isinstance(pins["k"], dict) else pins["k"]
                window = depth.overlap(records, 0) if n > 1 else None
                minimum = pins["steady_minimum"]
                stream_results, ident = [], []
                for s, r in enumerate(records):
                    bounds = window["window_ns"] if window else None
                    try:
                        res = lib.steady_intervals(r, minimum, k, bounds)
                    except lib.Fail as error:
                        if not str(error).startswith("insufficient steady"):
                            raise
                        hard.append(f"{boot.name}/depth-{dep}/s{s}: {error}")
                        res = lib.steady_intervals(r, 0, k, bounds)
                    identity = lib.token_identity(r)
                    if len(identity[0]) != res["timings"]["predicted_n"]:
                        raise lib.Fail("missing final token identity", "insufficient_data")
                    if res["timings"]["predicted_n"] < d.get("min_predicted", 0):
                        missing.append(f"{boot.name}/depth-{dep}/s{s}: too few predicted tokens")
                    ident.append(identity)
                    t = res["timings"]
                    total_rounds += t["draft_n"] / k / n
                    split = lib.request_split(t, k, res["steady_ms_per_round_D"])
                    if n > 1:
                        split["prompt_ms_M"] = depth.final_timings(record["prefill"][s])["prompt_ms"]
                    if not all(math.isfinite(v) for v in (*split.values(), res["steady_ms_per_round_D"])):
                        raise lib.Fail("invalid timing", "integrity")
                    stream_results.append(res | split | {"predicted_ms_M": t["predicted_ms"]})
                key = (dep, arm) if mode == "within-arm" else dep
                if mode in ("within-arm", "cross-arm"):
                    if key in identities and identities[key] != ident:
                        hard.append(f"{boot.name}/depth-{dep}: {mode} identity failed (tokens or counters)")
                    identities.setdefault(key, ident)
                tok_s = window["agg_tok_s"] if window else 1000 * stream_results[0]["timings"]["predicted_n"] / stream_results[0]["timings"]["predicted_ms"]
                results[dep][arm].append({"arm": arm, "boot": boot.name, "streams": stream_results, "agg_tok_s_M": tok_s,
                                         "overlap_frac_D": window["overlap_frac"] if window else None})
                if window and window["overlap_frac"] < pins.get("min_overlap", 0.8):
                    warn.append(f"{boot.name}/depth-{dep}: overlap {window['overlap_frac']:.3f} below minimum")
            except (OSError, ValueError, KeyError, TypeError, statistics.StatisticsError) as error:
                where = f"{boot.name}/depth-{dep}: {error}"
                if isinstance(error, lib.Fail) and error.cls in ("integrity", "deterministic"):
                    hard.append(where)
                else:
                    missing.append(where)
        path = boot / "round-trace.jsonl"
        trace[arm].append({"boot": boot.name, "traced": path.exists(),
                           "bytes_per_round_D": lib.trace_bytes_per_round(boot, total_rounds)})
        if env.exists() and bool(load(env).get("LLAMA_ROUND_TRACE")) != path.exists():
            hard.append(f"{boot.name}: trace file/env mismatch")
    out["trace"] = trace
    traced = {t["traced"] for ts in trace.values() for t in ts}
    if len(traced) > 1:
        hard.append("traced/untraced mix")
    if traced == {True}:
        values = {a: [t["bytes_per_round_D"] for t in ts if t["bytes_per_round_D"] is not None] for a, ts in trace.items()}
        if all(values.values()):
            out["trace_kb_per_round_D"] = {a: statistics.mean(vs) / 1000 for a, vs in values.items()}
            delta = out["trace_kb_per_round_D"][arms[1]] - out["trace_kb_per_round_D"][arms[0]]
            out["trace_bias_ms_D"] = delta * lib.TRACE_MS_PER_KB
            if abs(delta) > lib.TRACE_DELTA_KB_MAX:
                hard.append(f"trace bytes/round imbalance {delta:+.1f} KB (D); writer bias {out['trace_bias_ms_D']:+.3f} ms/round (D)")
        else:
            missing.append("trace bytes/round unavailable")
    # Paths differ by boot; compare the remaining recorded environment within each arm.
    for arm, records in envs.items():
        clean = [{k: v for k, v in e.items() if k != "LLAMA_ROUND_TRACE"} for e in records]
        if clean and any(e != clean[0] for e in clean[1:]):
            hard.append(f"{arm}: environment changes between boots")
    out["vram_min_free_mib_M"] = min((v for b in vram.values() for v in b.values()), default=None)
    out["vram_boots_M"] = vram
    if out["vram_min_free_mib_M"] is not None and out["vram_min_free_mib_M"] < 1536:
        warn.append("VRAM min-free below 1.5 GiB")
    out["xid_count_M"] = sum(len(re.findall(r"\bXid\b", p.read_text(errors="replace"))) for p in root.rglob("journal.txt"))
    shape = f"np{n}-{'traced' if traced == {True} else 'untraced'}"
    reference = load(AA_REFERENCE).get(shape) if AA_REFERENCE.exists() else None
    floor = reference["sd_ms_M"] if reference else lib.AA_BOOT_SD_MS.get((n, traced == {True}), 0.10)
    out["aa_reference"] = reference or {"sd_ms_G": floor}
    out["aa_floor_ms"] = floor
    classes = []
    for dep, by_arm in results.items():
        if any(len(by_arm[a]) != counts[a] for a in arms):
            missing.append(f"depth {dep}: incomplete boot records")
        if not all(by_arm.values()):
            continue
        row = {"metrics": {}}
        if n > 1:
            row["overlap_frac_min_D"] = {a: min(r["overlap_frac_D"] for r in runs) for a, runs in by_arm.items()}
        for metric in ("steady_ms_per_round_D", "predicted_ms_M", "prompt_ms_M", "fixed_ms_D"):
            row["metrics"][metric] = lib.boot_delta(by_arm, n, metric) if n > 1 else lib.boot_delta(
                {a: [r["streams"][0] for r in runs] for a, runs in by_arm.items()}, 1, metric)
        row["acceptance_M"] = {a: sum(s["timings"]["draft_n_accepted"] for r in runs for s in r["streams"]) /
                               sum(s["timings"]["draft_n"] for r in runs for s in r["streams"]) for a, runs in by_arm.items()}
        low = d.get("min_acceptance", 0)
        for a, value in row["acceptance_M"].items():
            if value < low:
                missing.append(f"depth {dep}: arm {a} acceptance {value:.3f} < min_acceptance {low}")
        if n > 1:
            row |= {"counter_groups": (match := counter_match(by_arm))["groups"], "counter_matched": match["matched"]}
            for sig, boots_by_arm in match["groups"].items():
                if not all(boots_by_arm.values()):
                    warn.append(f"depth {dep}: counter signature {sig} only in arm {next(a for a, c in boots_by_arm.items() if c)}")
        row["acceptance_equal"] = math.isclose(*row["acceptance_M"].values(), rel_tol=0, abs_tol=1e-12)
        row["agg_tok_s_M"] = {a: statistics.mean(r["agg_tok_s_M"] for r in runs) for a, runs in by_arm.items()} if row["acceptance_equal"] else None
        if not row["acceptance_equal"]:
            warn.append(f"depth {dep}: acceptance differs; tok/s n/a")
        steady = row["metrics"]["steady_ms_per_round_D"]
        row["class"] = classify(steady["delta"], steady["se"], floor)
        if row["class"] == "null":
            warn.append(f"depth {dep}: effect below 2 SE or A/A reference; class: null")
        classes.append(row["class"])
        prompt = row["metrics"]["prompt_ms_M"]
        base_prompt = statistics.mean(prompt["arms"][arms[0]])
        row["prompt_delta_pct_D"] = 100 * prompt["delta"] / base_prompt if base_prompt else None
        if row["prompt_delta_pct_D"] is not None and abs(row["prompt_delta_pct_D"]) > 1:
            warn.append(f"depth {dep}: prefill delta {row['prompt_delta_pct_D']:+.2f}% (D)")
        if abs(row["metrics"]["fixed_ms_D"]["delta"]) > 5:
            warn.append(f"depth {dep}: fixed/request delta {row['metrics']['fixed_ms_D']['delta']:+.1f} ms (D)")
        expected = 0 if aa else d.get("expected", {}).get(str(dep))
        if expected is not None and steady["se"] is not None and abs(steady["delta"] - expected) > 2 * steady["se"]:
            warn.append(f"depth {dep}: measured minus expected outside 2 SE")
        if aa:
            row["aa_boot_sd_ms_M"] = {a: statistics.stdev(vs) if len(vs) > 1 else None for a, vs in steady["arms"].items()}
        out["depths"][str(dep)] = row
    effects = set(classes) - {"null"}
    out["class"] = next(iter(effects)) if len(effects) == 1 else "null"
    if len(effects) > 1:
        warn.append("effect changes sign across depths; inspect per-depth classes")
    if not classes:
        missing.append("no depth deltas")
    finish_report(out)
    if out["exit_code"]:
        for row in out["depths"].values():
            row["class"] = "provisional"
    if aa and not out["exit_code"]:
        if any("measured minus expected" in w for w in warn):
            warn.append("A/A reference unchanged: effect exceeds 2 SE")
        else:
            references = load(AA_REFERENCE) if AA_REFERENCE.exists() else {}
            references[shape] = {"sd_ms_M": max(v for row in out["depths"].values() for v in row["aa_boot_sd_ms_M"].values()),
                                 "run_dir": str(root)}
            AA_REFERENCE.parent.mkdir(parents=True, exist_ok=True)
            AA_REFERENCE.write_text(json.dumps(references, indent=2) + "\n")
            out["aa_reference_refreshed"] = references[shape]
    return out


def perf_summary(records):
    samples = {}
    for record in records:
        for row in record["rows"]:
            match = re.search(r"type_a=(\w+),type_b=f32,m=(\d+),n=(\d+),k=(\d+),bs=\[1,1\],nr=\[1,1\]", row["op_params"])
            if not match:
                continue
            typ, m, n, k = match.groups()
            if not row["supported"] or not row["passed"] or not math.isfinite(row["time_us"]) or row["time_us"] <= 0 or row["n_runs"] < 1:
                raise ValueError(f"invalid perf row: {row['op_params']}")
            key = f"{'cold' if record['cold'] else 'hot'}:{typ}:{m}:{k}:{n}"
            samples.setdefault(key, {}).setdefault(record["mode"], []).append(row["time_us"])
    result = {key: {mode + "_us_M": statistics.median(vs) for mode, vs in modes.items()} for key, modes in samples.items()}
    for values in result.values():
        if "base_us_M" in values and "patch_us_M" in values:
            values["delta_us_D"] = values["patch_us_M"] - values["base_us_M"]
    return result


def perf_report(root):
    out = {"shape": "perf", "run_dir": str(root), "hard_gates": [], "warnings": [], "insufficient": [], "class": "null"}
    try:
        samples = perf_summary(load(root / "results.json"))
    except ValueError as error:
        out["hard_gates"].append(str(error))
        samples = {}
    out["cases"] = samples
    shapes = {key.split(":", 1)[1] for key in samples}
    for shape in sorted(shapes):
        for heat in ("hot", "cold"):
            if f"{heat}:{shape}" not in samples:
                out["insufficient"].append(f"{heat}:{shape}: missing hot/cold case")
    groups, slopes, ranked = {}, {}, []
    for key, values in samples.items():
        heat, typ, m, k, n = key.split(":")
        group = groups.setdefault((typ, int(n)), {"type": typ, "columns": int(n)})
        for mode in ("base", "patch"):
            if mode + "_us_M" in values:
                group.setdefault(heat + "_" + mode + "_us_M", []).append(values[mode + "_us_M"])
                slopes.setdefault((heat, typ, m, k, mode), []).append((int(n), values[mode + "_us_M"]))
        if "base_us_M" in values and "patch_us_M" in values:
            ratio = values["patch_us_M"] / values["base_us_M"]
            group.setdefault(heat + "_ratios", []).append(ratio)
            ranked.append({"key": key, "delta_pct_D": 100 * (ratio - 1)})
        else:
            out["insufficient"].append(f"{key}: insufficient data for deltas (missing base or patch)")
    for group in groups.values():
        for heat in ("hot", "cold"):
            for mode in ("base", "patch"):
                values = group.get(heat + "_" + mode + "_us_M")
                if values:
                    group[heat + "_" + mode + "_us_M"] = statistics.geometric_mean(values)
            ratios = group.pop(heat + "_ratios", [])
            group[heat + "_delta_pct_D"] = 100 * (statistics.geometric_mean(ratios) - 1) if ratios else None
    out["table"] = [groups[key] for key in sorted(groups)]
    slope_groups = {}
    for (heat, typ, m, k, mode), points in slopes.items():
        points = sorted((n, t) for n, t in points if 6 <= n <= 12)
        if len(points) >= 2:
            # Endpoint slope within each shape; averaging shapes never pairs different (m,k).
            value = (points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0])
            slope_groups.setdefault(f"{heat}:{typ}:{mode}", []).append(value)
    out["slope_us_per_column_D"] = {key: statistics.mean(vs) for key, vs in slope_groups.items()}
    out["worst"] = sorted(ranked, key=lambda r: r["delta_pct_D"], reverse=True)[:5]
    out["best"] = sorted(ranked, key=lambda r: r["delta_pct_D"])[:5]
    if ranked:
        # |Δ| ≤ 2 % is kernel timing noise (G); a shape-dependent result is "mixed", never "null"
        out["cells_D"] = {"faster": sum(r["delta_pct_D"] < -2 for r in ranked),
                          "slower": sum(r["delta_pct_D"] > 2 for r in ranked), "total": len(ranked)}
        signs = {c for c in ("faster", "slower") if out["cells_D"][c]}
        out["class"] = "mixed" if len(signs) == 2 else next(iter(signs)) if signs else "null"
    if not samples:
        out["insufficient"].append("no kernel perf rows")
    return finish_report(out)


def tail(path):
    if path.name == "round-trace.jsonl":
        return ["trace body intentionally not opened"]
    with path.open("rb") as f:
        pos, parts = path.stat().st_size, []
        while pos and sum(p.count(b"\n") for p in parts) < 9:
            start = max(0, pos - 8192)
            f.seek(start)
            parts.insert(0, f.read(pos - start))
            pos = start
        return b"".join(parts).decode(errors="replace").splitlines()[-8:]


def build_report(root):
    logs = [root / "payload.log"] if (root / "payload.log").exists() else sorted(root.glob("*/payload.log"))
    out = {"shape": "build-window", "run_dir": str(root), "payloads": [], "hard_gates": [], "warnings": [], "insufficient": [], "class": "null"}
    for log in logs:
        text = log.read_text(errors="replace")
        failure = re.search(r"FAIL rc=(\d+) ([^\n]+)", text)
        payload_rc = re.search(r"payload rc=(\d+)", text)
        stages = []
        for line in text.splitlines():
            body = re.sub(r"^[^:]+: (?:\d\d:\d\d:\d\d )?", "", line)
            if body.endswith(" ok") or body.startswith(("PASS ", "OK ")) or " frozen at " in body or "item checked" in body:
                stages.append(body)
            elif re.search(r"stage \d+ verdict rc=(?:0|3)\b", body):
                stages.append(body)
            elif re.search(r"stage \d+ passed\b", body):
                stages.append(body)
        row = {"name": log.parent.name, "last_stage_passed": stages[-1] if stages else "start", "payload_log": str(log)}
        if failure:
            named_log = re.search(r"\(see ([^)]+)\)", failure[2])
            row.update(rc=int(failure[1]), failing_stage=failure[2].split(" (see ", 1)[0],
                       failing_log=named_log[1] if named_log else str(log))
            p = Path(row["failing_log"])
            if not p.is_absolute():
                p = log.parent / p
            if not p.exists() and (log.parent / p.name).exists():
                p = log.parent / p.name
                out["warnings"].append(f"{row['name']}: named log relocated; using {p}")
            row["resolved_log"] = str(p)
            row["tail"] = tail(p) if p.exists() else ["named log missing"]
            out["hard_gates"].append(f"{row['name']}: FAIL rc={row['rc']} {row['failing_stage']}")
        elif payload_rc and int(payload_rc[1]):
            row.update(rc=int(payload_rc[1]), failing_stage="payload", failing_log=str(log), tail=tail(log))
            out["hard_gates"].append(f"{row['name']}: FAIL rc={row['rc']} payload")
        elif re.search(r"(?:payload rc=0|\b(?:DONE|SUCCESS)\b|\bdone(?: stage\d+ rc=0)?\s*$)", text, re.M):
            row["rc"] = 0
        else:
            out["insufficient"].append(f"{row['name']}: no completion marker")
        out["payloads"].append(row)
    return finish_report(out)


def analyze(root, aa=False):
    root = Path(root).resolve()
    pins = load(root / "pins.json") if (root / "pins.json").exists() else {}
    if "order" in pins and "depths" in pins:
        return depth_report(root, aa)
    if (root / "results.json").exists():
        return perf_report(root)
    if (root / "payload.log").exists() or list(root.glob("*/payload.log")):
        return build_report(root)
    return finish_report({"shape": "unknown", "run_dir": str(root), "class": "provisional", "hard_gates": [],
                          "warnings": [], "insufficient": ["unrecognized run shape"]})


def fmt(value, signed=False):
    return "n/a" if value is None else f"{value:+.3f}" if signed else f"{value:.3f}"


def render(out):
    lines = [f"report {out['shape']} · {out['run_dir']}" + (" · PROVISIONAL" if out["exit_code"] else "")]
    if out["shape"] == "depth":
        counts = ", ".join(f"{a}={n}" for a, n in Counter(out["plan"]).items())
        lines[0] += f" · np{out['streams']} · plan {','.join(out['plan'])} · boots {counts}"
        trace = str(out["trace_kb_per_round_D"]) + " KB/round (D)" if "trace_kb_per_round_D" in out else "/".join(
            "traced" if any(t["traced"] for t in ts) else "untraced" for ts in out.get("trace", {}).values())
        vram = f"{out['vram_min_free_mib_M']} MiB (M)" if out.get("vram_min_free_mib_M") is not None else "n/a"
        lines.append(f"gates identity {out['identity']} · trace {trace} · verdicts {'failed' if any('verdict' in s for s in out['hard_gates']) else 'checked'} · vram min {vram} · Xids {out.get('xid_count_M', 0)} (M)")
        lines.append("depth  up ms/round (D)  patch−up (D) ± SE  per-boot Δ (D)  predicted_ms Δ (D) ± SE  prompt_ms Δ (D) ± SE  fixed/req Δ (D) ± SE  agg tok/s up→patch (M)")
        for dep, row in out["depths"].items():
            metrics = row["metrics"]
            steady = metrics["steady_ms_per_round_D"]
            delta = lambda m: f"{fmt(metrics[m]['delta'], True)} ± {fmt(metrics[m]['se'])}"
            toks = "n/a (acceptance differs)" if row["agg_tok_s_M"] is None else "→".join(fmt(v) for v in row["agg_tok_s_M"].values())
            lines.append(f"{dep}  {fmt(statistics.mean(next(iter(steady['arms'].values()))))}  {delta('steady_ms_per_round_D')}  {' / '.join(fmt(v, True) for v in steady['boots'])}  {delta('predicted_ms_M')}  {delta('prompt_ms_M')}  {delta('fixed_ms_D')}  {toks} · class: {row['class']}")
            lines.append(f"  acceptance (M) {row['acceptance_M']} · prompt Δ {fmt(row['prompt_delta_pct_D'], True)}% (D)")
            if "counter_groups" in row:
                lines.append("  counters accepted/drafted per stream (boots): " + " ; ".join(f"{sig} " + " ".join(f"{a}={c}" for a, c in counts.items()) for sig, counts in row["counter_groups"].items()))
                m = row["counter_matched"]
                lines.append("  counter-matched patch−up (D): " + (f"{fmt(m['delta_D'], True)} ms/round over boots {m['boots']}" if m["delta_D"] is not None else "no matched boots"))
            if "overlap_frac_min_D" in row:
                lines.append(f"  overlap fraction min per arm (D): {row['overlap_frac_min_D']}")
            for metric in ("predicted_ms_M", "prompt_ms_M", "fixed_ms_D"):
                values = metrics[metric]
                lines.append(f"  {metric} boots " + " · ".join(f"{a} " + "/".join(fmt(v) for v in vs) for a, vs in values["arms"].items()) +
                             " · per-boot Δ (D) " + "/".join(fmt(v, True) for v in values["boots"]) + f" · SD {fmt(values['sd'])} (D)")
            if out["aa"]:
                lines.append(f"  A/A boot SD (M): {row['aa_boot_sd_ms_M']}")
        label = "M" if "sd_ms_M" in out.get("aa_reference", {}) else "G"
        lines.append(f"noise A/A boot SD reference {out.get('aa_floor_ms', 0.1):.2f} ms/round ({label}); SE = sqrt(var_up/n_up + var_patch/n_patch), boot is unit")
        if "aa_reference_refreshed" in out:
            lines.append(f"A/A reference refreshed in {AA_REFERENCE}: {out['aa_reference_refreshed']}")
        values = " / ".join(fmt(row["metrics"]["steady_ms_per_round_D"]["delta"], True) for row in out["depths"].values())
        depths = "/".join(out["depths"])
        ses = "/".join(fmt(row["metrics"]["steady_ms_per_round_D"]["se"]) for row in out["depths"].values())
        boots = "; ".join("/".join(fmt(v, True) for v in row["metrics"]["steady_ms_per_round_D"]["boots"]) for row in out["depths"].values())
        acceptance = "equal" if all(row["acceptance_equal"] for row in out["depths"].values()) else "differs"
        fixed = "/".join(fmt(row["metrics"]["fixed_ms_D"]["delta"], True) for row in out["depths"].values())
        prompt = "/".join(fmt(row["prompt_delta_pct_D"], True) for row in out["depths"].values())
        evidence = f"{values} ms/round at {depths} (D; per-boot {boots}; SE {ses}); {counts} boots; identity {out['identity']}; acceptance {acceptance}; prompt Δ {prompt}% and fixed/request Δ {fixed} ms (D); class: {out['class']}"
    elif out["shape"] == "perf":
        lines.append("type n  base hot (µs, M) patch hot (µs, M) Δ% (D) base cold (µs, M) patch cold (µs, M) Δ% (D); geometric means over (m,k)")
        for row in out["table"]:
            lines.append(f"{row['type']} {row['columns']}  " + "  ".join(fmt(row.get(heat + '_' + key)) for heat in ('hot', 'cold') for key in ('base_us_M', 'patch_us_M', 'delta_pct_D')))
        lines.append(f"small-J endpoint slope µs/col (D): {out['slope_us_per_column_D']}")
        lines.append(f"worst Δ% (D): {out['worst']}; best: {out['best']}")
        cells = out.get("cells_D", {})
        evidence = (f"{len(out['cases'])} kernel cases (M); cells beyond ±2 %: {cells.get('slower', 0)} slower, "
                    f"{cells.get('faster', 0)} faster of {cells.get('total', 0)} (D); class: {out['class']}")
    elif out["shape"] == "build-window":
        for row in out["payloads"]:
            lines.append(f"{row['name']} · last passed: {row['last_stage_passed']} · " + (f"FAIL rc={row['rc']} {row['failing_stage']} · {row['failing_log']}\n" + "\n".join(row['tail']) if row.get('rc') else "ok" if row.get('rc') == 0 else "incomplete"))
        evidence = "; ".join(out["hard_gates"] or out["insufficient"] or ["build window completed"])
    else:
        evidence = "insufficient data"
    for label, key in (("HARD", "hard_gates"), ("WARN", "warnings"), ("INSUFFICIENT", "insufficient")):
        if out[key]:
            messages = out[key][:8] if out["shape"] == "perf" else out[key]
            extra = f" · {len(out[key]) - 8} more; complete list in --json" if len(messages) < len(out[key]) else ""
            lines.append(label + " " + " · ".join(messages) + extra)
    if out["hard_gates"] and out["shape"] == "depth":
        evidence += "; PROVISIONAL: " + "; ".join(out["hard_gates"])
    quote = chr(96)
    lines.extend([f"class: {out['class']}", f"notes - {quote}{out['run_dir']}{quote}: {evidence}",
                  f"ledger | <lever> | <axis> | <cause> | <relation> | {quote}{out['run_dir']}{quote}: {evidence} | <verdict> |"])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="harness report", description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--record", action="store_true", help="append notes, INDEX and a ledger-row draft only after all report gates pass; never commit")
    for field in ("lever", "axis", "cause", "relation", "verdict"):
        parser.add_argument(f"--{field}", help=f"required with --record: ledger {field}")
    from . import notes
    parser.add_argument("--section", choices=notes.SECTIONS, default="llama-cpp", help="ledger draft section (default: llama-cpp)")
    parser.add_argument("--aa", action="store_true", help="check A/A against zero; refresh this checkout's noise cache after gates pass (run inputs stay read-only)")
    args = parser.parse_args(argv)
    try:
        out = analyze(args.run_dir, args.aa)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        out = finish_report({"shape": "unknown", "run_dir": str(args.run_dir), "class": "provisional", "hard_gates": [f"unreadable records: {error}"], "warnings": [], "insufficient": []})
    rendered = render(out)
    if args.record and not out["exit_code"]:
        try:
            out["recorded"] = notes.record(out, rendered, **{field: getattr(args, field) for field in ("lever", "axis", "cause", "relation", "verdict", "section")})
        except (lib.Fail, OSError) as error:
            print(error, file=sys.stderr)
            return error.rc if isinstance(error, lib.Fail) else 6
    print(json.dumps(out, indent=2, allow_nan=False) if args.json else rendered)
    if args.record and not args.json and "recorded" in out:
        for path in out["recorded"]:
            print(f"wrote {path}")
    return out["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
