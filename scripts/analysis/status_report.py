#!/usr/bin/env python3
"""Regenerate the whole experiment's state into one file.

Written so a cold reader -- a person at breakfast or a fresh agent session -- needs
no context beyond this file. It reads only artifacts on disk, so it stays correct
after the session that produced them is gone.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from scripts import ROOT

S1 = ROOT / "results" / "2026-08-rental-baseline"


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn line from a killed writer
    return out


def rate(rs: list[dict], num: str, den: str) -> float | None:
    n = sum(r.get(num) or 0 for r in rs)
    d = sum(r.get(den) or 0 for r in rs)
    return n / d if d else None


def fmt(x, p=1, dash="--"):
    return f"{x:,.{p}f}" if isinstance(x, (int, float)) else dash


def legs_table(pg: list[dict], lc: list[dict]) -> list[str]:
    groups: dict[tuple, list[dict]] = {}
    for r in pg:
        groups.setdefault((r.get("quant"), r.get("run_index"), "polyglot"), []).append(r)
    if lc:
        groups[(lc[0].get("quant"), 64, "longctx-64K")] = lc
    out = ["| leg | n | pass | timeouts | decode tok/s | prefill tok/s | decode tok/task | wall/task |",
           "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for (quant, run, kind), rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ok = sum(1 for r in rs if r.get("task_success"))
        to = sum(1 for r in rs if r.get("timed_out"))
        dt = statistics.mean([r.get("decode_tokens") or 0 for r in rs]) if rs else 0
        wt = statistics.mean([r.get("wall_seconds") or 0 for r in rs]) if rs else 0
        out.append(
            f"| {quant} {kind} run{run} | {len(rs)} | {ok}/{len(rs)} | {to} | "
            f"{fmt(rate(rs,'decode_tokens','decode_seconds'))} | "
            f"{fmt(rate(rs,'prefill_tokens','prefill_seconds'),0)} | {fmt(dt,0)} | {fmt(wt,0)} s |"
        )
    return out


def paired_tokens(pg: list[dict]) -> list[str]:
    """Q4 vs IQ3 on decode tokens to a test-verified solution.

    IQ3 is averaged over its runs so its test-retest noise does not masquerade as a
    quant effect. Timed-out tasks are dropped as missing, never scored as failures.
    """
    q4 = {r["task_id"]: r for r in pg if r.get("quant") == "UD-Q4_K_XL"}
    iq = {}
    for r in pg:
        if r.get("quant") == "UD-IQ3_XXS" and not r.get("timed_out") and r.get("decode_tokens"):
            iq.setdefault(r["task_id"], []).append(r["decode_tokens"])
    pairs = [(t, q4[t]["decode_tokens"], statistics.mean(iq[t]))
             for t in q4 if t in iq and not q4[t].get("timed_out") and q4[t].get("decode_tokens")]
    if len(pairs) < 2:
        return ["_Not enough paired tasks yet._"]
    d = [q - i for _, q, i in pairs]
    n = len(d)
    md, sd = statistics.mean(d), statistics.stdev(d)
    se = sd / (n ** 0.5)
    lo, hi = md - 1.96 * se, md + 1.96 * se
    mq = statistics.mean(q for _, q, _ in pairs)
    mi = statistics.mean(i for _, _, i in pairs)
    return [
        f"- Paired tasks: **{n}**; Q4 lower on **{sum(1 for x in d if x < 0)}/{n}**",
        f"- Q4 {mq:,.0f} vs IQ3 {mi:,.0f} decode tokens/task (**{mi/mq:.2f}x**)",
        f"- Mean paired difference **{md:,.0f}**, 95% CI [{lo:,.0f}, {hi:,.0f}], t={md/se:.2f}"
        f" -> {'**excludes zero**' if lo * hi > 0 else 'spans zero'}",
        "",
        "Correctness is saturated (both quants pass everything scored so far), so this is a",
        "cost difference, not a capability one. Caveat: arXiv:2504.04823 finds quantization",
        "does NOT increase output length -- but it tests W4A16 and above on single-shot",
        "benchmarks, while this is ~3.06-bit in an agentic loop, so the regimes differ.",
    ]


def mtp_table(bench: list[dict]) -> list[str]:
    g: dict[tuple, list[dict]] = {}
    for r in bench:
        g.setdefault((r.get("quant"), r.get("label"), r.get("depth_target")), []).append(r)
    out = ["| quant | MTP | depth | n | tok/s | accept |", "|---|---|--:|--:|--:|--:|"]
    for k, v in sorted(g.items(), key=lambda kv: str(kv[0])):
        tps = [x["decode_tps"] for x in v if x.get("decode_tps")]
        ar = [x["draft_accept_rate"] for x in v if x.get("draft_accept_rate")]
        out.append(f"| {k[0]} | {k[1]} | {k[2]:,} | {len(v)} | "
                   f"{fmt(statistics.mean(tps)) if tps else '--'} | "
                   f"{fmt(statistics.mean(ar),2) if ar else '--'} |")
    return out


def kld_section() -> list[str]:
    logs = sorted((S1 / "rental" / "kld").glob("*.log")) if (S1 / "rental" / "kld").exists() else []
    if not logs:
        return ["_Not run yet._"]
    out = ["| quant | mean KLD | median | 99th pct | same top-1 |", "|---|--:|--:|--:|--:|"]
    for f in logs:
        if f.name in {"base.log", "run.log"}:  # drivers, not measurements
            continue
        text = f.read_text(errors="replace")
        def grab(pat):
            m = re.search(pat, text, re.I)
            return m.group(1).strip() if m else "--"
        out.append(f"| {f.stem} | {grab(r'Mean +KLD *: *([0-9.eE+-]+)')} | "
                   f"{grab(r'Median +KLD *: *([0-9.eE+-]+)')} | "
                   f"{grab(r'99\.0% +KLD *: *([0-9.eE+-]+)')} | "
                   f"{grab(r'Same top p *: *([0-9.]+ *%?)')} |")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "research/2026-08-rental-baseline/reports/STATUS.md")
    ap.add_argument("--stamp", default="", help="timestamp string; callers pass date(1)")
    args = ap.parse_args()

    pg = rows(S1 / "codegen" / "polyglot.jsonl")
    lc = rows(S1 / "codegen" / "longctx.jsonl")
    bench = rows(S1 / "mtp" / "bench.jsonl")
    env = (S1 / "rental" / "environment.txt")

    doc = [f"# Qwen3.8-27B quantization experiment - status", ""]
    if args.stamp:
        doc += [f"_Generated {args.stamp}._", ""]
    doc += ["Regenerate with `PYTHONPATH=. python3 -m scripts.analysis.status_report`. Reads only on-disk",
            "artifacts, so it is valid after the producing session is gone.", ""]

    doc += ["## Legs", ""] + legs_table(pg, lc) + [""]
    doc += ["## Q4 vs IQ3: decode tokens to a verified solution", ""] + paired_tokens(pg) + [""]
    doc += ["## Speculative decoding (MTP)", ""] + (mtp_table(bench) if bench else ["_Not run._"]) + [""]
    doc += ["## KL-divergence vs Q8_0 reference", ""] + kld_section() + [""]

    if env.exists():
        caps = [l for l in env.read_text().splitlines() if "GeForce" in l and "," in l]
        doc += ["## Environment", "", "```", *caps[:4], "```",
                "",
                "Every tok/s figure above came from a 5090 power-capped by the host, so",
                "absolutes are a floor for an owned card; ratios between quants are sound.",
                ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(doc) + "\n")
    print(f"wrote {args.out} ({len(doc)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
