#!/usr/bin/env python3
"""Paired per-task comparison of two legs on the same task set.

Unpaired means are not comparable here: the tasks differ in difficulty by more
than the quants differ from each other.

Companion to analyze_paired.py, which compares pass/fail via discordant pairs.
Use that one when outcomes differ; this one when they do not -- once both legs
solve everything, McNemar has nothing to work with and the whole signal is in
what each answer cost.

Report the sign test before the ratio. Costs are heavy-tailed (one leg's median
is 5.5K completion tokens against a 41K max), so a ratio built from means is set
by whichever task ran away: the same pair of legs read as 1.39x by mean of totals
and 1.28x paired, and the second is accompanied by 23 of 39 tasks -- a coin flip.
"""
import argparse, json, math, statistics as st, sys

FIELDS = ("decode_tokens", "completion_tokens", "wall_seconds", "decode_tps")


def load(paths, idx, quant=None):
    """Rows for one leg, keyed by task.

    Refuses a task that arrives twice under different quants. Run indices are unique
    within a file but not across quants in it, so keying on the index alone lets a
    second quant's rows overwrite the first's and the comparison then runs on a
    silently mixed leg. Pass `quant` to name which one is wanted.
    """
    rows = {}
    for path in paths:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("run_index") != idx or (quant is not None and d.get("quant") != quant):
                    continue
                seen = rows.get(d["task_id"])
                if seen is not None and seen.get("quant") != d.get("quant"):
                    sys.exit(f"run {idx} holds {seen.get('quant')} and {d.get('quant')} "
                             f"for {d['task_id']}; name one with --base-quant/--cand-quant")
                rows[d["task_id"]] = d
    return rows


def sign_p(wins: int, n: int) -> float:
    """Two-sided exact sign test. Assumes nothing about the cost distribution."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, k) * 0.5 ** n for k in range(n + 1)]
    return min(1.0, 2 * min(sum(pmf[: wins + 1]), sum(pmf[wins:])))


def summarise(ratios: list[float]) -> dict:
    logs = [math.log(r) for r in ratios]
    gm = math.exp(st.mean(logs))
    sd = st.stdev(logs) if len(logs) > 1 else 0.0
    se = sd / math.sqrt(len(logs)) if len(logs) > 1 else 0.0
    wins = sum(1 for r in ratios if r > 1)
    return {
        "n": len(ratios), "geometric": gm, "log_sd": sd,
        "lo": math.exp(math.log(gm) - 1.96 * se), "hi": math.exp(math.log(gm) + 1.96 * se),
        "wins": wins, "p": sign_p(wins, len(ratios)),
    }


def tasks_needed(log_sd: float, ratio: float) -> int:
    """Paired tasks for 80% power at alpha 0.05. Why 40 resolves 1.3x and no finer."""
    return math.ceil(((1.96 + 0.84) * log_sd / math.log(ratio)) ** 2)


def main_argv(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", action="append", default=None,
                    help="result JSONL; repeat to compare legs written to different files")
    ap.add_argument("--base", type=int, required=True)
    ap.add_argument("--cand", type=int, required=True)
    # Run indices are unique within one file but not across them, and the two legs worth
    # comparing now live in different sessions' files.
    ap.add_argument("--base-quant", default=None)
    ap.add_argument("--cand-quant", default=None)
    ap.add_argument("--solved-only", action="store_true",
                    help="pair only tasks both legs solved; a failed task's cost is not a cost")
    a = ap.parse_args(argv)
    paths = a.path or ["results/2026-08-rental-baseline/speed/3090-polyglot.jsonl"]

    base, cand = load(paths, a.base, a.base_quant), load(paths, a.cand, a.cand_quant)
    for rows, idx in ((base, a.base), (cand, a.cand)):
        # load() catches a task claimed by two quants; this catches a leg whose tasks
        # were split between them, which leaves no single task duplicated.
        quants = {r.get("quant") for r in rows.values()}
        if len(quants) > 1:
            sys.exit(f"run {idx} holds more than one quant {sorted(quants)}; not one leg")
    if a.solved_only:
        base = {t: r for t, r in base.items() if r.get("task_success")}
        cand = {t: r for t, r in cand.items() if r.get("task_success")}
    shared = sorted(set(base) & set(cand))
    if not shared:
        sys.exit(f"no shared tasks between run {a.base} and run {a.cand}")

    dropped = (set(base) ^ set(cand))
    print(f"run {a.base} n={len(base)}  run {a.cand} n={len(cand)}  paired={len(shared)}")
    if dropped:
        print(f"unpaired (excluded): {sorted(dropped)}")

    for r, tag in ((base, a.base), (cand, a.cand)):
        ok = sum(1 for t in shared if r[t].get("task_success"))
        e = {r[t].get("engine") for t in shared}
        q = {r[t].get("quant") for t in shared}
        print(f"  run {tag}: {'/'.join(sorted(map(str, e)))} {'/'.join(sorted(map(str, q)))} "
              f"solved {ok}/{len(shared)}")

    print(f"\n{'task':<34}" + "".join(f"{f:>26}" for f in FIELDS))
    ratios = {f: [] for f in FIELDS}
    for t in shared:
        cells = ""
        for f in FIELDS:
            b, c = base[t].get(f), cand[t].get(f)
            if b in (None, 0) or c in (None, 0):
                cells += f"{'-':>26}"
                continue
            ratios[f].append(c / b)
            cells += f"{b:>10.0f} {c:>7.0f} {c/b:>6.2f}x"
        print(f"{t[:33]:<34}{cells}")

    stats = {f: summarise(v) for f, v in ratios.items() if len(v) > 1}
    print(f"\n{'ratio cand/base':<34}" + "".join(f"{f:>26}" for f in FIELDS))

    def row(label, cell):
        print(f"{label:<34}" + "".join(
            f"{cell(stats[f]):>26}" if f in stats else f"{'-':>26}" for f in FIELDS))

    for label, fn in (("median", st.median), ("mean", st.fmean)):
        print(f"{label:<34}" + "".join(
            f"{fn(ratios[f]):>25.2f}x" if len(ratios[f]) > 1 else f"{'-':>26}" for f in FIELDS))
    row("geometric", lambda s: f"{s['geometric']:.2f}x")
    row("95% CI", lambda s: f"[{s['lo']:.2f}, {s['hi']:.2f}]")
    row("tasks cand > base", lambda s: f"{s['wins']}/{s['n']}")
    row("sign test p", lambda s: f"{s['p']:.3f}")

    if "completion_tokens" in stats:
        sd = stats["completion_tokens"]["log_sd"]
        need = "  ".join(f"{r:.2f}x:{tasks_needed(sd, r)}" for r in (1.5, 1.3, 1.2, 1.15, 1.1))
        print(f"\npaired tasks for 80% power on completion_tokens (log sd {sd:.3f}):\n  {need}")


if __name__ == "__main__":
    main_argv()
