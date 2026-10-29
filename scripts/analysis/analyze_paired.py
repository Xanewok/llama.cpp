#!/usr/bin/env python3
"""Paired per-task analysis of Polyglot result rows.

Task difficulty dominates the variance at n=40, so outcomes are compared task by
task (McNemar-style discordant pairs), never as two aggregate pass rates. The
noise floor is the same-quant repeat disagreement rate; any cross-quant delta
that does not clear it is not a result.

When both legs solve everything -- which Polyglot reaches on this model -- there
are no discordant pairs and this says nothing. Compare cost instead, with
paired_runs.py.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[tuple[str, int], dict]:
    """Rows written before a schema addition simply lack those keys.

    Normalising here rather than rewriting the file keeps the analysis off the
    runner's append target, which it must never race.
    """

    rows = {}
    raw = []
    for line in path.read_text().splitlines():
        if line.strip():
            raw.append(json.loads(line))
    keys = set().union(*(r.keys() for r in raw)) if raw else set()
    for r in raw:
        full = {k: r.get(k) for k in keys}
        rows[(full["quant"], full["run_index"], full["task_id"])] = derive_timed_out(full)
    return rows


TIMEOUT_NOTE = "task timed out after "


def derive_timed_out(row: dict) -> dict:
    """Recover the timeout flag for rows predating the column.

    This re-encodes the note the harness itself wrote, so it adds no information
    that was not already recorded. Everything else absent from an older row --
    transcripts, server counters -- stays null rather than being reconstructed.
    """

    if row.get("timed_out") is None:
        note = row.get("notes")
        row["timed_out"] = isinstance(note, str) and note.startswith(TIMEOUT_NOTE)
    return row


def outcomes(rows, quant, run_index, include_timeouts: bool = False) -> dict[str, bool]:
    """Per-task pass/fail, dropping timed-out tasks by default.

    A wall-clock cut-off scales with decode rate, so a larger quant hits it for
    being larger. Scoring that as a quality failure would let the harness
    parameter masquerade as a quantization effect; the task is treated as
    missing instead, which shrinks n honestly.
    """

    return {
        task: bool(r["task_success"])
        for (q, ri, task), r in rows.items()
        if q == quant and ri == run_index and (include_timeouts or not r.get("timed_out"))
    }


def discordant(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, int]:
    shared = sorted(set(a) & set(b))
    a_only = sum(1 for t in shared if a[t] and not b[t])
    b_only = sum(1 for t in shared if b[t] and not a[t])
    return len(shared), a_only, b_only


def paired_difference_ci(n: int, n01: int, n10: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided CI on the paired difference in pass rate, discordants only.

    The signed difference is what the decision needs; total disagreement is
    direction-blind and cannot distinguish symmetric churn from a real loss.
    """

    if n == 0:
        return (0.0, 0.0)
    diff = (n10 - n01) / n
    # Agresti-Min style: variance of the paired difference from the discordant counts.
    var = (n01 + n10 - (n10 - n01) ** 2 / n) / n**2
    half = z * math.sqrt(max(var, 0.0))
    return (max(-1.0, diff - half), min(1.0, diff + half))


def exact_mcnemar_p(n01: int, n10: int) -> float:
    """Two-sided exact binomial p on the discordant pairs only."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="polyglot result JSONL")
    ap.add_argument("--floor-quant", required=True,
                    help="quant run twice to establish the noise floor")
    args = ap.parse_args()

    rows = load(args.results)
    quants = sorted({q for (q, _, _) in rows})

    print("== per-run pass counts ==")
    seen = defaultdict(int)
    for (q, ri, _t), r in rows.items():
        seen[(q, ri)] += bool(r["task_success"])
    counts = defaultdict(int)
    for (q, ri, _t) in rows:
        counts[(q, ri)] += 1
    for key in sorted(counts):
        q, ri = key
        print(f"  {q:<12} run {ri}: {seen[key]:2d}/{counts[key]:2d}")

    timeouts = defaultdict(int)
    for (q, ri, _t), r in rows.items():
        if r.get("timed_out"):
            timeouts[(q, ri)] += 1
    if timeouts:
        print("\n== timeouts (a harness cut-off, not a model verdict) ==")
        for key in sorted(timeouts):
            print(f"  {key[0]:<12} run {key[1]}: {timeouts[key]}")

    floor = None
    a, b = outcomes(rows, args.floor_quant, 1), outcomes(rows, args.floor_quant, 2)
    if a and b:
        n, x, y = discordant(a, b)
        floor = (x + y) / n if n else None
        print(f"\n== noise floor ({args.floor_quant}, run 1 vs run 2) ==")
        print(f"  shared tasks {n}, disagreements {x + y} "
              f"({x} run1-only, {y} run2-only) = {floor:.1%}")
        print("  This is how much churn the same model shows against itself. It is NOT"
              " a threshold: total disagreement is direction-blind, so symmetric churn"
              " can exceed it with no capability difference while a small one-sided"
              " loss can matter. Read the signed difference and its interval instead.")

    print("\n== paired cross-quant comparisons (run 1) ==")
    for i, q1 in enumerate(quants):
        for q2 in quants[i + 1:]:
            a, b = outcomes(rows, q1, 1), outcomes(rows, q2, 1)
            if not a or not b:
                continue
            n, x, y = discordant(a, b)
            rate = (x + y) / n if n else 0.0
            p = exact_mcnemar_p(x, y)
            lo, hi = paired_difference_ci(n, x, y)
            print(f"  {q1} vs {q2}: n={n} discordant={x + y} ({rate:.1%}) "
                  f"[{q1}-only {x}, {q2}-only {y}]")
            print(f"      exact McNemar p={p:.3f}; "
                  f"{q2}-{q1} difference {(y - x) / n:+.1%} "
                  f"(95% CI {lo:+.1%} to {hi:+.1%})")
            if floor is not None:
                print(f"      same-quant repeat disagreed on {floor:.1%} of tasks "
                      "-- context for how much churn is normal, not a threshold")


if __name__ == "__main__":
    main()
