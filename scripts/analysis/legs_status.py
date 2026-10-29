#!/usr/bin/env python3
"""Progress of the in-flight cost legs, counted by run_index.

Line count is the wrong gauge: an orphaned runner from a killed leg kept appending to the
same file under its own run_index, and `wc -l` read 85/78 on a leg that had done 42.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
import sys

from scripts import ROOT

# Void legs stay out: one that can never reach TOTAL holds the completion signal open
# forever. Run 11 is void -- see results/2026-08-serving-comparisons/polyglot-cost/void/README.md.
LEGS = {  # run_index -> (experiment dir, label)
    12: ("polyglot-cost-q2", "Q2_K_XL 5070Ti"),
    13: ("polyglot-cost-q4b", "Q4_K_XL 3090 #2"),
}
TOTAL = 78
root = ROOT / "results" / "2026-08-serving-comparisons"

parts, done = [], True
for run, (exp, label) in sorted(LEGS.items()):
    counts: collections.Counter = collections.Counter()
    path = root / exp / "polyglot.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    counts[json.loads(line).get("run_index")] += 1
                except ValueError:
                    pass
    n = counts.get(run, 0)
    stray = {k: v for k, v in counts.items() if k != run}
    parts.append(f"{label} r{run} {n}/{TOTAL}" + (f" [STRAY {stray}]" if stray else ""))
    done = done and n >= TOTAL
print(" | ".join(parts))
sys.exit(0 if not done else 3)
