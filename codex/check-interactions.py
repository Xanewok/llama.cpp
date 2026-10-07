"""Same-seed integration controls for fold plus chain and the np2 chain fallback."""
import argparse
import json
import os
from pathlib import Path

import chain_cpu_identity as chain

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
a = ap.parse_args()
chain.RES = a.out.resolve()
chain.RES.mkdir(parents=True, exist_ok=True)
cl = chain.cases(64, {"short": chain.SHORT, "mid": chain.MID}, [(1.0, 1729)])
report = {}
for meta in (False, True):
    tag = "meta" if meta else "plain"
    off = chain.boot(tag + "-off", 5, meta, False, cl)
    both = chain.boot(tag + "-both", 5, meta, True, cl, {"LLAMA_MTP_FOLD_C": "1"})
    cmp = chain.compare(off, both)
    assert all(cmp["tokens"].values()) and all(cmp["counters_equal"].values()), cmp
    assert cmp["dump_ids_equal"] and cmp["dump_rows"][0] > 0, cmp
    assert both["chain_line"] > 0, tag
    folds = sum(v for (ph, kind), v in both["counters"].items() if kind == "mtp_fold")
    rounds = both["counters"].get(("DC", "mtp_chain"), 0)
    assert folds >= 0.9 * rounds and rounds > 0, (tag, folds, rounds)
    report[tag] = {"comparison": cmp, "folds_M": folds, "rounds_M": rounds}
(a.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print("fold plus chain, plain and Meta, same-seed identity and engagement PASS")
