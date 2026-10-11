"""DC end -> next phase begin (us), by DC rows, from np2-matrix round traces. Usage: dcgap.py TRACE..."""
import json, sys, statistics as st
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f)]
    ph = [p for r in rows for p in r["phases"]]
    ph.sort(key=lambda p: p["begin_ns"])
    gaps = {}
    for a, b in zip(ph, ph[1:]):
        if a["phase"] == "DC":
            gaps.setdefault(a["n_tokens"], []).append((b["begin_ns"] - a["end_ns"]) / 1e3)
    print(f.split("/")[-1], {k: f"n={len(v)} med={st.median(v):.0f}us" for k, v in sorted(gaps.items())})
