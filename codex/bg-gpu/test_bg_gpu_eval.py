"""Gate logic of bg_gpu_check.evaluate on real stack7 round traces (verify-path-j5q canary: prod = candidate path,
fallback = host scan). Usage: test_bg_gpu_eval.py PROD_TRACE FALLBACK_TRACE"""
import copy
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent))
import bg_gpu_check as g  # noqa: E402

prod, fallback = Path(sys.argv[1]), Path(sys.argv[2])


def span(trace):
    b = [g.json.loads(l)["phases"][0]["begin_ns"] for l in trace.read_text().splitlines() if l.strip()]
    return min(b), max(b)


class FakeBoot:
    def __init__(self, name, trace):
        self.name, self.trace = name, trace


def row(boot, key, t, seed=1, text="x", warn=0, guard=True):
    return {"boot": boot, "key": key, "seed": seed, "t0_ns": t[0], "t1_ns": t[1], "error": None, "reasoning": text,
            "content": "", "tool_calls": [["exec_command", "{}"]], "completion_tokens": 10, "finish": "tool_calls",
            "disable_warnings": warn, "schedule_guard": guard, "prompt_ms": 400}


def base():
    tp, tf = span(prod), span(fallback)
    boots = {"cand-np1": FakeBoot("cand-np1", prod), "host-np1": FakeBoot("host-np1", fallback)}
    rows = []
    for c in ("auto", "required"):
        for s in g.SEEDS:
            rows.append(row("cand-np1", f"np1/{c}/{s}", tp, s, f"t{s}"))
            rows.append(row("host-np1", f"np1/{c}/{s}", tf, s, f"t{s}"))
    return rows, boots


def check(name, rows, boots, want_pass, want=None):
    res = g.evaluate(rows, boots, False)
    ok = (not res["failures"]) == want_pass and (want is None or any(want in f for f in res["failures"] + res["insufficient"]))
    print(("PASS " if ok else "FAIL ") + name, json.dumps(res["engagement"]), res["failures"][:2], res["insufficient"][:2])
    return ok


results = []
rows, boots = base()
results.append(check("identity + engagement", rows, boots, True))
r2 = copy.deepcopy(rows); r2[1]["reasoning"] = "changed"
results.append(check("mutant: one host token differs", r2, boots, False, "cross-arm identity failed"))
b3 = dict(boots, **{"host-np1": FakeBoot("host-np1", prod)})
r3 = [dict(r, t0_ns=span(prod)[0], t1_ns=span(prod)[1]) if r["boot"] == "host-np1" else r for r in rows]
results.append(check("mutant: host arm also on candidates", r3, b3, False, "path-engaged proof failed"))
r4 = copy.deepcopy(rows)
for r in r4:
    if r["boot"] == "host-np1" and r["seed"] == g.SEEDS[1]:
        r["reasoning"] = f"t{g.SEEDS[0]}"
for r in r4:
    if r["boot"] == "cand-np1" and r["seed"] == g.SEEDS[1]:
        r["reasoning"] = f"t{g.SEEDS[0]}"
results.append(check("mutant: blind comparator", r4, boots, False, "comparator"))
r5 = copy.deepcopy(rows); r5[0]["disable_warnings"] = 1
results.append(check("mutant: cand logged a disable warning", r5, boots, False, "disable"))
print("ALL PASS" if all(results) else "SOME FAIL")
sys.exit(0 if all(results) else 1)
