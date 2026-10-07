"""Judge np2-matrix.sh. Unit: one slot of one phase (tokens + draft/accepted/rounds), plus the draft dump rows.

Identity: off vs on equal tokens and counters, dump ids equal, |dq| <= 2e-7. Engagement: 'MTP chain: k=<k>' and DC decodes
of 2k rows (both slots) and k rows (one slot); no step-loop D phases unless slot 1 is greedy, where both must occur.
Controls: off-repeat equal (floor), every mutant differs from its off cell. Exit 0 pass, 2 identity, 3 engagement, 4 escape,
5 floor, 6 missing cell. Writes OUT/summary.json."""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
QUICK = len(sys.argv) > 2 and sys.argv[2] == "quick"
Q_TOL = 2e-7


def cell(tag):
    p = out / f"{tag}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    units = {}
    for ph in data["phases"]:
        for i, st in enumerate(ph["stats"]):
            units[f"{ph['phase']}.{st['slot']}"] = (ph["tokens"][i], st["draft"], st["accepted"], st["rounds"])
    dump = [l.split() for l in (out / f"{tag}.dump.txt").read_text().splitlines()]
    phases = [ph for line in (out / f"{tag}.trace.jsonl").read_text().splitlines() for ph in json.loads(line)["phases"]]
    log = (out / f"{tag}.log").read_text(errors="replace")
    return dict(units=units, dump=dump, phases=phases, log=log)


def compare(a, b):
    bad_units = sorted(k for k in a["units"] if a["units"][k] != b["units"].get(k))
    ids_equal = len(a["dump"]) == len(b["dump"]) and all(ra[:13] == rb[:13] for ra, rb in zip(a["dump"], b["dump"]))
    dq = max((abs(float(x) - float(y)) for ra, rb in zip(a["dump"], b["dump"]) for x, y in zip(ra[13:], rb[13:])), default=0.0)
    return dict(units_differ=bad_units, dump_rows=(len(a["dump"]), len(b["dump"])), dump_ids_equal=ids_equal, max_dq=dq,
                identical=not bad_units and ids_equal and len(a["dump"]) > 0 and dq <= Q_TOL)


def shapes(c):
    dc = sorted({ph["n_tokens"] for ph in c["phases"] if ph["phase"] == "DC"})
    steps = sum(1 for ph in c["phases"] if ph["phase"].startswith("D") and ph["phase"] != "DC")
    failed = sum(1 for ph in c["phases"] if ph["result"] != 0)
    return dict(dc_rows=dc, step_decodes=steps, failed_decodes=failed)


rc = 0
summary = {}


def fail(code, msg):
    global rc
    print("FAIL", msg)
    rc = rc or code


def pair(name, off_tag, on_tag, k, greedy=False):
    off, on = cell(off_tag), cell(on_tag)
    if off is None or on is None:
        fail(6, f"{name}: missing cell")
        return
    cmp = compare(off, on)
    s_off, s_on = shapes(off), shapes(on)
    summary[name] = dict(cmp=cmp, off=s_off, on=s_on)
    print(("ok  " if cmp["identical"] else "FAIL"), name, json.dumps(cmp), "on:", json.dumps(s_on))
    if not cmp["identical"]:
        fail(2, f"{name}: identity")
    want = {k, 2 * k} if not greedy else {k}
    engaged = (f"MTP chain: k={k}" in on["log"] and want <= set(s_on["dc_rows"]) and s_on["failed_decodes"] == 0
               and f"MTP chain: k=" not in off["log"] and not s_off["dc_rows"]
               and (s_on["step_decodes"] > 0 if greedy else s_on["step_decodes"] == 0))
    if not engaged:
        fail(3, f"{name}: engagement {s_on} (want DC rows {sorted(want)})")


for meta in (0, 1):
    for k in ((5,) if QUICK else (3, 5)):
        for seed in ((1729,) if QUICK else (1729, 2718)):
            t = f"{meta}-k{k}-s{seed}"
            pair(t, f"{t}-off", f"{t}-on", k)
KB = 5 if QUICK else 3  # k of the variant cells
B = f"0-k{KB}-s1729"
if not QUICK:
    pair("0-k3-s1729-unified", "0-k3-s1729-uni-off", "0-k3-s1729-uni-on", 3)
for meta in ((0,) if QUICK else (0, 1)):
    pair(f"{meta}-k{KB}-s1729-greedy1", f"{meta}-k{KB}-s1729-greedy1-off", f"{meta}-k{KB}-s1729-greedy1-on", KB, greedy=True)

base, rep = cell(f"{B}-off"), cell(f"{B}-off-repeat")
if base and rep:
    floor = compare(base, rep)
    summary["floor_off_repeat"] = floor
    if not (floor["identical"] and floor["max_dq"] == 0.0):
        fail(5, f"off-vs-off floor {floor}")
else:
    fail(6, "floor cells missing")

MUTANTS = [("mut-swap", f"{B}-off"), ("greedy1-mut-norng", f"{B}-greedy1-off")]
if not QUICK:
    MUTANTS += [("mut-swap-uni", f"{B}-uni-off"), ("mut-revu", f"{B}-off"), ("mut-shift", f"{B}-off")]
for mut, ref in MUTANTS:
    tag = f"{B}-{mut}"
    m, r = cell(tag), cell(ref)
    rcf = out / f"{tag}.rc"
    if m is None:
        if rcf.exists() and rcf.read_text().strip() != "0":
            summary[mut] = f"caught: exit {rcf.read_text().strip()}"
            print("ok  ", mut, summary[mut])
            continue
        fail(6, f"{mut}: missing")
        continue
    cmp = compare(r, m)
    engaged = f"MTP chain: k={KB}" in m["log"]
    summary[mut] = dict(cmp=cmp, engaged=engaged)
    print(("ok  " if engaged and not cmp["identical"] else "FAIL"), mut, "differs" if not cmp["identical"] else "ESCAPED",
          json.dumps(cmp["units_differ"]), "dump_ids_equal", cmp["dump_ids_equal"])
    if not engaged or cmp["identical"]:
        fail(4, f"{mut}: escaped or not engaged")

(out / "summary.json").write_text(json.dumps(dict(rc=rc, cells=summary), indent=1) + "\n")
print(f"verdict rc={rc}")
sys.exit(rc)
