"""Judge cells.sh OUT. Unit: one slot of one phase (tokens + draft/accepted/rounds), plus the chain dump rows (ids exact,
probabilities |dq| <= 2e-7). Each level cell vs the L0 cell of its group (0, 0d, 1); the floor (0-L0-repeat) must match and the
slotshift mutant must differ. Engagement: the hostless log strings of each level, none at L0, and on Meta(CPU,CPU) L1+ a
tok_embd buffer that is not the CPU one. Exit 0 pass, 2 identity, 3 engagement, 4 mutant escaped, 5 floor, 6 failed cell.
Writes OUT/summary.json."""
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
Q_TOL = 2e-7
rc = 0
summary = {}


def fail(code, msg):
    global rc
    print("FAIL", msg)
    rc = rc or code


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
    log = (out / f"{tag}.log").read_text(errors="replace")
    return dict(units=units, dump=dump, log=log)


def compare(a, b):
    bad = sorted(k for k in a["units"] if a["units"][k] != b["units"].get(k))
    ids = len(a["dump"]) == len(b["dump"]) and all(ra[:13] == rb[:13] for ra, rb in zip(a["dump"], b["dump"]))
    dq = max((abs(float(x) - float(y)) for ra, rb in zip(a["dump"], b["dump"]) for x, y in zip(ra[13:], rb[13:])), default=0.0)
    return dict(units_differ=bad, dump_rows=(len(a["dump"]), len(b["dump"])), dump_ids_equal=ids, max_dq=dq,
                identical=not bad and ids and len(a["dump"]) > 0 and dq <= Q_TOL)


def engaged(tag, c, level, meta):
    log = c["log"]
    want = ["MTP chain: k=5 seqs=2"]
    if level >= 1:
        want.append(f"hostless: level={level}")
    if level >= 2:
        want += ["hostless: draft id slots=64", "hostless: chain drafts deferred k=5 seqs=2",
                 "hostless: target batch reads device draft ids", "hostless: device order engaged"]
    if level >= 3:
        want.append("hostless: output lag engaged")
    missing = [w for w in want if w not in log]
    if level == 0 and "hostless:" in log:
        missing.append("no hostless strings at L0")
    m = re.search(r"hostless: tok_embd buffer=(\S+)", log)
    buf = m.group(1) if m else None
    if meta and level >= 1 and (buf is None or buf == "CPU"):
        missing.append(f"tok_embd off the host (buffer={buf})")
    summary.setdefault("engagement", {})[tag] = dict(missing=missing, tok_embd_buffer=buf)
    if missing:
        fail(3, f"{tag}: engagement missing {missing}")
    else:
        print("ok   engaged", tag, f"tok_embd_buffer={buf}")


cells = {}
for p in sorted(out.glob("*.rc")):
    tag = p.stem
    if p.read_text().strip() != "0":
        fail(6, f"{tag}: rc={p.read_text().strip()}")
        continue
    cells[tag] = cell(tag)

for tag, c in cells.items():
    m = re.fullmatch(r"(0d|0|1)-L(\d)(-.*)?", tag)
    group, level, suffix = m.group(1), int(m.group(2)), m.group(3) or ""
    ref = cells.get(f"{group}-L0")
    if ref is None:
        fail(6, f"{tag}: no {group}-L0 reference")
        continue
    engaged(tag, c, level, group == "1")
    if tag == f"{group}-L0":
        continue
    cmp = compare(ref, c)
    summary[tag] = cmp
    if suffix.startswith("-mut-"):
        print(("ok   differs" if not cmp["identical"] else "FAIL escaped"), tag, json.dumps(cmp))
        if cmp["identical"]:
            fail(4, f"{tag}: mutant escaped")
    elif suffix == "-repeat":
        print(("ok   floor" if cmp["identical"] else "FAIL floor"), tag, json.dumps(cmp))
        if not cmp["identical"]:
            fail(5, f"{tag}: floor")
    else:
        print(("ok   identical" if cmp["identical"] else "FAIL identity"), tag, json.dumps(cmp))
        if not cmp["identical"]:
            fail(2, f"{tag}: identity")

(out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
print("judge rc", rc)
sys.exit(rc)
