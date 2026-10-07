"""Judge the np2 CPU matrix written by bw-host-trims-v2.sh.

Per (meta, variant): OUT/<meta>-<variant>.json from bw-np2-driver (phases -> tokens, stats per slot) and .log.
The unit is one stream (slot) of one phase: tokens and the draft / accepted / rounds counters must be equal to the
reference variant, which runs both switches off (= stack6 code path).

Exit codes: 0 pass, 2 identity failure, 3 negative control "legacy b115f8180" did not fail (the test cannot reproduce
the GPU defect; everything else passed), 4 a mutant escaped, 5 reference not repeatable, 6 test lacks the reuse path,
7 verify hooks failed or never engaged.
"""
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
metas = [int(m) for m in sys.argv[2].split(",")]
IDENTICAL = ["seqrm", "bulk", "fix", "fixv"]
MUTANTS = ["mut-bulk", "mut-seqrm"]


def load(meta, name):
    return json.loads((out / f"{meta}-{name}.json").read_text())


def units(data):
    """(phase, slot) -> tokens and counters."""
    res = {}
    for ph in data["phases"]:
        for slot, st in enumerate(ph["stats"]):
            res[(ph["phase"], st["slot"])] = dict(tokens=ph["tokens"][slot], draft=st["draft"], accepted=st["accepted"],
                                                   rounds=st["rounds"], prompt=st["prompt"], cached=st["cached"])
    return res


def counters(u):
    return {k: (v["draft"], v["accepted"], v["rounds"]) for k, v in u.items()}


def differs(a, b):
    return sorted(k for k in a if a[k] != b[k])


rc = 0
summary = {}
for meta in metas:
    ref = units(load(meta, "ref"))
    row = {}
    print(f"== meta={meta} reference (both switches 0): per-stream counters")
    for (phase, slot), v in sorted(ref.items()):
        acc = v["accepted"] / v["draft"] if v["draft"] else float("nan")
        print(f"   phase {phase} slot {slot}: draft {v['draft']:4d} accepted {v['accepted']:4d} rounds {v['rounds']:3d} "
              f"acceptance {acc:.3f} prompt {v['prompt']} cached {v['cached']}")
    row["ref"] = {f"{p}.{s}": dict(draft=v["draft"], accepted=v["accepted"], rounds=v["rounds"], prompt=v["prompt"], cached=v["cached"])
                  for (p, s), v in sorted(ref.items())}
    # a stream far below its peer is the GPU defect's signature (stream 1 acceptance collapse)
    for phase in {p for p, _ in ref}:
        acc = [ref[(phase, s)]["accepted"] / max(1, ref[(phase, s)]["draft"]) for s in (0, 1) if (phase, s) in ref]
        if len(acc) == 2 and min(acc) < 0.5 * max(acc):
            row.setdefault("reference_stream_collapse", []).append(dict(phase=phase, acceptance=acc))
            print(f"   NOTE: reference already shows a collapsed stream in phase {phase}: acceptance {acc}")

    if not all(v["cached"] > 0 for (p, _), v in ref.items() if p == 1):
        print(f"FAIL meta={meta}: phase 2 reused no cached prompt tokens; the prefix-reuse path was not exercised")
        rc = rc or 6
    if units(load(meta, "ref2")) != ref:
        print(f"FAIL meta={meta}: the reference is not repeatable, identity checks would be meaningless")
        rc = rc or 5
        row["ref_repeatable"] = False
        summary[meta] = row
        continue
    row["ref_repeatable"] = True

    for name in IDENTICAL:
        bad = differs(ref, units(load(meta, name)))
        row[name] = "equal" if not bad else f"DIFFERENT in {bad}"
        print(f"{'ok  ' if not bad else 'FAIL'} meta={meta} {name}: tokens and per-stream counters {'equal to reference' if not bad else 'differ in ' + str(bad)}")
        if bad:
            rc = rc or 2

    log = (out / f"{meta}-fixv.log").read_text(errors="replace")
    bad_log = re.findall(r"host_trims_verify: (?:BULK MISMATCH|SEQRM RESIDUAL).*", log)
    engaged = "host_trims_verify: bulk checked" in log and "host_trims_verify: seqrm checked" in log
    row["verify"] = dict(violations=bad_log[:5], engaged=engaged)
    print(f"{'ok  ' if engaged and not bad_log else 'FAIL'} meta={meta} verify hooks: engaged={engaged} violations={len(bad_log)}")
    if bad_log or not engaged:
        rc = rc or 7

    for name in MUTANTS:
        rcf = out / f"{meta}-{name}.rc"
        if rcf.exists() and not (out / f"{meta}-{name}.json").exists():
            pat = r"^.*(?:failed|error|GGML_ASSERT).*$"
            ref_lines = {re.sub(r"^[\d.]+ ", "", l) for l in re.findall(pat, (out / f"{meta}-ref.log").read_text(errors="replace"), re.M)}
            err = [l for l in re.findall(pat, (out / f"{meta}-{name}.log").read_text(errors="replace"), re.M)
                   if re.sub(r"^[\d.]+ ", "", l) not in ref_lines]
            row[name] = f"caught: exit {rcf.read_text().strip()}, first error: {err[0].strip() if err else '?'}"
            print(f"ok   meta={meta} negative control {name}: {row[name]}")
            continue
        bad = differs(ref, units(load(meta, name)))
        engaged = name != "mut-bulk" or "host_trims_mutant: bulk_off engaged" in (out / f"{meta}-{name}.log").read_text(errors="replace")
        row[name] = f"caught in {bad}" if bad else "ESCAPED" if engaged else "NOT ENGAGED (the mutant never ran)"
        print(f"{'ok  ' if bad else 'FAIL'} meta={meta} negative control {name}: {'caught in ' + str(bad) if bad else row[name]}")
        if not bad:
            rc = rc or 4

    legacy = differs(ref, units(load(meta, "legacy")))
    row["legacy_b115"] = f"DIFFERS in {legacy}" if legacy else "equal"
    print(f"{'ok  ' if legacy else 'WARN'} meta={meta} legacy b115f8180 behaviour (SEQRM=1 BULKH=2): "
          f"{'differs from reference in ' + str(legacy) if legacy else 'EQUAL to the reference, the CPU test does not reproduce the GPU defect'}")
    row["legacy_reproduces"] = bool(legacy)
    summary[meta] = row

if rc == 0 and not all(r.get("legacy_reproduces") for r in summary.values()):
    rc = 3
(out / "summary.json").write_text(json.dumps(dict(rc=rc, metas=summary), indent=2) + "\n")
print(f"verdict rc={rc}")
sys.exit(rc)
