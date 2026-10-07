"""Summarize recorded np2 driver evidence; do not run a model."""
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for meta in (0, 1):
    off = json.loads((root / f"{meta}-0.json").read_text())
    on = json.loads((root / f"{meta}-1.json").read_text())
    rows = []
    for chain in (0, 1):
        log = (root / f"{meta}-{chain}.log").read_text()
        rounds = []
        for match in re.finditer(r"NP2_ROUND round=(\d+) path=(\w+) slots=(\d+) active=(\d+) seq0=(\d+) seq1=(\d+)", log):
            r, path, slots, active, n0, n1 = match.groups()
            rounds.append(dict(round=int(r), path=path, slots=int(slots), active=int(active), lengths=[int(n0), int(n1)]))
        rows.append(rounds)
    dumps = [(root / f"{meta}-{chain}.dump.txt").read_text().splitlines() for chain in (0, 1)]
    mismatch = next(((i, a.split(), b.split()) for i, (a, b) in enumerate(zip(*dumps)) if a.split()[2] != b.split()[2]), None)
    first = None
    if mismatch:
        i, a, b = mismatch
        first = dict(row=i, off=a, on=b, candidate_ids_equal=a[3:13] == b[3:13], q_equal=a[13:] == b[13:])
    tokens = []
    for a, b in zip(off["phases"], on["phases"]):
        tokens.append(dict(phase=a["phase"], off_stats=a["stats"], on_stats=b["stats"],
                           first_token_mismatch=[next((i for i, (x, y) in enumerate(zip(xrow, yrow)) if x != y), None)
                                                 for xrow, yrow in zip(a["tokens"], b["tokens"])]))
    shapes = []
    for chain in (0, 1):
        phases = [ph for line in (root / f"{meta}-{chain}.trace.jsonl").read_text().splitlines()
                  for ph in json.loads(line)["phases"]]
        shapes.append([(ph["phase"], ph["n_tokens"]) for ph in phases])
    summary[str(meta)] = dict(equal=off == on, rounds=rows, first_draft_id_mismatch=first,
                              phases=tokens, shapes_equal=shapes[0] == shapes[1], shapes=shapes,
                              dump_equal=dumps[0] == dumps[1])
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for meta, cell in summary.items():
    print(meta, json.dumps({k: v for k, v in cell.items() if k not in ("rounds", "shapes")}))
