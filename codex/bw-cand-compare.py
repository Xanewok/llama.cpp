#!/usr/bin/env python3
"""Compare the bw-cand-driver runs in OUT: usage bw-cand-compare.py OUT

Runs are OUT/<meta>-<spec>-<temp>-<arm>.{json,log}. Per (meta, spec, temp) every arm present must give the same tokens,
draft counters and (spec 0) post-sampling candidate sets with probabilities, bitwise. Fallback counts are diagnostic:
the 4B fixture has a tied LM head, so its Meta contexts refuse backend sampling before testing the sampler chain.
The CUDA depth item's canary_log proves candidate engagement and the stack6 refusal.
Exit: 0 pass | 2 identity failure | 8 missing or unreadable run.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REFUSAL = re.compile(r"set_sampler: [^\n]*using CPU")
TP_REFUSAL = re.compile(r"set_sampler: seq_id = \d+: backend sampling with SPLIT_MODE_TENSOR needs a sampler chain whose first active sampler is top-k")
NAME = re.compile(r"(\d)-(\d)-(\d)-(cand|host|refused)\.json")


def main(out):
    out = Path(out)
    groups, rc = defaultdict(dict), 0
    for path in sorted(out.glob("*.json")):
        m = NAME.fullmatch(path.name)
        if not m:
            continue
        meta, spec, temp, arm = m.groups()
        try:
            groups[(meta, spec, temp)][arm] = (json.loads(path.read_text()), path.with_suffix(".log").read_text(errors="replace"))
        except (OSError, ValueError) as error:
            print(f"FAIL unreadable run {path}: {error}")
            return 8
    expected = {(str(meta), str(spec), str(temp)): set(arms)
                for meta, spec, arms in ((1, 1, ('cand', 'host', 'refused')), (1, 0, ('cand', 'host', 'refused')), (0, 1, ('cand', 'host')))
                for temp in (0, 1)}
    if {key: set(arms) for key, arms in groups.items()} != expected:
        print(f"FAIL incomplete run matrix in {out}: { {key: sorted(arms) for key, arms in groups.items()} }")
        return 8

    for (meta, spec, temp), arms in sorted(groups.items()):
        tag = f"meta={meta} spec={spec} temp={temp}"
        if "cand" not in arms or len(arms) < 2:
            print(f"FAIL {tag}: need the cand arm and a reference, have {sorted(arms)}")
            return 8
        ref, _ = arms["cand"]
        if "host" not in arms:
            print(f"FAIL {tag}: missing host baseline")
            return 8
        n_fallback = {arm: len(REFUSAL.findall(log)) for arm, (_, log) in arms.items()}
        n_tp = {arm: len(TP_REFUSAL.findall(log)) for arm, (_, log) in arms.items()}
        for arm, (run, log) in sorted(arms.items()):
            if arm == "cand":
                continue
            for key in ("tokens", "stats", "probs"):
                if run[key] != ref[key]:
                    first = next((i for i, (a, b) in enumerate(zip(run[key], ref[key])) if a != b), None) if isinstance(ref[key], list) else None
                    print(f"FAIL {tag}: {arm} vs cand: {key} differ (first index {first})")
                    rc = 2
        if spec == "0" and not all(p["top"] for p in ref["probs"]):
            print(f"FAIL {tag}: an empty candidate set in the cand arm")
            rc = rc or 2
        print(f"{tag}: arms {sorted(arms)}, set_sampler CPU fallbacks {n_fallback}, top-k refusals {n_tp}, {len(ref['tokens'])} tokens, stats {ref['stats']}, "
              f"{sum(len(p['top']) for p in ref['probs'])} candidate entries: {'OK' if rc == 0 else 'see above'}")
    print("PASS CPU identity; backend path proof belongs to the CUDA canary" if rc == 0 else f"FAIL rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
