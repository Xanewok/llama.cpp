"""Replay real Meta fold-off artifacts as a silently disengaged fold-on arm."""
import argparse
import json
from pathlib import Path

from fold_cpu_identity import compare, engaged, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("off", type=Path)
    args = ap.parse_args()
    off = load(args.off)
    assert "Meta(" in Path(off["log"]).read_text(), "control requires a real Meta artifact"
    silent_on = dict(off, fold_on=1)
    identity = compare(off, silent_on)
    errors = engaged("meta-silent-disengagement", off, silent_on)
    assert identity["identical"], "control must preserve tokens, IDs, probabilities and counters"
    assert any("< 0.9 x draft rounds" in error for error in errors), errors
    assert any("C phases" in error for error in errors), errors
    print(json.dumps({"identity_would_pass": True, "engagement_rejected": errors}, indent=2))


if __name__ == "__main__":
    main()
