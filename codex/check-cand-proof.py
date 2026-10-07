#!/usr/bin/env python3
"""Replay canary signatures and CPU identity controls without running a model."""
import contextlib
import importlib.util
import io
import json
import re
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path


def main(out):
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("compare", root / "bw-cand-compare.py")
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    config = tomllib.loads((root / "cand-neginf-depth-j5q.toml.in").read_text())
    patterns = {arm: [p for p in config["arms"][arm]["canary_log"] if "sampler" in p or "set_logits" in p]
                for arm in ("s6", "cn")}
    def engaged(arm, log):
        return all(re.search(p, log) for p in patterns[arm])

    recorded = (root / "fixtures/stack6-tp-sampling.log").read_text()
    assert engaged("s6", recorded), "recorded stack6 must prove the top-k refusal"
    assert not engaged("cn", recorded), "frozen stack6 must fail the candidate proof"
    assert not engaged("cn", ""), "absence of fallback alone must not prove engagement"
    # No recorded candidate GPU run exists yet; check the new message's format separately.
    candidate = "sampler chain: logits -> logit-bias -> top-k -> dist\nset_logits: using backend candidates: n_cand = 24, n_ban = 3, top_k = 20\n"
    assert engaged("cn", candidate), "candidate message format"
    assert not engaged("cn", candidate.replace("n_ban = 3", "n_ban = 0")), "nonempty ban list required"
    assert not engaged("cn", candidate + recorded), "a fallback must fail candidate engagement"
    print("PASS canary signatures: real stack6 refusal accepted; stack6, empty log, zero bans and fallback rejected as cn")

    def verdict(path):
        with contextlib.redirect_stdout(io.StringIO()):
            return compare.main(path)

    assert verdict(out) == 0, "recorded CPU identity must pass"
    with tempfile.TemporaryDirectory(prefix="proof-control-", dir=Path(out).parent) as temp:
        dest = Path(temp)
        for path in Path(out).iterdir():
            if path.suffix in (".json", ".log"):
                shutil.copyfile(path, dest / path.name)
        path = dest / "1-1-1-host.json"
        run = json.loads(path.read_text())
        for key in ("tokens", "stats", "probs"):
            changed = json.loads(json.dumps(run))
            if key == "tokens":
                changed[key][0] += 1
            elif key == "stats":
                changed[key]["draft"] += 1
            else:
                changed[key] = [{"tok": -1, "prob": 0, "top": []}]
            path.write_text(json.dumps(changed))
            assert verdict(dest) == 2, f"{key} mutation escaped"
        path.write_text(json.dumps(run))
        assert verdict(dest) == 0, "restored CPU identity"
        path.unlink()
        assert verdict(dest) == 8, "missing run escaped"
    print("PASS CPU comparer controls: token, counter and probability mutations fail; missing run fails")


if __name__ == "__main__":
    main(sys.argv[1])
