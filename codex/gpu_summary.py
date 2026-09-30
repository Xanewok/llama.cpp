#!/usr/bin/env python3
import json
import pathlib
import statistics
import sys

out = pathlib.Path(sys.argv[1])
arms = ("up", "full", "sub3", "sub4", "sub5")
for request in ("tool", "think", "long"):
    runs = {arm: [json.loads((out / f"{arm}-{boot}-{request}.json").read_text()) for boot in (1, 2)] for arm in arms}
    assert len({r["sha256"] for arm in ("up", "full") for r in runs[arm]}) == 1, f"{request}: up/full seed identity failed"
    for arm in arms:
        values = []
        for boot, r in enumerate(runs[arm], 1):
            t, c = r["timings"], r["counters"]
            rounds = c["spec_decode_num_drafts_total"]
            drafted = c["spec_decode_num_draft_tokens_total"]
            accepted = c["spec_decode_num_accepted_tokens_total"]
            assert rounds > 0 and drafted > 0, f"{arm}: no speculative rounds"
            assert drafted == t["draft_n"] and accepted == t["draft_n_accepted"], "counter/response mismatch"
            values.append((t["predicted_ms"] / rounds, accepted / drafted, 1 + accepted / rounds,
                           t["predicted_n"] * 1000 / t["predicted_ms"]))
            print(f"M: {arm} boot={boot} {request} rounds={rounds:g} timings={t}")
        avg = [statistics.mean(v[i] for v in values) for i in range(4)]
        print(f"D: {arm} {request}: {avg[0]:.3f} ms/round, acceptance={avg[1]:.5f}, "
              f"{avg[2]:.3f} tokens/round, {avg[3]:.3f} tok/s; mean of measured boots")
    print(f"M: {request}: up/full seed identity passed")
print("D: ms/round uses generation wall time / measured verify-round count; includes partial rounds and host work.")
print("D: tokens/round is 1 + accepted/rounds; tok/s uses emitted tokens, including the initial target-only draw.")
