#!/usr/bin/env python3
import csv
import json
import pathlib
import statistics
import sys

out = pathlib.Path(sys.argv[1])
for request in ("tool", "think", "long"):
    for k in (3, 4, 5):
        runs = {arm: [json.loads((out / f"{arm}{k}-{boot}-{request}.json").read_text()) for boot in (1, 2)]
                for arm in ("plain", "chain")}
        assert len({r["sha256"] for arm in runs for r in runs[arm]}) == 1, f"{request} k={k}: completion identity failed"
        assert len({json.dumps(r["counters"], sort_keys=True) for arm in runs for r in runs[arm]}) == 1, f"{request} k={k}: acceptance/counter identity failed"
        times = {}
        for arm in runs:
            values = []
            for boot, r in enumerate(runs[arm], 1):
                t, c = r["timings"], r["counters"]
                rounds = c["spec_decode_num_drafts_total"]
                drafted = c["spec_decode_num_draft_tokens_total"]
                accepted = c["spec_decode_num_accepted_tokens_total"]
                assert rounds > 0 and drafted > 0, f"{arm}: no speculative rounds"
                assert drafted == t["draft_n"] and accepted == t["draft_n_accepted"], "counter/response mismatch"
                values.append(t["predicted_ms"] / rounds)
                print(f"M: {arm}{k} boot={boot} {request} rounds={rounds:g} timings={t}")
            times[arm] = values
            print(f"D: {arm}{k} {request}: {statistics.mean(values):.6f} ms/round; mean of measured boots")
        delta = [b - a for a, b in zip(times["plain"], times["chain"])]
        print(f"D: k={k} {request} chain - plain: {statistics.mean(delta):.6f} ms/round; paired deltas {delta}")
        print(f"M: k={k} {request}: completion and acceptance identity passed in both boots")
for k in (3, 4, 5):
    memory = {}
    for arm in ("plain", "chain"):
        by_gpu = {}
        for boot in (1, 2):
            for uuid, value in csv.reader((out / f"{arm}{k}-{boot}.memory.csv").read_text().splitlines()):
                by_gpu.setdefault(uuid.strip(), []).append(float(value.strip().split()[0]))
        memory[arm] = {uuid: statistics.mean(values) for uuid, values in by_gpu.items()}
    assert memory["plain"].keys() == memory["chain"].keys(), "GPU set changed"
    for uuid in memory["plain"]:
        print(f"D: k={k} {uuid} chain - plain memory: {memory['chain'][uuid] - memory['plain'][uuid]:.1f} MiB; includes scratch and allocator rounding")
print("D: ms/round is measured generation wall time / measured verify-round count; it includes partial rounds and host work.")
