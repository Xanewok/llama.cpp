#!/usr/bin/env python3
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from spec_tf import array, load_stream, harmonic


def distance(ids, probs, row):
    a = {int(i): float(p) for i, p in zip(ids, probs) if i >= 0 and p > 0}
    b = {int(i): float(p) for i, p in row if p > 0}
    return max((abs(a.get(i, 0) - b.get(i, 0)) for i in a.keys() | b.keys()), default=0), a.keys() != b.keys()


def check(target, arm, trace, streams, tolerance):
    header = json.loads((target / "header.json").read_text())
    b = json.loads((arm / "header.json").read_text())
    rounds = defaultdict(list)
    for line in trace.read_text().splitlines():
        record = json.loads(line)
        rounds[record["r"]].append(record)
    records, errors = [], []
    for i, stream in enumerate(streams):
        n, prompt = len(stream["prompt"]) + len(stream["continuation"]), len(stream["prompt"])
        pids = array(target / str(i), "p_ids", "<i4", [n, header["p_width"]])
        pprobs = array(target / str(i), "p_probs", "<f4", pids.shape)
        qids = array(arm / str(i), "q_sample_ids", "<i4", [n - prompt - 1, b["k"], 10])
        qprobs = array(arm / str(i), "q_probs", "<f4", qids.shape)
        t = prompt
        p_error = q_error = 0.0
        p_count = q_count = support_changes = censored = 0
        emitted = []
        for row in rounds[i]:
            L = row["L"]
            if "pall" not in row or "qall" not in row:
                raise ValueError("trace must contain full probability rows")
            for j, prow in enumerate(row["pall"]):
                if t + j >= n - 1:
                    censored += 1
                    continue
                error, support = distance(pids[t + j], pprobs[t + j], prow)
                p_error = max(p_error, error)
                p_count += 1
                support_changes += support
            for j in range(min(L + 1, row["k"])):
                if t + j >= n - 1:
                    censored += 1
                    continue
                error, support = distance(qids[t - prompt, j], qprobs[t - prompt, j], row["qall"][j])
                q_error = max(q_error, error)
                q_count += 1
                support_changes += support
            emitted.extend(row["d"][:L] + [row["emit"]])
            t += L + 1
        actual = stream["continuation"][1:]
        if emitted[:min(len(actual), len(emitted))] != actual[:len(emitted)] or len(actual) - len(emitted) > 1:
            errors.append(f"stream {i}: trace emitted tokens do not match the stream")
        if p_error > tolerance or q_error > tolerance:
            errors.append(f"stream {i}: p/q error {p_error}/{q_error} > {tolerance}")
        _, beta, _, tt = load_stream(target, arm, i)
        measured = float(np.mean([r["L"] + 1 for r in rounds[i]]))
        predicted = harmonic(tt)
        records.append({"stream": i, "rounds_M": len(rounds[i]), "p_rows_M": p_count, "q_rows_M": q_count,
                        "p_max_abs_error_M": p_error, "q_max_abs_error_M": q_error,
                        "support_changes_M": support_changes, "censored_rows_M": censored,
                        "measured_tokens_per_round_M": measured, "predicted_tokens_per_round_G": predicted,
                        "selection_bias_D": predicted - measured, "beta1_mean_D": float(beta[:, 0].mean())})
    return {"streams": records, "errors": errors, "tolerance": tolerance,
            "measured_tokens_per_round_M": float(np.mean([s["measured_tokens_per_round_M"] for s in records])),
            "predicted_tokens_per_round_G": float(np.mean([s["predicted_tokens_per_round_G"] for s in records]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--arm", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--streams", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=1e-3)
    args = p.parse_args()
    result = check(args.target, args.arm, args.trace, json.loads(args.streams.read_text()), args.tolerance)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return bool(result["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
