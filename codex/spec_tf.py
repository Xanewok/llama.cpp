#!/usr/bin/env python3
"""Teacher-forced acceptance. Absolute tokens/round estimates are labelled G."""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np


def estimate_rows(p_ids, p, q_ids, q, tokens):
    """Arrays are [starts, steps, support]; tokens are the true next tokens."""
    overlap = p_ids[..., :, None] == q_ids[..., None, :]
    beta = np.minimum(p[..., :, None], q[..., None, :])
    beta = (beta * overlap).sum(axis=(-2, -1))
    px = (p * (p_ids == tokens[..., None])).sum(axis=-1)
    qx = (q * (q_ids == tokens[..., None])).sum(axis=-1)
    r = np.minimum(1.0, np.divide(qx, px, out=np.zeros_like(px), where=px > 0))
    weights = np.concatenate((np.ones_like(r[:, :1]), np.cumprod(r[:, :-1], axis=1)), axis=1)
    return beta, r, 1 + (weights * beta).sum(axis=1)


def draft_probs(ids, logits, subset=None, temp=1.0, top_k=10, suppress=()):
    logits = np.array(logits, dtype=np.float64, copy=True)
    valid = (ids >= 0) & ~np.isin(ids, suppress)
    if subset is not None:
        valid &= np.isin(ids, subset)
    count = valid.sum(axis=-1)
    if np.any(count < top_k):
        raise ValueError("top-128 coverage insufficient: fewer than top_k subset ids; use a real sliced-head run")
    if subset is not None:
        warnings.warn("Offline subsets are exact only if their top_k is covered by the recorded full-head top-128", stacklevel=2)
    logits[~valid] = -np.inf
    order = np.argsort(-logits, axis=-1, kind="stable")[..., :top_k]
    ids = np.take_along_axis(ids, order, axis=-1)
    logits = np.take_along_axis(logits, order, axis=-1) / temp
    q = np.exp(logits - logits.max(axis=-1, keepdims=True))
    q /= q.sum(axis=-1, keepdims=True)
    return ids, q


def array(root, name, dtype, shape):
    path = root / (name + ".bin")
    expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {path.stat().st_size}")
    return np.memmap(path, dtype=dtype, mode="r", shape=tuple(shape))


def load_stream(target, arm, stream, subset=None):
    a = json.loads((target / "header.json").read_text())
    b = json.loads((arm / "header.json").read_text())
    if a["format"] != "spec-tf-v1" or b["format"] != a["format"]:
        raise ValueError("unknown format")
    if b["target_header"] != a or b["stage"] != "B":
        raise ValueError("arm does not reference this exact target header")
    info = a["streams"][stream]
    root_a, root_b = target / str(stream), arm / str(stream)
    n, pk, k, qk = info["n_tokens"], a["p_width"], b["k"], b["q_width"]
    starts = array(root_b, "starts", "<i4", [info["n_continuation"] - 1, 1])[:, 0]
    ids = array(root_b, "q_ids", "<i4", [len(starts), k, qk])
    logits = array(root_b, "q_logits", "<f4", ids.shape)
    full = starts + k < n
    starts = starts[full]
    ids, logits = ids[full], logits[full]
    # Stored q uses the actual common_sampler chain, including model suppress tokens.
    if subset is None:
        q_ids = array(root_b, "q_sample_ids", "<i4", [info["n_continuation"] - 1, k, 10])[full]
        q = array(root_b, "q_probs", "<f4", [info["n_continuation"] - 1, k, 10])[full].astype(np.float64)
    else:
        q_ids, q = draft_probs(ids, logits, subset, b["temperature"], suppress=b["suppress_ids"])
    tokens = array(root_a, "tokens", "<i4", [n])
    idx = starts[:, None] + np.arange(k)[None, :]
    p_ids = array(root_a, "p_ids", "<i4", [n, pk])[idx]
    p = array(root_a, "p_probs", "<f4", [n, pk])[idx].astype(np.float64)
    beta, r, t = estimate_rows(p_ids, p, q_ids, q, tokens[idx + 1])
    return starts, beta, r, t


def harmonic(t):
    return float(1 / np.mean(1 / t))


def paired(base, arm):
    if base.shape != arm.shape or not len(base):
        raise ValueError("paired arms must have the same nonempty starts")
    w = 1 / base
    return float(np.sum(w * (arm - base)) / w.sum())


def bootstrap(base, arm, rng, draws=2000):
    values = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(len(base), size=len(base))
        values[i] = paired(base[idx], arm[idx])
    return float(values.std(ddof=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--arm", type=Path, action="append", required=True)
    parser.add_argument("--spec-draft-vocab", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error("--bootstrap must be >= 2")
    header = json.loads((args.target / "header.json").read_text())
    arms = [(p.name, p, None) for p in args.arm]
    for vocab in args.spec_draft_vocab:
        ids = json.loads(vocab.read_text())
        if not isinstance(ids, list) or len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("draft vocab must be a JSON list of unique nonnegative ids")
        arms.append((vocab.stem, args.arm[-1], ids))
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"absolute_label": "G", "delta_label": "G", "sample_unit": "stream", "arms": {}}
    rng = np.random.default_rng(args.seed)
    base = [load_stream(args.target, arms[0][1], i)[-1] for i in range(len(header["streams"]))]
    for label, root, subset in arms:
        streams, values, all_t = [], [], []
        for i in range(len(base)):
            starts, beta, r, t = load_stream(args.target, root, i, subset)
            np.savez(args.output / f"{label}-{i}.npz", starts=starts, beta=beta, r=r, T=t)
            delta = paired(base[i], t)
            values.append(delta)
            all_t.append(t)
            streams.append({"stream": i, "starts": len(t), "tokens_per_round_G": harmonic(t),
                            "paired_delta_G": delta, "per_start_bootstrap_se_D": bootstrap(base[i], t, rng, args.bootstrap)})
        values = np.array(values)
        report["arms"][label] = {"streams": streams, "tokens_per_round_G": harmonic(np.concatenate(all_t)),
                                 "paired_delta_G": float(values.mean()),
                                 "stream_se_D": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
