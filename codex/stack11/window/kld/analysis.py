"""KLDO stays the historical top-64 estimator; KLF1 supplies full-vocab tail KL."""
import json
import struct
from pathlib import Path

import numpy as np


def tokens(path):
    data = Path(path).read_bytes()
    if data[:4] != b"KLDT" or len(data) < 8:
        raise ValueError("bad KLDT header")
    count = struct.unpack_from("<i", data, 4)[0]
    out, off = [], 8
    if not 0 < count < 100:
        raise ValueError("bad sequence count")
    for _ in range(count):
        n = struct.unpack_from("<i", data, off)[0]
        off += 4
        if n < 2 or off + n * 4 > len(data):
            raise ValueError("bad token length")
        out.append(np.frombuffer(data, "<i4", n, off))
        off += n * 4
    if off != len(data):
        raise ValueError("trailing tokens")
    return out


def records(path, lengths=None):
    data = Path(path).read_bytes()
    if len(data) < 16 or data[:4] != b"KLDO":
        raise ValueError("bad KLDO header")
    k, vocab, count = struct.unpack_from("<3i", data, 4)
    if not 0 < k <= vocab or vocab > 1000000 or not 0 < count < 100:
        raise ValueError("bad KLDO dimensions")
    dtype = np.dtype([("lp_next", "<f4"), ("entropy", "<f4"),
                      ("ids", "<i4", (k,)), ("lps", "<f4", (k,))])
    seqs, off = [], 16
    for _ in range(count):
        n = struct.unpack_from("<i", data, off)[0]
        off += 4
        if n < 1 or off + n * dtype.itemsize > len(data):
            raise ValueError("bad KLDO row count")
        rows = np.frombuffer(data, dtype, n, off)
        if not all(np.isfinite(rows[f]).all() for f in ("lp_next", "entropy", "lps")):
            raise ValueError("non-finite KLDO")
        if np.any(rows["ids"] < 0) or np.any(rows["ids"] >= vocab):
            raise ValueError("bad top-K id")
        if np.any(rows["lps"] > 1e-5) or np.any(np.diff(rows["lps"], axis=1) > 0):
            raise ValueError("bad sorted log probabilities")
        seqs.append(rows)
        off += n * dtype.itemsize
    if off != len(data):
        raise ValueError("trailing KLDO bytes")
    if lengths is not None and [len(s) for s in seqs] != list(lengths):
        raise ValueError("KLDO shape mismatch")
    return vocab, seqs


def full_rows(path, toks, tail, vocab):
    data = np.memmap(path, mode="r", dtype="u1")
    if bytes(data[:4]) != b"KLF1" or len(data) < 12:
        raise ValueError("bad KLF1 header")
    v, count = struct.unpack_from("<2i", data, 4)
    if (v, count) != (vocab, len(toks)):
        raise ValueError("KLF1 dimensions mismatch")
    out, off = [], 12
    for t in toks:
        start, n = struct.unpack_from("<2i", data, off)
        off += 8
        if (start, n) != (max(0, len(t) - tail), min(tail, len(t)) - 1):
            raise ValueError("KLF1 positions mismatch")
        size = n * v * 4
        if off + size > len(data):
            raise ValueError("truncated KLF1")
        out.append(np.ndarray((n, v), "<f4", data, off))
        off += size
    if off != len(data):
        raise ValueError("trailing KLF1 bytes")
    return out


def assistant(t):
    inside = False
    mask = np.zeros(len(t), bool)
    i = 0
    while i < len(t):
        if tuple(t[i:i + 3]) == (248045, 74455, 198):
            inside, i = True, i + 3
            continue
        mask[i] = inside
        if t[i] == 248046:
            inside = False
        i += 1
    return mask[1:]


def normalized(rows):
    x = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("non-finite full rows")
    mx = x.max(axis=1, keepdims=True)
    z = mx + np.log(np.exp(x - mx).sum(axis=1, keepdims=True))
    if np.any(np.abs(z) > 1e-5):
        raise ValueError("full log-prob mass mismatch")
    return x - z


def exact(p, q):
    out = []
    for i in range(0, len(p), 8):
        lp, lq = normalized(p[i:i + 8]), normalized(q[i:i + 8])
        d = lp - lq
        # KL = sum(p * (log(p/q) + q/p - 1)); this avoids first-order cancellation.
        term = np.exp(lp) * (d + np.expm1(-np.maximum(d, -40)))
        # Use q directly where q/p can overflow.
        far = d < -40
        term[far] = np.exp(lp[far]) * (d[far]-1) + np.exp(lq[far])
        kl = term.sum(axis=1)
        if not np.isfinite(kl).all() or np.any(kl < -1e-12):
            raise ValueError("invalid exact KL")
        out.extend(np.maximum(kl, 0))
    return np.asarray(out)


# Two-sided 95% Student t quantiles by df = n_sequences - 1.
T95 = {1: 12.706204736432095, 2: 4.302652729911275, 3: 3.182446305284263, 4: 2.776445105197793,
       5: 2.5705818366147395, 6: 2.4469118487916806, 7: 2.3646242510102993, 8: 2.306004135033371,
       9: 2.2621571627409915, 10: 2.2281388519649385}


def stats(xs):
    if not xs or any(len(x) == 0 for x in xs):
        raise ValueError("insufficient rows")
    a = np.concatenate(xs)
    means = np.array([x.mean() for x in xs])
    se = float(means.std(ddof=1) / np.sqrt(len(means))) if len(means) > 1 else None
    return dict(n=int(len(a)), mean=float(a.mean()), p99=float(np.percentile(a, 99)),
                max=float(a.max()), min=float(a.min()), positive=int((a > 0).sum()),
                negative=int((a < 0).sum()), zero=int((a == 0).sum()),
                per_sequence=means.tolist(), sequence_equal_mean=float(means.mean()), sequence_se=se,
                # Widen the sequence-mean interval to center on the row-pooled estimate.
                halfwidth95=(T95[len(means)-1] * se + abs(means.mean()-a.mean())) if len(means)-1 in T95 else None)


def validate_arm(out, name, toks, tail):
    vocab, rec = records(out / f"{name}.kldo", [len(t) - 1 for t in toks])
    full = full_rows(out / f"{name}.klf", toks, tail, vocab)
    for t, r, f in zip(toks, rec, full):
        start = len(t) - tail
        for i in range(0, len(f), 8):
            raw = np.asarray(f[i:i + 8], dtype=np.float64)
            normalized(raw)
            row = r[start + i:start + i + len(raw)]
            if not np.array_equal(raw[np.arange(len(raw)), t[start + i + 1:start + i + 1 + len(raw)]], row["lp_next"]):
                raise ValueError("next-token row alignment mismatch")
            top = np.take_along_axis(raw, row["ids"], axis=1)
            if not np.array_equal(top, row["lps"]):
                raise ValueError("top-K row alignment mismatch")
    return vocab, rec, full


def compare(p, q):
    """Per row of p vs q: KL(p||q) in nats, top-1 flip, max |dlogp|, bit-identical row."""
    kl, flip, dmax, same = [], [], [], []
    for i in range(0, len(p), 8):
        a, b = np.asarray(p[i:i + 8]), np.asarray(q[i:i + 8])
        same.extend((a == b).all(axis=1))
        kl.extend(exact(a, b))
        la, lb = normalized(a), normalized(b)
        flip.extend(la.argmax(axis=1) != lb.argmax(axis=1))
        dmax.extend(np.abs(la - lb).max(axis=1))
    return dict(kl=np.asarray(kl), flip=np.asarray(flip), dmax=np.asarray(dmax), same=np.asarray(same))


def summarize(cmps, masks, tail):
    """cmps: one compare() per sequence. Buckets: every tail row; rows whose next token is an assistant token."""
    out = {}
    for bucket in ("all_tail", "assistant_tail"):
        sel = [np.ones(tail - 1, bool) if bucket == "all_tail" else m[-(tail - 1):] for m in masks]
        pick = {k: [c[k][m] for c, m in zip(cmps, sel)] for k in ("kl", "flip", "dmax", "same")}
        keep = [i for i, x in enumerate(pick["kl"]) if len(x)]
        out[bucket] = None if not keep else dict(
            kl=stats([pick["kl"][i] for i in keep]), rows=int(sum(len(pick["kl"][i]) for i in keep)),
            top1_flips=int(sum(pick["flip"][i].sum() for i in keep)),
            rows_bitwise_equal=int(sum(pick["same"][i].sum() for i in keep)),
            max_abs_dlogp=float(max(pick["dmax"][i].max() for i in keep)))
    return out


def verdict(main, spread, expected=1e-6, falsifier=1e-5, ratio=10.0, f16=None):
    """Report-only: pre-registered expected and falsifier (absolute, or > ratio x the repeat arms' spread). The f16-KV control's mean
    KL to base at the same depth is a recorded yardstick, not a falsifier (gdn4 alone measured about its size, 10-09)."""
    mean = main["mean"]
    worst = max(spread)
    over_spread = bool(worst > 0 and mean > ratio * worst)
    over_f16 = bool(f16 is not None and mean > f16)
    return dict(mean_kl_candidate_to_base=mean, expected_le=expected, falsifier_gt=falsifier, spread_ratio=ratio,
                repeat_spread_mean_kl=worst, f16_control_mean_kl=f16, meets_expected=bool(mean <= expected),
                exceeds_absolute_falsifier=bool(mean > falsifier), exceeds_spread_falsifier=over_spread,
                exceeds_f16_control=over_f16, falsified=bool(mean > falsifier or over_spread),
                note="the spread criterion applies only when a repeat arm differs from its original (spread > 0)")


def sensitivity_control(rows, eps=2e-3):
    """Known-size perturbation (expected KL ~ eps^2/4 ~ 1e-6, the pre-registered bound): measured must match 0.5*Var_p(delta)."""
    lp = normalized(np.asarray(rows, dtype=np.float64))
    delta = eps * np.sin(np.arange(lp.shape[1]) * 0.37)[None, :] * np.linspace(0.5, 1.5, len(lp))[:, None]
    q = lp + delta
    q = (q - (q.max(axis=1, keepdims=True) + np.log(np.exp(q - q.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)))).astype("<f4")
    d = q.astype(np.float64) - lp
    p = np.exp(lp)
    want = 0.5 * ((p * d * d).sum(axis=1) - (p * d).sum(axis=1) ** 2)
    got = exact(lp.astype("<f4"), q)
    ratio = float(got.sum() / want.sum())
    if not 0.95 < ratio < 1.05:
        raise ValueError(f"sensitivity control off: measured/expected = {ratio}")
    return dict(eps=eps, mean_expected=float(want.mean()), mean_measured=float(got.mean()), ratio=ratio)


PAIRS = {"cand_to_base": ("cand", "base"), "cand_repeat_to_base_repeat": ("cand_repeat", "repeat"),
         "base_repeat_to_base": ("repeat", "base"), "cand_repeat_to_cand": ("cand_repeat", "cand"),
         "negative_f16_to_base": ("f16", "base")}


def analyze(config, out, stub=False):
    out = Path(out)
    result = {"labels": "M: CPU stub dumps; D: KL/statistics" if stub else "M: GPU dumps; D: KL/statistics",
              "report_only": "no resolution gate: the measured KL is the answer", "depths": {}}
    for depth in config["depths"]:
        d = out / depth["name"]
        toks, tail = tokens(depth["tokens"]), depth["tail"]
        arms = {n: validate_arm(d, n, toks, tail) for n in depth["arms"]}
        if len({a[0] for a in arms.values()}) != 1:
            raise ValueError("vocab mismatch")
        masks = [assistant(t) for t in toks]
        r = {"tail": tail, "sequence_tokens": [len(t) for t in toks], "arms": depth["arms"], "pairs": {}}
        for label, (a, b) in PAIRS.items():
            if a in arms and b in arms:
                r["pairs"][label] = summarize([compare(x, y) for x, y in zip(arms[a][2], arms[b][2])], masks, tail)
        if "negative_f16_to_base" in r["pairs"] and not r["pairs"]["negative_f16_to_base"]["all_tail"]["kl"]["max"] > 1e-12:
            raise ValueError("negative control has zero KL")
        spread = [r["pairs"][k]["all_tail"]["kl"]["mean"] for k in ("base_repeat_to_base", "cand_repeat_to_cand")]
        f16 = r["pairs"].get("negative_f16_to_base")
        vp = config.get("verdict", {})
        r["verdict_all_tail"] = verdict(r["pairs"]["cand_to_base"]["all_tail"]["kl"], spread, f16=f16 and f16["all_tail"]["kl"]["mean"],
                                        **vp.get("all_tail", {}))
        a = r["pairs"]["cand_to_base"]["assistant_tail"]
        fa = f16 and f16["assistant_tail"]
        r["verdict_assistant_tail"] = verdict(a["kl"], spread, f16=fa and fa["kl"]["mean"], **vp.get("assistant_tail", {})) if a else None
        if "sensitivity_control" not in result:
            result["sensitivity_control"] = sensitivity_control(arms["base"][2][0][:32])
        result["depths"][depth["name"]] = r
    (out / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
