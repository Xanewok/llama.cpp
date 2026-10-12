"""Top-K KL-divergence with a tail bucket, shared by analysis and the synthetic tests.

Per position, S = reference top-K ids plus the scored (next) token. p_i is known on S. q_i is known when i is
in the test top-K or is the next token; otherwise it is imputed as min(q_K, R/m): any id outside the test top-K
has q <= q_K (the test's K-th prob), and the m imputed ids share at most R, the test mass not yet accounted for.
KL is then taken between the two (|S|+1)-bin distributions (S, rest). With every q_i exact this coarse-grained
KL is a lower bound on the full-vocab KL (log-sum inequality); imputation keeps the bins a valid distribution.
"""
import struct
from pathlib import Path

import numpy as np

TINY = 1e-12


def load_kldo(path):
    b = Path(path).read_bytes()
    if b[:4] != b"KLDO":
        raise ValueError(f"{path}: not a KLDO file")
    k, n_vocab, n_seq = struct.unpack_from("<3i", b, 4)
    off, seqs = 16, []
    rec = np.dtype([("lp_next", "<f4"), ("entropy", "<f4"), ("ids", "<i4", (k,)), ("lps", "<f4", (k,))])
    for _ in range(n_seq):
        n_pos = struct.unpack_from("<i", b, off)[0]
        off += 4
        seqs.append(np.frombuffer(b, dtype=rec, count=n_pos, offset=off))
        off += n_pos * rec.itemsize
    full = None
    if off < len(b):
        if b[off:off + 4] != b"FULL":
            raise ValueError(f"{path}: trailing bytes")
        s, a, m = struct.unpack_from("<3i", b, off + 4)
        full = dict(seq=s, start=a, n=m, kld=np.frombuffer(b, "<f4", m, off + 16).astype(np.float64))
    return dict(k=k, n_vocab=n_vocab, seqs=seqs, full=full)


def kld_topk(ref, test, next_ids, k=None):
    """ref/test: structured record arrays of one sequence; next_ids: scored token per position.

    Returns per-position (kld, top1_agree, dlp_next)."""
    k = k or min(ref["ids"].shape[1], test["ids"].shape[1])
    rid, rlp = ref["ids"][:, :k], ref["lps"][:, :k].astype(np.float64)
    tid, tlp = test["ids"][:, :k], test["lps"][:, :k].astype(np.float64)
    r_next, t_next = ref["lp_next"].astype(np.float64), test["lp_next"].astype(np.float64)
    nxt = next_ids[:, None]

    match = rid[:, :, None] == tid[:, None, :]                      # (n, k, k)
    in_test = match.any(axis=2)
    q_lp = np.where(in_test, np.take_along_axis(tlp, match.argmax(axis=2), axis=1), np.nan)
    is_next = rid == nxt
    q_lp = np.where(is_next, t_next[:, None], q_lp)                 # next token's test logprob is always exact

    # Append the next token as an extra bin when it is outside the reference top-K.
    next_in_ref = is_next.any(axis=1)
    p_lp = np.concatenate([rlp, np.where(next_in_ref, -np.inf, r_next)[:, None]], axis=1)
    q_lp = np.concatenate([q_lp, np.where(next_in_ref, -np.inf, t_next)[:, None]], axis=1)

    q_known = np.where(np.isnan(q_lp), 0.0, np.exp(q_lp))
    next_in_test = (tid == nxt).any(axis=1)
    accounted = np.exp(tlp).sum(axis=1) + np.where(next_in_test, 0.0, np.exp(t_next))
    remaining = np.clip(1.0 - accounted, 0.0, None)
    missing = np.isnan(q_lp)
    m = missing.sum(axis=1)
    q_imp = np.minimum(np.exp(tlp[:, -1]), remaining / np.maximum(m, 1))
    q = np.where(missing, np.maximum(q_imp, TINY)[:, None], q_known)
    p = np.exp(p_lp)

    active = p > 0
    ps, qs = p.sum(axis=1), np.where(active, q, 0.0).sum(axis=1)
    p_tail = np.clip(1.0 - ps, TINY, None)
    q_tail = np.clip(1.0 - qs, TINY, None)
    body = np.where(active, p * (np.log(np.maximum(p, TINY)) - np.log(np.maximum(q, TINY))), 0.0).sum(axis=1)
    kl = body + p_tail * (np.log(p_tail) - np.log(q_tail))
    return np.maximum(kl, 0.0), rid[:, 0] == tid[:, 0], t_next - r_next


def exact_kld(p_logits, q_logits):
    lp = p_logits - np.logaddexp.reduce(p_logits, axis=-1, keepdims=True)
    lq = q_logits - np.logaddexp.reduce(q_logits, axis=-1, keepdims=True)
    return (np.exp(lp) * (lp - lq)).sum(axis=-1)


def records_from_logits(logits, next_ids, k):
    """Build KLDO-style records from dense logits (tests only)."""
    lp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    order = np.argsort(-lp, axis=-1)[:, :k]
    rec = np.zeros(len(lp), dtype=[("lp_next", "<f4"), ("entropy", "<f4"), ("ids", "<i4", (k,)), ("lps", "<f4", (k,))])
    rec["ids"] = order
    rec["lps"] = np.take_along_axis(lp, order, axis=1)
    rec["lp_next"] = lp[np.arange(len(lp)), next_ids]
    rec["entropy"] = -(np.exp(lp) * lp).sum(axis=1)
    return rec


ASSISTANT_HEADER = (248045, 74455, 198)   # "<|im_start|>assistant\n"
IM_END = 248046


def scored_rows(ids, kind):
    """Rows (row j predicts ids[j+1]) that enter the main aggregate.

    chat: rows whose target is assistant-authored (content, <think>, tool calls, closing <|im_end|>);
    doc: every row; anything else: none."""
    n = len(ids) - 1
    if kind == "doc":
        return np.ones(n, bool)
    if kind != "chat":
        return np.zeros(n, bool)
    ids = [int(x) for x in ids]
    inside, target, i = False, np.zeros(len(ids), bool), 0
    while i < len(ids):
        if tuple(ids[i:i + 3]) == ASSISTANT_HEADER:
            inside, i = True, i + 3
            continue
        target[i] = inside
        if ids[i] == IM_END:
            inside = False
        i += 1
    return target[1:]
