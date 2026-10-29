"""PLAN.md ledger KLD values for HQ checkpoints (and the llama quants quoted beside them).

Contract: these are `ledger-prior` rows under their own reference key `kld@ledger-q6k-8k`: KL against the
Swift 1.5 Q6_K reference on 30k assistant/doc rows at contexts <= 8k, run through the vLLM lane for HQ
checkpoints (PLAN.md Executive summary item 1). That scale is not REF6 (64k, llama lane) and not REFBF; the
report shows these values as context and never lets them satisfy or fail a constraint.
Each value is re-found in PLAN.md at ingest time; an entry whose text no longer matches is flagged and dropped.
"""
import re
from pathlib import Path

from ..store import Skip, sha256_file, stable_id
from .base import Batch, measurement
from .kld import weights_config

METRIC = "kld@ledger-q6k-8k"
# (model_id, value, regex that must match one PLAN.md line, what the line says)
ENTRIES = [
    ("swift-1.5:IQ4_XS", 0.0109, r"Q5_K_M 0\.0040, IQ4_XS 0\.0109", "Executive summary item 1"),
    ("swift-1.5:Q5_K_M", 0.0040, r"Q5_K_M 0\.0040, IQ4_XS 0\.0109", "Executive summary item 1"),
    ("swift-1.5:W4A16-syv-fast", 0.0254, r"W4A16 0\.0254\*\* \(2\.3", "Executive summary item 1 (HQ 1.5 g128, includes vLLM numerics)"),
    ("swift-1.0:W4A16-syv-fast", 0.0353, r"HQ 1\.0 0\.0353", "Executive summary item 1 (HQ 1.0 g128)"),
    ("swift-1.5:W4A16-g32-attn-int8", 0.0159, r"g32 \+ int8 attn 0\.0159", "goal matrix 10:45 (g32 + int8 attn)"),
    ("swift-1.5:W4A16-g32-mixed-int8", 0.0133, r"mixed 0\.0133 \(does not fit", "goal matrix 10:45 (mixed int8)"),
    ("swift-1.5:W4A16-g32-attn-int8-codexh", 0.0134, r"best `qkvo_codexh`.*KLD 0\.0134 \(M\)", "V5 result 13:35 (qkvo_codexh = the codexh checkpoint recipe)"),
]


def parse(path):
    p = Path(path)
    lines = p.read_text(errors="replace").splitlines()
    b = Batch()
    for model_id, value, pat, what in ENTRIES:
        hit = next((i for i, ln in enumerate(lines, 1) if re.search(pat, ln)), None)
        if hit is None:
            b.flag(p, f"ledger entry {model_id} {value} no longer found in the file (pattern {pat!r}); not ingested")
            continue
        cfg = weights_config(model_id, "unknown")
        cid = b.add_config(cfg)
        text = lines[hit - 1]
        prov = {"out_dir": str(p.parent), "summary_path": str(p), "summary_sha256": sha256_file(p)}
        b.measurements.append(measurement(
            source="ledger", metric=METRIC, value=value, unit="nats", config_id=cid, workload_id="kld:ledger-q6k-8k",
            dims={"bucket": "all_prefill", "scale": "Q6_K reference, 30k rows, contexts <= 8k", "where": what}, n=None,
            status="ledger-prior", reason=f"{p.name}:{hit}: {what}", prov=prov, item_id=f"{p.name}:{hit}",
            ident=stable_id(model_id, value, text)))
    if not b.measurements:
        raise Skip(f"{p}: a PLAN.md without the ledger KLD entries; not the ledger")
    return b
