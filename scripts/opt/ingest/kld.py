"""KLD result tables (kld-j5 / vast-kld-l / vast-gguf-n / f32acc text tables, vast-refbf results.json).

Contract: the reference is part of the metric key (`kld@REF6`, `kld@REFBF`, ...). A table whose
reference cannot be read from its own text is skipped, never defaulted: REF6 (Q6_K f16 KV) and REFBF
(BF16 f16 KV) differ by ~0.001 and must not mix.
Rows describe *weights arms*; an arm is mapped to a model identity (`swift-1.5:J4-bf16`) and KV type by
the explicit ARMS table below, never by fuzzy matching. An arm absent from the table is stored as
`unmapped` and can never be joined to an engine config. Unknown KV is `unknown`, which also never joins.
Row ids derive from the file's content hash, so two different files with the same directory name do not collide.
"""
import json
import re
from pathlib import Path

from ..store import Reject, Skip, canonical_toml, config_id_of, sha256_file
from .base import Batch, load_json, measurement, need

KNOWN_REFS = {"REF6", "REF8", "REFBF"}
FLOAT = r"[-+]?\d+\.\d+"
BUCKETS = r"all_prefill|0-8k|8-32k|32k\+|8-16k|16-32k|32-48k|48k\+|decode_tail|pre_tail_prefill"
V = "swift-1.5:"

# arm label -> (model recipe, kv, evidence). Every entry is a statement about the dump, cited.
ARMS = {
    # vast-refbf/results.{json,txt}: loc_*/gn_* dumps use kld-j5's flags, f16 KV (kld-j5/99-j5-kld64k.sh KV=${KV:-f16});
    # l40_* labels carry their KV; loc_j4q8 is J4-fromQ8 (J4 quantized from the Q8_0 file), NOT q8_0 KV (REPORT.md claim 3)
    "l40_IQ4_f16": ("IQ4_XS", "f16/f16", "vast-refbf label"),
    "l40_J1_f16": ("IQ4_XS-ssmout6", "f16/f16", "J1 = IQ4_XS + ssm_out Q6_K = the ssmout6 file (sha 90a3b0e5, vast-gguf-m RECIPE)"),
    "l40_Q5KS_q8": ("Q5_K_S", "q8_0/q8_0", "vast-refbf label"),
    "loc_q5ks": ("Q5_K_S", "f16/f16", "vast-refbf RECIPE: same flags as kld-j5 (f16)"),
    "loc_j1": ("IQ4_XS-ssmout6", "f16/f16", "vast-refbf RECIPE: same flags as kld-j5 (f16)"),
    "loc_j4q8": ("J4-fromQ8", "f16/f16", "vast-refbf REPORT claim 3: J4-fromQ8"),
    "loc_j4bf16": ("J4-bf16", "f16/f16", "vast-refbf RECIPE: = kld-j5/out-b1/arms/J4.kldo"),
    "gn_J2": ("J2-bf16", "f16/f16", "vast-gguf-n RECIPE (f16 KV, BF16-built)"),
    "gn_J3": ("J3-bf16", "f16/f16", "vast-gguf-n RECIPE"),
    "gn_J4": ("J4-bf16", "f16/f16", "vast-gguf-n RECIPE / vast-refbf REPORT 'J4-bf16 (6000Ada)'"),
    "gn_Q5KS": ("Q5_K_S", "f16/f16", "vast-gguf-n RECIPE"),
    "REF6": ("Q6_K", "f16/f16", "vast-refbf header: REF6 = Q6_K f16 KV (L40)"),
    "REF8": ("Q8_0", "q8_0/q8_0", "vast-refbf header: REF8 = Q8_0 q8_0 KV"),
    "refbf_q8": ("BF16", "q8_0/q8_0", "vast-refbf RECIPE: BF16 weights, q8_0 KV"),
    "Q5KS_q8_ctl": ("Q5_K_S", "q8_0/q8_0", "vast-refbf RECIPE: cross-GPU control, q8_0 KV"),
    # kld-j5/out-b1 (local, f16 KV) and its L40 reference arms
    "J4": ("J4-bf16", "f16/f16", "kld-j5 99-j5-kld64k.sh: f16 KV; vast-refbf RECIPE: J4.kldo is J4-bf16"),
    "J5a": ("J5a-bf16", "f16/f16", "kld-j5 99-j5-kld64k.sh"),
    "J5q": ("J5q-bf16", "f16/f16", "kld-j5 99-j5-kld64k.sh"),
    "attnout8": ("attnout8-bf16", "f16/f16", "kld-j5 99-j5-kld64k.sh"),
    "fdown6": ("fdown6-bf16", "f16/f16", "kld-j5 99-j5-kld64k.sh"),
    "fdown6mid": ("fdown6mid-bf16", "f16/f16", "kld-j5 99-j5-kld64k.sh; file Swift-1.5-...-fdown6mid-bf16.gguf"),
    "gate5": ("gate5-bf16", "f16/f16", "kld-j5 99-j5-kld64k.sh"),
    "qkv6": ("qkv6-bf16", "f16/f16", "kld-j5 99-j5-kld64k.sh"),
    "l40_J4": ("J4-bf16", "f16/f16", "kld-j5 table: L40 reference arm"),
    "l40_Q5KS": ("Q5_K_S", "f16/f16", "kld-j5 table: L40 reference arm"),
    # vast-gguf-n/results.txt 64k columns (f16 KV everywhere, except l_Q5KS_q8)
    "J1ctl": ("IQ4_XS-ssmout6", "f16/f16", "vast-gguf-n: sha 90a3b0e5 = kld-l J1"),
    "J2": ("J2-bf16", "f16/f16", "vast-gguf-n"),
    "J3": ("J3-bf16", "f16/f16", "vast-gguf-n"),
    "Q5KS": ("Q5_K_S", "f16/f16", "vast-gguf-n"),
    "l_J1_f16": ("IQ4_XS-ssmout6", "f16/f16", "vast-gguf-n control row = kld-l J1_f16"),
    "l_Q5KS_q8": ("Q5_K_S", "q8_0/q8_0", "vast-gguf-n control row = kld-l Q5_K_S q8_0"),
    # f32acc/kld-smoke: <model>_<kv>_<off|on>[_rep]
    "J5q_f16_off": ("J5q-bf16", "f16/f16", "f32acc header: scored against REFBF, GGML_CUDA_FA_F32_ACC off"),
    "J5q_f16_off_rep": ("J5q-bf16", "f16/f16", "f32acc determinism re-run"),
    "J5q_f16_on": ("J5q-bf16", "f16/f16", "f32acc: GGML_CUDA_FA_F32_ACC on"),
}
# Determinism re-runs of an arm: stored, never a verdict source.
CONTROL_ARMS = {"J5q_f16_off_rep"}
VARIANT_ARMS = {"J5q_f16_on": "GGML_CUDA_FA_F32_ACC=1"}
_KLD_L_BASE = {"IQ4_XS": "IQ4_XS", "Q5_K_S": "Q5_K_S", "Q6_K": "Q6_K", "J1": "IQ4_XS-ssmout6"}


def arm_identity(arm):
    """(model_id | None, kv, note). kv is 'unknown' unless the table or the label states it."""
    a = arm.strip()
    if a in ARMS:
        recipe, kv, note = ARMS[a]
        return V + recipe, kv, note
    m = re.match(r"(IQ4_XS|Q5_K_S|Q6_K|J1)(?: \(IQ4_XS \+ ssm_out Q6_K\))?(?:, KV (?:(K (f16|q8_0) / V (f16|q8_0))|(f16)|(q8_0/q8_0)))?(?:, (?:run \d+|re-run))?$", a)
    if m:
        kv = f"{m.group(3)}/{m.group(4)}" if m.group(2) else "f16/f16" if m.group(5) else "q8_0/q8_0" if m.group(6) else "unknown"
        return V + _KLD_L_BASE[m.group(1)], kv, "vast-kld-l arm label"
    return None, "unknown", "arm not in the explicit table"


def weights_config(model_id, kv, arm=None, variant=None):
    canon = {"kind": "weights", "model_id": model_id or "unmapped", "kv": kv}
    if variant:
        canon["variant"] = variant
    if model_id is None:
        canon["arm"] = arm
    cid = config_id_of(canon)
    fam = model_id.split(":")[0] if model_id else None
    return {"id": cid, "config_id": cid, "engine": "weights", "canon": canon, "canonical_toml": canonical_toml(canon),
            "model_id": model_id, "model_label": model_id.split(":", 1)[1] if model_id else arm,
            "model_family": fam, "kv_k": kv.split("/")[0], "kv_v": kv.split("/")[-1], "kv": kv,
            "variant_id": variant, "derived_from": "kld-arm"}


class _Out:
    def __init__(self, path):
        self.b = Batch()
        self.path = Path(path)
        self.sha = sha256_file(path)
        self.unmapped = set()

    def _cfg(self, arm):
        model_id, kv, _ = arm_identity(arm)
        if model_id is None:
            self.unmapped.add(arm)
        return self.b.add_config(weights_config(model_id, kv, arm, VARIANT_ARMS.get(arm)))

    def _prov(self):
        return {"out_dir": str(self.path.parent), "summary_path": str(self.path), "summary_sha256": self.sha}

    def row(self, arm, ref, bucket, kld, se, n=None, extra_dims=None, status="measured", reason=None, control=False,
            kind="kld"):
        if not isinstance(kld, float) or kld < 0 or (se is not None and (not isinstance(se, float) or se < 0)):
            raise Reject(f"{self.path}: bad KLD cell for {arm}/{bucket}: {kld!r} +- {se!r}")
        cid = self._cfg(arm)
        dims = {"bucket": bucket, "arm": arm, **(extra_dims or {})}
        if control or arm in CONTROL_ARMS:
            dims["control"] = True
        self.b.measurements.append(measurement(
            source="kld", metric=f"{kind}@{ref}", value=kld, unit="nats", config_id=cid, workload_id="kld:agentic-sessions",
            dims=dims, n=n, spread=se, status=status, reason=reason, prov=self._prov(), item_id=self.path.parent.name,
            ident=self.sha))

    def finish(self):
        for a in sorted(self.unmapped):
            self.b.flag(self.path, f"arm {a!r} is not in the explicit arm table; stored as unmapped (never joined to a model)")
        return self.b


def _parse_j5_table(out, lines, ref, nseq):
    """Header: arm +GB MiB/GPU KLD seqSE | d vs J4 +-se neg ..."""
    cols = None
    for ln in lines:
        if ln.startswith("arm "):
            cols = ln
            continue
        if cols is None or not ln.strip() or ln.startswith("control drift") or ln.startswith("  "):
            continue
        m = re.match(rf"(?P<arm>\S+)\s+(?P<gb>{FLOAT})\s+(?P<mib>[+-]\d+)\s+(?P<kld>{FLOAT})\s+(?P<se>{FLOAT})\s*\|\s*(?P<rest>.*)$", ln)
        if not m:
            raise Reject(f"{out.path}: unparsable KLD row: {ln[:90]!r}")
        out.row(m["arm"], ref, "all_prefill", float(m["kld"]), float(m["se"]), n=nseq,
                extra_dims={"gb": float(m["gb"]), "mib_per_gpu": int(m["mib"])}, control=m["rest"].strip() == "control")
        d = re.match(rf"\s*(?P<d>{FLOAT})\s+(?P<dse>{FLOAT})\s+(?P<neg>\d+)/(?P<tot>\d+)", m["rest"])
        if d:
            _delta(out, m["arm"], ref, d)


def _delta(out, arm, ref, d):
    out.b.measurements.append(measurement(
        source="kld", metric=f"kld_delta@{ref}", value=float(d["d"]), unit="nats", config_id=out._cfg(arm),
        workload_id="kld:agentic-sessions", dims={"bucket": "all_prefill", "arm": arm, "vs": "J4", "neg_seqs": int(d["neg"])},
        n=int(d["tot"]), spread=float(d["dse"]), status="derived", prov=out._prov(), item_id=out.path.parent.name,
        ident=out.sha))


def _parse_by_reference(out, lines, nseq):
    """vast-refbf / kld-j5 ref8 'KLD by reference' table: `arm REF cell cell ...`, cell = mean+-seqSE per bucket."""
    start = next((i for i, ln in enumerate(lines) if ln.startswith("== KLD by reference")), None)
    if start is None:
        return 0
    cols = None
    n = 0
    for ln in lines[start + 1:]:
        if ln.startswith("== "):
            break
        if ln.startswith("arm "):
            cols = [c for c in ln.split()[2:] if re.fullmatch(BUCKETS, c)]
            continue
        m = re.match(r"(?P<arm>\S+)\s+(?P<ref>REF\w+)\s+(?P<cells>(?:\d+\.\d+\u00b1\d+\.\d+\s*)+)", ln)
        if not m or not cols:
            continue
        if m["ref"] not in KNOWN_REFS:
            raise Reject(f"{out.path}: unknown reference {m['ref']!r} in the by-reference table")
        for bucket, cell in zip(cols, re.findall(r"(\d+\.\d+)\u00b1(\d+\.\d+)", m["cells"])):
            out.row(m["arm"], m["ref"], bucket, float(cell[0]), float(cell[1]), n=nseq)
        n += 1
    return n


def _parse_f32acc(out, lines):
    """f32acc/kld-smoke: header states `scored against REFBF`; rows `arm bucket rows KLD seqSE top1`."""
    n = 0
    for ln in lines:
        m = re.match(rf"(?P<arm>\S+)\s+(?P<bucket>{BUCKETS})\s+(?P<rows>\d+)\s+(?P<kld>{FLOAT})\s+(?P<se>{FLOAT})\s+\d+\.\d+%\s*$", ln)
        if m:
            out.row(m["arm"], "REFBF", m["bucket"], float(m["kld"]), float(m["se"]), n=6, extra_dims={"rows": int(m["rows"])})
            n += 1
    return n


def _parse_gguf_n(out, lines):
    """vast-gguf-n/results.txt: the 64k columns are KL(REF_q6k_f16 || arm) = REF6; the 8k columns use another
    (local) Q6_K reference and are not read."""
    n = 0
    for ln in lines:
        m = re.match(r"(?P<arm>\S+)\s+\d+\s+[+-]\d.*\|\s*(?P<tail>all_prefill .*)$", ln)
        if not m:
            continue
        for bucket, kld, se in re.findall(rf"({BUCKETS}) (\d+\.\d+)\u00b1(\d+\.\d+)", m["tail"]):
            out.row(m["arm"], "REF6", bucket, float(kld), float(se), n=6)
        n += 1
    return n


def _sibling_json_is_kld(p):
    j = p.with_suffix(".json")
    if not j.exists():
        return False
    try:
        d = json.loads(j.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(d, dict) and {"refs", "kld"} <= set(d)


def parse_text(path):
    p = Path(path)
    if _sibling_json_is_kld(p) and p.name == "results.txt":
        raise Skip(f"{p}: rendered view of the sibling results.json, which is ingested instead")
    text = p.read_text(errors="replace")
    lines = text.splitlines()
    out = _Out(p)
    found = 0
    # kld-j5 style: "== KL(REF6 || arm), ...; n_seq = 6; ..." followed by an arm table
    for i, ln in enumerate(lines):
        m = re.match(r"== KL\((?P<ref>\w+) \|\| arm\)(?P<rest>.*)$", ln)
        if not m:
            continue
        ref = m["ref"]
        end = next((j for j in range(i + 1, len(lines)) if lines[j].startswith("== KL(")), len(lines))
        block = lines[i + 1:end]
        if ref in KNOWN_REFS:
            ns = re.search(r"n_seq = (\d+)", m["rest"])
            _parse_j5_table(out, block, ref, int(ns.group(1)) if ns else None)
            found += 1
        elif ref.startswith("ref_"):
            ns = re.search(r"over (\d+) sequences", m["rest"])
            for bl in block:
                mm = re.match(rf"(?P<arm>\S+)\s+(?:(?P<bytes>\d+)\s+)?(?P<kld>{FLOAT})\s+(?P<se>{FLOAT})\s+{FLOAT}\s+{FLOAT}\s+\d+\.\d+%", bl)
                if mm:
                    out.row(mm["arm"], ref, "all_prefill", float(mm["kld"]), float(mm["se"]), n=int(ns.group(1)) if ns else None,
                            extra_dims={"bytes": int(mm["bytes"])} if mm["bytes"] else None)
            found += 1
        else:
            raise Reject(f"{p}: unknown reference {ref!r} in table header")
    # vast-kld-l style: header names the reference in prose, one row per (arm, bucket)
    if found == 0 and lines and lines[0].startswith("KL(REF || arm), REF = Q6_K, KV f16"):
        for ln in lines[3:]:
            if ln.startswith("REF NLL by bucket"):
                break
            if not ln.strip() or "depth trend" in ln:
                continue
            m = re.match(rf"(?P<arm>\S.*?)\s+(?P<bucket>{BUCKETS})\s+(?P<rows>\d+)\s+(?P<kld>{FLOAT})\s+(?P<se>{FLOAT})\s", ln)
            if not m:
                raise Reject(f"{p}: unparsable KLD row: {ln[:90]!r}")
            note = re.search(r"\[(.*)\]", m["arm"])
            ctrl = bool(note and "determinism" in note.group(1))
            out.row(re.sub(r"\s*\[.*\]", "", m["arm"]), "REF6", m["bucket"], float(m["kld"]), float(m["se"]),
                    extra_dims={"rows": int(m["rows"]), **({"note": note.group(1)} if note else {})}, control=ctrl)
        found += 1
    if found == 0:
        ns = re.search(r"n_seq = (\d+)", text)
        if any(ln.startswith("== KLD by reference") for ln in lines):
            found += bool(_parse_by_reference(out, lines, int(ns.group(1)) if ns else None))
        elif re.search(r"scored against REFBF", "\n".join(lines[:3])):
            found += bool(_parse_f32acc(out, lines))
        elif lines and re.match(r"gguf-n: .*KL\(REF Q6_K \|\| arm\)", lines[0]) and "REF_q6k_f16.kldo" in text[:600]:
            found += bool(_parse_gguf_n(out, lines))
    if not found or not out.b.measurements:
        raise Skip(f"{p}: no KLD table with a stated reference (REF6/REF8/REFBF or ref_*); not ingested")
    return out.finish()


def parse_refbf_json(path):
    p = Path(path)
    d = load_json(p)
    need(d, ["refs", "kld"], str(p))
    out = _Out(p)
    for arm, byref in d["kld"].items():
        for ref, buckets in byref.items():
            if ref not in KNOWN_REFS:
                raise Reject(f"{p}: unknown reference {ref!r} for arm {arm}")
            for bucket, cell in buckets.items():
                if not (isinstance(cell, list) and len(cell) >= 2):
                    raise Reject(f"{p}: malformed cell {arm}/{ref}/{bucket}")
                out.row(arm, ref, bucket, float(cell[0]), float(cell[1]), n=len(cell[2]) if len(cell) > 2 else None)
    if not out.b.measurements:
        raise Reject(f"{p}: no KLD cells")
    return out.finish()


def parse(path):
    p = Path(path)
    if p.suffix == ".json":
        return parse_refbf_json(p)
    return parse_text(p)
