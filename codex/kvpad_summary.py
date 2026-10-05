"""RESULTS.json of the CPU n_kv_pad A/B: bitwise identity across pads (token IDs + final logits, 1 thread) and graph hit/miss counts.
Reads codex/kvpad-results/bitwise/pad*/{prefill,decode}-m{0,1}-pad*.{bin,jsonl}. Steady decode = rounds whose verify batch has the full k+1 rows,
excluding the first (prompt tail) and the final shrinking rounds."""
import hashlib, json
from pathlib import Path

R = Path(__file__).resolve().parent / "kvpad-results/bitwise"
K1 = 4
out = {}
for scenario in ("prefill", "decode"):
    for meta in (0, 1):
        runs = {}
        for d in sorted(R.glob("pad*")):
            pad = d.name[3:]
            b, t = d / f"{scenario}-m{meta}-pad{pad}.bin", d / f"{scenario}-m{meta}-pad{pad}.jsonl"
            if not b.exists() or b.stat().st_size == 0 or not t.exists():
                continue
            g = {"prefill": [0, 0, 0, 0], "steady": [0, 0, 0], "rounds": 0}  # prefill V hit/miss, C hit/miss; steady hit/miss/rounds
            first_full = True
            for line in t.read_text().splitlines():
                row = json.loads(line)
                ph = row["phases"]
                cnt = lambda p, k: sum(c["count"] for c in p["counters"] if c["kind"] == k)
                v = ph[0]
                if v["phase"] != "V":
                    continue
                if v["n_tokens"] > 6:
                    for p in ph:
                        if p["phase"] in ("V", "C"):
                            i = 0 if p["phase"] == "V" else 2
                            g["prefill"][i] += cnt(p, "graph_hit"); g["prefill"][i + 1] += cnt(p, "graph_miss")
                elif v["n_tokens"] == K1:
                    if first_full:
                        first_full = False
                        continue
                    g["steady"][0] += sum(cnt(p, "graph_hit") for p in ph); g["steady"][1] += sum(cnt(p, "graph_miss") for p in ph); g["rounds"] += 1
            data = b.read_bytes()
            runs[pad] = {"sha256": hashlib.sha256(data).hexdigest(), "prefill_V_hit_miss_C_hit_miss": g["prefill"],
                         "steady_rounds": g["rounds"], "steady_hit_miss_all_phases": g["steady"][:2]}
        if runs:
            shas = {r["sha256"] for r in runs.values()}
            out[f"{scenario}-m{meta}"] = {"bitwise_identical_across_pads": len(shas) == 1, "runs": runs}
import struct
import numpy as np
def load(p):
    d = p.read_bytes(); n = struct.unpack("<Q", d[:8])[0]
    return np.frombuffer(d[8:8 + 4 * n], dtype=np.int32), np.frombuffer(d[8 + 4 * n:], dtype=np.float32)
for key, v in out.items():
    sc, m = key.rsplit("-m", 1)
    pads = sorted(v["runs"], key=int)
    base = load(R / f"pad{pads[0]}/{sc}-m{m}-pad{pads[0]}.bin")
    for pad in pads[1:]:
        t, l = load(R / f"pad{pad}/{sc}-m{m}-pad{pad}.bin")
        v["runs"][pad]["vs_first_pad"] = {"tokens_equal": bool(np.array_equal(t, base[0])), "max_abs_logit_diff": float(np.abs(l - base[1]).max())}
C = R.parent / "controls"
ref = load(R.parent / "t1/decode-m0-pad0.bin")
ctl = {"mask_mutant_vs_unmutated_1thread_ref": {}}
for pad in ("0", "4096"):
    f = C / f"maskB-pad{pad}/decode-m0-pad{pad}.bin"
    if f.exists():
        t, l = load(f)
        ctl["mask_mutant_vs_unmutated_1thread_ref"][pad] = {"tokens_equal": bool(np.array_equal(t, ref[0])), "logits_bitwise": bool(np.array_equal(l, ref[1])),
                                                            "max_abs_logit_diff": float(np.abs(l - ref[1]).max())}
a = C / "assert-mutA/assert-m0-pad4096.log"
if a.exists():
    ctl["n_ctx_not_padded_mutant_pad4096_aborts"] = "GGML_ASSERT(kv_size % n_pad == 0) failed" in a.read_text()
out["controls"] = ctl
(R.parent / "RESULTS.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=1))
