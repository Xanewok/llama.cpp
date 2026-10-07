"""Enforce the existing chain report's token/counter/ID gates and negative controls."""
import json
import sys
from pathlib import Path

r = json.loads((Path(sys.argv[1]) / "report.json").read_text())

def identical(c):
    return (bool(c["tokens"]) and all(c["tokens"].values()) and all(c["counters_equal"].values())
            and c["dump_rows"][0] > 0 and c["dump_rows"][0] == c["dump_rows"][1] and c["dump_ids_equal"])

for name in ("plain-k3", "plain-k5", "meta-k3", "meta-k5"):
    v = r[name]
    assert identical(v["cmp"]), (name, v["cmp"])
    assert v["cmp"]["dump_max_abs_dq"] <= 2e-7, (name, v["cmp"])
    assert v["on_chain_lines"] > 0 and v["off_chain_lines"] == 0, name
    assert any(k.endswith("/mtp_chain") and n > 0 for k, n in v["on_counters"].items()), name
assert identical(r["noise_floor_off_vs_off"]) and r["noise_floor_off_vs_off"]["dump_q_bit_equal"]
for name in ("control_reverse_u", "control_shift_embd"):
    assert not identical(r[name]["cmp"]), (name, "escaped")
    tag = "plain-k5-on-reverse-u" if name == "control_reverse_u" else "plain-k5-on-shift-embd"
    log = (Path(sys.argv[1]) / tag / "log.txt").read_text()
    assert "MTP chain: k=5" in log, (name, "not engaged")
print("chain identity, engagement and negative controls PASS")
