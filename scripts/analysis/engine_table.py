#!/usr/bin/env python3
"""Assemble the cross-engine table from the bench rows and the drivers' FIT lines.

Context ceilings are not in the bench rows: each engine reports its own, in its own words,
in its server log -- so the drivers echo a `FIT <label> ...` line and this reads that back
rather than re-deriving a number no client can see.
"""

from __future__ import annotations

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from scripts import ROOT

SPEED = ROOT / "results" / "2026-08-rental-baseline" / "speed"


def rows() -> dict[tuple, dict[str, list[float]]]:
    """Group by label AND the hardware it ran on.

    Grouping on the label alone merged the same quant benched on a 2x5070 Ti and a
    2x5060 Ti into one row of six samples, with whichever card was read last
    overwriting the other. The power cap joins the key because it is an experiment
    variable: the same card at two caps is two configurations.
    """

    out: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for line in (SPEED / "engines.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["label"], r.get("gpu"), r.get("gpu_count"), r.get("power_limit_w"))
        for field in ("decode_tok_s", "ttft_s", "prefill_tok_s"):
            if r.get(field):
                out[key][field].append(r[field])
        if r.get("completion_tokens") and r.get("total_s"):
            out[key]["overall"].append(r["completion_tokens"] / r["total_s"])
        out[key]["engine"] = r.get("engine")
    return out


def card_name(gpu: str | None, count: int | None) -> str:
    """`NVIDIA GeForce RTX 3090` -> `3090`, `RTX 5070 Ti 300W` -> `5070 Ti`.

    Anything without an RTX marker is passed through unchanged so a bad value stays
    visibly bad: 36 rows here carry a rental label in this field, and normalising
    them would hide that rather than fix it.
    """

    if not gpu:
        return "?"
    words = gpu.split()
    if "RTX" in words:
        words = words[words.index("RTX") + 1:]
        words = [w for w in words if not re.fullmatch(r"\d+W", w)]
        gpu = " ".join(words) or gpu
    return f"{count}x{gpu}" if count and count > 1 else gpu


def ceilings() -> dict[str, str]:
    """Every `FIT <label> <free text>` line any driver wrote, last one winning."""
    found: dict[str, str] = {}
    for log in SPEED.glob("*.log"):
        for line in log.read_text(errors="replace").splitlines():
            m = re.search(r"\bFIT\s+(\S+)\s+(.*)", line)
            if not m:
                continue
            # Each engine states its ceiling in its own log line; keep the token count and
            # drop the rest, or ninfer's startup line alone is wider than the table.
            text = m.group(2).strip()
            n = re.search(r"(?:GPU KV cache size:|resolved=)\s*([0-9,]+)\s*tokens", text)
            found[m.group(1)] = f"{n.group(1)} tokens" if n else text
    return found


def main() -> int:
    data, fits = rows(), ceilings()
    width = max((len(k[0]) for k in data), default=10)
    # The card is not decoration: 5090 and 3090 rows sit in the same file, and a table
    # that ranks them together without naming the card invites exactly the wrong reading.
    print(f"{'engine · config':{width}s} {'card':>22s} {'decode':>8s} {'overall':>8s} "
          f"{'ttft':>6s} {'prefill':>8s} {'n':>3s}  context ceiling")
    for (label, gpu, count, watts), v in sorted(
        data.items(), key=lambda kv: -st.mean(kv[1]["decode_tok_s"] or [0])
    ):
        def m(field: str) -> str:
            if not v[field]:
                return "-"
            # Median for the latency columns: the first request against a freshly started
            # vLLM pays graph capture (36 s against 1.4 s for the next one), and a mean
            # over four tasks turns that one-off into the reported prefill rate.
            f = st.median if field in ("ttft_s", "prefill_tok_s") else st.mean
            return f"{f(v[field]):.1f}"
        fit = fits.get(label) or fits.get(label.split("-", 1)[-1], "")
        card = card_name(gpu, count) + (f" {watts:.0f}W" if watts else "")
        print(f"{label:{width}s} {card:>22s} {m('decode_tok_s'):>8s} {m('overall'):>8s} "
              f"{m('ttft_s'):>6s} {m('prefill_tok_s'):>8s} {len(v['overall']):>3d}  {fit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
