"""Pre-registered expectations vs measured deltas. Contract: `[question] expected` is `<lo>[..<hi>] <unit> (D|G)` and is
read from the run's pinned experiment snapshot, so the prediction cannot be edited after the run."""
import re

UNIT = "ms/round"
HEADER = ("# Calibration: predicted -> measured\n\n"
          "One row per recorded run (`harness report RUN --record`, expected from the pinned experiment's `[question]`). "
          "Ratio is mean measured / midpoint of the expected range; sign hit compares their signs (n/a when the range contains 0); in range counts depths whose "
          "point delta lies in the expected interval. Persistent ratio < 1 or sign misses mean our D/G estimates are optimistic. "
          "Pre-register in the experiment: `[question]` with `expected = \"-0.4..-0.55 ms/round (D|G)\"`, `falsifier`, `answers`; "
          "`harness.mkitem` copies them into the gpuq item `note` (`gpuq note ITEM`).\n\n"
          "| date | item | lever | expected (label) | measured (M/D) | verdict | ratio/sign hit |\n"
          "|---|---|---|---|---|---|---|\n")
NUMBER = r"[+-]?\d+(?:\.\d+)?"
PATTERN = re.compile(rf"^(?P<lo>{NUMBER})(?:\s*(?:\.\.|…|\bto\b)\s*(?P<hi>{NUMBER}))?\s+(?P<unit>[^()|]+?)\s*\((?P<label>[DG])\)$")


def parse_expected(text):
    """-> dict(lo, hi, unit, label) with lo <= hi; ValueError names the expected shape."""
    if not isinstance(text, str):
        raise ValueError("expected must be a string")
    m = PATTERN.fullmatch(text.replace("−", "-").strip())
    if not m:
        raise ValueError(f"expected {text!r} is not '<lo>..<hi> {UNIT} (D|G)'")
    lo, hi = float(m["lo"]), float(m["hi"] if m["hi"] is not None else m["lo"])
    return {"lo": min(lo, hi), "hi": max(lo, hi), "unit": m["unit"].strip(), "label": m["label"]}


def assess(out):
    """-> {"row": {...}} when the report carries an expectation it can score, else {"warn": reason}. Depth reports only:
    the measured scalar is the per-depth patch-up delta of steady ms/round."""
    expected = out.get("question", {}).get("expected")
    if not expected:
        return {"warn": "no [question] expected in the run's pinned experiment: no calibration row"}
    if out.get("shape") != "depth" or not out.get("depths"):
        return {"warn": f"calibration scores depth reports only (shape {out.get('shape')}): no calibration row"}
    try:
        e = parse_expected(expected)
    except ValueError as error:
        return {"warn": f"{error}: no calibration row"}
    if e["unit"] != UNIT:
        return {"warn": f"expected unit {e['unit']!r} is not {UNIT!r}, the unit report measures: no calibration row"}
    deltas = {dep: row["metrics"]["steady_ms_per_round_D"]["delta"] for dep, row in out["depths"].items()}
    mean = sum(deltas.values()) / len(deltas)
    mid = (e["lo"] + e["hi"]) / 2
    inside = sum(e["lo"] <= v <= e["hi"] for v in deltas.values())
    # A range containing 0 (an A/A, a null prediction) predicts no direction: no sign to hit or miss.
    sign = "n/a" if e["lo"] <= 0 <= e["hi"] else "hit" if (mean > 0) == (mid > 0) and mean != 0 else "miss"
    ratio = f"{mean / mid:.2f}x" if mid else "n/a"
    measured = "/".join(f"{v:+.3f}" for v in deltas.values()) + f" {UNIT} at " + "/".join(deltas) + " (D)"
    return {"row": {"expected": expected.strip(), "measured": measured,
                    "hit": f"{ratio}, sign {sign}, {inside}/{len(deltas)} in range"}}


def summary(cal):
    r = cal["row"]
    return f"predicted {r['expected']} → measured {r['measured']} ({r['hit']})"


def row(date, item, lever, cal, verdict):
    r = cal["row"]
    cells = [date, item, lever, r["expected"], r["measured"], verdict, r["hit"]]
    if any("|" in c or "\n" in c for c in cells):
        raise ValueError("calibration row cells must be single-line without table separators")
    return "| " + " | ".join(cells) + " |\n"
