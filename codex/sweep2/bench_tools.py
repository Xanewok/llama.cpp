#!/usr/bin/env python3
"""Validate MMVQ rows before writing tables. M = measured, D = derived, G = guess."""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

OLD_VARIANTS = ("base", "default", "w2r1", "w4r1", "w4r2", "w2r4", "w8r1")
NEW_VARIANTS = ("w1r4", "w1r8", "w2r8", "w4r4", "w2r6")
VARIANTS = OLD_VARIANTS + NEW_VARIANTS
TYPES = ("q6_K", "q5_K", "iq4_xs", "q4_K", "q8_0")  # run order: J5q bytes first, so a cut-off pass keeps the important types
SHAPES = {"gate-up": (5120, 8704), "down": (8704, 5120), "qkv": (5120, 5120), "q": (5120, 6144),
          "attn-gate": (5120, 3072), "ssm-out": (3072, 5120), "head": (5120, 124160)}
NCOLS = (6, 5, 7, 8, 1, 2, 4)
PRIMARY_COLS = (6, 5, 7, 8)
SMALL_COLS = (1, 2, 4)
BLOCK = {"q5_K": (256, 176), "q6_K": (256, 210), "q4_K": (256, 144), "iq4_xs": (256, 136), "q8_0": (32, 34)}
PEAK_GBS, HARD_MAX_GBS = 448.0, 520.0
FULL = {"repeats": 100, "trials": 7}
CANARY = {"repeats": 5, "trials": 1}


class Integrity(Exception):
    pass


class Insufficient(Exception):
    pass


def expected_cases(kind):
    if kind == "canary":
        return {(t, "qkv", n, v) for t in TYPES for n in NCOLS for v in VARIANTS}
    cols, variants = ((1, 2, 4, 6, 8), OLD_VARIANTS) if kind == "legacy" else ((6,) if kind == "decision" else PRIMARY_COLS, VARIANTS)
    return {(t, s, n, v) for t in TYPES for s in SHAPES if s != "head" or t == "q6_K" for n in cols for v in variants}


def check_row(row, params):
    key = row["type"], row["shape"], row["ncols"], row["variant"]
    shape = SHAPES.get(row["shape"])
    if shape is None or (row["k"], row["m"]) != shape:
        raise Integrity(f"row {key}: shape does not match the frozen table")
    if row["cc"] != 1200 or row["repeats"] != params["repeats"] or row["trials"] != params["trials"]:
        raise Integrity(f"row {key}: cc {row['cc']} repeats {row['repeats']} trials {row['trials']}")
    for name in ("us_M", "weight_GB_s_D", "nmse_D", "cpu_nmse_D"):
        v = row[name]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise Integrity(f"row {key}: {name} is not finite")
    if row["us_M"] <= 0 or row["weight_GB_s_D"] <= 0:
        raise Integrity(f"row {key}: non-positive time or bandwidth")
    if not 0 <= row["nmse_D"] < 1e-8 or not 0 <= row["cpu_nmse_D"] < 2e-5:
        raise Integrity(f"row {key}: correctness gate value out of range (nmse {row['nmse_D']}, cpu {row['cpu_nmse_D']})")
    block, size = BLOCK[row["type"]]
    nbytes = row["k"] * row["m"] // block * size
    if row["weight_bytes"] != nbytes or not math.isclose(row["weight_GB_s_D"], nbytes / row["us_M"] / 1000, rel_tol=1e-5):
        raise Integrity(f"row {key}: bandwidth denominator changed")
    if row["l2_bytes_M"] <= 0 or row["copies_D"] < 1 or row["working_set_bytes_D"] != nbytes * row["copies_D"] \
            or row["working_set_bytes_D"] < 4 * row["l2_bytes_M"]:
        raise Integrity(f"row {key}: weight working set can fit in L2")
    if row["weight_GB_s_D"] > HARD_MAX_GBS:
        raise Integrity(f"rows outside band: {key} {row['weight_GB_s_D']:.0f} GB/s > {HARD_MAX_GBS:.0f}")
    return key


def load(files):
    rows = []
    for f in files:
        for n, line in enumerate(Path(f).read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append((str(f), json.loads(line)))
            except ValueError as error:
                raise Insufficient(f"{f}:{n}: unreadable bench row: {error}") from error
    return rows


def gather(files, kind):
    params = CANARY if kind == "canary" else FULL
    want = expected_cases(kind)
    passes = defaultdict(dict)
    for f, row in load(files):
        try:
            key = check_row(row, params)
        except (KeyError, TypeError, ValueError) as error:
            raise Integrity(f"{f}: malformed row: {error}") from error
        tag = Path(f).stem.split("-")[0]
        optional = kind == "full" and tag == "r3" and key[2] in SMALL_COLS and key[3] in (*NEW_VARIANTS, "base", "w2r4")
        if key in passes[tag] or (key not in want and not optional):
            raise Integrity(f"{f}: unexpected or duplicate case {key}")
        if tag not in ({"r1", "r2"} if kind == "legacy" else {"r1", "r3"}):
            raise Integrity(f"{f}: unexpected pass {tag}")
        passes[tag][key] = row
    if "r1" not in passes:
        raise Insufficient("no r1 rows")
    missing = want - passes["r1"].keys()
    if missing:
        raise Insufficient(f"r1 is missing {len(missing)} cases, e.g. {sorted(missing)[0]}")
    if kind == "full":
        picks = best_new(passes["r1"])
        for t in TYPES:
            present = {k for k in passes.get("r3", {}) if k[0] == t}
            if not present:
                continue
            expected = {(t, s, n, v) for s in SHAPES if s != "head" or t == "q6_K"
                        for n in SMALL_COLS for v in ("base", "w2r4", *picks[t])}
            if present - expected:
                raise Integrity(f"r3 {t}: not the best two new n=6 variants")
            if expected - present:
                raise Insufficient(f"r3 {t}: incomplete optional sweep")
    return passes


def mix(rows, t, n, v, shapes=None):
    """Effective GB/s of one (type, n, variant) over the non-head shapes: total bytes / total time."""
    cells = [r for (tt, s, nn, vv), r in rows.items() if (tt, nn, vv) == (t, n, v) and s != "head" and (shapes is None or s in shapes)]
    if not cells:
        return None
    return sum(r["weight_bytes"] for r in cells) / sum(r["us_M"] for r in cells) / 1000


def best_new(rows):
    picks = {}
    for t in TYPES:
        scores = {v: mix(rows, t, 6, v) for v in NEW_VARIANTS}
        if any(x is None for x in scores.values()):
            raise Insufficient(f"n=6 {t}: missing ranking rows")
        picks[t] = sorted(NEW_VARIANTS, key=lambda v: (-scores[v], VARIANTS.index(v)))[:2]
    return picks


def summarize(passes, kind):
    variants = OLD_VARIANTS if kind == "legacy" else VARIANTS
    out = ["# MMVQ second launch-shape sweep", "",
           "M = measured kernel time; D = derived GB/s and gains. Decimal weight GB/s = logical weight bytes / median kernel time.",
           "Direct MMVQ launch bypasses production routing; these tables do not establish serving speed.",
           "Non-head shapes use total bytes / total time; head is separate. Padded tail bytes are excluded from the denominator.",
           "Mandatory order: n=6, then n=5/7/8. Optional small columns test the best two new n=6 variants per type with base/w2r4 controls.", ""]
    tables = []
    for tag, rows in passes.items():
        for t in TYPES:
            cells = []
            for n in NCOLS:
                values = {v: mix(rows, t, n, v) for v in variants}
                values = {v: x for v, x in values.items() if x is not None}
                if not values:
                    continue
                winner = max(values, key=values.get)
                base, old = values.get("base"), values.get("w2r4")
                cells.append((n, values, winner, base, old))
                tables.append({"pass": tag, "type": t, "ncols": n, "GB_s_D": values, "best": winner,
                               "gain_base_pct_D": 100*(values[winner]/base-1) if base else None,
                               "gain_w2r4_pct_D": 100*(values[winner]/old-1) if old else None})
            if not cells:
                continue
            out += [f"## {tag} {t}: effective weight GB/s (D)", "",
                    "| ncols | " + " | ".join(variants) + " | best | gain/base % (D) | gain/w2r4 % (D) |",
                    "|---|" + "---|"*(len(variants)+3)]
            for n, values, winner, base, old in cells:
                gain = lambda ref: f"{100*(values[winner]/ref-1):+.2f}" if ref else "-"
                out.append(f"| {n} | " + " | ".join(f"{values[v]:.2f}" if v in values else "-" for v in variants)
                           + f" | {winner} | {gain(base)} | {gain(old)} |")
            out.append("")
        if any(k[1] == "head" for k in rows):
            out += [f"## {tag} q6_K head: weight GB/s (D)", "",
                    "| ncols | " + " | ".join(variants) + " | best | gain/base % (D) | gain/w2r4 % (D) |",
                    "|---|" + "---|"*(len(variants)+3)]
            for n in NCOLS:
                values = {v: rows[("q6_K", "head", n, v)]["weight_GB_s_D"] for v in variants if ("q6_K", "head", n, v) in rows}
                if not values:
                    continue
                winner = max(values, key=values.get)
                gains = [f"{100*(values[winner]/values[v]-1):+.2f}" if v in values else "-" for v in ("base", "w2r4")]
                out.append(f"| {n} | " + " | ".join(f"{values[v]:.2f}" if v in values else "-" for v in variants)
                           + f" | {winner} | " + " | ".join(gains) + " |")
            out.append("")
    return "\n".join(out) + "\n", tables


def parse(files, kind, md=None, js=None):
    passes = gather(files, kind)
    text, summary = summarize(passes, kind)
    if md:
        Path(md).write_text(text)
    if js:
        Path(js).write_text(json.dumps({"kind": kind, "rows": {p: len(v) for p, v in passes.items()}, "tables": summary}, indent=2))
    return {p: len(v) for p, v in passes.items()}


def synth_row(key, repeats, trials, **patch):
    t, s, n, v = key
    k, m = SHAPES[s]
    block, size = BLOCK[t]
    nbytes = k * m // block * size
    us = nbytes / (300 + 7 * VARIANTS.index(v)) / 1000
    row = {"type": t, "shape": s, "k": k, "m": m, "ncols": n, "variant": v, "us_M": us, "weight_GB_s_D": nbytes / us / 1000,
           "weight_bytes": nbytes, "nmse_D": 1e-13, "cpu_nmse_D": 1e-13, "repeats": repeats, "trials": trials, "cc": 1200,
           "l2_bytes_M": 33554432, "working_set_bytes_D": nbytes * max(1, -(-4 * 33554432 // nbytes)), "copies_D": max(1, -(-4 * 33554432 // nbytes))}
    row["working_set_bytes_D"] = nbytes * row["copies_D"]
    return row | patch


def selftest():
    import tempfile

    def run(rows, kind, name="r1-x.jsonl", extra=()):
        with tempfile.TemporaryDirectory() as d:
            files = []
            for nm, rs in [(name, rows), *extra]:
                p = Path(d) / nm
                p.write_text("\n".join(json.dumps(r) for r in rs) + "\n")
                files.append(p)
            return parse(files, kind, Path(d) / "o.md", Path(d) / "o.json")

    full = [synth_row(k, **FULL) for k in sorted(expected_cases("full"))]
    canary = [synth_row(k, **CANARY) for k in sorted(expected_cases("canary"))]
    assert run(full, "full") == {"r1": len(full)} and run(canary, "canary") == {"r1": len(canary)}
    picks = best_new({tuple(r[k] for k in ("type", "shape", "ncols", "variant")): r for r in full})
    optional = [synth_row((t, s, n, v), **FULL) for t in TYPES for s in SHAPES if s != "head" or t == "q6_K"
                for n in SMALL_COLS for v in ("base", "w2r4", *picks[t])]
    assert run(full, "full", extra=[("r3-x.jsonl", optional)]) == {"r1": len(full), "r3": len(optional)}
    try:
        run(full, "full", extra=[("r3-x.jsonl", optional[1:])])
    except Insufficient:
        pass
    else:
        raise AssertionError("partial optional pass accepted")
    bad = {
        "wrong-cc": lambda r: r | {"cc": 890}, "nan": lambda r: r | {"us_M": float("nan")}, "nmse": lambda r: r | {"nmse_D": 1e-6},
        "cpu-nmse": lambda r: r | {"cpu_nmse_D": 1e-3}, "denominator": lambda r: r | {"weight_GB_s_D": r["weight_GB_s_D"] * 1.5},
        "l2": lambda r: r | {"copies_D": 1, "working_set_bytes_D": r["weight_bytes"], "l2_bytes_M": 10 * r["weight_bytes"]},
        "repeats": lambda r: r | {"repeats": 7}, "shape": lambda r: r | {"k": r["k"] + 256},
        "band": lambda r: r | {"us_M": r["weight_bytes"] / 600 / 1000, "weight_GB_s_D": 600.0},
    }
    for name, f in bad.items():
        rows = [f(full[0])] + full[1:]
        try:
            run(rows, "full")
        except Integrity:
            continue
        raise AssertionError(f"selftest: {name} accepted")
    for rows, exc in ((full[1:], Insufficient), (full + [full[0]], Integrity), (full + [synth_row(("q8_0", "head", 6, "base"), **FULL)], Integrity)):
        try:
            run(rows, "full")
        except exc:
            continue
        raise AssertionError("selftest: structural defect accepted")
    print("selftest ok")


def recorded_test(directory):
    import tempfile
    raw = Path(directory)
    files = sorted(raw.glob("r[12]-*.jsonl"))
    legacy = gather(files, "legacy")
    md, tables = summarize(legacy, "legacy")
    assert tables and "gain/w2r4" in md
    fixtures = Path("/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures")
    assert hazard((fixtures / "xid-lines.txt").read_text())
    assert not hazard("stop_xid_hash_s_G")
    device((raw.parent / "env.txt").read_text())
    for text in ("0, NVIDIA RTX 4090", "1, NVIDIA RTX 5060 Ti"):
        try:
            device(text)
        except (Integrity, Insufficient):
            pass
        else:
            raise AssertionError("wrong/missing device accepted")
    assert "variant disagrees with baseline" in (raw / "negative.err").read_text()
    source = legacy["r1"]
    mapped = []
    for t, shape, n, variant in sorted(expected_cases("full")):
        old_n = {5: 6, 7: 8}.get(n, n)
        old_v = variant if variant in OLD_VARIANTS else "w2r4"
        mapped.append(source[(t, shape, old_n, old_v)] | {"ncols": n, "variant": variant})
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "r1-recorded.jsonl"
        def write(rows):
            f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        write(mapped)
        rows = gather([f], "full")
        assert len(best_new(rows["r1"])) == len(TYPES)
        assert summarize(rows, "full")[1]
        picks = best_new(rows["r1"])
        optional = [source[(t, shape, n, "w2r4")] | {"variant": v} for t in TYPES for shape in SHAPES
                    if shape != "head" or t == "q6_K" for n in SMALL_COLS for v in ("base", "w2r4", *picks[t])]
        opt = Path(d) / "r3-recorded.jsonl"
        opt.write_text("\n".join(json.dumps(r) for r in optional) + "\n")
        assert gather([f, opt], "full")["r3"]
        canary = [source[(t, "qkv", {5: 6, 7: 8}.get(n, n), v if v in OLD_VARIANTS else "w2r4")]
                  | {"ncols": n, "variant": v, **CANARY} for t, shape, n, v in sorted(expected_cases("canary"))]
        write(canary)
        assert gather([f], "canary")["r1"]
        for patch in ({"nmse_D": 1e-6}, {"cpu_nmse_D": 1e-3}, {"cc": 890}, {"weight_bytes": 1}, {"us_M": float("nan")},
                      {"working_set_bytes_D": 1}, {"repeats": 1}, {"k": 1}):
            write([mapped[0] | patch, *mapped[1:]])
            try:
                gather([f], "full")
            except Integrity:
                continue
            raise AssertionError(f"recorded corruption accepted: {patch}")
        write(mapped[1:])
        try:
            gather([f], "full")
        except Insufficient:
            pass
        else:
            raise AssertionError("recorded missing case accepted")
    print(json.dumps({"legacy_rows_M": {k: len(v) for k, v in legacy.items()},
                      "recorded_artifact": str(raw), "remapped_schema_tests": "full/canary/optional/ranking; no new GPU measurements", "negative_controls": "passed"}))


def hazard(text):
    import re
    return bool(re.search(r"\bXid\b|fallen off the bus", text))


def device(text):
    import csv
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if row and row[0] == "0":
            if len(row) < 2 or "RTX 5060 Ti" not in row[1]:
                raise Integrity("device 0 is not an RTX 5060 Ti")
            return
    raise Insufficient("no CUDA device in recorded inventory")


def execute(binary_args, prefix, cap):
    import signal
    import subprocess
    prefix = Path(prefix)
    proc = None
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        with open(str(prefix) + ".jsonl", "w") as out, open(str(prefix) + ".err", "w") as err:
            proc = subprocess.Popen(binary_args, stdout=out, stderr=err)
            Path(str(prefix) + ".pid").write_text(str(proc.pid) + "\n")
            try:
                return proc.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                print("bench_tools: hit its cap (budget)", file=sys.stderr)
                return 7
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("recorded-test")
    p.add_argument("directory")
    p = sub.add_parser("device")
    p.add_argument("file")
    p = sub.add_parser("hazard")
    p.add_argument("file")
    p = sub.add_parser("execute")
    p.add_argument("--cap", required=True, type=int)
    p.add_argument("--prefix", required=True)
    p.add_argument("argv", nargs=argparse.REMAINDER)
    p = sub.add_parser("select")
    p.add_argument("files", nargs="+")
    p.add_argument("--canary", action="store_true")
    p = sub.add_parser("parse")
    p.add_argument("files", nargs="+")
    p.add_argument("--expect", required=True, choices=("canary", "full", "legacy"))
    p.add_argument("--md")
    p.add_argument("--json")
    args = ap.parse_args()
    try:
        if args.cmd == "selftest":
            selftest()
        elif args.cmd == "recorded-test":
            recorded_test(args.directory)
        elif args.cmd == "device":
            device(Path(args.file).read_text())
        elif args.cmd == "hazard":
            if hazard(Path(args.file).read_text()):
                print("bench_tools: Xid or fallen off the bus", file=sys.stderr)
                return 3
        elif args.cmd == "execute":
            argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
            if not argv or args.cap < 1:
                raise Integrity("invalid execute arguments")
            return execute(argv, args.prefix, args.cap)
        elif args.cmd == "select":
            picks = best_new(gather(args.files, "canary" if args.canary else "decision")["r1"])
            for t, variants in picks.items():
                print(t, ",".join(("base", "w2r4", *variants)))
        else:
            print(json.dumps(parse(args.files, args.expect, args.md, args.json)))
    except Integrity as error:
        print(f"bench_tools: integrity: {error}", file=sys.stderr)
        return 6
    except (Insufficient, OSError) as error:
        print(f"bench_tools: insufficient data: {error}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
