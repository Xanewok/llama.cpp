#!/usr/bin/env python3
import argparse
import array
import json
import math
from pathlib import Path


def records(path):
    result = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("M: case ") or line.startswith("M: follow "):
            kind = line.split()[1]
            row = {"kind": kind}
            for part in line.split()[2:]:
                key, value = part.split("=")
                row[key] = float(value) if any(c in value for c in ".eE") else int(value)
            assert all(math.isfinite(v) for v in row.values() if isinstance(v, float))
            result.append(row)
    assert result, f"No measurements: {path}"
    return result


def summary(rows):
    cases = [r for r in rows if r["kind"] == "case"]
    result = {"measured_cases": len(cases), "measured_follows": len(rows) - len(cases)}
    for key in ("pre_nmse", "conv_nmse", "gdn_nmse", "post_nmse", "matched_post_nmse"):
        result["measured_max_" + key] = max(r[key] for r in cases)
    result["measured_logit_exact_cases"] = sum(r["bit_equal"] for r in cases)
    result["measured_state_exact_cases"] = sum(r["state_equal"] for r in cases)
    return result


def error(a, b):
    assert len(a) == len(b) and len(a) % 4 == 0 and a, "Invalid artifact shape"
    x, y = array.array("f"), array.array("f")
    x.frombytes(a); y.frombytes(b)
    assert all(math.isfinite(v) for v in x) and all(math.isfinite(v) for v in y)
    return sum((u-v)**2 for u, v in zip(x, y)) / max(sum(v*v for v in y), 1e-300)


def compare(a, b):
    left = {p.name: p.read_bytes() for p in a.glob("*.f32")}
    right = {p.name: p.read_bytes() for p in b.glob("*.f32")}
    assert left and left.keys() == right.keys(), "Missing or mismatched artifacts"
    return {"measured_rows": len(left),
            "measured_bit_equal_rows": sum(left[n] == right[n] for n in left),
            "measured_max_nmse": max(error(left[n], right[n]) for n in left)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        a = array.array("f", [1, 2, 3]).tobytes()
        b = array.array("f", [1, 2, 4]).tobytes()
        assert error(a, a) == 0 and error(a, b) > 0
        positive_zero = array.array("f", [0.0]).tobytes()
        negative_zero = array.array("f", [-0.0]).tobytes()
        assert positive_zero != negative_zero and error(positive_zero, negative_zero) == 0
        try:
            error(a, a[:-4])
        except AssertionError:
            pass
        else:
            raise AssertionError("Missing row negative control escaped")
        print("M: equal, perturbed, signed-zero and truncated artifact controls pass")
        return
    if args.gpu:
        root = Path(args.paths[0])
        runs = [root / name for name in ("0-up", "1-patch", "2-patch", "3-up")]
        report = {p.name: summary(records(p / "probe.log")) for p in runs}
        report["up_repeat"] = compare(runs[0] / "logits", runs[3] / "logits")
        report["patch_repeat"] = compare(runs[1] / "logits", runs[2] / "logits")
        report["cross_arm"] = compare(runs[0] / "logits", runs[1] / "logits")
        print(json.dumps(report, indent=2))
        for name in ("up_repeat", "patch_repeat"):
            assert report[name]["measured_rows"] == report[name]["measured_bit_equal_rows"], "Repeated boot changed rows"
        return
    report = {}
    for index, path in enumerate(args.paths):
        rows = records(path)
        if args.sweep and index == 0:
            actual = [(r["prefix"], r["k"], r["r"]) for r in rows if r["kind"] == "case"]
            expected = {(n, k, r) for n in (*range(120, 141), *range(250, 261))
                        for k in range(1, 7) for r in range(1, k + 1)}
            assert len(actual) == len(expected) and set(actual) == expected, "Incomplete or duplicate sweep"
        report[path] = summary(rows)
        if args.exact:
            for row in rows:
                assert row["bit_equal"] == 1, "Logit exactness failed"
                if row["kind"] == "case":
                    assert row["state_equal"] == 1, "State exactness failed"
                    assert all(v == 0 for k, v in row.items() if k.endswith("nmse")), "Noise in deterministic oracle"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
