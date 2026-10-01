#!/usr/bin/env python3
import argparse
import json
import re
import sqlite3
import statistics
from pathlib import Path


def suite(path, arm):
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text())
    cases = [line for line in text.splitlines() if "swift_verify=1" in line and re.search(r"\): (OK|FAIL)$", line)]
    failed = [line for line in cases if line.endswith("FAIL")]
    expected = 362 if arm == "old" else 386
    assert len(cases) == expected, f"expected {expected} cases, got {len(cases)}"
    assert "not supported" not in text.lower()
    if arm == "old":
        assert len(failed) == 30
        assert all("pattern=large," in line and re.search(r"kv=(65536|102400|102399),", line) for line in failed)
        assert text.count("[CONCAT] ERR = inf") == 30
    else:
        assert not failed
    print(json.dumps({"arm": arm, "pass": len(cases) - len(failed), "fail": len(failed)}))


def kernels(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = db.execute("SELECT k.end-k.start, s.value, k.gridX, k.blockX, k.blockY, k.registersPerThread, k.dynamicSharedMemory "
                      "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start").fetchall()
    for kind in ("flash_attn_ext_f16<", "flash_attn_stream_k_fixup"):
        selected = [r for r in rows if kind in r[1]]
        if not selected:
            if kind == "flash_attn_ext_f16<":
                raise RuntimeError("no MMA kernel recorded")
            continue
        selected = selected[len(selected)//10:]
        times = [r[0]/1000.0 for r in selected]
        print(json.dumps({"kernel": kind, "calls": len(times), "median_us": statistics.median(times),
                          "mean_us": statistics.mean(times), "launches": sorted(set(tuple(r[2:]) for r in selected))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["suite", "kernels"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--arm", choices=["old", "new"])
    args = parser.parse_args()
    if args.mode == "suite":
        suite(args.path, args.arm)
    else:
        kernels(args.path)
