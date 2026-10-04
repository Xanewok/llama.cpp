"""Replay real recorded GPU output and negative controls without GPU calls."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mmvq_gpu as m

FIXTURES = Path("/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures")


def reject(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("negative control escaped")


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    short = json.loads((FIXTURES / "llama-sse/long-tool-s1-1.json").read_text())
    assert m.h.completion(short["chunks"]) == short["response"]
    assert m.h.steady(short, 5, 1)["steady_rounds_M"] == 24
    reject(lambda: m.parse_request(short, 5, 40))
    long_path = Path("/home/xanewok/repos/graph-reuse/codex/runs/graph-reuse-tandem/full-a2/k5-1-up/long-tool-s1-1.json")
    real = json.loads(long_path.read_text())
    expected = m.parse_request(real, 5, 40)
    changed = copy.deepcopy(real)
    changed["chunks"].insert(2, {"timings": {"draft_n": None}})
    changed["chunks"].append({"choices": [{"delta": {}}]})
    assert m.parse_request(changed, 5, 40) == expected
    for kind in ("counter", "tail", "length", "response"):
        changed = copy.deepcopy(real)
        if kind == "counter":
            changed["metrics"]["accepted_tokens"] += 1
        elif kind == "tail":
            changed["chunks"] = changed["chunks"][:5]
        elif kind == "length":
            changed = short
        else:
            changed["response"]["timings"]["predicted_ms"] += 10
        reject(lambda: m.parse_request(changed, 5, 40))
    memory = (FIXTURES / "nvidia-smi-query.csv").read_text()
    assert [row[2] for row in m.h.gpu_memory(memory)] == [108, 15]
    reject(lambda: m.h.gpu_memory(memory.replace("108 MiB", "unknown")))
    harness = object.__new__(m.h.Harness)
    harness.env = {"XID_SINCE": "fixture-cutoff"}
    with patch.object(harness, "run", return_value=SimpleNamespace(stdout=(FIXTURES / "xid-lines.txt").read_text())):
        reject(harness.xid)
    with patch.object(harness, "run", return_value=SimpleNamespace(stdout="ordinary kernel message stop_xid_hash_s_G")):
        harness.xid()
    # This protocol is new. Its first recorded GPU output is validated in the bench canary.
    rows = []
    for typ in m.TYPES:
        block, size = {"q5_K": (256, 176), "q6_K": (256, 210), "q4_K": (256, 144),
                       "iq4_xs": (256, 136), "q8_0": (32, 34)}[typ]
        for shape, (k, nrows) in m.SHAPES.items():
            if shape == "head" and typ != "q6_K":
                continue
            for n in (1, 2, 4, 6, 8):
                for variant in m.VARIANTS:
                    count = k*nrows//block*size
                    l2 = 32*1024*1024
                    copies = (4*l2 + count - 1)//count
                    rows.append(dict(type=typ, shape=shape, k=k, m=nrows, ncols=n, variant=variant,
                                     us_M=100.0, weight_GB_s_D=count/100.0/1000, weight_bytes=count,
                                     nmse_D=0, cpu_nmse_D=0, repeats=5, trials=1, cc=1200,
                                     l2_bytes_M=l2, copies_D=copies, working_set_bytes_D=count*copies))
    def raw(data):
        return "\n".join(json.dumps(row) for row in data)
    assert len(m.parse_bench(raw(rows), 5, 1)) == 1085
    reject(lambda: m.parse_bench(raw(rows[:-1]), 5, 1))
    reject(lambda: m.parse_bench(raw(rows + rows[:1]), 5, 1))
    for key, value in (("us_M", float("nan")), ("nmse_D", 0.1), ("cpu_nmse_D", 0.1),
                       ("weight_GB_s_D", 500), ("cc", 890), ("weight_bytes", 100), ("working_set_bytes_D", 1)):
        changed = copy.deepcopy(rows)
        changed[0][key] = value
        reject(lambda: m.parse_bench(raw(changed), 5, 1))
    print(json.dumps({"real_artifacts": [str(FIXTURES), str(long_path)], "steady_rounds_M": expected["steady_rounds_M"],
                      "synthetic_bench_cases": len(rows), "negative_controls_passed": True}))


if __name__ == "__main__":
    main()
