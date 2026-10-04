"""Replay recorded GPU np2 artifacts through the exact parsers and gates used by this draft."""
import copy
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, "/home/xanewok/repos/qwen38-eval")
from harness import depth, lib

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path("/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures")
RECORDED = Path("/home/xanewok/repos/qwen38-eval/results/2026-10-04-duo-np2-depth/canary-a2/1-patch")


def rejects(fn, classification):
    try:
        fn()
    except lib.Fail as error:
        assert error.cls == classification, error
    else:
        raise AssertionError("negative control escaped")


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    directory = ROOT / "build-tmp/np2-harness"
    directory.mkdir(parents=True, exist_ok=True)
    depth.fixture_test(FIXTURES)
    x = lib.read_experiment(ROOT / "codex/mask-np2.experiment.toml")
    x["depth"]["depths"] = [8192, 32768]
    results = {}
    failures = depth.evaluate_streams(x, RECORDED, "patch", results)
    assert not failures, failures
    memory = lib.min_free_vram(RECORDED / "smi.csv", x["devices"])
    assert memory == json.loads((RECORDED / "vram.json").read_text())
    partial = directory / "partial-smi.csv"
    partial.write_text((RECORDED / "smi.csv").read_text() + "header, free\npartial,\nerror\n")
    assert lib.min_free_vram(partial, x["devices"]) == memory
    missing = directory / "missing-smi.csv"
    missing.write_text("\n".join(line for line in partial.read_text().splitlines() if x["devices"][0] in line))
    rejects(lambda: lib.min_free_vram(missing, x["devices"]), "insufficient_data")
    negative = ["missing VRAM rank"]
    for n in x["depth"]["depths"]:
        record = json.loads((RECORDED / f"depth-{n}.json").read_text())
        identities = [lib.token_identity(r) for r in record["streams"]]
        reference = {}
        depth.settle(reference, "cross-arm", "up", n, identities, multi=True)
        broken = copy.deepcopy(identities)
        broken[0][0][0] ^= 1
        rejects(lambda: depth.settle(reference, "cross-arm", "patch", n, broken, multi=True), "integrity")
        shifted = copy.deepcopy(record["streams"])
        for chunk in shifted[1]["chunks"]:
            if "t" in chunk:
                chunk["t"] += 10**18
        rejects(lambda: depth.overlap(shifted, 0.8), "insufficient_data")
        negative += [f"token identity {n}", f"no overlap {n}"]
    summary = {"recorded": str(RECORDED), "fixtures": str(FIXTURES), "depths": x["depth"]["depths"],
               "streams_M": 2, "negative_controls": negative, "vram_min_free_mib_M": memory,
               "parser_gates": "native token/draft identity, final timings, prefill replay, steady intervals, overlap, VRAM, fault signatures"}
    (directory / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
