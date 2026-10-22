#!/usr/bin/env python3
"""MMA78 subset of the shared flatcost bench; GPU execution requires gpuq."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

SHARED = Path("/home/xanewok/repos/qwen38-eval-worktrees/wt/flatcost-sweep")
spec = importlib.util.spec_from_file_location("flatcost", SHARED / "bench.py")
flatcost = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flatcost)
SHAPES = ("iq4_xs:5120:8704", "iq4_xs:8704:5120", "q6_K:5120:5120", "q6_K:124160:5120")


def select_shapes(manifest, filters):
    selected = []
    for key in filters:
        matches = [row for row in manifest["shapes"]
                   if f"{row['type']}:{row['m']}:{row['k']}" == key and not row["gdn"]]
        if not matches:
            raise ValueError(f"shape {key} is absent; use a type:m:k from the reviewed inventory")
        selected.append(matches[0])
    return selected


def gpu_guard(args):
    if os.environ.get("GPUQ_ACTIVE") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("GPU work requires gpuq with GPUQ_ACTIVE=1 and CUDA_VISIBLE_DEVICES=0")
    if args.binary is None or not args.binary.is_file():
        raise ValueError("GPU work requires --binary pointing to the preflight test-backend-ops")
    if "GGML_CUDA_MMVQ_MAX_COLS" in os.environ or "GGML_CUDA_MMA78_ARM" in os.environ:
        raise ValueError("unset GGML_CUDA_MMVQ_MAX_COLS and GGML_CUDA_MMA78_ARM; --arms owns dispatch")


def correctness(args, shapes):
    env = dict(os.environ, GGML_CUDA_DISABLE_GRAPHS="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    for arm in args.arms:
        env["GGML_CUDA_MMA78_ARM"] = arm
        cases = [(row, n) for row in shapes for n in (1, 6, 7, 8)]
        for i, (row, n) in enumerate(cases):
            env["GGML_TEST_FLATCOST_CASE"] = f"{row['type_id']},{row['m']},{n},{row['k']},1"
            check_case(args, env, args.out / f"correctness-{arm}-{i}.log", 1)
        env.pop("GGML_TEST_FLATCOST_CASE")
        env["GGML_TEST_MMA78_CASES"] = "1"
        check_case(args, env, args.out / f"correctness-{arm}-edges.log", 32)
        env.pop("GGML_TEST_MMA78_CASES")
    flatcost.write_json(args.out / "correctness.json", dict(arms=args.arms, shapes=shapes, widths=[1, 6, 7, 8],
                                                           edge_cases_per_arm=32, nmse_limit=5e-4))


def check_case(args, env, log, expected):
    with log.open("w") as stream:
        process = subprocess.run([str(args.binary), "test", "-o", "MUL_MAT", "-b", "CUDA0", "-j", "1"],
                                 env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
    output = log.read_text()
    if process.returncode or f"{expected}/{expected} tests passed" not in output:
        raise RuntimeError(f"CPU-reference correctness failed or skipped; inspect {log}")
    if "Device 0: NVIDIA GeForce RTX 5060 Ti, compute capability 12.0," not in output:
        raise ValueError(f"unexpected GPU; inspect {log} and fix the gpuq device selection")


def measure(args, row, n, arm, log):
    os.environ["GGML_CUDA_MMA78_ARM"] = arm
    try:
        # The shared runner owns cold rotation, timing, GPU identity and record validation.
        return flatcost.measure(args.binary, row, n, 1, "default", log)
    finally:
        os.environ.pop("GGML_CUDA_MMA78_ARM")


def run(args, shapes):
    result = dict(shapes=shapes, widths=[6, 7, 8], arms=args.arms, repeats=args.repeats,
                  metric="complete operation us; shared flatcost cold rotation and deterministic inputs",
                  pairing="alternating arm order; process-level pairs; no default/default bookends",
                  controls="MMQ at IQ4 down n=8 remeasured from the prior sweep winner; other cells use served dispatch",
                  rows=[])
    for shape_id, row in enumerate(shapes):
        for n in (6, 7, 8):
            arms = list(args.arms)
            if row["type"] == "iq4_xs" and row["m"] == 5120 and row["k"] == 8704 and n == 8 and "mmq" not in arms:
                arms.append("mmq")
            samples = {arm: [] for arm in arms}
            for repeat in range(args.repeats):
                order = arms if repeat % 2 == 0 else list(reversed(arms))
                for arm in order:
                    log = args.out / f"gpu0-s{shape_id}-n{n}-b1-r{repeat}-{arm}.log"
                    samples[arm].append(measure(args, row, n, arm, log))
            entry = dict(shape_id=shape_id, n=n, arms={arm: flatcost.ci(values) for arm, values in samples.items()})
            for arm, values in samples.items():
                entry["arms"][arm]["speedup_vs_default"] = flatcost.ci([a / b for a, b in zip(samples["default"], values)])
            result["rows"].append(entry)
            flatcost.write_json(args.out / "results.json", result)
    # The gate requires both IQ4 shapes at both widths, regardless of a caller filter.
    wins = []
    for key in SHAPES[:2]:
        ids = [i for i, row in enumerate(shapes) if f"{row['type']}:{row['m']}:{row['k']}" == key]
        if not ids:
            raise ValueError("promotion gate requires both IQ4 shapes; rerun without --shape")
        for n in (7, 8):
            entry = next(e for e in result["rows"] if e["shape_id"] == ids[0] and e["n"] == n)
            baseline = min(value["mean"] for arm, value in entry["arms"].items() if arm != "mma")
            wins.append(baseline / entry["arms"]["mma"]["mean"])
    geomean = math.exp(sum(math.log(win) for win in wins) / len(wins))
    result["gate"] = dict(iq4_speedups=wins, iq4_geomean=geomean,
                          pass_gate=geomean >= 1.10 and min(wins) >= 1.0,
                          best_existing="prior winner selection, remeasured in this item; not an exhaustive new crossover sweep")
    flatcost.write_json(args.out / "results.json", result)
    if not result["gate"]["pass_gate"]:
        raise RuntimeError("MMA kill gate: IQ4 n7/8 geomean <1.10 or a large-shape cell loses; inspect results.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="verify GGUF headers against the shared frozen inventory; no GPU execution")
    mode.add_argument("--correctness", action="store_true", help="compare selected shapes and edge cases with the CPU reference")
    mode.add_argument("--run", action="store_true", help="measure n=6,7,8 and apply the IQ4 kill gate")
    parser.add_argument("--out", type=Path, required=True, help="artifact directory")
    parser.add_argument("--binary", type=Path, help="preflight-built test-backend-ops; required for GPU modes")
    parser.add_argument("--shape", nargs="+", choices=SHAPES, default=list(SHAPES), help="type:m:k subset; default is the four served shapes")
    parser.add_argument("--arms", nargs="+", choices=("default", "mma", "mmq", "mmvq"), default=["default", "mma"],
                        help="dispatch arms; default restores pre-MMA served dispatch, mma enables the candidate")
    parser.add_argument("--repeats", type=int, default=4, help="paired repetitions; minimum 4, default 4")
    args = parser.parse_args()
    if args.repeats < 4 or len(set(args.arms)) != len(args.arms) or len(set(args.shape)) != len(args.shape):
        parser.error("use --repeats >=4 and unique --arms and --shape values")
    if args.prepare and (args.binary or args.arms != ["default", "mma"] or args.repeats != 4):
        parser.error("--binary, --arms and --repeats apply only to GPU modes; omit them with --prepare")
    if args.run and not {"default", "mma"}.issubset(args.arms):
        parser.error("--run requires default and mma in --arms for the paired gate")
    if args.run and not set(SHAPES[:2]).issubset(args.shape):
        parser.error("--run requires both IQ4 shapes for the kill gate; include both in --shape")
    if args.correctness and args.repeats != 4:
        parser.error("--repeats applies only to --run; omit it with --correctness")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        manifest = flatcost.prepare(args.out)
        if manifest != json.loads((SHARED / "evidence/shapes.json").read_text()):
            raise ValueError("GGUF headers changed; review the shared frozen inventory before queueing")
        select_shapes(manifest, args.shape)
        return
    gpu_guard(args)
    manifest = json.loads((SHARED / "evidence/shapes.json").read_text())
    shapes = select_shapes(manifest, args.shape)
    print(f"mma78: shapes={args.shape}, arms={args.arms}, repeats={args.repeats}, GPU0 (gpuq pin)", flush=True)
    if args.correctness:
        correctness(args, shapes)
    else:
        run(args, shapes)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"mma78: {error}", file=sys.stderr)
        sys.exit(5)
