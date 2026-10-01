#!/usr/bin/env python3
"""Parent-only Marlin benchmark. Dry runs import only the standard library."""
import argparse
import hashlib
import json
import os
import pathlib
import statistics
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--shapes", type=pathlib.Path, default=pathlib.Path(__file__).parent / "evidence/shapes.json")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--budget-s", type=int, default=180)
    args = parser.parse_args()
    shapes = json.loads(args.shapes.read_text())
    variants = [("w4g128", 4, 128), ("w4g32", 4, 32), ("w4g64", 4, 64), ("w8channel", 8, -1)]
    if not args.execute:
        print(json.dumps(dict(parent_only=True, shapes=shapes, variants=variants, rows=list(range(1, 7)),
                              cache_modes=["warm", "cold"], devices="run each TP rank separately", budget_s=args.budget_s)))
        return
    if os.environ.get("PARENT_GPU") != "1" or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("--execute requires PARENT_GPU=1 and explicit CUDA_VISIBLE_DEVICES")
    if args.repeats < 2 or args.budget_s < 1:
        parser.error("invalid repeat count or budget")

    import torch
    import vllm
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    frozen = pathlib.Path(os.environ["FROZEN_ROOT"]).resolve()
    for module in [torch, vllm, ops]:
        if not pathlib.Path(module.__file__).resolve().is_relative_to(frozen):
            raise RuntimeError(f"module escaped frozen environment: {module.__file__}")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    props = torch.cuda.get_device_properties(device)
    print(json.dumps(dict(kind="metadata", gpu=props.name, capability=[props.major, props.minor],
                          torch=torch.__version__, tune=os.environ.get("VLLM_MARLIN_TUNE", "0"),
                          vllm_path=vllm.__file__, torch_path=torch.__file__)), flush=True)
    deadline = time.monotonic() + args.budget_s
    # Flush more than this card's L2 between cold samples; exclude the flush from timing.
    eviction = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    for shape_index, shape in enumerate(shapes):
        k, n = shape["k"], shape["n"]
        for variant, bits, group in variants:
            if time.monotonic() >= deadline:
                raise TimeoutError("microbenchmark budget exhausted")
            torch.manual_seed(913 + shape_index)
            w = torch.randn(k, n, dtype=torch.bfloat16, device=device) * 0.02
            wtype = scalar_types.uint4b8 if bits == 4 else scalar_types.uint8b128
            ref, packed, scales = marlin_quantize(w, wtype, group)
            del w
            workspace = marlin_make_workspace_new(device)
            for m in range(1, 7):
                torch.manual_seed(321 + m)
                a = torch.randn(m, k, dtype=torch.bfloat16, device=device)
                def call():
                    return ops.marlin_gemm(a, None, packed, None, scales, None, None, None,
                                           workspace, wtype, m, n, k, False, True, False)

                got = call()
                expected = a @ ref
                if not torch.allclose(got.float(), expected.float(), rtol=0.05, atol=0.03):
                    raise AssertionError(f"dequantized GEMM mismatch: {shape['name']} {variant} m={m}")
                broken = got.clone()
                broken[0, 0] += 10000
                if torch.allclose(broken.float(), expected.float(), rtol=0.05, atol=0.03):
                    raise AssertionError("negative control failed to reject corrupted output")
                raw = got.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                digest = hashlib.sha256(raw).hexdigest()
                for _ in range(10):
                    call()
                torch.cuda.synchronize()
                for mode in ["warm", "cold"]:
                    start = torch.cuda.Event(enable_timing=True, external=True)
                    end = torch.cuda.Event(enable_timing=True, external=True)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        if mode == "cold":
                            eviction.zero_()
                        start.record()
                        output = call()
                        end.record()
                    samples = []
                    for _ in range(args.repeats):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("microbenchmark budget exhausted")
                        graph.replay()
                        torch.cuda.synchronize()
                        samples.append(start.elapsed_time(end) * 1000)
                    if not torch.equal(output, got):
                        raise AssertionError("graph replay changed output")
                    print(json.dumps(dict(kind="latency", label="M", shape=shape["name"], k=k, n=n,
                                          count=shape["count"], variant=variant, m=m, cache=mode,
                                          p50_us=statistics.median(samples), min_us=min(samples),
                                          max_us=max(samples), output_sha256=digest)), flush=True)
                    del graph, output
            del ref, packed, scales, workspace, a, got, broken, expected


if __name__ == "__main__":
    main()
