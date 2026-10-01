#!/usr/bin/env python3
"""CPU model: round each 16-key MMA result to half; this is not CUDA execution."""
import argparse
from pathlib import Path

import numpy as np


def accumulate(v, weights, precision):
    total = np.zeros(v.shape[1:], dtype=precision)
    for start in range(0, len(v), 16):
        update = (v[start:start + 16] * weights[start:start + 16, None, None]).sum(axis=0, dtype=np.float64)
        total = (total.astype(np.float64) + update).astype(precision)
    return total.astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tokens", type=int, default=149)
    args = parser.parse_args()
    # This control must distinguish the overflowing path from the corrected law.
    v = np.full((512, 1, 1), 1024.0)
    w = np.full(512, 0.125)
    with np.errstate(over="ignore"):
        assert np.isinf(accumulate(v, w, np.float16)).all()
    assert np.all(accumulate(v, w, np.float32) == 65536.0)
    print("CONTROL,half=inf,float=65536")
    print("MODEL,layer,keys,flatness,l2_half_error,l2_float_error,max_channel_mean,max_abs")
    for path in sorted(args.directory.glob("Vcur-*.f32"), key=lambda p: int(p.stem.split("-")[1]) if " " not in p.stem else 10000):
        if " " in path.stem:
            continue
        values = np.fromfile(path, dtype=np.float32).reshape(args.tokens, 4, 256).astype(np.float16).astype(np.float64)
        for keys in (464, 928, 1424, 2848, 4272):
            extended = values[np.arange(keys) % args.tokens]
            # G scenarios: flat logits and one peak with a diffuse tail, repeated measured values.
            for flatness in (1.0, 0.1, 0.01):
                w = np.full(keys, flatness/8.0, dtype=np.float16).astype(np.float64)
                w[-1] = 0.125
                ref = (extended * w[:, None, None]).sum(axis=0)
                norm = np.linalg.norm(ref)
                half_err = np.linalg.norm(accumulate(extended, w, np.float16) - ref)/norm
                float_err = np.linalg.norm(accumulate(extended, w, np.float32) - ref)/norm
                print(f"MODEL,{path.stem},{keys},{flatness},{half_err:.9g},{float_err:.9g},{np.abs(values.mean(axis=0)).max():.9g},{np.abs(values).max():.9g}")
        for keys in (32768, 102400):
            for np_warps in (2, 1):
                for scenario in ("repeat", "persistent_mean"):
                    if scenario == "repeat":
                        extended = values[np.arange(keys) % args.tokens]
                    else:
                        mean = values.mean(axis=0).astype(np.float16).astype(np.float64)
                        extended = np.broadcast_to(mean, (keys, 4, 256))
                    ref = extended.sum(axis=0)/8.0
                    total_half = np.zeros((4, 256), dtype=np.float64)
                    total_float = np.zeros((4, 256), dtype=np.float64)
                    tiles = keys//32
                    for block in range(36):
                        lo, hi = (block*tiles//36)*32, ((block + 1)*tiles//36)*32
                        for warp in range(np_warps):
                            selected = np.arange(lo, hi).reshape(-1, 16)[warp::np_warps].reshape(-1)
                            local = extended[selected]
                            w = np.full(len(local), 0.125)
                            total_half += accumulate(local, w, np.float16)
                            total_float += accumulate(local, w, np.float32)
                    half_err = np.linalg.norm(total_half - ref)/np.linalg.norm(ref)
                    float_err = np.linalg.norm(total_float - ref)/np.linalg.norm(ref)
                    print(f"OUTPUT,{path.stem},{keys},np={np_warps}:{scenario},{half_err:.9g},{float_err:.9g},{np.abs(values.mean(axis=0)).max():.9g},{np.abs(values).max():.9g}")


if __name__ == "__main__":
    main()
