#!/usr/bin/env python3
"""RMS-normalized input bound, computed from the real GGUF on CPU."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
from gguf import GGUFReader
from gguf.quants import dequantize


parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()
reader = GGUFReader(args.model, "r")
tensors = {t.name: t for t in reader.tensors}
print("layer,max_rms_bound")
for layer in range(3, 64, 4):
    v = tensors[f"blk.{layer}.attn_v.weight"]
    norm = tensors[f"blk.{layer}.attn_norm.weight"]
    gain = dequantize(norm.data, norm.tensor_type).astype(np.float64).reshape(-1)
    weights = dequantize(v.data, v.tensor_type).astype(np.float64).reshape(-1, len(gain))
    bound = np.sqrt(len(gain)) * np.linalg.norm(weights * gain, axis=1)
    print(f"{layer},{bound.max():.9g}")
