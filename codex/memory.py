#!/usr/bin/env python3
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "gguf-py"))
import gguf

assert len(sys.argv) == 3, "usage: memory.py MODEL VOCAB_JSON"
ids = json.loads(pathlib.Path(sys.argv[2]).read_text())
reader = gguf.GGUFReader(sys.argv[1])
for tensor in reader.tensors:
    if "embed_tokens" in tensor.name or tensor.name == "token_embd.weight":
        block, size = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
        row_bytes = int(tensor.shape[0]) * size // block
        total = len(ids) * row_bytes
        print(f"M: {tensor.name} {tensor.tensor_type.name} shape={tensor.shape.tolist()} subset rows={len(ids)}")
        print(f"D: row={row_bytes} B; subset table={total} B/device ({total / 2**20:.9f} MiB/device); compact indices={len(ids)*4} B total, vocab-sharded")
