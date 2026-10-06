"""Read GGUF tensor metadata; never load tensors into a backend."""
import collections
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gguf-py'))
from gguf import GGUFReader

result = {}
for quant in ('IQ4_XS', 'J5q-bf16'):
    reader = GGUFReader('/home/xanewok/models/Swift-1.5-Qwen3.8-27B-' + quant + '.gguf')
    hist = collections.Counter()
    target = collections.Counter()
    shapes = []
    for tensor in reader.tensors:
        hist[tensor.tensor_type.name] += tensor.n_bytes
        match = re.match(r'blk\.(\d+)\.', tensor.name)
        if len(tensor.shape) == 2 and 'conv1d' not in tensor.name and (
            (match and int(match[1]) < 64) or tensor.name == 'output.weight'
        ):
            target[tensor.tensor_type.name] += tensor.n_bytes
        if tensor.name.startswith(('blk.0.', 'blk.3.')) or tensor.name == 'output.weight':
            shapes.append(dict(name=tensor.name, type=tensor.tensor_type.name,
                               shape=[int(n) for n in tensor.shape], bytes=tensor.n_bytes))
    result[quant] = dict(histogram_bytes=dict(hist), target_gemm_bytes=dict(target), shapes=shapes,
                         target_floor_ms_D=sum(target.values()) / 2 / 448e9 * 1000,
                         target_weight_read_twice_floor_ms_D=sum(target.values()) / 448e9 * 1000)
print(json.dumps(result, indent=2))
