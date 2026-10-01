"""CPU-only, one-tensor-at-a-time audit of two prepared checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '8'
from safetensors import safe_open
import torch

torch.set_num_threads(8)


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(root):
    tensors = {}
    for path in sorted(root.glob('*.safetensors')):
        with path.open('rb') as f:
            header = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
        with safe_open(path, framework='pt', device='cpu') as f:
            for key in f.keys():
                if key in tensors:
                    raise ValueError(f'duplicate tensor: {key}')
                tensor = f.get_tensor(key)
                digest = hashlib.sha256(memoryview(tensor.view(torch.uint8).numpy())).hexdigest()
                del tensor
                info = header[key]
                tensors[key] = dict(file=path.name, dtype=info['dtype'], shape=info['shape'],
                                    bytes=info['data_offsets'][1] - info['data_offsets'][0], sha256=digest)
    files = {p.name: sha_file(p) for p in sorted(root.iterdir())
             if p.is_file() and p.suffix in ('.json', '.yaml', '.jinja', '.txt', '.pt')
             and not p.name.startswith(('SHA256', 'RECIPE'))}
    return dict(root=str(root), tensors=tensors, files=files)


def compare(a, b):
    keys = sorted(a['tensors'].keys() | b['tensors'].keys())
    changed = [k for k in keys if a['tensors'].get(k) != b['tensors'].get(k)]
    return dict(tensor_count=[len(a['tensors']), len(b['tensors'])], changed=changed,
                file_diffs=[k for k in sorted(a['files'].keys() | b['files'].keys())
                            if a['files'].get(k) != b['files'].get(k)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('up', type=Path)
    p.add_argument('patch', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    start = time.perf_counter()
    arms = [audit(args.up), audit(args.patch)]
    result = dict(arms=arms, comparison=compare(*arms), wall_seconds_M=time.perf_counter()-start)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    changed = result['comparison']['changed']
    print(json.dumps(dict(wall_seconds_M=result['wall_seconds_M'],
                         tensor_count=result['comparison']['tensor_count'],
                         changed=len(changed), changed_mtp=[k for k in changed if 'mtp' in k],
                         file_diffs=result['comparison']['file_diffs']), indent=2))


if __name__ == '__main__':
    main()
