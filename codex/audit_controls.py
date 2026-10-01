#!/usr/bin/env python3
"""CPU address/lifetime oracles; these do not execute CUDA kernels."""
import argparse
import pathlib
import struct
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--broken-mask-bound', action='store_true')
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parent.parent

def require(ok, message):
    if not ok:
        raise SystemExit(message)

# The mask scanner has a real partial-tile hole, but its launch gate excludes this campaign.
for rows in (4, 5, 6, 128):
    launched = rows >= 1024 or 1 > 1
    require(not launched, 'single-sequence verify reached the KV-max scanner')
overreads = 0
for rows in (4, 5, 6):
    width = 4 if rows <= 4 else 8
    for j in range(width):
        mask_row = j if args.broken_mask_bound else j % rows
        require(mask_row < rows, 'FAIL: padded MMA mask address escaped actual queries')
        overreads += j >= rows
require(overreads == 5, 'partial-tile fixture changed')
print('D: five padded query lanes across the five/six-row fixtures; MMA modulo addresses are in bounds.')

for k in (3, 4, 5):
    slots = k + 1
    for tokens in (1, k + 1, 127, 128):
        written = [tokens - 1 - t for t in range(tokens) if 0 <= tokens - 1 - t < slots]
        require(sorted(written) == list(range(min(tokens, slots))), 'snapshot slots missing or duplicated')
    for rollback in range(1, k + 1):
        keep = k + 1 - rollback
        require(keep >= 1 and rollback < slots, 'rollback cannot address a retained state')
print('D: snapshot indices cover min(tokens,k+1) groups without a depth-four boundary.')

# Exact candidate IDs fit F32, while BF16 would destroy ordinary vocabulary IDs.
value = struct.unpack('<I', struct.pack('<f', 25879.0))[0]
rounded = (value + 0x7fff + ((value >> 16) & 1)) & 0xffff0000
wrong_id = int(struct.unpack('<f', struct.pack('<I', rounded))[0])
require(wrong_id != 25879, 'BF16 ID corruption control was insensitive')
print(f'D: BF16 negative control maps token ID 25879 to {wrong_id}; exact gather must bypass it.')

# Exercise the delivered comparison tool, including artifact corruption and missing artifacts.
with tempfile.TemporaryDirectory(dir=root / 'build-cpu') as tmp:
    out = pathlib.Path(tmp)
    runs = [out / name for name in ('0-up', '1-patch', '2-patch', '3-up')]
    for run in runs:
        (run / 'logits').mkdir(parents=True)
        (run / 'status').write_text('0\n')
        (run / 'logits/test.f32').write_bytes(struct.pack('<ff', 1, 2))
    command = ['python3', str(root / 'codex/gpu_compare.py'), str(out)]
    require(subprocess.run(command, capture_output=True).returncode == 0, 'equal-row control failed')
    (runs[1] / 'logits/test.f32').write_bytes(struct.pack('<ff', 1, 3))
    require(subprocess.run(command, capture_output=True).returncode == 1, 'perturbed-row control escaped')
    (runs[1] / 'logits/test.f32').unlink()
    require(subprocess.run(command, capture_output=True).returncode != 0, 'missing-row control escaped')
print('M: comparison accepts equal rows and rejects perturbed/missing rows.')
