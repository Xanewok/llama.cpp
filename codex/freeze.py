"""Freeze local diagnostic inputs; no network or model execution."""
import hashlib
from pathlib import Path
import shutil
import time

HERE=Path(__file__).resolve().parent
FROZEN=HERE/'frozen'
PY=Path('/home/xanewok/repos/HyperQwen/venv/bin/python')
PKG=PY.parent.parent/'lib/python3.12/site-packages'
MODELS={
 'up':Path('/mnt/scratch/work/models/hq/Swift-1.5-Qwen3.8-27B-W4A16-g32-attn-int8'),
 'patch':Path('/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-g32-attn-int8-codexh'),
}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

start=time.perf_counter()
FROZEN.mkdir(exist_ok=True)
for name in ('teacher_probe.py','compare_probe.py','tokens.json'):
    shutil.copyfile(HERE/name,FROZEN/name)
for arm,root in MODELS.items():
    paths=sorted(root.glob('*.safetensors'))+[root/name for name in ('config.json','model.safetensors.index.json','mtp_draft_vocab_ids.pt')]
    (FROZEN/f'{arm}.sha256').write_text(''.join(f'{digest(p)}  {p}\n' for p in paths))
(FROZEN/'SOURCE').write_text('llama.cpp base 68d99c67d; HyperQwen/vLLM checkpoint diagnostic; streamed Transformers reference; BF16 KV; first-slot overlap only\n')
paths=sorted(p for p in FROZEN.iterdir() if p.is_file() and p.name!='SHA256SUMS')
# Freeze the installed CPU/GPU engine code and native libraries, including the interpreter.
paths += [PY]
for package in ('torch','transformers','safetensors','compressed_tensors'):
    paths += sorted(p for p in (PKG/package).rglob('*') if p.is_file() and p.suffix in ('.py','.so'))
paths=sorted(set(paths))
(FROZEN/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p}\n' for p in paths))
print(f'CPU freeze: files_M={len(paths)}; wall_seconds_M={time.perf_counter()-start:.3f}')
