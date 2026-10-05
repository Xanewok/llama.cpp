#!/usr/bin/env python3
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / 'codex/cpu-results'
out.mkdir(exist_ok=True)
model = '/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf'
gate = '/home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate'
base = os.environ | dict(CUDA_VISIBLE_DEVICES='', GIT_NO_LAZY_FETCH='1', LLAMA_ARG_ALLOW_CPU_DEVICE='1', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONDONTWRITEBYTECODE='1')
base['PATH'] = str(root / 'codex/gate-bin') + os.pathsep + base['PATH']
for name in ('GGML_BACKEND_PATH', 'LD_PRELOAD', 'GDN_TEST_META_SEQ_CP', 'GDN_FIXTURE_TRACE'):
    base.pop(name, None)
for line in (root / 'codex/base-cpu/SHA256SUMS').read_text().splitlines():
    digest, name = line.split('  ', 1)
    with (root / 'codex/base-cpu' / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name
results = []
for meta in (0, 1):
    for np in (1, 2):
        for arm in ('base', 'indexed'):
            name = f'{arm}-meta{meta}-np{np}'
            env = base | {'LD_LIBRARY_PATH': str(root / ('codex/base-cpu/bin' if arm == 'base' else 'build-cpu/bin'))}
            cmd = [gate, str(root / 'build-cpu/bin/gdn-cpu'), model, str(meta), str(np), str(out / f'{name}.txt'), '0', '-c', '4096', '3']
            with (out / f'{name}.log').open('w') as log:
                subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            text = (out / f'{name}.log').read_text()
            counters = re.search(r'M: .*indexed_calls=(\d+).*wall_ms=([\d.]+)', text)
            assert counters, text[-2000:]
            assert (int(counters[1]) > 0) == (arm == 'indexed'), text[-2000:]
            results.append(dict(arm=arm, meta=meta, np=np, indexed_calls_M=int(counters[1]), wall_ms_M=float(counters[2])))
        a = (out / f'base-meta{meta}-np{np}.txt').read_bytes()
        b = (out / f'indexed-meta{meta}-np{np}.txt').read_bytes()
        assert a == b, f'identity failed: meta={meta} np={np}'
        print(f'M: sampled tokens, logits/hidden fingerprints and per-layer R/S fingerprints: meta={meta} np={np}', flush=True)
for mutant, label in ((1, 'wrong-slot'), (2, 'stale-after-clear')):
    name = f'negative-{label}'
    cmd = [gate, str(root / 'build-cpu/bin/gdn-cpu'), model, '0', '1', str(out / f'{name}.txt'), str(mutant), '-c', '4096', '3']
    with (out / f'{name}.log').open('w') as log:
        subprocess.run(cmd, cwd=root, env=base | {'LD_LIBRARY_PATH': str(root / 'build-cpu/bin')}, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert 'mutated=1' in (out / f'{name}.log').read_text()
    assert (out / f'{name}.txt').read_bytes() != (out / 'base-meta0-np1.txt').read_bytes(), f'{label} escaped'
    print(f'M: {label} changed the compared output', flush=True)
(out / 'identity-results.json').write_text(json.dumps({'runs': results, 'identity': 'PASS', 'negative_controls': 'PASS'}, indent=2) + '\n')
