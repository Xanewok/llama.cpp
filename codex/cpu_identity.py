"""Gated pad identity: token IDs and final logits, CPU and Meta(CPU,CPU), probabilistic MTP."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
MODEL = '/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf'
GATE = '/home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
OUT = ROOT / 'codex/cpu-results'
OUT.mkdir(exist_ok=True)
results = []
for meta in (0, 1):
    expected = None
    for pad in (256, 4096):
        path = OUT / f'api-meta{meta}-pad{pad}.bin'
        env = dict(os.environ, CPU_KV_PAD=str(pad), CPU_PREFILL_WORDS='400', CPU_UBATCH='256', CPU_CACHE_PROMPT='0',
                   OMP_NUM_THREADS='4' if meta else '8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8', LLAMA_ARG_LOG_VERBOSITY='4')
        start = time.monotonic()
        with path.with_suffix('.log').open('w') as log:
            subprocess.run([GATE, str(ROOT / 'build-cpu/bin/fa-identity-cpu'), MODEL, str(path), str(meta), '3', '1729', '32', '2', '0'],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        data = path.read_bytes()
        if expected is None:
            expected = data
        assert data == expected, f'pad identity failed: meta={meta} pad={pad}'
        stats = json.loads(Path(str(path) + '.stats.json').read_text())
        assert sum(s['metrics']['draft_tokens'] for s in stats) > 0, 'MTP draft did not engage'
        if meta:
            assert 'Meta' in path.with_suffix('.log').read_text(), 'Meta did not engage'
        results.append(dict(meta=meta, pad=pad, bytes_M=len(data), sha256=hashlib.sha256(data).hexdigest(), wall_s_M=time.monotonic()-start, records_M=stats))
        (OUT / 'identity.json').write_text(json.dumps(dict(runs=results), indent=2) + '\n')
        print(f'M: meta={meta} pad={pad} bytes={len(data)} identical; MTP draft engaged', flush=True)
for mutant in (False, True):
    start = time.monotonic()
    with (OUT / ('tail-mutant.log' if mutant else 'tail-positive.log')).open('w') as log:
        rc = subprocess.run([GATE, str(ROOT / 'build-cpu/bin/fa-tail-cpu')] + (['drop-last-live-tile'] if mutant else []), stdout=log, stderr=subprocess.STDOUT).returncode
    assert rc == (1 if mutant else 0), f'tail control failed: mutant={mutant} rc={rc}'
    print(f'M: tail control mutant={mutant} expected rc={rc} wall_s={time.monotonic()-start:.6f}', flush=True)
print('M: CPU and Meta pad identity, MTP engagement, padded/unpadded FA and drop-last-live-tile negative control PASS', flush=True)
