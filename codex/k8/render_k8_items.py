"""Render the K8 perf item into the gpuq drafts and run gpuq submit --check (never submits).

Usage: render_k8_items.py COMMIT LOG_DIR     env K8_DRAFTS overrides the drafts dir (dry runs)
"""
import datetime
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = Path('/home/xanewok/repos/qwen38-eval')
DRAFTS = Path(os.environ.get('K8_DRAFTS', EVAL / 'orchestration/gpuq/items/drafts'))
CCX01 = Path('/home/xanewok/repos/qwen38-eval-worktrees/work/aa-load/ccx01.sh')
BIN = ROOT / 'k8-bin'


def main():
    commit, log = sys.argv[1], Path(sys.argv[2]).resolve()
    assert (BIN / 'SOURCE').read_text().split()[0] == commit
    libs = sorted(p for p in BIN.glob('libggml*.so.*') if p.is_file() and not p.is_symlink())
    assert libs
    name = 'k8-mmq-perf'
    text = (ROOT / f'codex/k8/items/{name}.toml.in').read_text()
    text = (text.replace('@ROOT@', str(ROOT)).replace('@COMMIT@', commit).replace('@DATE@', datetime.date.today().isoformat())
            .replace('@LIB_PINS@', '\n'.join(f'  "{p}",' for p in libs)))
    assert not re.search(r'@[A-Z_]+@', text)
    parsed = tomllib.loads(text)
    assert parsed['priority'] == 29 and parsed['host_sensitive'] is False
    for stage in ('run', 'canary'):
        assert parsed[stage]['cmd'][:2] == ['bash', str(CCX01)]
    item = DRAFTS / f'{name}.toml'
    assert not item.exists(), f'{item} exists; not overwriting'
    item.write_text(text)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', GIT_NO_LAZY_FETCH='1', PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run(['gpuq', 'submit', '--check', str(item)], env=env, capture_output=True, text=True)
    (log / f'{name}-contract.log').write_text(r.stdout + r.stderr)
    assert r.returncode == 0 and 'contract violated' not in (r.stdout + r.stderr).lower(), (item, r.returncode, r.stdout, r.stderr)
    print(f'checked, NOT submitted: {item}')


if __name__ == '__main__':
    main()
