"""Render the chain-m2 GPU items (np2 A/B, np1 no-regression) plus CCX0+1 copies; gpuq submit --check only.

Usage: make_items.py COMMIT CPU_SUMMARY OUT_LOG_DIR
"""
import datetime
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = Path('/home/xanewok/repos/qwen38-eval-worktrees')
EVAL = Path('/home/xanewok/repos/qwen38-eval')
DRAFTS = EVAL / 'orchestration/gpuq/items/drafts'
CCX01 = W / 'work/aa-load/ccx01.sh'
WRAPPER = W / 'wt/stack7/codex/gpu_ab.sh'  # untraced stack7 entry, so ms/round compares with the stack7 items
FROZEN = W / 'frozen/chain-m2'


def ccx01_copy(text):
    item = tomllib.loads(text)
    out, section = [], None
    for line in text.splitlines():
        if line.startswith('['):
            section = line.strip('[]')
        if section in ('run', 'canary') and line.startswith('cmd = '):
            assert item[section]['cmd'][0] == 'bash'
            line = 'cmd = ' + __import__('json').dumps(['bash', str(CCX01), *item[section]['cmd']])
        out.append(line)
        if line.startswith('priority = ') and 'host_sensitive' not in item:
            out.append('host_sensitive = false')
    text = '\n'.join(out) + '\n'
    text, n = re.subn(r'(?m)^files = \[\n', f'files = [\n  "{CCX01}",\n', text, count=1)
    assert n == 1
    parsed = tomllib.loads(text)
    assert parsed['run']['cmd'][1] == str(CCX01) and parsed['canary']['cmd'][1] == str(CCX01)
    assert parsed['host_sensitive'] is False and str(CCX01) in parsed['pins']['files']
    return text


def check(item, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', GIT_NO_LAZY_FETCH='1', PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run(['gpuq', 'submit', '--check', str(item)], env=env, capture_output=True, text=True)
    (log / f'{item.stem}-contract.log').write_text(r.stdout + r.stderr)
    assert r.returncode == 0 and 'contract violated' not in (r.stdout + r.stderr).lower(), (item, r.returncode, r.stdout, r.stderr)
    print(f'checked, NOT submitted: {item}')


def main():
    commit, summary, log = sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
    assert summary.is_file() and not summary.is_relative_to(ROOT)
    assert (FROZEN / 'SOURCE').read_text().split()[0] == commit
    date = datetime.date.today().isoformat()
    sys.path.insert(0, str(EVAL))
    from harness import mkitem
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', GIT_NO_LAZY_FETCH='1', PYTHONDONTWRITEBYTECODE='1')
    for name in ('chain-m2-np2-real-j5q', 'chain-m2-np1-depth-j5q'):
        experiment = EVAL / f'harness/experiments/{name}.toml'
        assert not experiment.exists(), f'{experiment} exists; not overwriting'
        text = (ROOT / f'codex/m2/items/{name}.toml.in').read_text()
        text = text.replace('@COMMIT@', commit).replace('@CPU_SUMMARY@', str(summary)).replace('@DATE@', date)
        assert not re.search(r'@[A-Z_]+@', text)
        experiment.write_text(text)
        item = DRAFTS / f'{name}.toml'
        subprocess.run([sys.executable, '-m', 'harness.mkitem', str(experiment), '--out', str(item)], cwd=EVAL, env=env, check=True)
        item.write_text(mkitem.wrap_stages(item.read_text(), WRAPPER))
        parsed = tomllib.loads(item.read_text())
        assert str(summary) in parsed['pins']['files'] and str(FROZEN / 'build/bin/round-trace') in parsed['pins']['files']
        assert parsed['canary']['timeout_s'] <= 600
        check(item, log)
        pinned = DRAFTS / f'{name}-ccx01.toml'
        pinned.write_text(ccx01_copy(item.read_text()))
        check(pinned, log)


if __name__ == '__main__':
    main()
