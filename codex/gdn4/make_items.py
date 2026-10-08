"""Render the gdn4 GPU items (cuda checks, np1 depth, np2 real prompts), CCX0+1 pinned; gpuq submit --check only.

Usage: make_items.py COMMIT WINDOW_SUMMARY OUT_LOG_DIR
"""
import datetime
import json
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
ITEMS = ROOT / 'codex/gdn4/items'
CCX01 = W / 'work/aa-load/ccx01.sh'
WRAPPER = W / 'wt/stack8/codex/gpu_ab.sh'  # untraced stack8 entry; takes the saved harness run.sh first
FROZEN = W / 'frozen/gdn-4col-s8'
VARIANTS = W / 'work/gdn-4col-s8-variants'
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', GIT_NO_LAZY_FETCH='1', PYTHONDONTWRITEBYTECODE='1')


def with_ccx01(text):
    item = tomllib.loads(text)
    out, section = [], None
    for line in text.splitlines():
        if line.startswith('['):
            section = line.strip('[]')
        if section in ('run', 'canary') and line.startswith('cmd = '):
            assert item[section]['cmd'][0] == 'bash'
            if item[section]['cmd'][1] != str(CCX01):
                line = 'cmd = ' + json.dumps(['bash', str(CCX01), *item[section]['cmd']])
        out.append(line)
        if line.startswith('priority = ') and 'host_sensitive' not in item:
            out.append('host_sensitive = false')
    text = '\n'.join(out) + '\n'
    if str(CCX01) not in item['pins']['files']:
        text, n = re.subn(r'(?m)^files = \[\n', f'files = [\n  "{CCX01}",\n', text, count=1)
        assert n == 1
    parsed = tomllib.loads(text)
    assert parsed['run']['cmd'][1] == str(CCX01) and parsed['canary']['cmd'][1] == str(CCX01)
    assert parsed['host_sensitive'] is False and str(CCX01) in parsed['pins']['files'] and parsed['priority'] == 29
    return text


def check(item, log):
    r = subprocess.run(['gpuq', 'submit', '--check', str(item)], env=ENV, capture_output=True, text=True)
    (log / f'{item.stem}-contract.log').write_text(r.stdout + r.stderr)
    assert r.returncode == 0 and 'contract violated' not in (r.stdout + r.stderr).lower(), (item, r.returncode, r.stdout, r.stderr)
    print(f'checked, NOT submitted: {item}')


def bin_pins():
    pins = []
    for d in (FROZEN / 'build/bin', VARIANTS / 'base/bin', VARIANTS / 'mutant/bin'):
        libs = sorted(p for p in d.glob('libggml*.so.*') if p.is_file() and not p.is_symlink())
        assert libs, d
        pins += [d / 'SHA256SUMS', d / 'test-backend-ops', *libs]
    return '\n'.join(f'  "{p}",' for p in pins)


def render_checks(commit, date, log):
    text = (ITEMS / 'gdn4-cuda-checks.toml.in').read_text()
    text = text.replace('@ROOT@', str(ROOT)).replace('@COMMIT@', commit).replace('@DATE@', date).replace('@BIN_PINS@', bin_pins())
    assert not re.search(r'@[A-Z_]+@', text)
    item = DRAFTS / 'gdn4-cuda-checks.toml'
    assert not item.exists(), f'{item} exists; not overwriting'
    item.write_text(with_ccx01(text))
    check(item, log)


def render_depth(name, commit, summary, date, log):
    from harness import mkitem
    experiment = EVAL / f'harness/experiments/{name}.toml'
    item = DRAFTS / f'{name}.toml'
    assert not experiment.exists() and not item.exists(), f'{name} exists; not overwriting'
    text = (ITEMS / f'{name}.toml.in').read_text()
    text = text.replace('@COMMIT@', commit).replace('@WINDOW_SUMMARY@', str(summary)).replace('@DATE@', date)
    assert not re.search(r'@[A-Z_]+@', text)
    experiment.write_text(text)
    subprocess.run([sys.executable, '-m', 'harness.mkitem', str(experiment), '--out', str(item)], cwd=EVAL, env=ENV, check=True)
    text = mkitem.wrap_stages(item.read_text(), WRAPPER)
    # gpuq treats every .py/.sh beside the entry as its library: pin gpu_ab.sh's siblings too.
    have = set(tomllib.loads(text)['pins']['files'])
    siblings = sorted(str(q) for q in WRAPPER.parent.iterdir() if q.suffix in ('.py', '.sh') and q.is_file()
                      and not q.name.startswith(('gen_', 'test_', '.gpuq-')) and str(q) not in have)
    head, sep, tail = text.partition('[pins]\nfiles = [\n')
    assert sep, 'pins.files is not a multi-line array'
    text = head + sep + ''.join(f'  {json.dumps(q)},\n' for q in siblings) + tail
    parsed = tomllib.loads(text)
    assert parsed['canary']['timeout_s'] <= 600
    assert str(FROZEN / 'build/bin/round-trace') in parsed['pins']['files'] and str(summary) in parsed['pins']['files']
    item.write_text(with_ccx01(text))
    check(item, log)


def main():
    commit, summary, log = sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
    assert summary.is_file() and not summary.is_relative_to(ROOT)
    assert (FROZEN / 'SOURCE').read_text().split()[0] == commit
    date = datetime.date.today().isoformat()
    sys.path.insert(0, str(EVAL))
    render_checks(commit, date, log)
    for name in ('gdn4-depth-j5q', 'gdn4-np2-real-j5q'):
        render_depth(name, commit, summary, date, log)


if __name__ == '__main__':
    main()
