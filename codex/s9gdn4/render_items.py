"""Render the two s9-gdn4 timing items (np1 depth, np2 real prompts) from codex/items/*.toml.in and run gpuq submit --check only.

  render_items.py --out DIR [--g4-root DIR]

--g4-root defaults to frozen/stack9-gdn4 (exists after the bw-s9-gdn4 window). Its SOURCE commit must be an ancestor of HEAD with no
non-codex difference. A stand-in root (hardlinked copy of frozen/stack9 with its own SOURCE) checks the items before the freeze;
its output is a draft for the contract check only.
"""
import argparse
import hashlib
import json
import re
import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = Path('/home/xanewok/repos/qwen38-eval')
W = ROOT.parent.parent
sys.path.insert(0, str(EVAL))
from harness import lib, mkitem, snapshot  # noqa: E402

NAMES = ('s9gdn4-depth-j5q', 's9gdn4-np2-real-j5q')


def local_snapshot(x):
    files = {f'harness/{p.name}': p.read_bytes() for p in [*mkitem.H.glob('*.py'), *mkitem.H.glob('*.sh')]}
    experiment = x['file'].read_bytes()
    assert hashlib.sha256(experiment).hexdigest() == x['_file_sha'], 'experiment changed during rendering'
    files['experiment.toml'] = experiment
    files['source-root.txt'] = (str(EVAL) + '\n').encode()
    return snapshot.create(files, ROOT / 'codex/snapshots', 'harness')


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--g4-root', type=Path, default=W / 'frozen/stack9-gdn4')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    s9 = (W / 'frozen/stack9/SOURCE').read_text().splitlines()[0]
    g4 = (args.g4_root / 'SOURCE').read_text().splitlines()[0]
    subprocess.check_call(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', g4, 'HEAD'])
    assert git('diff', '--name-only', g4, 'HEAD', '--', '.', ':!codex') == '', 'HEAD differs from the frozen commit outside codex/'
    mkitem.harness_snapshot = local_snapshot
    for name in NAMES:
        experiment = args.out / f'{name}.toml'
        source = (ROOT / f'codex/items/{name}.toml.in').read_text()
        source = source.replace('@S9_SHA@', s9).replace('@G4_SHA@', g4).replace('@G4_ROOT@', str(args.g4_root))
        assert not re.search(r'@[A-Z0-9_]+@', source)
        experiment.write_text(source)
        x = lib.read_experiment(experiment)
        text = mkitem.render(x)
        text = mkitem.wrap_stages(text, ROOT / 'codex/gpu_ab.sh')
        text = text.replace('kind = "measure"', 'host_sensitive = false\nkind = "measure"', 1)
        item = tomllib.loads(text)
        # Keep the preflight on its normal CPU path; both GPU stages inherit CCX0+1 affinity.
        for stage in ('run', 'canary'):
            old = 'cmd = ' + json.dumps(item[stage]['cmd'])
            new = 'cmd = ' + json.dumps(['bash', str(W / 'work/aa-load/ccx01.sh'), *item[stage]['cmd']])
            assert text.count(old) == 1
            text = text.replace(old, new)
        # gpuq treats every .py/.sh beside the entry as its library: pin gpu_ab.sh's siblings too.
        entry_dir = (ROOT / 'codex/gpu_ab.sh').resolve().parent
        have = set(tomllib.loads(text)['pins']['files'])
        siblings = sorted(str(q) for q in entry_dir.iterdir() if q.suffix in ('.py', '.sh') and q.is_file()
                          and not q.name.startswith(('gen_', 'test_', '.gpuq-')) and str(q) not in have)
        if siblings:
            head, sep, tail = text.partition('[pins]\nfiles = [\n')
            assert sep, 'pins.files is not a multi-line array'
            text = head + sep + ''.join(f'  {json.dumps(q)},\n' for q in siblings) + tail
        parsed = tomllib.loads(text)
        assert parsed['canary']['timeout_s'] <= 600
        assert str(W / 'work/aa-load/ccx01.sh') in parsed['pins']['files']
        assert str(args.g4_root / 'build/bin/round-trace') in parsed['pins']['files']
        assert str(W / 'frozen/stack9/build/bin/round-trace') in parsed['pins']['files']
        assert parsed['priority'] == 29 and parsed['host_sensitive'] is False
        assert 'expect:' in parsed['note'] and 'falsifier:' in parsed['note']
        path = args.out / f'{name}.gpuq.toml'
        path.write_text(text)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', GIT_NO_LAZY_FETCH='1', PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run(['gpuq', 'submit', '--check', str(path)], env=env, text=True, capture_output=True)
        log = result.stdout + result.stderr
        (args.out / f'{name}-contract.log').write_text(log)
        assert result.returncode == 0 and 'contract violated' not in log.lower(), log
        print(f'checked, NOT submitted: {path}')


if __name__ == '__main__':
    main()
