"""Render the fa-q8 depth item with the parent harness into this worktree; check only after freezing."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parent.parent
EVAL = Path('/home/xanewok/repos/qwen38-eval')
W = ROOT.parent.parent
sys.path.insert(0, str(EVAL))
from harness import lib, mkitem, snapshot


def local_snapshot(x):
    files = {f'harness/{p.name}': p.read_bytes() for p in [*mkitem.H.glob('*.py'), *mkitem.H.glob('*.sh')]}
    experiment = x['file'].read_bytes()
    assert hashlib.sha256(experiment).hexdigest() == x['_file_sha'], 'experiment changed during rendering'
    files['experiment.toml'] = experiment
    files['source-root.txt'] = (str(EVAL) + '\n').encode()
    return snapshot.create(files, ROOT / 'codex/snapshots', 'harness')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base_sha = (W / 'frozen/stack10-cand2/SOURCE').read_text().splitlines()[0]
    sha = (W / 'frozen/fa-q8-inkernel/SOURCE').read_text().splitlines()[0]
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    assert sha == head, 'freeze must match HEAD'
    subprocess.check_call(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', base_sha, head])
    for tested in (ROOT/'codex/parser-harness/harness').glob('*.py'):
        assert tested.read_bytes() == (EVAL/'harness'/tested.name).read_bytes(), f'parent parser changed since Mac replay: {tested.name}'
    mkitem.harness_snapshot = local_snapshot
    for name in ('fa-q8-depth-j5q',):
        experiment = args.out / f'{name}.toml'
        source = (ROOT / f'codex/items/{name}.toml.in').read_text()
        source = source.replace('@STACK10_SHA@', base_sha).replace('@NEW_SHA@', sha)
        assert '@' not in source
        experiment.write_text(source)
        x = lib.read_experiment(experiment)
        text = mkitem.render(x)
        text = mkitem.wrap_stages(text, ROOT / 'codex/gpu_fa_q8.sh')
        text = text.replace('kind = "measure"', 'host_sensitive = false\nkind = "measure"', 1)
        item = tomllib.loads(text)
        # Keep the preflight on its normal CPU path; both GPU stages inherit CCX0+1 affinity.
        import json
        for stage in ('run', 'canary'):
            old = 'cmd = ' + json.dumps(item[stage]['cmd'])
            new = 'cmd = ' + json.dumps(['bash', str(W / 'work/aa-load/ccx01.sh'), *item[stage]['cmd']])
            assert text.count(old) == 1
            text = text.replace(old, new)
        # gpuq treats every .py/.sh beside the entry as its library: pin gpu_ab.sh's siblings too.
        entry_dir = (ROOT / 'codex/gpu_fa_q8.sh').resolve().parent
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
        for frozen in ('stack10-cand2', 'fa-q8-inkernel'):
            assert str(W / f'frozen/{frozen}/build/bin/round-trace') in parsed['pins']['files']
        assert not any('build-bw-mac-results' in p for p in parsed['pins']['files'])
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
