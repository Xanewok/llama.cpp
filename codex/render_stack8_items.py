"""Render with the parent harness into this worktree; check only after freezing."""
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
    parser.add_argument('--duo-fold-c', choices=('0', '1'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sha = (W / 'frozen/stack8/SOURCE').read_text().splitlines()[0]
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    assert sha == head, 'freeze must match HEAD'
    mkitem.harness_snapshot = local_snapshot
    for name in ('stack8-depth-j5q', 'stack8-np2-chain-j5q', 'stack8-duo-soak-j5q'):
        experiment = args.out / f'{name}.toml'
        source = (ROOT / f'codex/items/{name}.toml.in').read_text()
        source = source.replace('@STACK8_SHA@', sha).replace('@DUO_FOLD_C@', args.duo_fold_c)
        assert '@' not in source
        experiment.write_text(source)
        x = lib.read_experiment(experiment)
        text = mkitem.render(x)
        text = mkitem.wrap_stages(text, ROOT / 'codex/gpu_ab.sh')
        text = text.replace('kind = "measure"', 'host_sensitive = false\nkind = "measure"', 1)
        item = tomllib.loads(text)
        # Keep the preflight on its normal CPU path; both GPU stages inherit CCX0+1 affinity.
        import json
        for stage in ('run', 'canary'):
            old = 'cmd = ' + json.dumps(item[stage]['cmd'])
            new = 'cmd = ' + json.dumps(['bash', str(W / 'work/aa-load/ccx01.sh'), *item[stage]['cmd']])
            assert text.count(old) == 1
            text = text.replace(old, new)
        parsed = tomllib.loads(text)
        assert parsed['canary']['timeout_s'] <= 600
        assert str(W / 'work/aa-load/ccx01.sh') in parsed['pins']['files']
        assert str(W / 'frozen/stack8/build/bin/round-trace') in parsed['pins']['files']
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
