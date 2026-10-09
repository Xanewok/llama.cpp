"""Render the np2 real J5q item of the ar-duplex window with the parent harness; gpuq submit --check only, never submit.
Base arm: frozen/stack10 if it exists, else frozen/stack9-p55. Dry run before the freeze: AR_DUPLEX_ROOT=<any frozen dir> overrides the ard arm root."""
import argparse
import hashlib
import json
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
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    ard_root = Path(os.environ.get('AR_DUPLEX_ROOT', W / 'frozen/ar-duplex'))
    sha = (ard_root / 'SOURCE').read_text().splitlines()[0]
    assert os.environ.get('AR_DUPLEX_ROOT') or sha == head, 'freeze must match HEAD'
    if (W / 'frozen/stack10/SOURCE').exists():
        base = dict(label='stack10', root=W / 'frozen/stack10', repo=W / 'wt/stack10',
                    canary=['persistent rotations backend=', r'alpha_gate_fusion: device=0 width=[1-9]\d* rows=[1-9]\d*',
                            r'alpha_gate_fusion: device=1 width=[1-9]\d* rows=[1-9]\d*', r'mtp_fold_decode: engaged n_seq=2\b',
                            r'llama_kv_cache: size =[^\n]*,\s+1 layers,[^\n]*K \(f16\):[^\n]*V \(f16\):'])
    else:
        base = dict(label='s9p55', root=W / 'frozen/stack9-p55', repo=W / 'wt/stack9-p55',
                    canary=[r'llama_kv_cache: size =[^\n]*,\s+1 layers,[^\n]*K \(f16\):[^\n]*V \(f16\):'])
    base['sha'] = (base['root'] / 'SOURCE').read_text().splitlines()[0]
    print(f"base arm: {base['label']} {base['sha']}  ard arm: {ard_root} {sha}")
    mkitem.harness_snapshot = local_snapshot
    for name in ('ardup-np2-real-j5q',):
        experiment = args.out / f'{name}.toml'
        source = (ROOT / f'codex/items/{name}.toml.in').read_text()
        for key, val in (('@BASE_LABEL@', base['label']), ('@BASE_SHA@', base['sha']), ('@BASE_ROOT@', str(base['root'])),
                         ('@BASE_REPO@', str(base['repo'])), ('@BASE_CANARY@', json.dumps(base['canary'])), ('@ARDUP_SHA@', sha),
                         ('@ARDUP_ROOT@', str(ard_root)), ('@ARDUP_REPO@', str(ROOT))):
            source = source.replace(key, val)
        assert '@' not in source.replace('@' + 'ar-duplex', '')
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
        assert str(base['root'] / 'build/bin/round-trace') in parsed['pins']['files']
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
