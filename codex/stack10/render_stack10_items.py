"""Render the stack10 A/B items (frozen/stack9-p55 vs frozen/stack10) with the parent harness and `gpuq submit --check` each.

Contract: nothing is submitted. Every run and canary command is wrapped in work/aa-load/ccx01.sh (CCX0+1 affinity, inherited by the
harness and its server); the candidate arm carries no env switch; canary patterns and log envs are those of the selected components
only (engagement.py). --stack10-root and --evidence exist for dry renders against a labelled stand-in, never for the real window.

usage: render_stack10_items.py --out DIR [--components "p55 k7 ..."] [--only ID]... [--stack10-root DIR] [--base-root DIR]
                               [--evidence FILE] [--snapshots DIR]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s10items
from s10items import COMPONENTS, CCX01, EVAL, ROOT, W

sys.path.insert(0, str(EVAL))
from harness import lib, mkitem, snapshot

NAMES = ('s10-depth-j5q', 's10-depth-iq4', 's10-np2-real-j5q', 's10-np2-real-iq4', 's10-duo-soak-j5q', 's10-duo-soak-iq4')
GPU_AB = ROOT / 'codex/stack10/gpu/gpu_ab.sh'


def snapshotter(snapshots):
    def local_snapshot(x):
        files = {f'harness/{p.name}': p.read_bytes() for p in [*mkitem.H.glob('*.py'), *mkitem.H.glob('*.sh')]}
        experiment = x['file'].read_bytes()
        assert hashlib.sha256(experiment).hexdigest() == x['_file_sha'], 'experiment changed during rendering'
        files['experiment.toml'] = experiment
        files['source-root.txt'] = (str(EVAL) + '\n').encode()
        return snapshot.create(files, snapshots, 'harness')
    return local_snapshot


def wrap_ccx(text):
    item = tomllib.loads(text)
    for stage in ('run', 'canary'):
        old = 'cmd = ' + json.dumps(item[stage]['cmd'])
        new = 'cmd = ' + json.dumps(['bash', str(CCX01), *item[stage]['cmd']])
        assert text.count(old) == 1, f'{stage} cmd not found once'
        text = text.replace(old, new)
    return text


def pin_siblings(text):
    """gpuq treats every .py/.sh beside the entry as its library: pin gpu_ab.sh's siblings too."""
    have = set(tomllib.loads(text)['pins']['files'])
    siblings = sorted(str(q) for q in GPU_AB.parent.iterdir() if q.suffix in ('.py', '.sh') and q.is_file()
                      and not q.name.startswith(('gen_', 'test_', '.gpuq-')) and str(q) not in have)
    if siblings:
        head, sep, tail = text.partition('[pins]\nfiles = [\n')
        assert sep, 'pins.files is not a multi-line array'
        text = head + sep + ''.join(f'  {json.dumps(q)},\n' for q in siblings) + tail
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--components', default=' '.join(COMPONENTS))
    parser.add_argument('--only', choices=NAMES, action='append')
    parser.add_argument('--stack10-root', type=Path, default=W / 'frozen/stack10')
    parser.add_argument('--base-root', type=Path, default=W / 'frozen/stack9-p55')
    parser.add_argument('--evidence', type=Path, default=ROOT / 'codex/stack10/mac-evidence/summary.json')
    parser.add_argument('--snapshots', type=Path, default=ROOT / 'codex/stack10/snapshots')
    args = parser.parse_args()
    components = args.components.split()
    assert set(components) <= set(COMPONENTS), f'unknown component in {components}'
    args.out.mkdir(parents=True, exist_ok=True)
    args.snapshots.mkdir(parents=True, exist_ok=True)
    sha = (args.stack10_root / 'SOURCE').read_text().splitlines()[0]
    base_sha = (args.base_root / 'SOURCE').read_text().splitlines()[0]
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    # the items run the frozen build; codex/ (harness glue, proofs) may move after the freeze
    changed = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--name-only', sha, head, '--', '.', ':!codex'], text=True).split()
    assert not changed, f'frozen/stack10 is {sha}, HEAD {head} differs outside codex/: {changed}'
    mkitem.harness_snapshot = snapshotter(args.snapshots)
    for name in args.only or NAMES:
        source = (ROOT / f'codex/stack10/items/{name}.toml.in').read_text()
        text = s10items.fill(source, sha=sha, base_sha=base_sha, components=components, tree=ROOT, evidence=args.evidence,
                             template_name=f'{name}.toml.in')
        text = text.replace(f'"{W}/frozen/stack10"', f'"{args.stack10_root}"').replace(f'"{W}/frozen/stack9-p55"', f'"{args.base_root}"')
        experiment = args.out / f'{name}.toml'
        experiment.write_text(text)
        x = lib.read_experiment(experiment)
        rendered = mkitem.render(x)
        rendered = mkitem.wrap_stages(rendered, GPU_AB)
        rendered = rendered.replace('kind = "measure"', 'host_sensitive = false\nkind = "measure"', 1)
        rendered = pin_siblings(wrap_ccx(rendered))
        parsed = tomllib.loads(rendered)
        assert parsed['canary']['timeout_s'] <= 600
        pins = parsed['pins']['files']
        assert str(CCX01) in pins
        assert str(args.stack10_root / 'build/bin/round-trace') in pins and str(args.base_root / 'build/bin/round-trace') in pins
        assert not any('build-bw-mac-results' in p for p in pins)
        path = args.out / f'{name}.gpuq.toml'
        path.write_text(rendered)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', GIT_NO_LAZY_FETCH='1', PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run(['gpuq', 'submit', '--check', str(path)], env=env, text=True, capture_output=True)
        log = result.stdout + result.stderr
        (args.out / f'{name}-contract.log').write_text(log)
        assert result.returncode == 0 and 'contract violated' not in log.lower(), log
        print(f'checked, NOT submitted: {path}')


if __name__ == '__main__':
    main()
