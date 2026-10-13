"""Render the stack11-s2 A/B items and `gpuq submit --check` each (never submits).

  s2-depth-{j5q,iq4}       np1 cross-arm token identity at 8k/32k + timing: frozen/stack11-base (s11) vs frozen/stack11-s2 (s2)
  s2-np2-real-{j5q,iq4}    np2 real prompts, 8k/32k/90k: s11 vs s2
  s2-duo-soak-iq4          duo serving soak, two Codex agents: s11 vs s2
  s2-drop-np2-real-j5q     frozen/stack11-s2-nok (K7 and K10 removed from the source) vs s2 (priced, not decided)

Contract: every run/canary/analysis command is wrapped in work/aa-load/ccx01.sh (CCX0+1 affinity, inherited by the harness and the
server); host_sensitive = false. --dry renders against a labelled stand-in (--s2-root, --nok-root, --evidence) and writes a DRY banner
into every item: those items are contract checks only, never submitted.

usage: render_items.py --out DIR [--components "p55 k7 ..."] [--only ID]... [--dry --s2-root DIR --nok-root DIR --evidence FILE]
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
import s2items
from s2items import CCX01, COMPONENTS, CROOT, EVAL, HERE, ROOT, W

sys.path.insert(0, str(EVAL))
from harness import lib, mkitem, snapshot  # noqa: E402

AB = ('s2-depth-j5q', 's2-depth-iq4', 's2-np2-real-j5q', 's2-np2-real-iq4', 's2-duo-soak-iq4', 's2-drop-np2-real-j5q')


def snapshotter(snapshots):
    def local_snapshot(x):
        files = {f'harness/{p.name}': p.read_bytes() for p in [*sorted(mkitem.H.glob('*.py')), *sorted(mkitem.H.glob('*.sh'))]}
        experiment = x['file'].read_bytes()
        assert hashlib.sha256(experiment).hexdigest() == x['_file_sha'], 'experiment changed during rendering'
        files['experiment.toml'] = experiment
        files['source-root.txt'] = (str(EVAL) + '\n').encode()
        return snapshot.create(files, snapshots, 'harness')
    return local_snapshot


def wrap_ccx(text, stages):
    item = tomllib.loads(text)
    for stage in stages:
        old = 'cmd = ' + json.dumps(item[stage]['cmd'])
        new = 'cmd = ' + json.dumps(['bash', str(CCX01), *item[stage]['cmd']])
        assert text.count(old) == 1, f'{stage} cmd not found once'
        text = text.replace(old, new)
    return text


def check(path, out):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', GIT_NO_LAZY_FETCH='1', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run(['gpuq', 'submit', '--check', str(path)], env=env, text=True, capture_output=True)
    log = result.stdout + result.stderr
    (out / f'{path.name.removesuffix(".gpuq.toml")}-contract.log').write_text(f'$ gpuq submit --check {path}\nrc={result.returncode}\n{log}')
    assert result.returncode == 0 and 'contract violated' not in log.lower(), log
    print(f'checked, NOT submitted: {path}')


def render_ab(name, args, s2_sha, nok_sha, base_sha, banner):
    source = (HERE / f'items/{name}.toml.in').read_text()
    text = s2items.fill(source, s2_sha=s2_sha, nok_sha=nok_sha, base_sha=base_sha, components=args.components.split(),
                        evidence=args.evidence, template_name=f'{name}.toml.in')
    text = text.replace(f'"{W}/frozen/stack11-s2"', f'"{args.s2_root}"').replace(f'"{W}/frozen/stack11-s2-nok"', f'"{args.nok_root}"')
    experiment = args.out / f'{name}.toml'
    experiment.write_text(banner + text)
    x = lib.read_experiment(experiment)
    rendered = mkitem.render(x)
    rendered = rendered.replace('kind = "measure"', 'host_sensitive = false\nkind = "measure"', 1)
    rendered = wrap_ccx(rendered, ('run', 'canary'))
    rendered = rendered.replace('\nfiles = [\n', f'\nfiles = [\n  {json.dumps(str(CCX01))},\n', 1) if str(CCX01) not in rendered else rendered
    parsed = tomllib.loads(rendered)
    assert parsed['canary']['timeout_s'] <= 600
    pins = parsed['pins']['files']
    assert str(CCX01) in pins and str(args.s2_root / 'build/bin/round-trace') in pins
    other = args.nok_root if name == 's2-drop-np2-real-j5q' else W / 'frozen/stack11-base'
    assert str(other / 'build/bin/round-trace') in pins
    path = args.out / f'{name}.gpuq.toml'
    path.write_text(banner + rendered)
    check(path, args.out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--components', default=' '.join(COMPONENTS))
    parser.add_argument('--only', choices=AB, action='append')
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--s2-root', type=Path, default=W / 'frozen/stack11-s2')
    parser.add_argument('--nok-root', type=Path, default=W / 'frozen/stack11-s2-nok')
    parser.add_argument('--evidence', type=Path, default=ROOT / 'codex/stack11-s2/mac-evidence/with-k.tsv')
    parser.add_argument('--snapshots', type=Path, default=HERE / 'snapshots')
    args = parser.parse_args()
    for key in ('out', 's2_root', 'nok_root', 'evidence', 'snapshots'):
        setattr(args, key, getattr(args, key).absolute())
    assert set(args.components.split()) <= set(COMPONENTS), f'unknown component in {args.components}'
    defaults = (args.s2_root == W / 'frozen/stack11-s2', args.nok_root == W / 'frozen/stack11-s2-nok')
    assert args.dry != all(defaults), '--dry needs the stand-ins (--s2-root, --nok-root); a real render takes none'
    args.out.mkdir(parents=True, exist_ok=True)
    args.snapshots.mkdir(parents=True, exist_ok=True)
    s2_sha = (args.s2_root / 'SOURCE').read_text().splitlines()[0]
    nok_sha = (args.nok_root / 'SOURCE').read_text().splitlines()[0]
    base_sha = (W / 'frozen/stack11-base/SOURCE').read_text().splitlines()[0]
    banner = ''
    if args.dry:
        banner = (f'# DRY RENDER, NOT AN ITEM TO SUBMIT: frozen/stack11-s2 and frozen/stack11-s2-nok do not exist yet; the stand-ins are '
                  f'{args.s2_root} (commit {s2_sha[:9]}) and {args.nok_root} (commit {nok_sha[:9]}), Mac evidence {args.evidence}.\n')
    else:
        for tree, sha, label in ((CROOT, s2_sha, 'frozen/stack11-s2'), (ROOT, nok_sha, 'frozen/stack11-s2-nok')):
            head = subprocess.check_output(['git', '-C', str(tree), 'rev-parse', 'HEAD'], text=True).strip()
            changed = subprocess.check_output(['git', '-C', str(tree), 'diff', '--name-only', sha, head, '--', '.', ':!codex'], text=True).split()
            assert not changed, f'{label} is {sha}, {tree} HEAD {head} differs outside codex/: {changed}'
    mkitem.harness_snapshot = snapshotter(args.snapshots)
    for name in args.only or AB:
        render_ab(name, args, s2_sha, nok_sha, base_sha, banner)


if __name__ == '__main__':
    main()
