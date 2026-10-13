"""Render the stack11-base items and `gpuq submit --check` each (never submits).

  A/B (parent harness, kind depth/campaign): s11-{depth,np2-real,duo-soak}-{j5q,iq4}: frozen/stack10 (s10) vs frozen/stack11-base (s11);
       s11-drop-np2-real-j5q: s11 vs s11 with K7 and K10 switched off.
  KLD (kld/runner.py): kld-s11-j5q, kld-s11-iq4: paired full-vocab KL of s11 against s10 (one model per runner config).

Contract: every run/canary/analysis command is wrapped in work/aa-load/ccx01.sh (CCX0+1 affinity, inherited by the harness and the
server); host_sensitive = false. --dry renders against a labelled stand-in (--s11-root, --evidence, --kld-gen) and writes a DRY banner
into every item: those items are contract checks only, never submitted.

usage: render_items.py --out DIR [--components "p55 k7 ..."] [--only ID]... [--dry --s11-root DIR --evidence FILE --kld-gen DIR]
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
import s11items
from s11items import CCX01, COMPONENTS, EVAL, HERE, ROOT, W

sys.path.insert(0, str(EVAL))
from harness import lib, mkitem, snapshot  # noqa: E402

AB = ('s11-depth-j5q', 's11-depth-iq4', 's11-np2-real-j5q', 's11-np2-real-iq4', 's11-duo-soak-j5q', 's11-duo-soak-iq4', 's11-drop-np2-real-j5q')
KLD = ('kld-s11-j5q', 'kld-s11-iq4')
KLD_DIR = HERE / 'kld'
KLD_SCRATCH = '/mnt/scratch/work/kld-s11'
UUIDS = 'GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f'
KLD_NOTE = ("why: paired full-vocab KLD of frozen/stack11-base vs frozen/stack10, width-6 verify, tails at ~32k and ~64k | expect: assistant-tail "
            "mean KL <= 6e-4, all-tail <= 0.06 (D: gdn4 alone 3.5e-4/5.6e-4 and 0.044/0.058 vs stack9) | falsifier: twice that, or > 10x the "
            "repeat spread | answers: does the master rebase add numerics beyond gdn4's; token identity is not expected")


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


def render_ab(name, args, s11_sha, base_sha, banner):
    source = (HERE / f'items/{name}.toml.in').read_text()
    text = s11items.fill(source, s11_sha=s11_sha, base_sha=base_sha, components=args.components.split(), evidence=args.evidence,
                         template_name=f'{name}.toml.in')
    text = text.replace(f'"{W}/frozen/stack11-base"', f'"{args.s11_root}"')
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
    assert str(CCX01) in pins and str(args.s11_root / 'build/bin/round-trace') in pins
    assert name.startswith('s11-drop') or str(W / 'frozen/stack10/build/bin/round-trace') in pins
    path = args.out / f'{name}.gpuq.toml'
    path.write_text(banner + rendered)
    check(path, args.out)


def build_pins(root):
    b = Path(root) / 'build/bin'
    return [str(Path(root) / 'SOURCE'), str(b / 'SHA256SUMS'), *sorted(str(p) for p in b.iterdir() if p.name.startswith('lib') and '.so' in p.name)]


def render_kld(name, args, banner):
    model = name.rsplit('-', 1)[1]
    config_path = args.kld_gen / f'config-{model}.json'
    config = json.loads(config_path.read_text())
    assert Path(config['builds']['cand']['root']) == args.s11_root and Path(config['builds']['base']['root']) == W / 'frozen/stack10'
    run = f"{config['scratch']}/{{id}}/a{{attempt}}"
    entry = lambda stage, sub, *rest: ['bash', str(CCX01), 'bash', str(KLD_DIR / 'run.sh'), stage, '--config', str(config_path),
                                       '--out', f'{run}/{sub}', *rest]
    pins = [str(CCX01), *sorted(str(p) for p in KLD_DIR.iterdir() if p.suffix in ('.py', '.sh')),
            str(KLD_DIR / 'src/kld_dump.cpp'), str(KLD_DIR / 'src/batch_compat.h'), str(config_path),
            config['builds']['base']['binary'], config['builds']['cand']['binary'], config['model'],
            *config['sha256'], *build_pins(W / 'frozen/stack10'), *build_pins(args.s11_root)]
    pins = list(dict.fromkeys(pins))
    q = json.dumps
    assert len(KLD_NOTE) <= 600, len(KLD_NOTE)
    text = f"""{banner}# Rendered by codex/stack11/window/render_items.py; config and kld_dump binaries from bw-stack11-base stage 71 (kld/build_dump.sh).
# Dumps take ~10.3 GB of /mnt/scratch per item plus a 5 GiB margin (preflight checks): remove one item's .klf dumps before the other runs
# if /mnt/scratch is short.
id = "{name}"
title = "Paired {model.upper()} KLD at six-column verify width: frozen/stack10 vs frozen/stack11-base, tails at ~32k and ~64k depth"
priority = 28
host_sensitive = false
kind = "measure"
ab = true
retry = "fresh"
# canary 300 + run --budget-s 4400 + analysis headroom; est_s: G (d32k 4 arms x ~215 s, d64k 5 arms x ~430 s, kld-s9-gdn4 measured)
budget_s = 5200
est_s = 3300
note = {q(KLD_NOTE)}
exit_codes = {{ "3" = "hazard", "4" = "transient", "5" = "insufficient_data", "6" = "integrity", "7" = "timeout", "8" = "deterministic" }}

[run]
cmd = {q(entry('run', 'run', '--budget-s', '4400', '--arm-timeout-s', '800'))}
cwd = {q(str(KLD_DIR))}
ports = []

[run.env]
CUDA_HOME = "/usr/local/cuda-13.2"
CUDA_VISIBLE_DEVICES = "{UUIDS}"

[preflight]
cmd = {q(['bash', str(KLD_DIR / 'preflight.sh'), '--config', str(config_path), '--out', f"{config['scratch']}/{name}/preflight"])}
timeout_s = 900

[canary]
cmd = {q(entry('canary', 'canary', '--budget-s', '300'))}
timeout_s = 300

[analysis]
cmd = {q(entry('analysis', 'run'))}
timeout_s = 1800

[outputs]
paths = [{q(run + '/run')}, {q(run + '/canary')}]

[fix]
editable = []

[pins]
files = [
{os.linesep.join('  ' + q(p) + ',' for p in pins)}
]
"""
    path = args.out / f'{name}.gpuq.toml'
    path.write_text(text)
    check(path, args.out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--components', default=' '.join(COMPONENTS))
    parser.add_argument('--only', choices=AB + KLD, action='append')
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--s11-root', type=Path, default=W / 'frozen/stack11-base')
    parser.add_argument('--evidence', type=Path, default=ROOT / 'codex/stack11/mac-evidence/summary.json')
    parser.add_argument('--snapshots', type=Path, default=HERE / 'snapshots')
    parser.add_argument('--kld-gen', type=Path, default=HERE / 'kld/gen')
    args = parser.parse_args()
    for key in ('out', 's11_root', 'evidence', 'snapshots', 'kld_gen'):
        setattr(args, key, getattr(args, key).absolute())
    assert set(args.components.split()) <= set(COMPONENTS), f'unknown component in {args.components}'
    defaults = (args.s11_root == W / 'frozen/stack11-base', args.kld_gen == HERE / 'kld/gen')
    assert args.dry != all(defaults), '--dry needs the stand-ins (--s11-root, --kld-gen); a real render takes none'
    args.out.mkdir(parents=True, exist_ok=True)
    args.snapshots.mkdir(parents=True, exist_ok=True)
    s11_sha = (args.s11_root / 'SOURCE').read_text().splitlines()[0]
    base_sha = (W / 'frozen/stack10/SOURCE').read_text().splitlines()[0]
    banner = ''
    if args.dry:
        banner = (f'# DRY RENDER, NOT AN ITEM TO SUBMIT: frozen/stack11-base does not exist yet; its stand-in is {args.s11_root} '
                  f'(commit {s11_sha[:9]}), Mac evidence stand-in {args.evidence}, KLD stand-in configs {args.kld_gen}.\n')
    else:
        head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        changed = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--name-only', s11_sha, head, '--', '.', ':!codex'], text=True).split()
        assert not changed, f'frozen/stack11-base is {s11_sha}, HEAD {head} differs outside codex/: {changed}'
    mkitem.harness_snapshot = snapshotter(args.snapshots)
    for name in args.only or AB + KLD:
        if name in AB:
            render_ab(name, args, s11_sha, base_sha, banner)
        else:
            render_kld(name, args, banner)


if __name__ == '__main__':
    main()
