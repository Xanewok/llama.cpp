"""Render and contract-check comparison items after the vanilla freeze; never submit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
W = ROOT.parent.parent
EVAL = Path('/home/xanewok/repos/qwen38-eval')
CCX = W / 'work/aa-load/ccx01.sh'
sys.path.insert(0, str(EVAL))
from harness import lib, mkitem, snapshot


def snapshotter(directory):
    def save(x):
        files = {f'harness/{p.name}': p.read_bytes() for p in [*mkitem.H.glob('*.py'), *mkitem.H.glob('*.sh')]}
        data = x['file'].read_bytes()
        assert hashlib.sha256(data).hexdigest() == x['_file_sha']
        files['experiment.toml'] = data
        files['source-root.txt'] = (str(EVAL) + '\n').encode()
        return snapshot.create(files, directory, 'harness')
    return save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--vanilla-root', type=Path, default=W / 'frozen/vanilla-08246a28f')
    parser.add_argument('--dry', action='store_true', help='labelled stand-in only; never usable for GPU measurement')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sha = (args.vanilla_root / 'SOURCE').read_text().splitlines()[0]
    for ref in (sha, 'HEAD'):
        changed = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--name-only', '08246a28f', ref, '--', '.', ':!codex'], text=True).split()
        assert not changed, changed
    if args.dry:
        assert (args.vanilla_root / 'DRY-STAND-IN').is_file()
    else:
        assert args.vanilla_root == W / 'frozen/vanilla-08246a28f'
        assert not (args.vanilla_root / 'DRY-STAND-IN').exists()
    mkitem.harness_snapshot = snapshotter(args.out / 'snapshots')
    for template in sorted((ROOT / 'codex/items').glob('*.toml.in')):
        text = template.read_text().replace('@VANILLA_ROOT@', str(args.vanilla_root)).replace('@VANILLA_SHA@', sha)
        for name in ('stack9-p55', 'stack10'):
            text = text.replace('@' + name.upper().replace('-', '_') + '_SHA@', (W / f'frozen/{name}/SOURCE').read_text().splitlines()[0])
        path = args.out / template.name.removesuffix('.in')
        if args.dry:
            text = text.replace('id = "vanilla-', 'id = "dry-vanilla-', 1)
            text = text.replace('title = "', 'title = "DRY STAND-IN, DO NOT SUBMIT: ', 1)
        path.write_text(text)
        x = lib.read_experiment(path)
        rendered = mkitem.wrap_stages(mkitem.render(x), ROOT / 'codex/gpu_ab.sh')
        item = tomllib.loads(rendered)
        for stage in ('run', 'canary'):
            before = 'cmd = ' + json.dumps(item[stage]['cmd'])
            rendered = rendered.replace(before, 'cmd = ' + json.dumps(['bash', str(CCX), *item[stage]['cmd']]), 1)
        rendered = rendered.replace('kind = "measure"', 'host_sensitive = true\nkind = "measure"', 1)
        item = tomllib.loads(rendered)
        assert item['canary']['timeout_s'] <= 600
        assert len(x['plan'])*mkitem.boot_wall(x) + x['budget']['canary_s'] <= 0.7*item['budget_s']
        dest = args.out / (path.stem + '.gpuq.toml')
        dest.write_text(rendered)
        proc = subprocess.run(['gpuq', 'submit', '--check', str(dest)], capture_output=True, text=True)
        (args.out / f'{path.stem}-contract.log').write_text(proc.stdout + proc.stderr)
        assert proc.returncode == 0 and 'contract violated' not in (proc.stdout + proc.stderr).lower(), proc.stdout + proc.stderr
        print(f'checked, NOT submitted: {dest}; est_s_G={item["est_s"]}; budget_s_D={item["budget_s"]}')


if __name__ == '__main__':
    main()
