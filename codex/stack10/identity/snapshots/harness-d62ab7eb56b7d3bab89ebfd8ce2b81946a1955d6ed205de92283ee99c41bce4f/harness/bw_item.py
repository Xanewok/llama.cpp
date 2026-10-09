"""Render a build-window filler item; submit it separately after preflight passes."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from zoneinfo import ZoneInfo

from . import lib, mkitem, snapshot

H = Path(__file__).resolve().parent


def git(worktree, *args):
    return subprocess.run(['git', '-C', str(worktree), *args], check=True, capture_output=True, text=True,
                          env=dict(os.environ, GIT_NO_LAZY_FETCH='1')).stdout.strip()


def render(payload, *, priority, out_root, budget_s=14400, est_s=5400, ok_rcs='0'):
    payload, out_root = Path(payload).resolve(), Path(out_root).resolve()
    worktree = Path(git(payload.parent, 'rev-parse', '--show-toplevel'))
    if payload.parent != worktree / 'codex' or payload.suffix != '.sh':
        raise lib.Fail('inputs changed: payload must be <worktree>/codex/*.sh', 'integrity')
    if out_root.is_relative_to(worktree):
        raise lib.Fail('inputs changed: out-root must be outside the payload worktree', 'integrity')
    name = payload.stem.removeprefix('bw-')
    if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,60}', name):
        raise lib.Fail('inputs changed: invalid build-window name', 'integrity')
    if not re.fullmatch(r'\d+(,\d+)*', ok_rcs) or budget_s < est_s + 60 or est_s <= 0:
        raise lib.Fail('inputs changed: invalid OK_RCS or budget (allow 60 seconds cleanup)', 'integrity')
    files = {'bw-run.sh': (H / 'bw-run.sh').read_bytes(), 'HEAD': (git(worktree, 'rev-parse', 'HEAD') + '\n').encode()}
    sources = sorted([*payload.parent.glob('*.sh'), *payload.parent.glob('*.py')])
    for path in sources:
        files[f'codex/{path.name}'] = path.read_bytes()
    files['SOURCES.sha256'] = ''.join(f'{lib.file_sha(path)}  codex/{path.name}\n' for path in sources).encode()
    code = snapshot.create(files, out_root, 'bw-code')
    entry = ['/usr/bin/bash', str(code / 'bw-run.sh'), str(code / 'codex' / payload.name), str(worktree), str(code / 'HEAD')]
    output = out_root / f'{name}-a{{attempt}}'
    env = {'PATH': f'{Path.home()}/.local/bin:/usr/local/bin:/usr/bin:/bin', 'OK_RCS': ok_rcs,
           'BUDGET_S': str(budget_s - 60), 'BW_OUT_ROOT': str(out_root), 'BW_NAME': name}
    j = json.dumps
    codes = '{ ' + ', '.join(f'"{rc}" = {j(cls)}' for rc, cls in lib.EXIT.items()) + ' }'
    env_toml = '{ ' + ', '.join(f'{k} = {j(v)}' for k, v in env.items()) + ' }'
    text = '\n'.join([
        f'id = {j("bw-" + name)}', f'title = {j("Build window: " + name + " (CPU only)")}', f'priority = {priority}',
        'kind = "filler"', 'ab = false', 'retry = "never"', f'budget_s = {budget_s}', f'est_s = {est_s}',
        f'exit_codes = {codes}', 'signatures = [{ class = "deterministic", regex = "bw-run: FAIL" }]',
        '[run]', f'cmd = {j(entry + [str(output)])}', f'cwd = {j(str(worktree))}', f'env = {env_toml}', 'ports = []',
        '[preflight]', f'cmd = {j(entry + ["--preflight"])}', 'timeout_s = 60',
        '[outputs]', f'paths = {j([str(output)])}', '[fix]', 'editable = []', 'protected = ["budget", "HEAD"]',
        '[pins]', f'files = {j([str(code / name) for name in sorted(files)])}', '',
    ])
    leaked = mkitem.scratch_paths(text)
    if leaked:
        raise lib.Fail(f'inputs changed: scratchpad paths in the item: {leaked[:3]}', 'integrity')
    return out_root / f'bw-{name}.toml', text


def main(argv=None):
    parser = argparse.ArgumentParser(prog='bw-item', description=__doc__)
    parser.add_argument('payload', type=Path, help='committed build-window payload in <worktree>/codex')
    parser.add_argument('--priority', type=int, required=True, help='explicit queue priority')
    parser.add_argument('--out-root', type=Path, default=Path.home() / 'repos/qwen38-eval-worktrees/work' /
                        datetime.now(ZoneInfo('Europe/Warsaw')).strftime('build-window-%m%d'), help='external window root (default work/build-window-MMDD)')
    parser.add_argument('--budget-s', type=int, default=14400, help='whole-window budget in seconds, including cleanup')
    parser.add_argument('--est-s', type=int, default=5400, help='estimated payload duration in seconds')
    parser.add_argument('--ok-rcs', default='0', help='comma-separated payload return codes normalized to success')
    args = parser.parse_args(argv)
    try:
        item, text = render(args.payload, priority=args.priority, out_root=args.out_root,
                            budget_s=args.budget_s, est_s=args.est_s, ok_rcs=args.ok_rcs)
        manifest = tomllib.loads(text)
        checked = subprocess.run(manifest['preflight']['cmd'], env=dict(os.environ, **manifest['run']['env']), timeout=60)
        if checked.returncode:
            return 10
        item.write_text(text)
        print(f'wrote {item}')
        return 0
    except lib.Fail as error:
        print(error, file=sys.stderr)
        return error.rc
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f'inputs changed: {error}', file=sys.stderr)
        return 6


if __name__ == '__main__':
    sys.exit(main())
