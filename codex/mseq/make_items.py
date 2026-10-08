"""Render and contract-check the fold items inside this worktree. Never submit.

Usage: make_items.py COMMIT CPU_SUMMARY OUT [--dry]
--dry renders with frozen/stack7 in the patch arm to check the contract before a CUDA freeze exists.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
W = Path('/home/xanewok/repos/qwen38-eval-worktrees')
EVAL = Path('/home/xanewok/repos/qwen38-eval')
CCX = W / 'work/aa-load/ccx01.sh'
sys.path.insert(0, str(EVAL))
from harness import lib, mkitem, snapshot

commit, summary, output = sys.argv[1:4]
output = Path(output).resolve()
output.mkdir(parents=True, exist_ok=True)
assert output.is_relative_to(ROOT), 'Outputs must stay in this worktree'
dry = '--dry' in sys.argv[4:]
summary = Path(summary).resolve()
assert summary.is_file(), summary
if not dry:
    assert json.loads(summary.read_text())['rc'] == 0, 'Mac gates must pass before rendering real items'
    assert (W / 'frozen/fold-mseq/SOURCE').read_text().split()[0] == commit


def local_snapshot(x):
    files = {f'harness/{p.name}': p.read_bytes() for p in [*mkitem.H.glob('*.py'), *mkitem.H.glob('*.sh')]}
    files['experiment.toml'] = x['file'].read_bytes()
    assert hashlib.sha256(files['experiment.toml']).hexdigest() == x['_file_sha']
    files['source-root.txt'] = (str(EVAL) + '\n').encode()
    return snapshot.create(files, output / 'snapshots', 'harness')


mkitem.harness_snapshot = local_snapshot
for np in (1, 2):
    name = f'fold-mseq-np{np}-' + ('real' if np == 2 else 'depth') + '-j5q'
    text = (ROOT / f'codex/mseq/items/{name}.toml.in').read_text()
    text = text.replace('@COMMIT@', commit).replace('@CPU_SUMMARY@', str(summary)).replace('@DATE@', datetime.date.today().isoformat())
    assert not re.search(r'@[A-Z_]+@', text)
    if dry:
        text = text.replace('/frozen/fold-mseq', '/frozen/stack7')
        text = text.replace(commit, '48b82b0d1559e936dbf612c871f2583cd4c79f3e')
    exp = output / f'{name}.experiment.toml'
    exp.write_text(text)
    rendered = mkitem.wrap_stages(mkitem.render(lib.read_experiment(exp)), ROOT / 'codex/gpu_ab.sh')
    parsed = tomllib.loads(rendered)
    lines = []
    section = None
    for line in rendered.splitlines():
        if line.startswith('['):
            section = line.strip('[]')
        if section in ('run', 'canary') and line.startswith('cmd = '):
            line = 'cmd = ' + json.dumps(['bash', str(CCX), *parsed[section]['cmd']])
        lines.append(line)
        if line.startswith('priority = '):
            lines.append('host_sensitive = false')
        if line == 'files = [':
            lines += ['  ' + json.dumps(str(f)) + ',' for f in (CCX, W / 'work/build-window-1007/mmq.sh')]
    item = output / f'{name}-ccx01.toml'
    item.write_text('\n'.join(lines) + '\n')
    r = subprocess.run(['gpuq', 'submit', '--check', str(item)], capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES='', GIT_NO_LAZY_FETCH='1', PYTHONDONTWRITEBYTECODE='1'))
    (output / f'{name}-contract.log').write_text(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(r.returncode)
    print('checked, NOT submitted:', item)
