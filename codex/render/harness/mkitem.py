"""Render p55 items with the existing harness and worktree-local snapshots."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

from . import lib, snapshot

ROOT = Path(__file__).resolve().parents[3]
EVAL = Path('/home/xanewok/repos/qwen38-eval')
spec = importlib.util.spec_from_file_location('harness._mkitem', EVAL / 'harness/mkitem.py')
mkitem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mkitem)
create_snapshot = snapshot.create
upstream_render = mkitem.render


def local_snapshot(files, directory, prefix):
    if Path(directory) != EVAL / 'orchestration/gpuq/snapshots':
        raise lib.Fail('inputs changed: unexpected snapshot destination', 'integrity')
    return create_snapshot(files, ROOT / 'codex/snapshots', prefix)


def render(x):
    text = upstream_render(x)
    text = mkitem.wrap_stages(text, ROOT / 'codex/gpu-p55.sh')
    item = tomllib.loads(text)
    wrapper = '/home/xanewok/repos/qwen38-eval-worktrees/work/aa-load/ccx01.sh'
    assert wrapper in item['pins']['files']
    section = None
    lines = []
    for line in text.splitlines():
        if line.startswith('['):
            section = line.strip('[]')
        if section in ('run', 'canary') and line.startswith('cmd = '):
            line = 'cmd = ' + json.dumps(['bash', wrapper, *item[section]['cmd']])
        lines.append(line)
        if line.startswith('priority = '):
            lines.append('host_sensitive = false')
    return '\n'.join(lines) + '\n'


def main():
    template = Path(sys.argv[1]).resolve()
    assert template.parent == ROOT / 'codex/experiments' and template.name.endswith('.toml.in')
    frozen = ROOT.parent.parent / 'frozen/stack7-p55/SOURCE'
    sha = frozen.read_text().split()[0] if frozen.exists() else subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    experiment = template.with_suffix('')
    experiment.write_text(template.read_text().replace('@P55_SHA@', sha))
    sys.argv[1] = str(experiment)
    os.environ['HARNESS_CODE_ROOT'] = str(EVAL)
    sys.path.insert(0, str(ROOT / 'codex'))
    import gpu_stack7
    import p55_checks
    gpu_stack7.fixture_checks()
    p55_checks.fixture_checks()
    from . import depth
    record = json.loads((gpu_stack7.REAL / 'depth-8192.json').read_text())
    identity = lib.token_identity(record)
    results = {}
    depth.settle(results, 'within-arm', 'patch', 8192, identity)
    depth.settle(results, 'within-arm', 'up', 8192, identity)
    changed = (identity[0][:-1] + [identity[0][-1] + 1], *identity[1:])
    try:
        depth.settle(results, 'within-arm', 'patch', 8192, changed)
    except lib.Fail:
        pass
    else:
        raise AssertionError('within-arm token mutant escaped')
    print('within-arm recorded token negative control PASS')
    snapshot.create = local_snapshot
    mkitem.render = render
    if '--submit' in sys.argv or '--stale' in sys.argv:
        raise lib.Fail('inputs changed: this entry only renders or checks', 'integrity')
    return mkitem.main()


if __name__ == '__main__':
    sys.exit(main())
