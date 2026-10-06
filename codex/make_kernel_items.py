"""Render serverless gpuq manifests; harness.mkitem probe requires a server."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
UP = Path('/home/xanewok/repos/qwen38-eval-worktrees/frozen/stack6')
PATCH = HERE / 'frozen/mmvq-np2'
EVAL = Path('/home/xanewok/repos/qwen38-eval/harness')


def render(kind):
    j = json.dumps
    run_budget = 1200 if kind == 'memcheck' else 1800
    canary_budget = 300
    entry = ['python3', str(HERE / 'kernel_item.py'), kind]
    out = HERE / 'gpu-results' / kind
    files = set(HERE.glob('*.py')) | set(HERE.glob('*.sh'))
    for path in (HERE / 'fixtures', HERE / 'cases', UP, PATCH):
        files.update(p for p in path.rglob('*') if p.is_file() or p.is_symlink())
    files.update(EVAL.glob('*.py'))
    files.add(HERE / 'REPORT-mmvq-np2.md')
    text = f'''id = "mmvq-np2-{kind}"
title = "Wide MMVQ {kind}: five types, TP2 projection shapes; memcheck must pass before perf"
kind = "measure"
ab = {'true' if kind == 'perf' else 'false'}
retry = "fresh"
budget_s = {run_budget + canary_budget + 150}
est_s = {700 if kind == 'memcheck' else 1300}
startup_grace_s = 120
exit_codes = {{ "3" = "hazard", "4" = "transient", "5" = "insufficient_data", "6" = "integrity", "7" = "timeout", "8" = "deterministic" }}
signatures = [
  {{ class = "integrity", regex = "probe: integrity" }},
  {{ class = "timeout", regex = "probe: budget" }}
]

[run]
cmd = {j(entry + ['--stage', 'run', '--out', str(out / 'full-a{attempt}'), '--budget', str(run_budget)])}
cwd = {j(str(HERE))}
env = {{ GPUQ_ACTIVE = "1", PYTHONDONTWRITEBYTECODE = "1" }}
ports = []

[preflight]
cmd = {j(entry + ['--stage', 'preflight'])}
timeout_s = 120

[canary]
cmd = {j(entry + ['--stage', 'canary', '--out', str(out / 'canary-a{attempt}'), '--budget', '300'])}
env = {{ GPUQ_ACTIVE = "1", PYTHONDONTWRITEBYTECODE = "1" }}
timeout_s = 300

[outputs]
paths = {j([str(out / 'full-a{attempt}')])}

[fix]
editable = []
protected = ["frozen", "budget", "coverage", "correctness", "memcheck_first"]

[pins]
files = [
''' + ''.join('  ' + j(str(path)) + ',\n' for path in sorted(files)) + ']\n'
    (HERE / 'items' / f'{kind}.gpuq.toml').write_text(text)

if __name__ == '__main__':
    render('memcheck')
    render('perf')
