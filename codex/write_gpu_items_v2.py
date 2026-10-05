"""Render the shared depth contract, adding the local engagement driver as its entry point."""
import json
from pathlib import Path
import tomllib

from harness import lib, mkitem

root = Path(__file__).resolve().parent
freeze = root / 'frozen/gdn-conv-v2-proof'
sha = (freeze / 'SOURCE').read_text().splitlines()[0]
experiment = root / 'gdn-conv-v2.experiment.toml'
text = experiment.read_text()
old = tomllib.loads(text)['arms']['patch']['commit']
text = text.replace(old, sha).replace(f'patch {old[:9]}', f'patch {sha[:9]}')
experiment.write_text(text)
x = lib.read_experiment(experiment)
rendered = mkitem.render(x)
manifest = tomllib.loads(rendered)
for stage in ('run', 'preflight', 'canary'):
    old_command = manifest[stage]['cmd']
    command = ['bash', str(root / 'gpu_ab.sh'), *old_command[4:]]
    rendered = rendered.replace('cmd = ' + json.dumps(old_command), 'cmd = ' + json.dumps(command))
(root / 'gdn-conv-v2.item.toml').write_text(rendered)

pins = [str(p) for p in mkitem.tree(freeze)]
pins += [str(root / name) for name in ('gpu_bench.sh', 'gdn_conv_gpu.py', 'test_gpu_parsers_v2.py',
                                      'gpu-parser-tests-v2.json', 'conv_state_bench.cpp',
                                      'bench-layout-v2.txt', 'bench-layout-negative.status')]
pins += [str(p) for p in Path('/home/xanewok/repos/qwen38-eval/harness').glob('*.py')]
pins += ['/home/xanewok/repos/qwen38-eval-worktrees/work/gdn-movers-diag/runs/bench-a2/gpu0-p1.txt',
         '/home/xanewok/repos/qwen38-eval-worktrees/work/gdn-movers-diag/runs/bench-a2/gpu0-p1.err']
bench = [
    'id = "gdn-conv-v2-kernel-bench"',
    'title = "V2 concat and snapshot fusion: real model graph order, gallocr, exact caches"',
    'kind = "measure"', 'ab = false', 'retry = "fresh"',
    '# G: full <= 450 s + canary <= 120 s + cleanup <= 60 s; 630 / 900 = 0.70.',
    'budget_s = 900', 'est_s = 450', 'startup_grace_s = 30',
    'exit_codes = { ' + ', '.join(f'"{code}" = {json.dumps(cls)}' for code, cls in lib.EXIT.items()) + ' }',
    'signatures = [' + ', '.join('{ class = ' + json.dumps(cls) + ', regex = ' + json.dumps(regex) + ' }' for cls, regex in lib.SIGNATURES) + ']',
]
for stage, mode in (('run', '--execute'), ('canary', '--canary'), ('preflight', '--preflight')):
    command = ['bash', str(root / 'gpu_bench.sh'), mode, '--out', str(root / 'bench-runs' / (stage + '-a{attempt}'))]
    bench += ['', '[' + stage + ']', 'cmd = ' + json.dumps(command)]
    if stage == 'run':
        bench += ['cwd = ' + json.dumps(str(root.parent))]
    if stage != 'preflight':
        bench += ['env = { GPUQ_ACTIVE = "1" }']
    if stage != 'run':
        bench += ['timeout_s = ' + ('180' if stage == 'canary' else '120')]
bench += ['', '[outputs]', 'paths = [' + json.dumps(str(root / 'bench-runs/run-a{attempt}')) + ']',
          '', '[fix]', 'editable = []', 'protected = ["frozen", "identity", "engagement", "budget"]',
          '', '[pins]', 'files = ' + json.dumps(pins), '']
(root / 'gdn-conv-v2-bench.item.toml').write_text('\n'.join(bench))
