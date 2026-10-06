#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = Path('/home/xanewok/repos/qwen38-eval')
experiment = HERE / 'depth-experiment.toml'
item = HERE / 'depth-item.toml'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', GIT_NO_LAZY_FETCH='1', PYTHONDONTWRITEBYTECODE='1')
subprocess.run([sys.executable, '-m', 'harness.mkitem', str(experiment), '--out', str(item)], cwd=EVAL, env=env, check=True)
text = item.read_text()
parsed = tomllib.loads(text)
for stage in ['run', 'preflight', 'canary']:
    argv = parsed[stage]['cmd']
    assert argv[:4] == ['bash', str(EVAL / 'harness/run.sh'), 'depth', str(experiment)]
    text = text.replace('cmd = ' + json.dumps(argv), 'cmd = ' + json.dumps(['bash', str(HERE / 'gpu_ab.sh'), *argv[4:]]))
# Keep mkitem's budgets, pins and gates; the wrapper adds offline git reads and a one-boot identity control.
item.write_text(text)
