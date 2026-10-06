"""Compile the MMVQ translation unit with ptxas resource diagnostics; no device execution."""
from pathlib import Path
import os
import shlex
import subprocess

here = Path(__file__).resolve().parent
root = here.parent
cuda = root / 'build-cuda/ggml/src/ggml-cuda'
flags = (cuda / 'CMakeFiles/ggml-cuda.dir/flags.make').read_text()
values = dict(line.split(' = ', 1) for line in flags.splitlines() if ' = ' in line)
command = ['/usr/local/cuda-13.2/bin/nvcc', '-forward-unknown-to-host-compiler']
for key in ('CUDA_DEFINES', 'CUDA_INCLUDES', 'CUDA_FLAGS'):
    command += shlex.split(values[key])
command += ['-Xptxas=-v', '-c', str(root / 'ggml/src/ggml-cuda/mmvq.cu'), '-o', str(here / 'mmvq-resources.o')]
with (here / 'registers.log').open('w') as log:
    log.write(shlex.join(command) + '\n')
    log.flush()
    result = subprocess.run(command, cwd=cuda, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=log)
raise SystemExit(result.returncode)
