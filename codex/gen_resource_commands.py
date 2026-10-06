"""Print buildgate commands for baseline MMQ resource diagnostics; never execute device code."""
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
cuda = ROOT / 'build-cuda/ggml/src/ggml-cuda'
flags = (cuda / 'CMakeFiles/ggml-cuda.dir/flags.make').read_text()
values = dict(line.split(' = ', 1) for line in flags.splitlines() if ' = ' in line)
base = ['/usr/local/cuda-13.2/bin/nvcc', '-forward-unknown-to-host-compiler']
for key in ('CUDA_DEFINES', 'CUDA_INCLUDES', 'CUDA_FLAGS'):
    base += shlex.split(values[key])
for quant in ('iq4_xs', 'q6_k', 'q5_k', 'q8_0', 'q4_k'):
    for mode in ('base', 'patch'):
        command = base + (['-DGGML_CUDA_MMQ_SMALLJ_DISABLE'] if mode == 'base' else []) + ['-c',
                          str(ROOT / f'ggml/src/ggml-cuda/template-instances/mmq-instance-{quant}.cu'),
                          '-o', str(HERE / 'resources' / f'{mode}-{quant}.o')]
        print(mode + '\t' + quant + '\t' + shlex.join(command))
