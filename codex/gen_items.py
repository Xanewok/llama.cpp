"""Clone the recorded depth experiment and render items without changing the shared harness."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

HERE = Path(__file__).resolve().parent
EVAL = Path('/home/xanewok/repos/qwen38-eval')
sys.path.insert(0, str(HERE))
from kernel_item import ROOTS


def value(obj):
    if isinstance(obj, dict):
        return '{ ' + ', '.join(f'{json.dumps(str(k))} = {value(v)}' for k, v in obj.items()) + ' }'
    if isinstance(obj, list):
        return '[' + ', '.join(value(v) for v in obj) + ']'
    return json.dumps(obj)


def write_toml(path, data):
    path.write_text('\n'.join(f'{key} = {value(obj)}' for key, obj in data.items()) + '\n')


def local_pins():
    return sorted(str(path) for path in HERE.iterdir() if path.suffix in ('.py', '.sh')) + \
        sorted(str(path) for path in (HERE / 'fixtures').iterdir() if path.is_file())


def kernel_items():
    for kind, run_s, estimate in [('memcheck', 3000, 1600), ('perf', 4800, 3200)]:
        entry = [sys.executable, str(HERE / 'kernel_item.py'), kind]
        out = HERE / 'gpu-results' / kind
        files = local_pins() + [str(p) for p in (EVAL / 'harness').glob('*.py')]
        for root in ROOTS.values():
            files += [str(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        data = dict(id='mmq-smallj-' + kind, title='Small-J MMQ: ' + kind + '; reference and safe negative control before perf',
                    kind='measure', ab=kind == 'perf', retry='fresh', budget_s=run_s+750, est_s=estimate,
                    startup_grace_s=120,
                    exit_codes={'3': 'hazard', '4': 'transient', '5': 'insufficient_data', '6': 'integrity',
                                '7': 'timeout', '8': 'deterministic'},
                    signatures=[{'class': 'integrity', 'regex': 'probe: integrity'},
                                {'class': 'timeout', 'regex': 'probe: budget'}],
                    run=dict(cmd=entry + ['--stage', 'run', '--out', str(out / 'full-a{attempt}'), '--budget', str(run_s)],
                             cwd=str(HERE), env={'GPUQ_ACTIVE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}, ports=[]),
                    preflight=dict(cmd=entry + ['--stage', 'preflight'], timeout_s=120),
                    canary=dict(cmd=entry + ['--stage', 'canary', '--out', str(out / 'canary-a{attempt}'), '--budget', '570'],
                                env={'GPUQ_ACTIVE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}, timeout_s=600),
                    outputs=dict(paths=[str(out / 'full-a{attempt}')]),
                    fix=dict(editable=[], protected=['frozen', 'budget', 'coverage', 'correctness', 'memcheck_first']),
                    pins=dict(files=sorted(set(files))))
        write_toml(HERE / 'items' / f'{kind}.gpuq.toml', data)


def arm(name):
    root = ROOTS[name]
    return dict(label='stack6 baseline' if name == 'base' else 'sm_120 I64 J8/16/24, 128 threads, occupancy 2',
                commit=(root / 'SOURCE').read_text().splitlines()[0], root=str(root), bin='build/bin',
                exe='build/bin/round-trace', env={'GGML_SCHED_INPUT_STAGING': '1', 'LLAMA_ARG_LOG_VERBOSITY': '4',
                                                'LD_LIBRARY_PATH': '{root}/build/bin'})


def depth_items():
    template = tomllib.loads((EVAL / 'harness/experiments/duo-np2-depth.toml').read_text())
    for quant, streams in [('iq4', 2), ('j5q', 2), ('j5q', 1)]:
        name = f'{quant}-np{streams}'
        x = copy.deepcopy(template)
        x['id'] = 'mmq-smallj-' + name
        x['title'] = f'stack6 vs small-J MMQ {quant}, TP2 np{streams} k5 ub512, ABBA; memcheck/perf required before queueing'
        x['out'] = str(HERE / 'gpu-results' / name)
        x['arms'] = {'up': arm('base'), 'patch': arm('patch')}
        x['extra_pins'] = local_pins() + [str(HERE / 'REPORT-mmq-smallj.md')]
        if quant == 'j5q':
            x['model'] = dict(id='swift15-j5q', path='/home/xanewok/models/Swift-1.5-Qwen3.8-27B-J5q-bf16.gguf',
                              bytes=19996051744, sha='f2c850ea3cd818b9d669fbd1e18767ef669ce012e614d819cce5ab6809e3a774')
        args = x['server']['args']
        args[args.index('-ub') + 1] = '512'
        args += ['--kv-pad', '4096']
        x['server']['knobs'] += ['GGML_CUDA_DEBUG_MMQ_SMALLJ']
        if streams == 1:
            args[args.index('-c') + 1] = '102400'
            args[args.index('-np') + 1] = '1'
        x['depth']['streams'] = streams
        x['depth']['identity'] = 'off' if streams == 2 else 'cross-arm'
        x['depth']['identity_off_reason'] = 'np2 batch composition is timing-dependent; CPU-reference MMQ test and np1 identity gate separately'
        x['depth']['distinct_builds'] = True
        x['depth']['wall_s'] = 1100 if streams == 2 else 420
        x['canary'] = dict(plan=['patch'], depths=[8192], proofs=[])
        x['canary']['env'] = {'GGML_CUDA_DEBUG_MMQ_SMALLJ': '1'}
        if streams == 2:
            proof_type = 'q5_K' if quant == 'j5q' else 'iq4_xs'
            x['arms']['patch']['canary_log'] = [
                f'mmq_smallj: device={device} type={proof_type} columns=12 J=16 I=64 threads=128 occupancy=2'
                for device in (0, 1)]
        x['budget'].update(canary_s=600, margin=1.5, est_s=5000 if streams == 2 else 2300)
        path = HERE / 'items' / (name + '.toml')
        write_toml(path, x)
        render(path)

    x = copy.deepcopy(x)
    x['id'] = 'mmq-smallj-j5q-prefill'
    x['title'] = 'stack6 vs small-J MMQ: J5q np1 ub512 cold 32k prefill per boot and decode, ABBA'
    x['out'] = str(HERE / 'gpu-results/j5q-prefill')
    x['depth']['depths'] = [32768]
    x['depth']['wall_s'] = 300
    x['budget'].update(canary_s=600, est_s=1500)
    path = HERE / 'items/j5q-prefill.toml'
    write_toml(path, x)
    render(path)


def render(path):
    subprocess.run([sys.executable, '-m', 'harness.mkitem', str(path), '--out', str(path.with_suffix('.gpuq.toml'))],
                   cwd=EVAL, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))


if __name__ == '__main__':
    (HERE / 'items').mkdir(exist_ok=True)
    kernel_items()
    depth_items()
