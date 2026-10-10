"""Small-context CPU model smoke and repeat check. Invoke through cpugate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import analyze
import item


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU check requires hidden CUDA devices')
    if args.out.exists():
        raise ValueError('CPU output must be fresh')
    args.out.mkdir(parents=True)
    inputs = {}
    for order, depths in [('short', [32, 96]), ('long', [32, 64, 96])]:
        inputs[order] = args.out / (order + '.bin')
        item.prepare(inputs[order], depths)
    plan = [('a', 'short', 0), ('a', 'long', 0), ('a', 'long', 1), ('a', 'short', 1),
            ('b', 'long', 0), ('c', 'long', 0), ('d', 'long', 0)]
    records = {}
    timings = []
    for i, (arm, order, repeat) in enumerate(plan):
        if int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))) < 10*1024*1024:
            raise ValueError('MemAvailable below CPU model floor')
        dest = args.out / f'{i}-{arm}-{order}-r{repeat}'
        dest.mkdir()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   HISTORY_INPUT=str(inputs[order]), HISTORY_OUT=str(dest), HISTORY_CLEAR=item.ARMS[arm][1],
                   HISTORY_DRAFT=str(int(item.ARMS[arm][2])), HISTORY_TRACE='1', LLAMA_ARG_KV_PAD='256',
                   GGML_SCHED_INPUT_STAGING='1', GGML_BACKEND_PATH=str(args.binary.parent))
        env.pop('LD_PRELOAD', None)
        env.pop('LD_LIBRARY_PATH', None)
        cmd = [str(args.binary), '-m', '/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf',
               '-c', '4096', '-np', '1', '-ngl', '0', '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0',
               '-b', '512', '-ub', '512', '-t', '8', '-tb', '8']
        item.write_json(dest / 'command.json', cmd)
        started = time.monotonic()
        with (dest / 'tool.log').open('w') as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        timings.append({'arm': arm, 'order': order, 'elapsed_s_M': time.monotonic() - started})
        nodes = analyze.read_trace(dest / 'trace.jsonl', draft=item.ARMS[arm][2])
        header, values = item.base.read_logits(dest / 'logits.bin')
        if header[:2] != (96, 4):
            raise ValueError('wrong CPU logit shape')
        key = arm, order
        if key in records:
            previous, logits = records[key]
            if previous != nodes or not item.base.np.array_equal(logits.view('<u4'), values.view('<u4')):
                raise ValueError('CPU repeat changed')
        records[key] = nodes, values.copy()
    short, short_logits = records['a', 'short']
    long, long_logits = records['a', 'long']
    item.write_json(args.out / 'SUMMARY.json', {'timings': timings, 'a_repeats_equal': True,
                                               'a_trace': analyze.compare(short, long),
                                               'a_logits': item.base.compare_rows(short_logits, long_logits),
                                               'controls_vs_a_long': {
                                                   arm: {'trace': analyze.compare(long, records[arm, 'long'][0]),
                                                         'logits': item.base.compare_rows(long_logits, records[arm, 'long'][1])}
                                                   for arm in ('b', 'c', 'd')}})


if __name__ == '__main__':
    main()
