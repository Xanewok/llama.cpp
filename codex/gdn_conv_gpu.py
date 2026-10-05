"""Queue-only drivers; reuse the shared harness for server ownership, identity and timing gates."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from harness import depth, lib
from harness.__main__ import main as harness_main

ENGAGED = re.compile(r'ssm_conv_state_fusion: engaged (.*?) device=(\d+) graph=(\d+) tokens=(\d+) channels=(\d+) seqs=(\d+) slots=(\d+) nodes=(\d+)')
CENSUS = re.compile(r'ssm_conv_state_fusion: graph device=(\d+) graph=(\d+) tokens=(-?\d+) conv=(\d+) fused=(\d+)')
REJECTED = re.compile(r'ssm_conv_state_fusion: rejected .*? device=(\d+) graph=(\d+) tokens=(\d+): (.*)')


def engagement(text, layers, tokens, devices, fragmented=False):
    groups = {}
    for name, dev, graph, count, channels, seqs, slots, nodes in ENGAGED.findall(text):
        if int(count) != tokens:
            continue
        key = (int(dev), int(graph))
        groups.setdefault(key, set()).add(name)
    rejected = [row for row in REJECTED.findall(text) if int(row[2]) == tokens]
    census = [row for row in CENSUS.findall(text) if int(row[2]) == tokens]
    census_keys = {(int(dev), int(uid)) for dev, uid, _, _, _ in census}
    if not census or census_keys != set(groups) or any(int(conv) < 1 or conv != fused for _, _, _, conv, fused in census):
        raise lib.Fail('path-engaged proof failed: missing or partial graph census', 'integrity')
    by_device = {dev: set() for dev in devices}
    for (dev, uid), names in groups.items():
        by_device.setdefault(dev, set()).update(names)
    counts = by_device.values() if fragmented else groups.values()
    if rejected or not groups or {dev for dev, _ in groups} != set(devices) or any(len(names) != layers for names in counts):
        raise lib.Fail(f'path-engaged proof failed: tokens={tokens}, graph layer counts={[len(v) for v in groups.values()]}, rejected={len(rejected)}', 'integrity')
    return [{'device': dev, 'graph': uid, 'layers_M': len(names), 'census_M': [{'conv': int(c), 'fused': int(f)} for d, g, t, c, f in census if int(d) == dev and int(g) == uid]} for (dev, uid), names in sorted(groups.items())]


def bench_results(text, legacy=False):
    rows = {}
    for line in text.splitlines():
        if not line.startswith('RESULT '):
            continue
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
        if 'mode' not in fields:
            continue
        mode = fields['mode']
        if legacy and mode == 'galloc':
            mode = 'galloc-model'
        values = {}
        for key in ('us_per_graph_sync_med', 'us_per_graph_async_med', 'us_per_layer_async', 'max_abs_out', 'max_abs_cache'):
            try:
                value = float(fields[key])
            except (KeyError, ValueError) as error:
                raise lib.Fail(f'bench parser: missing/invalid {key}', 'insufficient_data') from error
            if not math.isfinite(value) or value < 0:
                raise lib.Fail(f'bench parser: invalid {key}', 'integrity')
            values[key + '_M'] = value
        if values['max_abs_out_M'] != 0 or values['max_abs_cache_M'] != 0:
            raise lib.Fail('bench bitwise output/cache gate failed', 'integrity')
        rows[mode] = values
    if set(rows) != {'sep-unfused', 'sep-fused', 'galloc-model', 'galloc-pinned'}:
        raise lib.Fail('bench parser: missing modes', 'insufficient_data')
    return rows


def bench_engagement(text, layers, tokens, device):
    sections = re.split(r'^BENCH_MODE (\S+)\n', text, flags=re.MULTILINE)
    logs = dict(zip(sections[1::2], sections[2::2]))
    if set(logs) != {'sep-unfused', 'sep-fused', 'galloc-model', 'galloc-pinned'}:
        raise lib.Fail('path-engaged proof failed: missing bench mode logs', 'integrity')
    return {mode: engagement(logs[mode], layers, tokens, [device])
            for mode in ('sep-fused', 'galloc-model', 'galloc-pinned')}


def run_depth(experiment, args):
    original = depth.evaluate_boot

    def evaluate(x, dest, arm, results, canary=False):
        failures = original(x, dest, arm, results, canary)
        if arm == 'patch':
            try:
                proof = engagement((dest / 'server.log').read_text(errors='replace'), 48, 6, [0, 1], fragmented=True)
                (dest / 'conv-engagement.json').write_text(json.dumps(proof, indent=2))
            except lib.Fail as error:
                failures.append(error)
        return failures

    depth.evaluate_boot = evaluate
    return harness_main(['depth', str(experiment), *args])


def run_bench(root, mode, out):
    if mode == '--preflight':
        lib.check_frozen({'name': 'bench', 'root': root, 'runroot': None, 'bin': 'build/bin',
                          'bindir': root / 'build/bin', 'exe': root / 'build/bin/conv-state-bench',
                          'repo': None, 'pinned': {}, 'commit': (root / 'SOURCE').read_text().splitlines()[0]})
        return 0
    if os.environ.get('GPUQ_ACTIVE') != '1':
        raise lib.Fail('GPU execution requires gpuq')
    lib.verify_nice(os.getpriority(os.PRIO_PROCESS, 0))
    out.mkdir(parents=True, exist_ok=True)
    if any(p.name != 'console.log' for p in out.iterdir()):
        raise lib.Fail('benchmark output directory must be fresh')
    run_bench(root, '--preflight', out)
    lib.event('measure_start')
    # G: full <= 300 s, one-process canary <= 60 s. gpuq supplies the outer budget and GPU/RAM gates.
    canary = mode == '--canary'
    deadline = time.monotonic() + (120 if canary else 450)

    def interrupted(signum, frame):
        raise lib.Fail('benchmark budget interrupted', 'timeout')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    start = time.time()
    summaries = []
    try:
        for device in ([0] if canary else [0, 1]):
            for tokens in ([6] if canary else [1, 4, 6]):
                for slots in ([6] if canary else range(1, 7)):
                    tag = f'd{device}-t{tokens}-k{slots}'
                    command = [str(root / 'build/bin/conv-state-bench'), '--backend', f'CUDA{device}',
                               '--layers', '2' if canary else '48', '--channels', '128' if canary else '5120',
                               '--tokens', str(tokens), '--K', str(slots), '--iters', '5' if canary else '300']
                    env = dict(os.environ, GGML_CUDA_DEBUG_SSM_CONV='1', LD_LIBRARY_PATH=str(root / 'build/bin'))
                    (out / f'{tag}-command.json').write_text(json.dumps(command))
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise lib.Fail('benchmark budget exhausted', 'timeout')
                    with (out / f'{tag}.txt').open('w') as stdout, (out / f'{tag}.err').open('w') as stderr:
                        try:
                            child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
                            try:
                                code = child.wait(timeout=remaining)
                            finally:
                                lib.stop_child(child, 5)
                        except subprocess.TimeoutExpired as error:
                            raise lib.Fail('benchmark budget exhausted', 'timeout') from error
                    if code:
                        raise lib.Fail(f'benchmark exited {code}')
                    rows = bench_results((out / f'{tag}.txt').read_text())
                    proof = bench_engagement((out / f'{tag}.err').read_text(), 2 if canary else 48, tokens, device)
                    summaries.append({'device': device, 'tokens': tokens, 'K': slots, 'results': rows, 'engagement': proof})
                    # The benchmark also memcmp-checks every output and cache plane against its unfused mode.
                    (out / 'summary.json').write_text(json.dumps(summaries, indent=2))
    finally:
        lib.record_journal(start, out / 'journal.txt')
    run_bench(root, '--preflight', out)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['depth', 'bench'])
    parser.add_argument('source', type=Path)
    parsed, args = parser.parse_known_args()
    if parsed.kind == 'depth':
        return run_depth(parsed.source, args)
    stage = next((x for x in args if x in ('--preflight', '--canary', '--execute')), '--preflight')
    out = Path(args[args.index('--out') + 1]) if '--out' in args else parsed.source.parent / 'bench-preflight'
    return run_bench(parsed.source, stage, out)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except lib.Fail as error:
        print(str(error), file=sys.stderr)
        sys.exit(error.rc)
