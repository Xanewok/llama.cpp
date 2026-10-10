"""Frozen-library request-history diagnostic. Model execution is gpuq-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
WT = TREE.parents[1]
sys.path.insert(0, str(TREE / 'codex/stack10/identity2'))
import kld_item_v2 as base
import analyze as trace

lib = base.lib
ARMS = {
    'a': ('stack9-p55', 'none', True, True, {}),
    'b': ('stack9-p55', 'data', True, True, {}),
    'c': ('stack9-p55', 'meta', True, True, {}),
    'd': ('stack9-p55', 'none', False, True, {}),
    'e': ('stack9-p55', 'none', True, True, {'GGML_CUDA_DISABLE_GRAPHS': '1'}),
    'f': ('stack10', 'none', True, True, {}),
    'untraced': ('stack9-p55', 'none', True, False, {}),
}
# Reverse both arm and order traversal for the second repeat.
PLAN = [(arm, order, 0) for arm in ARMS for order in ('short', 'long')]
PLAN += [(arm, order, 1) for arm in reversed(ARMS) for order in ('long', 'short')]
DRAFT_STEPS = 5
DRAFT_VOCAB = Path('/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json')
DEPTHS = {'short': [8192, 90112], 'long': [8192, 32768, 90112]}
CANARY_DEPTHS = [128, 256, 512]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def prepare(path, depths, steps=4):
    lib.check_input({'path': base.CORPUS, 'sha': '753920ad0a0dbfc135aece41284c74e96d70d856df0351100a5305511aabf412'}, 'corpus')
    sequences = base.depth.kldt_sequences(base.CORPUS)
    body = [t for seq in sequences for t in seq]
    with Path(path).open('wb') as f:
        f.write(b'HRT1' + struct.pack('<2i', len(depths), DRAFT_STEPS))
        for depth in depths:
            continuation = body[depth:depth + steps + DRAFT_STEPS]
            if len(continuation) != steps + DRAFT_STEPS:
                raise ValueError('corpus is too short')
            tokens = base.depth.corpus_prompt(sequences, depth) + continuation
            f.write(struct.pack('<2i', depth, steps) + struct.pack(f'<{len(tokens)}i', *tokens))


def environment(name, inp, dest, arm, canary=False):
    frozen, clear, draft, traced, extra = ARMS[arm]
    env = base.environment(base.frozen(frozen)['bindir'], inp, dest, extra, canary)
    env.pop('S10_KLD_INPUT')
    env.pop('S10_KLD_OUT')
    env.update(HISTORY_INPUT=str(inp), HISTORY_OUT=str(dest), HISTORY_CLEAR=clear,
               HISTORY_DRAFT=str(int(draft)), HISTORY_TRACE=str(int(traced)), LLAMA_MTP_FOLD_C='1')
    return env


def command(binary):
    return [str(binary), '-m', str(base.MODEL), '-c', '102400', '-np', '1', '-ctk', 'q8_0', '-ctv', 'q8_0',
            '-ngl', '99', '-fa', 'on', '-sm', 'tensor', '--tensor-split', '50,50', '-b', '2048', '-ub', '512',
            '-t', '8', '-tb', '8', '--spec-draft-vocab', str(DRAFT_VOCAB)]


def build(out, name, remaining):
    target = out / ('build-' + name)
    target.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', TMPDIR=str(target), CCACHE_TEMPDIR=str(target),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', GGML_CCACHE='ON')
    gate = str(WT / 'bin/buildgate')
    commands = [['nice', '-n', '19', 'taskset', '-c', '6-11,18-23', gate, 'cmake', '-S', str(HERE), '-B', str(target),
                 '-DCMAKE_BUILD_TYPE=Release', '-DENGINE=' + str(TREE), '-DCOMMIT=' + base.COMMITS[name],
                 '-DFROZEN=' + str(base.frozen(name)['bindir']), '-DGGML_CUDA=OFF', '-DGGML_CCACHE=ON'],
                ['nice', '-n', '19', 'taskset', '-c', '6-11,18-23', gate, 'cmake', '--build', str(target), '-j4']]
    with (target / 'build.log').open('w') as log:
        for cmd in commands:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=min(180, remaining()), check=True)
    return target / 'history-root'


def validate(dest, arm, depth=90112):
    _, _, draft, traced, _ = ARMS[arm]
    header, values = base.read_logits(dest / 'logits.bin')
    if header != (depth, 4, 248320) or not base.np.isfinite(values).all():
        raise ValueError('invalid logits shape or values')
    text = (dest / 'tool.log').read_text()
    lib.verify_log_patterns(text, [r'history-root: complete nodes=\d+ traced=\d requests=[23]'], 'completion')
    if traced:
        lib.verify_log_patterns(text, [r'meta_completion_calls=[1-9]\d*'], 'meta AllReduce completion hook')
    if not traced:
        return []
    nodes = trace.read_trace(dest / 'trace.jsonl', draft=draft)
    target_tensors = [t for row in nodes if row['domain'] == 'target' for t in [row['output'], *row['inputs']]]
    if not any(t['type'] == 'q8_0' for t in target_tensors):
        raise ValueError('target q8_0 KV was not observed')
    if not any(len(row['output'].get('slices', [])) == 2 for row in nodes if row['domain'] == 'target'):
        raise ValueError('target tensor split was not observed')
    return nodes


def summarize(out, plan):
    records = {}
    for i, (arm, order, repeat) in enumerate(plan):
        dest = out / f'{i}-{arm}-{order}-r{repeat}'
        nodes = validate(dest, arm)
        _, logits = base.read_logits(dest / 'logits.bin')
        key = arm, order
        if key in records:
            previous, previous_logits = records[key]
            if nodes != previous or not base.np.array_equal(logits.view('<u4'), previous_logits.view('<u4')):
                write_json(out / 'SUMMARY.json', {'valid': False, 'repeat_failure': {'arm': arm, 'order': order,
                           'trace': trace.compare(previous, nodes), 'logits': base.compare_rows(previous_logits, logits)}})
                raise lib.Fail(f'repeat differs: {arm}/{order}', 'integrity')
        records[key] = nodes, logits
    results = {}
    for arm in dict.fromkeys(a for a, _, _ in plan):
        short, short_logits = records[arm, 'short']
        long, long_logits = records[arm, 'long']
        results[arm] = {'repeats_equal': True, 'logits_differ': not base.np.array_equal(short_logits.view('<u4'), long_logits.view('<u4')),
                        'logits': base.compare_rows(short_logits, long_logits),
                        'trace': trace.compare(short, long) if ARMS[arm][3] else None}
    baseline = results['a']['logits_differ']
    readings = []
    if results['untraced']['logits_differ'] and not baseline:
        readings.append('inconclusive: callback instrumentation removes target-logit history dependence')
    if not baseline:
        readings.append('not reproduced in traced teacher-forced target logits; no root-cause conclusion')
        if results['a']['trace']['by_domain'].get('target') is None and any(
                finding for domain, finding in results['a']['trace']['by_domain'].items() if domain != 'target'):
            readings.append('only draft/catchup tensors differ at the observed prefix; inspect draft state before attributing target arithmetic')
    else:
        if not results['b']['logits_differ'] and results['c']['logits_differ']:
            readings.append('stale-memory-byte bug class: data clear removes target dependence, metadata clear does not')
        if not results['d']['logits_differ']:
            readings.append('MTP/draft-context state or allocation interaction')
        if not results['e']['logits_differ']:
            readings.append('CUDA graph state under callback instrumentation; confirm without tracing')
        if not readings:
            readings.append('dependence survives all included s9p ablations')
    result = {'arms': results, 'reading': readings,
              'limits': ['Callback execution changes scheduler partitioning, cached meta plans, fusion and graph reuse.',
                         'Physical slice drift can be masked KV padding, inactive rank data, or a PARTIAL representation.',
                         'Identical observed arithmetic inputs are a drift candidate, not proof of benign reassociation.',
                         'The last prefill ubatch is the first observed interval, not necessarily the first causal divergence.',
                         'No sampled proposals, rejection verification, reduced-vocabulary chain, or four-row target verify is replayed.']}
    write_json(out / 'SUMMARY.json', result)
    return result


def negative_control(nodes):
    import copy
    changed = copy.deepcopy(nodes)
    first = changed[0]['output']['slices'][0]['data']
    first['sha256'] = ('1' if first['sha256'][0] != '1' else '0') + first['sha256'][1:]
    if trace.compare(nodes, changed)['first'] is None or trace.compare(nodes, nodes)['first'] is not None:
        raise ValueError('fingerprint comparison negative control failed')


def preflight(out):
    out.mkdir(parents=True, exist_ok=True)
    lib.check_input({'path': DRAFT_VOCAB, 'sha': 'e30af79520ade5b7ddb851d75cf271f07d407dc768bf646e4517607d562a65a6'}, 'draft vocabulary')
    result = base.parser_tests()
    # Parse callback records built from actual GPU logit bytes. Native callback GPU records do not exist yet.
    with base.REAL_KLF.open('rb') as f:
        f.read(4)
        vocab, _ = struct.unpack('<2i', f.read(8))
        depth, _ = struct.unpack('<2i', f.read(8))
        real = f.read(vocab*4)
    tensor = {'name': 'recorded-logits', 'op': 'MUL_MAT', 'type': 'f32', 'shape': [vocab, 1, 1, 1],
              'strides': [4, vocab*4, vocab*4, vocab*4], 'op_params': [0]*16, 'status': 'ok',
              'slices': [{'rank': 0, 'shape': [vocab, 1, 1, 1], 'strides': [4, vocab*4, vocab*4, vocab*4], 'flags': 1,
                          'data': {'sha256': hashlib.sha256(real).hexdigest(), 'bytes': len(real),
                                   'sum_bytes': sum(real), 'absmax_bytes': max(real)}}]}
    fixture = out / 'recorded-logits-trace.jsonl'
    fixture.write_text(''.join(json.dumps({'kind': 'node', 'domain': 'target', 'phase': phase, 'index': 0,
                                         'inputs': [], 'output': tensor}) + '\n'
                               for phase in ['prefill', *(f'decode-{i}' for i in range(4))]))
    nodes = trace.read_trace(fixture, draft=False)
    negative_control(nodes)
    bad = out / 'bad-trace.jsonl'
    bad.write_text(fixture.read_text().rsplit('\n', 2)[0] + '\n')
    try:
        trace.read_trace(bad, draft=False)
    except ValueError:
        pass
    else:
        raise ValueError('missing-phase control was accepted')
    prepare(out / 'short.bin', DEPTHS['short'])
    prepare(out / 'long.bin', DEPTHS['long'])
    if (out / 'short.bin').read_bytes()[8:] == (out / 'long.bin').read_bytes()[8:]:
        raise ValueError('history input control is not distinct')
    result.update(trace_fixture_origin='actual GPU KLF bytes serialized in callback schema; not native node capture',
                  comparison_mutation_rejected=True, missing_phase_rejected=True, plan=PLAN)
    write_json(out / 'PREFLIGHT.json', result)
    return result


def execute(args):
    if os.environ.get('GPUQ_ACTIVE') != '1':
        raise lib.Fail('execution requires gpuq', 'deterministic')
    out = args.out
    if out.exists():
        raise lib.Fail('output dir is not fresh', 'deterministic')
    out.mkdir(parents=True)
    deadline = time.monotonic() + args.budget_s
    def interrupted(signum, frame):
        raise lib.Fail('budget or queue interruption', 'timeout')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise lib.Fail('budget expired', 'timeout')
        return value
    canary = args.mode == 'canary'
    plan = [('a', 'long', 0)] if canary else PLAN
    inputs = {}
    for order in dict.fromkeys(order for _, order, _ in plan):
        inputs[order] = out / (order + '.bin')
        prepare(inputs[order], CANARY_DEPTHS if canary else DEPTHS[order])
    pins = {name: lib.check_frozen(base.frozen(name)) for name in dict.fromkeys(ARMS[a][0] for a, _, _ in plan)}
    if base.MODEL.stat().st_size != 15475951296:
        raise lib.Fail('model size changed', 'integrity')
    lib.check_input({'path': DRAFT_VOCAB, 'sha': 'e30af79520ade5b7ddb851d75cf271f07d407dc768bf646e4517607d562a65a6'}, 'draft vocabulary')
    write_json(out / 'plan.json', {'plan': plan, 'arms': ARMS, 'frozen': pins,
                                 'draft_steps_D': DRAFT_STEPS, 'inputs_sha256': {k: lib.file_sha(v) for k, v in inputs.items()}})
    binaries = {}
    lib.event('measure_start')
    for i, (arm, order, repeat) in enumerate(plan):
        name = ARMS[arm][0]
        if name not in binaries:
            binaries[name] = build(out, name, remaining)
        if i:
            lib.wait_gpu_idle(base.DEVICES, wait_s=min(90, remaining()))
        lib.check_frozen(base.frozen(name))
        dest = out / f'{i}-{arm}-{order}-r{repeat}'
        dest.mkdir()
        env = environment(name, inputs[order], dest, arm, canary)
        cmd = command(binaries[name])
        if canary:
            cmd[cmd.index('-c') + 1] = '4096'
        write_json(dest / 'command.json', cmd)
        write_json(dest / 'env.json', env)
        started = time.time()
        proc = None
        timed_out = False
        checked_maps = False
        try:
            with (dest / 'tool.log').open('w') as log:
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        preexec_fn=lambda: lib._own_group(os.getppid()))
                (dest / 'pid').write_text(str(proc.pid))
                arm_deadline = time.monotonic() + min(240, remaining())
                while proc.poll() is None:
                    if time.monotonic() > arm_deadline:
                        timed_out = True
                        break
                    if not checked_maps:
                        maps = Path(f'/proc/{proc.pid}/maps')
                        if maps.exists() and 'libllama' in maps.read_text():
                            outside = lib.libs_outside(proc.pid, base.frozen(name)['root'])
                            if outside:
                                raise lib.Fail('libraries outside frozen root: ' + ', '.join(outside), 'integrity')
                            checked_maps = True
                    time.sleep(.2)
                rc = proc.poll()
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            faults = lib.record_journal(int(started), dest / 'journal.txt')
        write_json(dest / 'status.json', {'rc': rc if proc else None, 'timed_out': timed_out, 'maps_checked': checked_maps,
                                        'elapsed_s_M': time.time() - started, 'faults': faults})
        if faults is None:
            raise lib.Fail('kernel journal unreadable', 'integrity')
        if faults:
            raise lib.Fail('GPU fault since arm start', 'hazard')
        if timed_out:
            raise lib.Fail('budget or arm timeout', 'timeout')
        if rc:
            raise lib.Fail(f'history tool exited {rc}', 'deterministic')
        if not checked_maps:
            raise lib.Fail('library provenance was not observed', 'integrity')
        nodes = validate(dest, arm, CANARY_DEPTHS[-1] if canary else DEPTHS[order][-1])
        lib.check_frozen(base.frozen(name))
        if canary:
            negative_control(nodes)
            # The same whole-run parsers and real-artifact negative controls run before full work.
            preflight(out / 'parser-checks')
            write_json(out / 'SUMMARY.json', {'canary': 'valid', 'nodes_D': len(nodes), 'comparison_negative_control': 'passed'})
    if not canary:
        summarize(out, plan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['preflight', 'canary', 'run'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--budget-s', type=int, default=6600)
    args = p.parse_args()
    try:
        if args.mode == 'preflight':
            preflight(args.out)
        else:
            execute(args)
        return 0
    except lib.Fail as error:
        print(str(error), file=sys.stderr)
        return error.rc
    except subprocess.TimeoutExpired:
        print('budget expired during build', file=sys.stderr)
        return 7
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 8


if __name__ == '__main__':
    sys.exit(main())
