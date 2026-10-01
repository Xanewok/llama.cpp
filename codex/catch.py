#!/usr/bin/env python3
"""Parent-only bounded prefix capture and onset continuations."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error

import catch_replay as replay
import experiment as study
import onset
from runaways import ci, spans

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = Path('/home/xanewok/repos/ra-capture/stack')
FROZEN = REFERENCE / 'codex/frozen/capture/build/bin'
BASE = Path('/home/xanewok/models/Qwen3.8-27B-IQ4_XS.gguf')
PREFIXES = ['swift15-llamacpp-5-776e6ec1', 'swift15-llamacpp-4-dd232125']
VARIANTS = ['served', 'no-spec', 'temp07', 'presence']
OFFSETS = [0, 64, 512]
PINS = json.loads((ROOT / 'codex/CAPTURE-PINS.json').read_text())


def gate():
    if os.environ.get('RUN_PARENT_GPU') != '1' or os.environ.get('PLAN', '1') != '0':
        raise RuntimeError('GPU execution requires RUN_PARENT_GPU=1 PLAN=0')


def seed(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:4], 'little') % (2**32 - 1)


def verify_base():
    sys.path.insert(0, str(ROOT / 'gguf-py'))
    from gguf import GGUFReader
    reader = GGUFReader(str(BASE), 'r')
    metadata = {name: field.contents() for name, field in reader.fields.items()
                if name.startswith('general.') or name in ('qwen35.block_count', 'qwen35.embedding_length', 'qwen35.nextn_predict_layers')}
    if (metadata.get('general.name') != 'Qwen3.8 27B' or metadata.get('general.basename') != 'Qwen3.8'
            or metadata.get('general.architecture') != 'qwen35' or metadata.get('general.file_type') != 30):
        raise RuntimeError('base metadata does not identify Qwen3.8-27B IQ4_XS')
    return dict(path=str(BASE), metadata_M=metadata, bytes_M=BASE.stat().st_size,
                tensor_count_M=len(reader.tensors),
                nextn_tensor_names=[t.name for t in reader.tensors if 'nextn' in t.name or 'mtp' in t.name],
                arm='no-spec regardless of preserved NextN tensors')


def freeze(source, dest):
    expected = PINS['sha256']
    for name, digest in expected.items():
        if name == 'llama-server':
            continue
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or study.digest(path) != digest:
            raise RuntimeError('frozen bytes changed/escaped: ' + name)
    if any(p.name not in expected for p in source.glob('*.so*')):
        raise RuntimeError('unmanifested frozen library')
    dest.mkdir(parents=True, exist_ok=True)
    for name, digest in expected.items():
        if name == 'llama-server':
            continue
        target = dest / name
        if not target.exists():
            shutil.copy2(source / name, target, follow_symlinks=True)
        if study.digest(target) != digest:
            raise RuntimeError('local frozen bytes changed: ' + name)
    (dest / 'SHA256SUMS').write_text(''.join(f'{v}  {k}\n' for k, v in sorted(expected.items()) if k != 'llama-server'))


def xid():
    gate()
    p = subprocess.run(['journalctl', '-k', '-b', '--no-pager'], capture_output=True, text=True, check=True, timeout=10)
    if not p.stdout.strip() or 'Xid' in p.stdout or 'fallen off' in p.stdout:
        raise RuntimeError('empty kernel journal or Xid/fallen-off in current boot')


def idle(args):
    gate()
    xid()
    p = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10)
    rows = {r[0].strip(): list(map(int, r[1:])) for r in (line.split(',') for line in p.stdout.splitlines())}
    for device in args.devices.split(','):
        if device not in rows or rows[device][0] > 750 or rows[device][1] > 5:
            raise RuntimeError('selected GPU missing/busy: ' + device)
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                       capture_output=True, text=True, check=True, timeout=10)
    if any(line.strip() and not any(x in line.lower() for x in ('1password', 'cosmic-', 'xdg-desktop-portal'))
           for line in p.stdout.splitlines()):
        raise RuntimeError('another GPU compute process')
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if (p / 'comm').read_text().strip() == 'llama-server':
                    raise RuntimeError('existing llama-server')
            except (FileNotFoundError, ProcessLookupError):
                pass
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', args.port)) == 0:
            raise RuntimeError('port is occupied')


@contextmanager
def server(args, arm, model, label):
    gate()
    binary = args.out / 'frozen/capture-srv'
    for name, digest in PINS['sha256'].items():
        if name != 'llama-server' and study.digest(binary.parent / name) != digest:
            raise RuntimeError('local frozen bytes changed before boot: ' + name)
    idle(args)
    directory = args.out / 'boots' / (label + '-' + str(time.time_ns()))
    captures = directory / 'captures'
    captures.mkdir(parents=True)
    cmd = study.command(binary, 'IQ4_XS', arm, args)
    cmd[cmd.index('-m') + 1] = str(model)
    cmd += ['--no-context-shift']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.devices, LLAMA_CAPTURE_DIR=str(captures),
               LLAMA_CAPTURE_PROBS='1', LLAMA_CAPTURE_LIMIT=str(args.length + args.branch_length),
               LD_LIBRARY_PATH=str(binary.parent), GGML_CUDA_FA_MMA_Q8_0='1',
               OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1')
    study.atomic(directory / 'launch.json', dict(command=cmd, capture_probs='1', model=str(model)))
    child = None
    try:
        with (directory / 'server.log').open('w') as log:
            child = subprocess.Popen(['nice', '-n', '19', *cmd], env=env, stdout=log, stderr=subprocess.STDOUT)
            ready = time.monotonic() + 300
            while time.monotonic() < ready:
                if child.poll() is not None:
                    raise RuntimeError('server exited before health: ' + str(directory))
                try:
                    with study.http(args, '/health', timeout=2) as response:
                        if json.load(response).get('status') == 'ok':
                            break
                except (OSError, ValueError):
                    pass
                time.sleep(.5)
            else:
                raise RuntimeError('server health timeout')
            with study.http(args, '/slots') as response:
                slots = json.load(response)
            if len(slots) != 1 or slots[0].get('n_ctx') != args.context:
                raise RuntimeError('actual slot context differs')
            yield captures
    finally:
        timer = signal.setitimer(signal.ITIMER_REAL, 0)
        study.stop(child)
        xid()
        if sys.exc_info()[0] is None and timer[1]:
            signal.setitimer(signal.ITIMER_REAL, 1, timer[1])


def capture(args, directory, body):
    url = 'http://127.0.0.1:' + str(args.port)
    result, path, tokens = replay.request_capture(url, directory, body, args.request_timeout)
    if result.get('timings', {}).get('predicted_n') != len(tokens):
        raise RuntimeError('capture/completion count mismatch')
    for token in tokens:
        onset.distribution(token)
    return result, path, tokens


def retain(args, path, ident):
    dest = args.out / 'retained' / (ident + '.jsonl')
    dest.parent.mkdir(exist_ok=True)
    path.replace(dest)
    return dest


def originals(args, fixtures, phase):
    cells = [('base', 'no-spec', p) for p in PREFIXES] if phase == 0 else [('swift', a, PREFIXES[0]) for a in ('no-spec', 'S3')]
    limit = args.n if phase == 0 else args.catch_n
    for block, start in enumerate(range(0, limit, args.per_boot)):
        order = cells[block % len(cells):] + cells[:block % len(cells)]
        if block % 2:
            order.reverse()
        for model, arm, prefix in order:
            group = f'p{phase}-{model}-{arm}-{prefix}'
            rows = [json.loads(p.read_text()) for p in args.out.glob('trials/' + group + '-*.json')]
            if phase == 1 and sum(r['repetition'] for r in rows) >= args.k:
                continue
            pending = [i for i in range(start, min(start + args.per_boot, limit))
                       if not (args.out / 'trials' / f'{group}-{i:04}.json').exists()]
            if not pending:
                continue
            with server(args, arm, BASE if model == 'base' else Path(study.MODELS['IQ4_XS']), group) as directory:
                ids, history, overrides = study.prepare_prefix(args, fixtures[prefix])
                if history or len(ids) + args.length >= args.context:
                    raise RuntimeError('unsupported sampler import or insufficient context')
                prefix_path = args.out / 'prefix-tokens' / (group + '.json')
                prefix_path.parent.mkdir(exist_ok=True)
                evaluated = dict(fixture=prefix, model=model, token_ids=ids)
                if prefix_path.exists() and json.loads(prefix_path.read_text()) != evaluated:
                    raise RuntimeError('tokenized prefix changed between boots')
                study.atomic(prefix_path, evaluated)
                for i in pending:
                    ident = f'{group}-{i:04}'
                    trial = dict(id=ident, seed=seed('catch-v1', group, i), prefix=prefix, index=i)
                    body = study.request_body(ids, trial, args, overrides)
                    body['stream'] = False
                    result, path, tokens = capture(args, directory, body)
                    found = onset.find_onset(tokens)
                    repetition = bool(spans(replay.text(tokens)))
                    controls = sum(not r['repetition'] and r.get('capture') is not None for r in rows)
                    saved = retain(args, path, ident) if repetition or controls < 3 else None
                    if saved:
                        onset.analyze(saved, args.out / 'analysis' / ident)
                    else:
                        path.unlink()
                    row = dict(trial, phase=phase, model=model, arm=arm, repetition=repetition, onset=found,
                               capture=str(saved) if saved else None, output_tokens_M=len(tokens),
                               limit_reached=len(tokens) >= args.length, final=result,
                               fidelity='reconstructed Swift-written history; exploratory')
                    study.atomic(args.out / 'trials' / (ident + '.json'), row)
                    rows.append(row)
                    summarize(args)
                    if phase == 1 and sum(r['repetition'] for r in rows) >= args.k:
                        break


def branch_request(path, position, length, branch_seed, variant, tolerance):
    body = replay.body(path, position, length, branch_seed, tolerance, allow_drift=variant != 'served')
    if variant == 'temp07':
        body['temperature'] = .7
    elif variant == 'presence':
        body.update(presence_penalty=1.0, repeat_last_n=102400)
    return body


def branches(args):
    sources = [json.loads(p.read_text()) for p in sorted(args.out.glob('trials/p1-*.json'))]
    sources = [s for s in sources if s['repetition'] and s.get('capture')]
    if not sources:
        raise RuntimeError('phase 2 has no captured loop; do not count this as a negative')
    for source in sources:
        found = source['onset']
        if not found or not found['exact_token_period']:
            study.atomic(args.out / 'branches' / (source['id'] + '-unsupported.json'), dict(reason='no exact token period'))
            continue
        path = Path(source['capture'])
        controls = [s for s in map(lambda p: json.loads(p.read_text()), args.out.glob('trials/p1-*.json'))
                    if not s['repetition'] and s['arm'] == source['arm'] and s.get('capture')]
        for control in controls:
            onset.analyze(control['capture'], args.out / 'matched-controls' / source['id'] / control['id'], match=found)
        for offset in OFFSETS:
            position = found['onset_token_M'] - offset
            key = source['id'] + f'-offset{offset}'
            if position <= 0:
                study.atomic(args.out / 'branches' / (key + '-unavailable.json'), dict(reason='capture build requires a positive generated-history boundary', position_D=position))
                continue
            validation_file = args.out / 'branches' / (key + '-fidelity.json')
            if not validation_file.exists():
                with server(args, source['arm'], Path(study.MODELS['IQ4_XS']), key + '-fidelity') as directory:
                    _, check, _ = capture(args, directory, replay.body(path, position, 1, tolerance=args.tolerance))
                    events = [json.loads(l) for l in check.read_text().splitlines()]
                    validations = [e for e in events if e['event'] == 'validation']
                    study.atomic(validation_file, dict(strict=True, capture_sha256=study.digest(path),
                                 position_M=position, validated_rows_M=len(validations), all_equal_M=all(e['equal'] for e in validations)))
                    check.unlink()
            for block, start in enumerate(range(0, args.continuations, args.per_boot)):
                order = VARIANTS[block % len(VARIANTS):] + VARIANTS[:block % len(VARIANTS)]
                if block % 2:
                    order.reverse()
                for variant in order:
                    pending = [i for i in range(start, min(start + args.per_boot, args.continuations))
                               if not (args.out / 'branches' / f'{key}-{variant}-{i:04}.json').exists()]
                    if not pending:
                        continue
                    arm = 'no-spec' if variant == 'no-spec' else source['arm']
                    with server(args, arm, Path(study.MODELS['IQ4_XS']), key + '-' + variant) as directory:
                        for i in pending:
                            # Matched branch seeds across interventions; speculation still consumes RNG differently.
                            branch_seed = seed('branch-v1', key, i)
                            body = branch_request(path, position, args.branch_length, branch_seed, variant, args.tolerance)
                            if len(body['prompt']) + position + args.branch_length >= args.context:
                                raise RuntimeError('branch exceeds slot context')
                            result, check, tokens = capture(args, directory, body)
                            events = [json.loads(l) for l in check.read_text().splitlines()]
                            branch_events = [e for e in events if e['event'] == 'branch']
                            if len(branch_events) != 1 or branch_events[0]['at'] != position:
                                raise RuntimeError('branch did not reset sampler at requested boundary')
                            suffix = tokens[position:]
                            validations = [e for e in events if e['event'] == 'validation']
                            study.atomic(args.out / 'branches' / f'{key}-{variant}-{i:04}.json',
                                dict(source=source['id'], offset=offset, variant=variant, arm=arm, index=i,
                                     seed=branch_seed, position_M=position, output_tokens_M=len(suffix),
                                     repetition=bool(spans(replay.text(suffix))), evidence=spans(replay.text(suffix)),
                                     token_ids=[t['id'] for t in suffix], final=result,
                                     validated_rows_M=len(validations), drift_rows_M=sum(not e['equal'] for e in validations),
                                     intervention={k: body[k] for k in ('temperature', 'presence_penalty', 'repeat_last_n')},
                                     strict_fidelity=str(validation_file)))
                            check.unlink()
                            summarize(args)


def summarize(args):
    groups = {}
    for path in args.out.glob('trials/*.json'):
        r = json.loads(path.read_text())
        groups.setdefault((r['phase'], r['model'], r['arm'], r['prefix']), []).append(r)
    originals = []
    for key, rows in sorted(groups.items()):
        count = sum(r['repetition'] for r in rows)
        originals.append(dict(phase=key[0], model=key[1], arm=key[2], prefix=key[3],
                         n_M=len(rows), loops_M=count, rate_D=count / len(rows),
                         ci95_fixed_n_D=ci(count, len(rows)) if key[0] == 0 else None,
                         stopping='fixed N' if key[0] == 0 else 'K loops or N; stopped fraction is descriptive'))
    bg = {}
    for path in args.out.glob('branches/*.json'):
        r = json.loads(path.read_text())
        if 'variant' in r:
            bg.setdefault((r['source'], r['offset'], r['variant']), []).append(r)
    branch_rows = []
    for key, rows in sorted(bg.items()):
        count = sum(r['repetition'] for r in rows)
        branch_rows.append(dict(source=key[0], offset=key[1], variant=key[2], n_M=len(rows),
                                loops_M=count, rate_D=count / len(rows), complete=len(rows) == args.continuations))
    study.atomic(args.out / 'summary.json', dict(originals=originals, branches=branch_rows,
                 historical_swift_no_spec_M={PREFIXES[0]: dict(loops=2, n=24), PREFIXES[1]: dict(loops=2, n=24)},
                 caveat='Swift-written selected histories; not an unconditional base-vs-Swift task comparison'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'build-ra-catch')
    ap.add_argument('--frozen-dir', type=Path, default=FROZEN)
    ap.add_argument('--phases', nargs='+', type=int, choices=[0, 1, 2], default=[0, 1, 2])
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--catch-n', type=int, default=64)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--length', type=int, default=8192)
    ap.add_argument('--branch-length', type=int, default=4096)
    ap.add_argument('--continuations', type=int, default=16)
    ap.add_argument('--per-boot', type=int, default=8)
    ap.add_argument('--budget', type=int, default=86400)
    ap.add_argument('--request-timeout', type=int, default=1800)
    ap.add_argument('--context', type=int, default=102400)
    ap.add_argument('--port', type=int, default=8096)
    ap.add_argument('--devices', default='0,1')
    ap.add_argument('--disk-cap', type=int, default=5_000_000_000)
    ap.add_argument('--tolerance', type=float, default=1e-6)
    ap.add_argument('--subset', type=Path, default=Path('/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json'))
    args = ap.parse_args()
    if min(args.n, args.catch_n, args.k, args.length, args.branch_length, args.continuations, args.per_boot, args.request_timeout) < 1 or args.budget < 60:
        ap.error('positive counts required; budget must reserve cleanup time')
    if args.disk_cap < 100_000_000 or args.disk_cap > 5_000_000_000 or args.tolerance <= 0:
        ap.error('disk cap must be between 100 MB and 5 GB; tolerance must be positive')
    if not args.out.resolve().is_relative_to(ROOT) or args.out.resolve() == ROOT:
        ap.error('output must be a subdirectory inside this worktree')
    fixtures = {p: json.loads((ROOT / 'codex/prefixes' / (p + '.json')).read_text()) for p in PREFIXES}
    plan = dict(run_order=args.phases, phase0_trials_D=2 * args.n, phase1_max_trials_D=2 * args.catch_n,
                phase1_stop=dict(loops_D=args.k, max_trials_D=args.catch_n, arms=['no-spec', 'S3']),
                phase2_max_continuations_D=2 * args.k * len(OFFSETS) * len(VARIANTS) * args.continuations,
                offsets_tokens_D=OFFSETS, variants=VARIANTS, continuation_tokens_D=args.branch_length,
                length_tokens_D=args.length, disk_cap_bytes_D=args.disk_cap, budget_seconds_D=args.budget,
                time_scenario_G=dict(phase0_hours=2, phase1_max_hours=5, phase2_max_hours=48),
                capture_source=PINS['source'], commands={a: study.command(Path('frozen/capture-srv'), 'IQ4_XS', a, args) for a in ('no-spec', 'S3')},
                replay='strict source-arm fidelity before every boundary; changed arms retain row drift',
                fidelity='same reconstructed study histories; base forced on Swift-written histories')
    if os.environ.get('RUN_PARENT_GPU') != '1' or os.environ.get('PLAN', '1') != '0':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print(json.dumps(plan, indent=2))
        print('PLAN: no GPU query, server, subprocess, or HTTP request executed')
        return
    gate()
    args.out.mkdir(parents=True, exist_ok=True)
    owner = (args.out / 'owner.lock').open('w')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    held = []
    for name in ('qwen-hillclimb.lock', 'qwen-local-server.lock'):
        path = Path(f'/run/user/{os.getuid()}') / name
        if path.exists():
            handle = path.open('r')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held.append(handle)
    deadline = time.monotonic() + args.budget - 30
    def watchdog(*_):
        size = sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file())
        if time.monotonic() >= deadline:
            raise TimeoutError('wall budget exhausted; resume identical settings')
        if size >= args.disk_cap - 20_000_000 or shutil.disk_usage(args.out).free < 100_000_000:
            raise RuntimeError('disk cap/free-space guard; retain partial evidence and stop')
    signal.signal(signal.SIGALRM, watchdog)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.setitimer(signal.ITIMER_REAL, 1, 1)
    try:
        freeze(args.frozen_dir, args.out / 'frozen')
        study.atomic(args.out / 'BASE-METADATA.json', verify_base())
        settings = dict(n=args.n, catch_n=args.catch_n, k=args.k, length=args.length, branch_length=args.branch_length,
                        continuations=args.continuations, context=args.context, per_boot=args.per_boot,
                        tolerance=args.tolerance, disk_cap=args.disk_cap, devices=args.devices,
                        models={str(p): study.digest(p) for p in (BASE, Path(study.MODELS['IQ4_XS']))},
                        subset=study.digest(args.subset), template=study.digest(ROOT / 'codex/prefixes/chat-template.jinja'),
                        source=PINS, fixtures={p: hashlib.sha256(json.dumps(f, sort_keys=True).encode()).hexdigest() for p, f in fixtures.items()},
                        drivers={name: study.digest(ROOT / 'codex' / name) for name in ('catch.py', 'catch_replay.py', 'onset.py', 'experiment.py', 'runaways.py')})
        manifest = args.out / 'RUN.json'
        if manifest.exists() and json.loads(manifest.read_text()) != settings:
            raise RuntimeError('resume inputs changed; use a new output directory')
        study.atomic(manifest, settings)
        study.atomic(args.out / 'PLAN.json', plan)
        for name in ('trials', 'branches'):
            (args.out / name).mkdir(exist_ok=True)
        for phase in args.phases:
            originals(args, fixtures, phase) if phase in (0, 1) else branches(args)
        summarize(args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for handle in held:
            handle.close()
        owner.close()


if __name__ == '__main__':
    main()
