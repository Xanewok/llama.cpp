#!/usr/bin/env python3
"""Parent-only capture campaign and continuations. Plan mode does not spawn or query."""
import argparse
import bisect
import codecs
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import replay

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / 'codex/frozen/capture/build/bin'
UP = Path('/home/xanewok/repos/qwen38-weekend/frozen/stack2/build/bin')
MODEL = Path('/home/xanewok/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf')
IDS = Path('/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json')
TEMPLATE = Path('/home/xanewok/repos/ra-rates/stack/codex/prefixes/chat-template.jinja')
DETECTOR = Path('/home/xanewok/repos/ra-rates/stack/codex/runaways.py')
ARMS = {'no-spec': (0, False, 'q8_0', '1'), 'S0': (3, False, 'q8_0', '1'),
        'S3': (3, True, 'q8_0', '1'), 'S4': (4, True, 'q8_0', '1'), 'S5': (5, True, 'q8_0', '1'),
        'S4-f16': (4, True, 'f16', '1'), 'S4-convert': (4, True, 'q8_0', '0')}


def execution_gate():
    if os.environ.get('RUN_PARENT_GPU') != '1' or os.environ.get('PLAN', '1') != '0':
        raise RuntimeError('GPU execution requires RUN_PARENT_GPU=1 PLAN=0')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen(directory, server):
    entries = {}
    for line in (directory / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip('*')
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != digest:
            raise RuntimeError('frozen hash mismatch/escape: ' + name)
        entries[name] = digest
    if server not in entries or 'llama-server' not in entries or entries[server] != entries['llama-server']:
        raise RuntimeError('renamed executable is not pinned to server bytes')
    for library in directory.glob('*.so*'):
        if library.name not in entries:
            raise RuntimeError('unmanifested library: ' + str(library))
    return entries


def xid():
    execution_gate()
    p = subprocess.run(['journalctl', '-k', '-b', '--no-pager'], capture_output=True, text=True, timeout=10, check=True)
    if not p.stdout.strip() or any(word in p.stdout for word in ('Xid', 'fallen off')):
        raise RuntimeError('empty kernel journal or Xid/fallen-off found')


def idle(port):
    execution_gate()
    xid()
    for process in Path('/proc').iterdir():
        if process.name.isdigit():
            try:
                if (process / 'comm').read_text().strip() == 'llama-server':
                    raise RuntimeError('existing llama-server')
            except (FileNotFoundError, ProcessLookupError):
                pass
    p = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, timeout=10, check=True)
    rows = [[int(x.strip()) for x in line.split(',')] for line in p.stdout.splitlines()]
    if len(rows) != 2 or sum(row[0] for row in rows) >= 1500 or any(row[1] > 5 for row in rows):
        raise RuntimeError('GPUs are busy: ' + p.stdout.strip())
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                       capture_output=True, text=True, timeout=10, check=True)
    for line in p.stdout.splitlines():
        name = line.split(',', 1)[-1].strip().lower()
        if line.strip() and not any(word in name for word in ('1password', 'cosmic-', 'xdg-desktop-portal')):
            raise RuntimeError('another compute process: ' + line)
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError('port is busy')


def locks():
    execution_gate()
    handles = []
    try:
        for name in ('qwen-hillclimb.lock', 'qwen-local-server.lock'):
            handle = (Path(f'/run/user/{os.getuid()}') / name).open('r')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles.append(handle)
        return handles
    except BaseException:
        for handle in handles:
            handle.close()
        raise


def stop(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def command(binary, arm, port):
    k, subset, kv, _ = ARMS[arm]
    cmd = [str(binary), '-m', str(MODEL), '-c', '102400', '-np', '1', '-ctk', kv, '-ctv', kv,
           '-fa', 'on', '-ngl', '99', '-sm', 'tensor', '--tensor-split', '50,50', '-b', '2048', '-ub', '128',
           '-t', '8', '-tb', '8', '--fit', 'off', '--no-context-shift', '--jinja',
           '--chat-template-file', str(TEMPLATE), '--host', '127.0.0.1', '--port', str(port),
           '--backend-sampling', '--temp', '1', '--top-k', '20', '--top-p', '.95', '--min-p', '.05', '--seed', '1729']
    if k:
        cmd += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', str(k), '--spec-draft-n-min', '0',
                '--spec-draft-p-min', '0', '--spec-draft-sampling', 'probabilistic',
                '--spec-draft-backend-sampling', '--spec-draft-type-k', 'f16', '--spec-draft-type-v', 'f16']
    else:
        cmd += ['--spec-type', 'none']
    if subset:
        cmd += ['--spec-draft-vocab', str(IDS)]
    return cmd


def boot(arm, directory, deadline, up=False, capture=True):
    execution_gate()
    source = UP if up else FROZEN
    name = 'weekend-srv' if up else 'capture-srv'
    verify_frozen(source, name)
    idle(18128)
    directory.mkdir(parents=True, exist_ok=False)
    captures = directory / 'captures'
    captures.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', LLAMA_CAPTURE_DIR=str(captures), LLAMA_CAPTURE_PROBS='1',
               LLAMA_CAPTURE_LIMIT='40000', GGML_CUDA_FA_MMA_Q8_0=ARMS[arm][3],
               OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', TMPDIR=str(ROOT / 'build-tmp'))
    if not capture:
        for key in ('LLAMA_CAPTURE_DIR', 'LLAMA_CAPTURE_PROBS', 'LLAMA_CAPTURE_LIMIT'):
            env.pop(key, None)
    cmd = command(source / name, arm, 18128)
    (directory / 'launch.json').write_text(json.dumps(dict(command=cmd, env={k: env[k] for k in
        ('CUDA_VISIBLE_DEVICES', 'GGML_CUDA_FA_MMA_Q8_0')}), indent=2))
    log = (directory / 'server.log').open('w')
    child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        boot_end = min(deadline, time.time() + 300)
        while time.time() < boot_end:
            if child.poll() is not None:
                raise RuntimeError('server died: ' + str(directory))
            try:
                req = urllib.request.Request('http://127.0.0.1:18128/props')
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=2) as response:
                    props = json.load(response)
                if props.get('default_generation_settings', {}).get('n_ctx') != 102400:
                    raise RuntimeError('slot context mismatch')
                return child, log, captures
            except urllib.error.HTTPError as error:
                if error.code != 503:
                    raise
            except OSError:
                pass
            time.sleep(.5)
        raise RuntimeError('server boot timed out')
    except BaseException:
        stop(child)
        log.close()
        xid()
        raise


def onset(path):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(DETECTOR.parent))
    from runaways import spans
    try:
        _, _, tokens = replay.load(path)
    except (ValueError, KeyError, json.JSONDecodeError):
        return None
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    text, ends = '', []
    for token in tokens:
        text += decoder.decode(bytes.fromhex(token['piece_hex']))
        ends.append(len(text))
    matches = spans(text)
    if not matches:
        return None
    match = min(matches, key=lambda m: m['start'])
    token_index = bisect.bisect_right(ends, match['start'])
    position = token_index - 2000
    if position < 1:
        return None
    return dict(capture=str(path.resolve()), onset_token_M=token_index,
                branch_position_D=position, detector=match)


def scan(directory):
    for path in sorted(directory.rglob('request-*.jsonl')):
        found = onset(path)
        if found:
            return found
    return None


def bounded(child, deadline, watch=None):
    try:
        while child.poll() is None and time.time() < deadline:
            if watch and watch():
                stop(child)
                return 0
            time.sleep(5)
        if child.poll() is None:
            stop(child)
            return None
        return child.returncode
    finally:
        stop(child)


def smoke(args, deadline):
    rows = []
    for i, up in enumerate((True, False, False, True)):
        if time.time() + 30 >= deadline:
            raise RuntimeError('smoke budget exhausted')
        child, log, _ = boot('S4', args.out / f'ab-{i}', deadline, up=up, capture=not up)
        try:
            started = time.monotonic()
            result = replay.post('http://127.0.0.1:18128', '/completion', dict(
                prompt='Count from zero to one hundred.', n_predict=48, temperature=0,
                seed=1729, cache_prompt=False, return_tokens=True), timeout=max(1, deadline-time.time()))
            rows.append(dict(arm='up' if up else 'capture', tokens=result['tokens'],
                             wall_seconds_M=time.monotonic()-started, final=result))
        finally:
            stop(child)
            log.close()
            xid()
    if any(row['tokens'] != rows[0]['tokens'] for row in rows):
        raise RuntimeError('capture A/B changed greedy emitted IDs')
    (args.out / 'ab-smoke.json').write_text(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'build-parent-capture')
    parser.add_argument('--budget', type=int, default=7200)
    parser.add_argument('--capture-budget', type=int, default=3600)
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--length', type=int, default=2048)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=['S0', 'S4', 'no-spec'])
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--ab-smoke', action='store_true', default=True)
    parser.add_argument('--skip-ab-smoke', action='store_false', dest='ab_smoke')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if min(args.budget, args.capture_budget, args.n, args.length) <= 0 or args.budget <= 30:
        parser.error('positive counts and a cleanup budget are required')
    if not math.isfinite(args.tolerance) or not 0 <= args.tolerance <= .01:
        parser.error('tolerance must be finite and within [0, .01]')
    if args.length > 40000:
        parser.error('branch length exceeds response cap')
    plan = dict(tasks=['rust/alphametics', 'rust/nucleotide-codons'], capture_arm='S4',
                response_cap_D=40000, before_onset_tokens_D=2000, arms=args.arms,
                continuations_per_arm_D=args.n, length_D=args.length, budget_seconds_D=args.budget,
                capture_budget_seconds_D=args.capture_budget, probability_tolerance_D=args.tolerance, interleaved_ab=['up', 'capture', 'capture', 'up'])
    if args.plan or os.environ.get('PLAN', '1') != '0' or os.environ.get('RUN_PARENT_GPU') != '1':
        print(json.dumps(plan, indent=2))
        return
    execution_gate()
    if not args.out.resolve().is_relative_to(ROOT):
        raise RuntimeError('output must be inside this worktree')
    args.out.mkdir(parents=True, exist_ok=False)
    deadline = time.time() + args.budget - 30
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    pins = dict(capture=verify_frozen(FROZEN, 'capture-srv'), model=sha(MODEL), subset=sha(IDS),
                template=sha(TEMPLATE), detector=sha(DETECTOR), drivers={p.name: sha(p) for p in ROOT.joinpath('codex').glob('*.py')})
    (args.out / 'PLAN.json').write_text(json.dumps(plan, indent=2))
    (args.out / 'PINS.json').write_text(json.dumps(pins, indent=2))
    if args.ab_smoke:
        held = locks()
        try:
            smoke(args, deadline)
        finally:
            for handle in held:
                handle.close()
    found = onset(args.capture) if args.capture else None
    capture_end = min(deadline, time.time() + args.capture_budget)
    iteration = 0
    while found is None and time.time() + 30 < capture_end and not args.capture:
        idle(8081)
        campaign = args.out / f'campaign-{iteration}'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', TMPDIR=str(ROOT / 'build-tmp'))
        with (args.out / f'campaign-{iteration}.log').open('w') as log:
            child = subprocess.Popen(['nice', '-n', '19', sys.executable, str(ROOT / 'codex/capture_campaign.py'),
                '--out', str(campaign), '--deadline', str(capture_end - 20)], env=env, stdout=log, stderr=subprocess.STDOUT)
            code = bounded(child, capture_end - 20, lambda: scan(campaign))
        xid()
        found = scan(campaign)
        iteration += 1
        if code not in (0, None) and found is None:
            # Preserve the partial campaign and report failure; a transport fault is not a negative.
            (args.out / 'CENSORED.json').write_text(json.dumps(dict(reason='campaign failed', code=code)))
            raise RuntimeError('campaign failed; inspect log before retry')
    if found is None:
        (args.out / 'CENSORED.json').write_text(json.dumps(dict(reason='no replayable onset captured before budget')))
        print('No replayable onset captured; no continuation experiment ran.')
        return
    (args.out / 'ONSET.json').write_text(json.dumps(found, indent=2))
    source = Path(found['capture'])
    position = found['branch_position_D']
    if position + args.length > 40000:
        raise RuntimeError('prefix plus branch exceeds environment cap')
    held = locks()
    # Locks stay open through all continuation boots and close on process exit.
    # Establish fidelity under the capture arm before allowing numerical drift in changed arms.
    child, log, captures = boot('S4', args.out / 'fidelity-S4', deadline)
    try:
        replay.request_capture('http://127.0.0.1:18128', captures, replay.body(source, position, 1, tolerance=args.tolerance),
                               timeout=max(1, deadline-time.time()))
    finally:
        stop(child)
        log.close()
        xid()
    completed = []
    # Rotate/reverse arm order across boots; each seed gets an independent history reconstruction.
    for i in range(args.n):
        order = args.arms[i % len(args.arms):] + args.arms[:i % len(args.arms)]
        if i % 2:
            order.reverse()
        for arm in order:
            if time.time() + 30 >= deadline:
                (args.out / 'CENSORED.json').write_text(json.dumps(dict(reason='branch budget exhausted', completed=completed)))
                return
            directory = args.out / f'branch-{i}-{arm}'
            child, log, captures = boot(arm, directory, deadline)
            try:
                seed = int.from_bytes(hashlib.sha256(f'{sha(source)}:{arm}:{i}'.encode()).digest()[:4], 'little') % (2**32-1)
                request = replay.body(source, position, args.length, seed, tolerance=args.tolerance, allow_drift=arm != 'S4')
                # The capture validates every retained prefix row and the branch's next row before emitting it.
                result, path, tokens = replay.request_capture('http://127.0.0.1:18128', captures, request,
                                                             timeout=max(1, deadline-time.time()))
                from runaways import spans
                suffix = replay.text(tokens[position:])
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                drift = [e['index'] for e in rows if e['event'] == 'validation' and not e['equal']]
                value = dict(arm=arm, seed=seed, capture=str(path), numeric_drift_rows_M=drift, ids=[t['id'] for t in tokens[position:]],
                             repetition=spans(suffix), output_tokens_M=len(tokens)-position, final=result)
                (directory / 'result.json').write_text(json.dumps(value, indent=2))
                completed.append(directory.name)
            except BaseException as error:
                (directory / 'CENSORED.json').write_text(json.dumps(dict(reason=str(error))))
                raise
            finally:
                stop(child)
                log.close()
                xid()
    (args.out / 'COMPLETE.json').write_text(json.dumps(dict(completed=completed), indent=2))


if __name__ == '__main__':
    main()
