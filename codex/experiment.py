#!/usr/bin/env python3
"""Parent-only bounded continuation experiment; no GPU access in plan mode."""
import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from runaways import ci, spans

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent
MODELS = {
    'IQ4_XS': '/home/xanewok/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf',
    'J5q': '/mnt/scratch/work/models/j5/Swift-1.5-Qwen3.8-27B-J5q-bf16.gguf',
    'Q5_K_S': '/mnt/scratch/work/models/gguf/Swift-1.5-Qwen3.8-27B-Q5_K_S.gguf',
    'Q6_K': '/home/xanewok/models/Swift-1.5-Qwen3.8-27B-Q6_K.gguf',
}
ARMS = {
    'no-spec': (0, False, 'q8_0', '1'),
    'S0': (3, False, 'q8_0', '1'),
    'S3': (3, True, 'q8_0', '1'),
    'S5': (5, True, 'q8_0', '1'),
    'S5-f16': (5, True, 'f16', '1'),
    'S5-convert': (5, True, 'q8_0', '0'),
    'S4': (4, True, 'q8_0', '1'),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def seeds(quant, arm, prefix, index, build):
    key = f'runaway-v1:{quant}:{arm}:{prefix}:{index}:{build}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little') % (2**32 - 1)


def schedule(args, prefixes):
    cells = [(q, a, 'stack2') for q in args.quants for a in args.arms]
    if args.up_dir:
        cells += [(q, 'S0', 'up') for q in args.quants]
    plan = []
    for block, start in enumerate(range(0, args.n, args.per_boot)):
        order = cells[block % len(cells):] + cells[:block % len(cells)]
        if block % 2:
            order.reverse()
        for quant, arm, build in order:
            trials = []
            for prefix in prefixes:
                for i in range(start, min(start + args.per_boot, args.n)):
                    seed = seeds(quant, arm, prefix['id'], i, build)
                    trials.append(dict(prefix=prefix['id'], index=i, seed=seed,
                                       id=f'{build}-{quant}-{arm}-{prefix["id"]}-{i:04}'))
            plan.append(dict(block=block, quant=quant, arm=arm, build=build, trials=trials))
    return plan


def command(binary, quant, arm, args):
    k, subset, kv, _ = ARMS[arm]
    cmd = [str(binary), '-m', MODELS[quant], '-np', '1', '-fa', 'on', '-ctk', kv, '-ctv', kv,
           '-c', str(args.context), '-b', '2048', '-ub', '128', '-sm', 'tensor', '--tensor-split', '50,50',
           '-ngl', '99', '--fit', 'off', '-t', '8', '-tb', '8', '--host', '127.0.0.1', '--port', str(args.port),
           '--metrics', '--jinja', '--temp', '1', '--top-k', '20', '--top-p', '0.95', '--min-p', '0.05',
           '--backend-sampling', '--chat-template-file', str(HERE / 'prefixes/chat-template.jinja')]
    if k:
        cmd += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', str(k), '--spec-draft-n-min', '0',
                '--spec-draft-p-min', '0', '--spec-draft-sampling', 'probabilistic', '--spec-draft-backend-sampling',
                '--spec-draft-type-k', 'f16', '--spec-draft-type-v', 'f16']
    else:
        cmd += ['--spec-type', 'none']
    if subset:
        cmd += ['--spec-draft-vocab', str(args.subset)]
    return cmd


def verified_manifest(directory):
    p = directory / 'SHA256SUMS'
    if not p.is_file():
        raise RuntimeError('missing frozen manifest: ' + str(p))
    entries = {}
    for line in p.read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        source = (directory / name).resolve()
        if not source.is_relative_to(directory.resolve()):
            raise RuntimeError('manifest escapes frozen directory')
        if digest(source) != sha:
            raise RuntimeError('frozen hash mismatch: ' + str(source))
        entries[name] = sha
    if 'llama-server' not in entries:
        raise RuntimeError('server missing from frozen manifest')
    for source in directory.glob('*.so*'):
        if source.name not in entries:
            raise RuntimeError('unmanifested library: ' + str(source))
    return entries


def freeze(source, dest):
    entries = verified_manifest(source)
    dest.mkdir(parents=True, exist_ok=True)
    own = {}
    for name, sha in entries.items():
        if name != 'llama-server' and '.so' not in name:
            continue
        output_name = 'ra-continuation-srv' if name == 'llama-server' else Path(name).name
        target = dest / output_name
        if not target.exists():
            shutil.copy2(source / name, target, follow_symlinks=True)
        if digest(target) != sha:
            raise RuntimeError('local frozen bytes changed: ' + str(target))
        own[output_name] = sha
    manifest = ''.join(f'{sha}  {name}\n' for name, sha in sorted(own.items()))
    (dest / 'SHA256SUMS').write_text(manifest)
    return own


def http(args, path, data=None, timeout=10):
    request = urllib.request.Request('http://127.0.0.1:' + str(args.port) + path,
        data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    # Local requests must not inherit a proxy that could redirect them off-host.
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout)


def post(args, path, data, timeout=10):
    with http(args, path, data, timeout) as response:
        return json.load(response)


def prepare_prefix(args, fixture):
    if fixture.get('token_ids'):
        ids = fixture['token_ids']
        if not all(isinstance(t, int) for t in ids):
            raise RuntimeError('invalid token IDs')
        return ids, fixture.get('sampler_history_ids', []), fixture.get('request_overrides', {})
    body = dict(messages=fixture['messages'], tools=fixture['tools'], reasoning_effort='xhigh')
    prompt = post(args, '/apply-template', body)['prompt'] + fixture['continuation_text']
    ids = post(args, '/tokenize', dict(content=prompt, add_special=True, parse_special=True))['tokens']
    return ids, [], fixture.get('request_overrides', {})


def request_body(ids, trial, args, overrides):
    body = dict(prompt=ids, n_predict=args.length, seed=trial['seed'], stream=True,
                cache_prompt=False, temperature=1.0, top_k=20, top_p=.95, min_p=.05,
                repeat_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0, dry_multiplier=0.0,
                return_tokens=True)
    body.update(overrides)
    for key in ['prompt', 'n_predict', 'seed', 'stream', 'cache_prompt', 'return_tokens']:
        if body[key] != dict(prompt=ids, n_predict=args.length, seed=trial['seed'], stream=True,
                             cache_prompt=False, return_tokens=True)[key]:
            raise RuntimeError('override changes controlled request field: ' + key)
    return body


def collect(lines, trial, length):
    text = ''
    ids, final = [], None
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode()
        if not line.startswith('data: '):
            continue
        raw = line[6:].strip()
        if raw == '[DONE]':
            continue
        item = json.loads(raw)
        if 'error' in item:
            raise RuntimeError(str(item['error']))
        text += item.get('content', '')
        ids.extend(item.get('tokens', []))
        if item.get('stop'):
            final = item
    if final is None:
        raise RuntimeError('incomplete SSE: trial is censored, not a negative')
    predicted = final.get('timings', {}).get('predicted_n')
    if predicted is None:
        raise RuntimeError('missing completion token count')
    return dict(trial, output_tokens_M=predicted, content=text, token_ids=ids,
                repetition=bool(spans(text)), evidence=spans(text), final=final,
                bounded_length=length, limit_reached=predicted >= length)


def xid():
    p = subprocess.run(['journalctl', '-k', '-b', '--no-pager'], capture_output=True, text=True, timeout=10)
    if p.returncode:
        raise RuntimeError('cannot read kernel Xid log')
    return [line for line in p.stdout.splitlines() if 'Xid' in line or 'fallen off' in line]


def idle(args):
    if xid():
        raise RuntimeError('Xid/fallen-off exists in this boot')
    p = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10)
    selected = set(args.devices.split(','))
    rows = {x[0].strip(): x for x in (line.split(',') for line in p.stdout.splitlines())}
    for device in selected:
        if device not in rows or int(rows[device][1]) > 750 or int(rows[device][2]) > 5:
            raise RuntimeError('selected GPU missing or busy: ' + device)
    if subprocess.run(['pgrep', '-x', 'llama-server'], stdout=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError('another llama-server is active')
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1', args.port)) == 0:
            raise RuntimeError('port is occupied')


def stop(process):
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def summarize(out, plan):
    from scipy.stats import fisher_exact

    rows = []
    for boot in plan:
        group = []
        for t in boot['trials']:
            p = out / 'trials' / (t['id'] + '.json')
            if p.exists():
                group.append(json.loads(p.read_text()))
        # Summaries per prefix/arm aggregate below, without counting failed requests.
        rows.extend(dict(r, arm=boot['arm'], quant=boot['quant'], build=boot['build']) for r in group)
    groups = {}
    for r in rows:
        groups.setdefault((r['build'], r['quant'], r['arm'], r['prefix']), []).append(r)
    summary = []
    for key, group in sorted(groups.items()):
        count = sum(r['repetition'] for r in group)
        summary.append(dict(build=key[0], quant=key[1], arm=key[2], prefix=key[3],
                            n_M=len(group), loops_M=count, ci_95_D=ci(count, len(group))))
    atomic(out / 'summary.json', summary)
    expected = {}
    for boot in plan:
        for t in boot['trials']:
            key = (boot['build'], boot['quant'], boot['arm'], t['prefix'])
            expected[key] = expected.get(key, 0) + 1
    prefixes = sorted({t['prefix'] for b in plan for t in b['trials']})
    comparisons = []
    for prefix in prefixes:
        for left, right in [('IQ4_XS:S5', 'IQ4_XS:no-spec'), ('J5q:S5', 'J5q:no-spec'), ('IQ4_XS:S5', 'J5q:S5')]:
            q0, a0 = left.split(':')
            q1, a1 = right.split(':')
            k0, k1 = ('stack2', q0, a0, prefix), ('stack2', q1, a1, prefix)
            if k0 in expected and k1 in expected:
                comparisons.append((prefix, left, right, k0, k1))
    family_complete = all(len(groups.get(k, [])) == expected[k] for *_, k0, k1 in comparisons for k in [k0, k1])
    tests = []
    for prefix, left, right, k0, k1 in comparisons:
        g0, g1 = groups.get(k0, []), groups.get(k1, [])
        x0, x1 = sum(r['repetition'] for r in g0), sum(r['repetition'] for r in g1)
        p = float(fisher_exact([[x0, len(g0) - x0], [x1, len(g1) - x1]]).pvalue) if family_complete else None
        alpha = .05 / len(comparisons)
        tests.append(dict(prefix=prefix, left=left, right=right, family_complete=family_complete,
                          p_two_sided_D=p, family_alpha_D=alpha,
                          reject=None if p is None else p <= alpha))
    atomic(out / 'primary-tests.json', tests)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=WORKTREE / 'build-runaway-ab')
    ap.add_argument('--frozen-dir', type=Path, default=Path('/home/xanewok/repos/qwen38-weekend/frozen/stack2/build/bin'))
    ap.add_argument('--up-dir', type=Path)
    ap.add_argument('--prefix-dir', type=Path, default=HERE / 'prefixes')
    ap.add_argument('--quants', nargs='+', choices=MODELS, default=['IQ4_XS', 'J5q'])
    ap.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS)[:6])
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--length', type=int, default=8192)
    ap.add_argument('--per-boot', type=int, default=8)
    ap.add_argument('--budget', type=int, default=7200)
    ap.add_argument('--request-timeout', type=int, default=900)
    ap.add_argument('--context', type=int, default=102400)
    ap.add_argument('--port', type=int, default=8096)
    ap.add_argument('--devices', default='0,1')
    ap.add_argument('--subset', type=Path, default=Path('/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json'))
    ap.add_argument('--exploratory', action='store_true', help='permit reconstructed prefixes; results cannot prove production exactness')
    args = ap.parse_args()
    if min(args.n, args.length, args.per_boot, args.budget, args.request_timeout) <= 0:
        ap.error('budgets and counts must be positive')
    if args.budget < 30:
        ap.error('wall budget must leave at least 20 seconds for cleanup')
    if not args.out.resolve().is_relative_to(WORKTREE):
        ap.error('output must stay inside this worktree')
    prefixes = [json.loads(p.read_text()) for p in sorted(args.prefix_dir.glob('*.json')) if p.name != 'manifest.json']
    prefixes = [p for p in prefixes if p.get('primary')]
    if not prefixes:
        ap.error('no primary prefix fixtures')
    plan = schedule(args, prefixes)
    live = os.environ.get('RUN_PARENT_GPU') == '1' and os.environ.get('PLAN', '1') == '0'
    if not live:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        args.out.mkdir(parents=True, exist_ok=True)
        atomic(args.out / 'dry-plan.json', dict(boots_D=len(plan), trials_D=sum(len(b['trials']) for b in plan),
            maximum_output_tokens_D=sum(len(b['trials']) for b in plan) * args.length,
            available_models_M={q: Path(p).is_file() for q, p in MODELS.items()},
            exact_prefixes_M=all(p['exact_prefix'] for p in prefixes), plan=plan))
        print('PLAN: no GPU query, process, or HTTP request executed')
        print(f'D: {len(plan)} boots; {sum(len(b["trials"]) for b in plan)} trials; L={args.length}; budget={args.budget}s per invocation')
        return
    if not args.exploratory and any(not p['exact_prefix'] for p in prefixes):
        ap.error('original request/token/grammar state is missing; use --exploratory for the reconstructed study')
    for p in prefixes:
        if p['exact_prefix'] and (not p.get('token_ids') or not p.get('exactness_evidence') or not p.get('state_fidelity_verified')):
            ap.error('exact prefixes need token_ids, exactness_evidence, and state_fidelity_verified')
    args.out.mkdir(parents=True, exist_ok=True)
    work_budget = args.budget - 20
    deadline = time.monotonic() + work_budget
    def budget_expired(*_):
        raise TimeoutError('invocation wall budget reached; resume the identical command')
    signal.signal(signal.SIGALRM, budget_expired)
    signal.setitimer(signal.ITIMER_REAL, work_budget)
    own_dirs = {'stack2': args.out / 'frozen/stack2'}
    frozen = {'stack2': freeze(args.frozen_dir, own_dirs['stack2'])}
    if args.up_dir:
        own_dirs['up'] = args.out / 'frozen/up'
        frozen['up'] = freeze(args.up_dir, own_dirs['up'])
    settings = dict(n=args.n, length=args.length, context=args.context, per_boot=args.per_boot,
        frozen=frozen, models={q: digest(MODELS[q]) for q in args.quants}, subset=digest(args.subset),
        template=digest(HERE / 'prefixes/chat-template.jinja'),
        driver=digest(__file__), detector=digest(HERE / 'runaways.py'),
        prefixes={p['id']: hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest() for p in prefixes},
        plan=plan, exploratory=args.exploratory, devices=args.devices)
    manifest_path = args.out / 'RUN.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != settings:
        raise RuntimeError('resume configuration changed; use a new output directory')
    atomic(manifest_path, settings)
    (args.out / 'trials').mkdir(exist_ok=True)
    (args.out / 'logs').mkdir(exist_ok=True)
    fixtures = {p['id']: p for p in prefixes}
    process = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for boot_no, boot in enumerate(plan):
            pending = [t for t in boot['trials'] if not (args.out / 'trials' / (t['id'] + '.json')).exists()]
            if not pending:
                continue
            if deadline - time.monotonic() < 60:
                print('Budget reached; resume with the identical command')
                break
            idle(args)
            cmd = command(own_dirs[boot['build']] / 'ra-continuation-srv', boot['quant'], boot['arm'], args)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.devices, GGML_CUDA_FA_MMA_Q8_0=ARMS[boot['arm']][3],
                       LD_LIBRARY_PATH=str(own_dirs[boot['build']]), OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1')
            suffix = str(time.time_ns())
            with (args.out / 'logs' / f'{boot_no}-{suffix}.log').open('w') as log:
                process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                ready = min(deadline, time.monotonic() + 300)
                while time.monotonic() < ready:
                    if process.poll() is not None:
                        raise RuntimeError('server exited before health')
                    try:
                        with http(args, '/health', timeout=2) as response:
                            if json.load(response).get('status') == 'ok':
                                break
                    except (OSError, ValueError):
                        pass
                    time.sleep(1)
                else:
                    raise RuntimeError('health timeout')
                with http(args, '/slots', timeout=2) as response:
                    slots = json.load(response)
                if len(slots) != 1 or slots[0].get('n_ctx') != args.context:
                    raise RuntimeError('actual slot context differs from registered context')
                for trial in pending:
                    remaining = deadline - time.monotonic()
                    if remaining < 10:
                        break
                    fixture = fixtures[trial['prefix']]
                    ids, history, overrides = prepare_prefix(args, fixture)
                    if history:
                        raise RuntimeError('sampler state import is unsupported; add an audited replay helper')
                    if len(ids) + args.length >= args.context:
                        raise RuntimeError('prefix plus continuation exceeds context')
                    atomic(args.out / 'logs' / f'{trial["id"]}.prefix.json', dict(token_ids=ids, fixture=fixture['id']))
                    body = request_body(ids, trial, args, overrides)
                    # A watchdog bounds the whole operation, including a stream that keeps delivering bytes.
                    def timed_out(*_):
                        raise TimeoutError('trial exceeded wall budget')
                    signal.signal(signal.SIGALRM, timed_out)
                    signal.setitimer(signal.ITIMER_REAL, min(args.request_timeout, remaining))
                    try:
                        with http(args, '/completion', body, min(args.request_timeout, remaining)) as response:
                            result = collect(response, trial, args.length)
                    finally:
                        signal.signal(signal.SIGALRM, budget_expired)
                        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
                    result.update(exact_prefix=fixture['exact_prefix'], grammar=overrides.get('grammar'),
                                  source_build=boot['build'], request=body)
                    atomic(args.out / 'trials' / (trial['id'] + '.json'), result)
            stop(process)
            process = None
            if xid():
                raise RuntimeError('Xid after boot')
            summarize(args.out, plan)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        stop(process)
        summarize(args.out, plan)


if __name__ == '__main__':
    main()
