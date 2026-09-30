#!/usr/bin/env python3
"""Parent-only frozen A/B; default mode prints a plan without running tools."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
SP = Path('/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad')
ORDER = ('up', 'patch', 'patch', 'up')
COUNTERS = ('spec_decode_num_drafts_total', 'spec_decode_num_draft_tokens_total', 'spec_decode_num_accepted_tokens_total')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def manifest(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def verify(directory, expected):
    require(manifest(directory) == expected, f'Frozen bytes or file set changed: {directory}')


def freeze(source, dest, executable):
    require(Path(executable).name == executable, 'Executable must be a basename')
    source_id = (source / 'SOURCE').read_text().splitlines()[0]
    require(re.fullmatch('[0-9a-f]{40}', source_id), 'SOURCE must start with a full commit hash')
    entries = {}
    for line in (source / 'bin' / 'SHA256SUMS').read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64}) [ *](.+)', line)
        require(match is not None, 'Unsupported SHA256SUMS line')
        digest, name = match.groups()
        rel = Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe SHA256SUMS path')
        require(sha(source / 'bin' / rel) == digest, f'Source hash failed: {name}')
        entries[name.removeprefix('./')] = digest
    require(executable in entries, 'Executable missing from SHA256SUMS')
    # All shipped library bytes must be frozen, including dereferenced symlinks.
    for file in (source / 'bin').rglob('*'):
        if file.is_file() and '.so' in file.name:
            require(str(file.relative_to(source / 'bin')) in entries, f'Library missing from SHA256SUMS: {file}')
    dest.mkdir(parents=True)
    shutil.copy2(source / 'SOURCE', dest / 'SOURCE')
    shutil.copytree(source / 'bin', dest / 'bin', symlinks=False)
    require((dest / 'SOURCE').read_text().splitlines()[0] == source_id, 'SOURCE changed during copy')
    for name, expected in entries.items():
        require(sha(dest / 'bin' / name) == expected, f'Copied hash failed: {name}')
    renamed = dest / 'bin' / 'draft-loop-ab'
    original = dest / 'bin' / executable
    if original != renamed:
        original.rename(renamed)
    require(os.access(renamed, os.X_OK), 'Frozen executable is not executable')
    sums = ''.join(f'{sha(p)}  {p.relative_to(dest / "bin")}\n' for p in sorted((dest / 'bin').rglob('*')) if p.is_file() and p.name != 'SHA256SUMS')
    (dest / 'bin' / 'SHA256SUMS').write_text(sums)
    expected = manifest(dest)
    return renamed, expected


def arm_env(raw):
    values = json.loads(raw)
    require(isinstance(values, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in values.items()), 'Env must be an object of strings')
    protected = {'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LD_PRELOAD'}
    require(not protected.intersection(values), 'Arm env cannot replace GPU/thread/library controls')
    return values


def stop_owned(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def canonical(choices):
    normalized = []
    for choice in choices:
        message = dict(choice['message'])
        if message.get('tool_calls'):
            message['tool_calls'] = [{k: v for k, v in c.items() if k != 'id'} for c in message['tool_calls']]
        normalized.append((message, choice.get('finish_reason')))
    return normalized


def summarize(records, ks, seeds):
    output = []
    for k in ks:
        for seed in seeds:
            for kind in ('tool', 'think'):
                rows = [r for r in records if (r['k'], r['seed'], r['kind']) == (k, seed, kind)]
                boots = {(r['arm'], r['boot']) for r in rows}
                require(boots == {('up', 0), ('patch', 1), ('patch', 2), ('up', 3)}, 'Incomplete interleaved A/B')
                reference = rows[0]
                for row in rows:
                    require(canonical(row['choices']) == canonical(reference['choices']), f'Completion identity failed: k={k}, seed={seed}, {kind}')
                    require(row['counters'] == reference['counters'], 'Draft/acceptance/round identity failed')
                times = {}
                for arm, boot in boots:
                    group = [r for r in rows if (r['arm'], r['boot']) == (arm, boot)]
                    times[boot] = sum(r['timings']['predicted_ms'] for r in group) / sum(r['counters'][COUNTERS[0]] for r in group)
                deltas = [times[1] - times[0], times[2] - times[3]]
                output.append(f'D: k={k}, seed={seed}, {kind}: patch-up {statistics.mean(deltas):.6f} ms/round; paired deltas {deltas}')
    return '\n'.join(output) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--up', type=Path)
    ap.add_argument('--patch', type=Path)
    ap.add_argument('--up-exe', default='llama-server')
    ap.add_argument('--patch-exe', default='llama-server')
    ap.add_argument('--up-env', default='{}')
    ap.add_argument('--patch-env', default='{}')
    ap.add_argument('--patch-chain', action='store_true')
    ap.add_argument('--k', type=int, nargs='+', default=[3])
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--model', type=Path, default=Path('/home/xanewok/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf'))
    ap.add_argument('--model-bytes', type=int, default=15475951296)
    ap.add_argument('--template', type=Path)
    ap.add_argument('--vocab', type=Path)
    ap.add_argument('--out', type=Path, default=ROOT / 'codex/gpu-ab-out')
    ap.add_argument('--port', type=int, default=8096)
    ap.add_argument('--budget-s', type=int, default=1200)
    ap.add_argument('--min-rounds', type=int, default=20)
    args = ap.parse_args()
    require(120 < args.budget_s <= 2400, 'Budget must be in (120,2400] seconds')
    require(args.min_rounds > 0 and args.k and set(args.k) <= {3, 4, 5}, 'Invalid round count or k')
    require(len(set(args.k)) == len(args.k) and len(set(args.seeds)) == len(args.seeds) and args.seeds, 'Duplicate or empty cases')
    plan = dict(order=ORDER, k=args.k, seeds=args.seeds, budget_s=args.budget_s,
                up=str(args.up), patch=str(args.patch), patch_chain=args.patch_chain,
                up_env=arm_env(args.up_env), patch_env=arm_env(args.patch_env),
                mode='execute' if args.execute else 'plan', min_rounds_per_case=args.min_rounds,
                metric='D: generation wall ms / M: actual spec_decode_num_drafts_total delta',
                logger='--log-verbosity 3; library INFO is level 4',
                contract='same subset, target+draft backend candidates, seed, messages and counters')
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    require(os.environ.get('RUN_GPU_AB') == '1' and os.environ.get('CUDA_VISIBLE_DEVICES'), 'Parent-only execute guard')
    require(all(p is not None and p.is_dir() for p in (args.up, args.patch)), 'Supply frozen --up and --patch roots')
    require(all(p is not None and p.is_file() for p in (args.model, args.template, args.vocab)), 'Supply model/template/vocab')
    require(args.model.stat().st_size == args.model_bytes, 'Model size changed')
    require(args.out.resolve().is_relative_to(ROOT) and not args.out.exists(), 'OUT must be new and inside this worktree')
    server = None
    records = []
    begin = time.monotonic()

    def aborted(signum, frame):
        raise RuntimeError(f'Interrupted or wall budget exhausted: signal {signum}')

    for sig in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, aborted)
    signal.setitimer(signal.ITIMER_REAL, args.budget_s - 30)

    def remaining():
        return max(.01, args.budget_s - 30 - (time.monotonic() - begin))

    def run(command):
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=min(15, remaining()))

    def xid():
        log = run(['journalctl', '-k', '-b', '--no-pager']).stdout
        require(not re.search(r'Xid|fallen off', log, re.I), 'Xid/fallen-off in this boot')

    def idle():
        for name in ('llama-server', 'draft-loop-ab', 'chain-ab', 'vocab-ab'):
            checked = subprocess.run(['pgrep', '-x', name], capture_output=True, timeout=min(5, remaining()))
            require(checked.returncode == 1, f'Process already present or pgrep failed: {name}')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', args.port))
        text = run(['nvidia-smi', '--id=' + os.environ['CUDA_VISIBLE_DEVICES'], '--query-gpu=memory.used', '--format=csv,noheader,nounits']).stdout
        used = sum(int(x) for x in text.split())
        require(used <= 1500, f'Selected GPUs busy: {used} MiB')

    def http(path, body=None):
        require(server is not None and server.poll() is None, 'Owned server exited')
        request = urllib.request.Request(f'http://127.0.0.1:{args.port}{path}',
                    data=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=min(180, remaining())) as response:
            data = response.read().decode()
        return data if path == '/metrics' else json.loads(data)

    def metrics():
        values = {}
        for line in http('/metrics').splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0].removeprefix('llamacpp:') in COUNTERS:
                values[parts[0].removeprefix('llamacpp:')] = float(parts[1])
        require(set(values) == set(COUNTERS), 'Missing actual speculative round counters')
        return values

    try:
        idle()
        xid()
        args.out.mkdir(parents=True)
        (args.out / 'plan.json').write_text(json.dumps(plan, indent=2))
        frozen = {arm: freeze(src.resolve(), args.out / 'frozen' / arm, exe)
                  for arm, src, exe in [('up', args.up, args.up_exe), ('patch', args.patch, args.patch_exe)]}
        smoke_path = SP / 'prefilter2/codex_smoke.py'
        inputs = {str(p.resolve()): sha(p) for p in (args.model, args.template, args.vocab, smoke_path)}
        (args.out / 'inputs.json').write_text(json.dumps(inputs, indent=2))
        spec = importlib.util.spec_from_file_location('codex_smoke', smoke_path)
        smoke = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke)
        conversations = dict(smoke.conversations())
        for k in args.k:
            for boot, arm in enumerate(ORDER):
                require(remaining() > 60, 'Insufficient budget; incomplete A/B')
                idle()
                xid()
                executable, expected = frozen[arm]
                directory = executable.parent.parent
                verify(directory, expected)
                require(all(sha(Path(p)) == digest for p, digest in inputs.items()), 'Frozen input changed')
                env = dict(os.environ, LD_LIBRARY_PATH=str(executable.parent), OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8')
                env.update(plan[arm + '_env'])
                command = [str(executable), '-m', str(args.model), '-t', '8', '-tb', '8', '-np', '1', '-fa', 'on', '-fit', 'off',
                    '--spec-type', 'draft-mtp', '--spec-draft-n-max', str(k), '--spec-draft-n-min', '0', '--spec-draft-p-min', '0',
                    '--spec-draft-sampling', 'probabilistic', '--spec-draft-backend-sampling', '--backend-sampling', '--spec-draft-vocab', str(args.vocab),
                    '-b', '2048', '-ub', '128', '--jinja', '--temp', '1', '--top-k', '20', '--top-p', '.95', '--min-p', '.05',
                    '--host', '127.0.0.1', '--port', str(args.port), '-c', '32768', '-ctk', 'q8_0', '-ctv', 'q8_0', '-ngl', '99',
                    '--spec-draft-type-k', 'f16', '--spec-draft-type-v', 'f16', '-sm', 'tensor', '--tensor-split', '50,50',
                    '--chat-template-file', str(args.template), '--metrics', '--log-verbosity', '3', '--log-colors', 'off']
                if arm == 'patch' and args.patch_chain:
                    command.append('--spec-mtp-chain')
                stem = f'k{k}-{boot}-{arm}'
                (args.out / f'{stem}.command.json').write_text(json.dumps(command))
                with (args.out / f'{stem}.server.log').open('w') as log:
                    server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    deadline = time.monotonic() + min(90, remaining())
                    while True:
                        require(time.monotonic() < deadline and server.poll() is None, 'Server did not become healthy')
                        try:
                            if http('/health').get('status') == 'ok':
                                break
                        except (OSError, ValueError):
                            pass
                        time.sleep(.5)
                    def request(kind, seed):
                        return http('/v1/chat/completions', dict(messages=conversations['fresh' if kind == 'warmup' else 'follow'],
                            tools=smoke.TOOLS, tool_choice='auto', seed=seed, cache_prompt=False,
                            max_tokens=128 if kind == 'warmup' else 384 if kind == 'tool' else 1024,
                            chat_template_kwargs={'enable_thinking': kind != 'tool'}))
                    request('warmup', 1)
                    for seed in args.seeds:
                        for kind in ('tool', 'think'):
                            rounds = 0
                            for repeat in range(10):
                                before = metrics()
                                response = request(kind, seed)
                                after = metrics()
                                counters = {name: after[name] - before[name] for name in COUNTERS}
                                require(counters[COUNTERS[0]] > 0, 'No speculative rounds')
                                require(counters[COUNTERS[1]] == response['timings']['draft_n'] and counters[COUNTERS[2]] == response['timings']['draft_n_accepted'], 'Metric/response disagreement')
                                (args.out / f'{stem}-s{seed}-{kind}-{repeat}.json').write_text(json.dumps(response, indent=2))
                                records.append(dict(arm=arm, boot=boot, k=k, seed=seed, kind=kind, repeat=repeat,
                                                    counters=counters, timings=response['timings'], choices=response['choices']))
                                rounds += counters[COUNTERS[0]]
                                if rounds >= args.min_rounds:
                                    break
                            require(rounds >= args.min_rounds, 'Too few measured rounds')
                    memory = run(['nvidia-smi', '--id=' + os.environ['CUDA_VISIBLE_DEVICES'], '--query-gpu=uuid,memory.used', '--format=csv,noheader']).stdout
                    (args.out / f'{stem}.memory.csv').write_text(memory)
                    stop_owned(server)
                    server = None
                log_text = (args.out / f'{stem}.server.log').read_text()
                require(not re.search(r'backend offload failed|MTP chain unavailable|backend sampling.*using CPU', log_text, re.I), 'Fallback invalidates the arm')
                if arm == 'patch' and args.patch_chain:
                    require('MTP subset chain enabled' in log_text, 'No common-logger chain enablement marker')
                xid()
        require(all(sha(Path(p)) == digest for p, digest in inputs.items()), 'Input bytes changed')
        for arm, (exe, expected) in frozen.items():
            verify(exe.parent.parent, expected)
        summary = summarize(records, args.k, args.seeds)
        (args.out / 'summary.txt').write_text(summary)
        print(summary, end='')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        stop_owned(server)
        if args.out.exists():
            (args.out / 'records.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
