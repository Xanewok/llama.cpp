#!/usr/bin/env python3
"""Parent-only A/B for future frozen V2 overlays; no candidate is assumed to exist."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ORDER = ('up', 'patch', 'patch', 'up', 'up', 'patch', 'patch', 'up')


def within(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE.parent):
        raise ValueError('Writable artifacts and frozen snapshots must be inside this worktree')
    return path


def frozen(descriptor):
    path = within(descriptor)
    config = json.loads(path.read_text())
    manifest = config['sha256']
    if not manifest or not config.get('argv') or not config.get('env'):
        raise ValueError('Descriptor needs argv, env and a nonempty sha256 mapping')
    for name, digest in manifest.items():
        target = within(path.parent / name)
        if hashlib.file_digest(target.open('rb'), 'sha256').hexdigest() != digest:
            raise ValueError(f'Frozen bytes changed: {target}')
    # The executable may be the shared Python runtime but must be pinned too.
    executable = Path(config['argv'][0]).resolve()
    actual = hashlib.file_digest(executable.open('rb'), 'sha256').hexdigest()
    if actual != config['python_sha256']:
        raise ValueError('Python executable changed')
    if executable.name == 'llama-server' or any(Path(a).name == 'llama-server' for a in config['argv']):
        raise ValueError('A process named llama-server is forbidden')
    return config


def request(url, body=None, timeout=5):
    data = json.dumps(body).encode() if body is not None else None
    headers = {'Content-Type': 'application/json'} if data is not None else {}
    with urlopen(Request(url, data=data, headers=headers), timeout=timeout) as response:
        return response.read().decode()


def xid():
    result = subprocess.run(['journalctl', '-k', '-b', '--no-pager'], capture_output=True, text=True, check=True)
    if 'Xid' in result.stdout or 'fallen off' in result.stdout:
        raise RuntimeError('Xid/fallen-off in kernel log')


def idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True)
    if result.stdout.strip():
        raise RuntimeError('GPU compute process already running')


def metrics(text):
    out = {}
    for line in text.splitlines():
        if line and not line.startswith('#'):
            fields = line.rsplit(' ', 1)
            if len(fields) == 2:
                name = fields[0].split('{')[0]
                try:
                    out[name] = out.get(name, 0.) + float(fields[1])
                except ValueError:
                    pass
    return out


def decode_result(before, after):
    seconds = after.get('vllm:request_decode_time_seconds_sum', 0) - before.get('vllm:request_decode_time_seconds_sum', 0)
    tokens = after.get('vllm:generation_tokens_total', 0) - before.get('vllm:generation_tokens_total', 0)
    if seconds <= 0 or tokens <= 0:
        raise RuntimeError('Missing/nonpositive decode metrics; wall latency is not a decode result')
    return {'decode_s_M': seconds, 'tokens_M': tokens, 'decode_tps_D': tokens/seconds}


def settled_metrics(url, min_count, deadline):
    limit = min(deadline, time.monotonic() + 30)
    while time.monotonic() < limit:
        result = metrics(request(url + '/metrics'))
        if result.get('vllm:request_decode_time_seconds_count', 0) >= min_count:
            return result
        time.sleep(1)
    raise TimeoutError('Completed-request metrics did not reach the expected count')


def stop(process):
    # This group was created by this Popen, never discovered with a process-name pattern.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='store_true')
    p.add_argument('--up', type=Path)
    p.add_argument('--patch', type=Path)
    p.add_argument('--cases', type=Path)
    p.add_argument('--cases-sha256')
    p.add_argument('--out', type=Path, default=HERE / 'ab-runs')
    p.add_argument('--port', type=int, default=8097)
    p.add_argument('--budget', type=int, default=2400)
    args = p.parse_args()
    if not args.run:
        print('Parent-only plan:', ' '.join(ORDER))
        print('No active candidate delivered: replay did not clear +5% on current k=5 arms.')
        print('Real run needs frozen up/patch descriptors and a checksummed request corpus; see REPORT.md.')
        return
    if os.environ.get('PARENT_GPU_RUN') != '1' or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        p.error('--run requires PARENT_GPU_RUN=1 and explicitly selected CUDA_VISIBLE_DEVICES')
    if not all((args.up, args.patch, args.cases, args.cases_sha256)):
        p.error('Missing frozen descriptors or request corpus hash')
    descriptors = {'up': args.up, 'patch': args.patch}
    configs = {arm: frozen(path) for arm, path in descriptors.items()}
    if configs['up']['argv'] == configs['patch']['argv'] and configs['up']['env'] == configs['patch']['env']:
        raise ValueError('Arms have identical launch settings')
    case_path = within(args.cases)
    if hashlib.file_digest(case_path.open('rb'), 'sha256').hexdigest() != args.cases_sha256:
        raise ValueError('Request corpus changed')
    cases = json.loads(case_path.read_text())
    if not cases or any(case.get('stream') or 'seed' not in case for case in cases):
        raise ValueError('Cases must be nonstreaming completion requests with fixed seeds')
    out = within(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'cases.json').write_bytes(case_path.read_bytes())
    for arm, path in descriptors.items():
        (out / (arm + '.json')).write_bytes(path.read_bytes())
    url = f'http://127.0.0.1:{args.port}'
    deadline = time.monotonic() + args.budget
    rows = []
    for i, arm in enumerate(ORDER):
        xid()
        idle()
        try:
            request(url + '/health')
        except Exception:
            pass
        else:
            raise RuntimeError('A server already owns the selected port')
        config = frozen(descriptors[arm])
        if config != configs[arm]:
            raise ValueError('Launch descriptor changed since preflight')
        env = dict(os.environ, **config['env'])
        env.update(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8',
                   VLLM_USE_V2_MODEL_RUNNER='1', TOKENIZERS_PARALLELISM='false',
                   PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   XDG_CACHE_HOME=str(out / 'cache'), VLLM_CACHE_ROOT=str(out / 'cache/vllm'),
                   TRITON_CACHE_DIR=str(out / 'cache/triton'), CUDA_CACHE_PATH=str(out / 'cache/cuda'),
                   TORCHINDUCTOR_CACHE_DIR=str(out / 'cache/inductor'), TMPDIR=str(out / 'tmp'))
        (out / 'tmp').mkdir(exist_ok=True)
        if env.get('CUDA_VISIBLE_DEVICES') != os.environ['CUDA_VISIBLE_DEVICES']:
            raise ValueError('Descriptor must not replace the parent-selected devices')
        with (out / f'{i}-{arm}.server.log').open('w') as log:
            process = subprocess.Popen(config['argv'], env=env, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=out, start_new_session=True)
            try:
                healthy = False
                for _ in range(600):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('A/B wall budget exhausted')
                    if process.poll() is not None:
                        raise RuntimeError('Server exited during boot')
                    try:
                        request(url + '/health', timeout=1)
                        healthy = True
                        break
                    except Exception:
                        time.sleep(1)
                if not healthy:
                    raise TimeoutError('Server boot exceeded 600 seconds')
                initial = metrics(request(url + '/metrics'))
                request(url + '/v1/completions', dict(cases[0], max_tokens=64), timeout=min(120, deadline-time.monotonic()))
                before = settled_metrics(url, initial.get('vllm:request_decode_time_seconds_count', 0) + 1, deadline)
                for j, case in enumerate(cases):
                    response = request(url + '/v1/completions', case, timeout=min(240, deadline-time.monotonic()))
                    (out / f'{i}-{arm}-case{j}.json').write_text(response)
                after = settled_metrics(url, before['vllm:request_decode_time_seconds_count'] + len(cases), deadline)
                row = dict(decode_result(before, after), arm=arm, boot=i)
                rows.append(row)
                (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
            finally:
                stop(process)
        xid()
        for _ in range(30):
            try:
                idle()
                break
            except RuntimeError:
                time.sleep(1)
        else:
            raise RuntimeError('Owned server workers did not exit')
    paired = []
    for i in range(0, len(rows), 2):
        pair = {r['arm']: r for r in rows[i:i+2]}
        paired.append(100*(pair['patch']['decode_tps_D']/pair['up']['decode_tps_D']-1))
    (out / 'summary.json').write_text(json.dumps({'paired_speed_pct_D': paired}, indent=2) + '\n')
    print('Paired decode speed changes (D):', paired)


if __name__ == '__main__':
    main()
