#!/usr/bin/env python3
"""Queued replay runner. --preflight and --stub never execute GPU programs."""
import argparse
import hashlib
import http.client
import json
import math
import os
import random
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent
PLAN = (('base', 5), ('subset', 5), ('subset', 6), ('base', 6), ('base', 7), ('subset', 7),
        ('subset', 7), ('base', 7), ('base', 6), ('subset', 6), ('subset', 5), ('base', 5))
GPUS = ('GPU-ee9a3956-297c-604b-b4c1-f07930d27e46', 'GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f')
MIN_DECODE = 16
SAMPLING = dict(temperature=1.0, top_k=20, top_p=0.95, min_p=0.05)


class Failure(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def timing(rows):
    """Cumulative SSE counters are snapshots, never increments to sum."""
    candidates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        t = row.get('timings')
        if not isinstance(t, dict):
            continue
        names = ('predicted_n', 'predicted_ms', 'predicted_per_second', 'draft_n', 'draft_n_accepted')
        if any(type(t.get(n)) not in (int, float) or not math.isfinite(t[n]) for n in names):
            continue
        if t['predicted_n'] < 2 or t['predicted_ms'] <= 0 or t['predicted_per_second'] <= 0:
            continue
        if not 0 <= t['draft_n_accepted'] <= t['draft_n'] or t['draft_n'] <= 0:
            continue
        candidates.append(t)
    if not candidates:
        return None
    return max(candidates, key=lambda t: (t['predicted_n'], t['predicted_ms']))


def hazard(text):
    return bool(re.search(r'\bXid\b|fallen off the bus', text))


def gpu_memory(text):
    rows = {}
    for line in text.splitlines():
        fields = [s.strip() for s in line.split(',')]
        if not line.strip() or fields[0] == 'index':
            continue
        if len(fields) != 4:
            raise Failure(6, 'invalid GPU memory record')
        used, total = (int(s.removesuffix(' MiB')) for s in fields[2:])
        if fields[1] in rows or not 0 <= used <= total:
            raise Failure(6, 'invalid GPU memory bounds')
        rows[fields[1]] = used
    if set(rows) != set(GPUS):
        raise Failure(6, 'wrong selected GPUs')
    return rows


def argv(config, arm, k, port):
    args = list(config['argv'])
    if arm == 'subset':
        args += ['--spec-draft-vocab', config['vocab']]
    else:
        assert arm == 'base'
    args[args.index('--spec-draft-n-max') + 1] = str(k)
    args[args.index('--port') + 1] = str(port)
    return [config['binaries'][arm]] + args


def verify(config):
    for path, expected in config['hashes'].items():
        if not Path(path).is_file() or sha(path) != expected:
            raise Failure(6, 'frozen input changed: ' + path)
    canonical = argv(config, 'base', 5, 8080)[1:]
    for arm, k in PLAN:
        command = argv(config, arm, k, 8080)
        if Path(command[0]).name != 'round-trace':
            raise Failure(6, 'forbidden server process name')
        command = command[1:]
        if arm == 'subset':
            command = command[:-2]
        command[command.index('--spec-draft-n-max') + 1] = '5'
        if command != canonical:
            raise Failure(6, 'served argv drift')


def bootstrap(values, clusters=None):
    rng = random.Random(9001)
    n = len(values)
    if not n:
        raise Failure(5, 'no complete paired prefixes')
    groups = {}
    for i, value in enumerate(values):
        groups.setdefault(clusters[i] if clusters else i, []).append(value)
    samples = sorted(statistics.median([v for group in rng.choices(list(groups.values()), k=len(groups))
                                       for v in group]) for _ in range(10000))
    return dict(median_D=statistics.median(values), ci95_D=[samples[249], samples[9749]],
                prefixes=len(values), source_clusters=len(groups))


def summary(rows, prefixes, canary=False):
    failures = [r for r in rows if not r.get('valid')]
    groups = {}
    for r in rows:
        if r.get('valid'):
            groups.setdefault((r['arm'], r['k'], r['prefix']), []).append(r)
    arms = (('subset', 7),) if canary else sorted(set(PLAN))
    controls, seeded_mismatch = {}, []
    for arm, k in arms:
        ratios, clusters = [], []
        for prefix in prefixes:
            group = groups.get((arm, k, prefix), [])
            if len(group) != 2:
                failures.append(dict(arm=arm, k=k, prefix=prefix, error='need two valid repeats'))
                continue
            if group[0]['prompt_sha256'] != group[1]['prompt_sha256']:
                failures.append(dict(arm=arm, k=k, prefix=prefix, error='prefix tokens changed'))
            if group[0]['content_sha256'] != group[1]['content_sha256']:
                seeded_mismatch.append(dict(arm=arm, k=k, prefix=prefix))
            ratios.append(group[1]['tok_s_M'] / group[0]['tok_s_M'])
            clusters.append(group[0]['session'])
        if ratios:
            control = bootstrap(ratios, clusters)
            control['equivalence_margin_D'] = [0.98, 1.02]
            control['equivalent'] = 0.98 <= control['ci95_D'][0] <= control['ci95_D'][1] <= 1.02
            controls[f'{arm}-k{k}'] = control
            if not control['equivalent']:
                failures.append(dict(arm=arm, k=k, error='repeat CI is not contained in equivalence margin'))
    if failures:
        return dict(status='INVALID', failures=failures, control=controls)
    rates = {key: statistics.median(r['tok_s_M'] for r in group) for key, group in groups.items()}
    comparisons, acceptance = {}, {}
    if not canary:
        clusters = [groups['base', 5, p][0]['session'] for p in prefixes]
        for arm, k in arms:
            comparisons[f'{arm}-k{k}/base-k5'] = bootstrap(
                [rates[arm, k, p] / rates['base', 5, p] for p in prefixes], clusters)
        for k in (5, 6, 7):
            comparisons[f'subset-k{k}/base-k{k}'] = bootstrap(
                [rates['subset', k, p] / rates['base', k, p] for p in prefixes], clusters)
            comparisons[f'subset-k{k}/subset-k5'] = bootstrap(
                [rates['subset', k, p] / rates['subset', 5, p] for p in prefixes], clusters)
        for p in prefixes:
            hashes = {r['prompt_sha256'] for r in rows if r['prefix'] == p}
            if len(hashes) != 1:
                return dict(status='INVALID', reason='cross-arm prefix identity failed')
    for arm, k in arms:
        kr = [r for r in rows if r['arm'] == arm and r['k'] == k]
        drafted, accepted = sum(r['drafted_M'] for r in kr), sum(r['accepted_M'] for r in kr)
        acceptance[f'{arm}-k{k}'] = dict(drafted_M=drafted, accepted_M=accepted, fraction_D=accepted / drafted)
    return dict(status='ok', control=controls, ratios_D=comparisons, acceptance=acceptance,
                seeded_mismatch_M=seeded_mismatch, per_prefix_M=rows, steady_ms_per_round_M=None,
                steady_note='No steady-round logger enabled; decode rate excludes prefill.')


class Runner:
    def __init__(self, args):
        self.a = args
        self.config = json.loads((DATA / f'{args.model}.json').read_text())
        self.corpus = json.loads((DATA / 'prefixes.json').read_text())
        self.bounds = json.loads((DATA / 'prefix_bounds.json').read_text())
        if self.bounds['corpus_sha256'] != sha(DATA / 'prefixes.json'):
            raise Failure(6, 'prefix calibration corpus changed')
        self.out = Path(args.out).resolve()
        if not self.out.is_relative_to(ROOT):
            raise Failure(6, 'output must be in this worktree')
        self.out.mkdir(parents=True, exist_ok=True)
        if (self.out / 'results.jsonl').exists():
            raise Failure(6, 'fresh output directory required')
        self.deadline = time.monotonic() + args.budget_s
        self.child = None
        self.pid_start = None
        self.rows = []

    def left(self, cap=300, reserve=25):
        left = self.deadline - time.monotonic() - reserve
        if left <= 0:
            raise Failure(7, 'budget exhausted')
        return min(cap, left)

    def command(self, args, cap=60):
        try:
            return subprocess.run(args, check=True, capture_output=True, text=True, timeout=self.left(cap)).stdout
        except subprocess.TimeoutExpired as e:
            raise Failure(7, 'budget or stage cap exhausted') from e

    def check_hazard(self):
        if self.a.stub:
            return
        since = os.environ.get('XID_SINCE')
        if not since:
            raise Failure(6, 'missing queue journal cutoff')
        try:
            journal = self.command(['journalctl', '-k', '-b', '--since', since, '--no-pager'])
        except subprocess.CalledProcessError as e:
            raise Failure(6, 'unreadable kernel journal') from e
        (self.out / 'kernel-journal.txt').write_text(journal)
        if hazard(journal):
            raise Failure(3, 'kernel GPU hazard since cutoff')

    def idle(self):
        self.check_hazard()
        if not self.a.stub:
            while True:
                raw = self.command(['nvidia-smi', '--id=' + ','.join(GPUS),
                    '--query-gpu=index,uuid,memory.used,memory.total', '--format=csv,noheader,nounits'])
                if sum(gpu_memory(raw).values()) <= 1500:
                    break
                self.left()
                time.sleep(1)
                self.check_hazard()
        try:
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('127.0.0.1', self.a.port))
        except OSError as e:
            raise Failure(4, 'replay port busy') from e

    def http(self, path, body=None, stream=False, cap=180):
        limit = self.left(cap)
        conn = http.client.HTTPConnection('127.0.0.1', self.a.port, timeout=limit)
        previous_alarm = signal.getsignal(signal.SIGALRM)
        def expired(signum, frame):
            raise Failure(7, 'budget or HTTP stage cap exhausted')
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, limit)
        try:
            conn.request('POST' if body is not None else 'GET', path,
                None if body is None else json.dumps(body), {'Content-Type': 'application/json'})
            response = conn.getresponse()
            if response.status != 200:
                raw = response.read().decode(errors='replace')
                raise Failure(8, f'HTTP {response.status}: {raw}')
            if not stream:
                return json.loads(response.read())
            chunks = []
            while True:
                conn.sock.settimeout(self.left(cap)) if conn.sock else None
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    break
                try:
                    row = json.loads(data)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    chunks.append(row)
            return chunks
        except (socket.timeout, TimeoutError) as e:
            reason = 'budget exhausted' if self.deadline - time.monotonic() < 30 else 'request stage cap exhausted'
            raise Failure(7, reason) from e
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)
            conn.close()

    def stop(self):
        child = self.child
        if child is None:
            return
        try:
            current = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
            if child.poll() is None and current == self.pid_start:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        except FileNotFoundError:
            child.wait(timeout=5)
        self.child = None

    def boot(self, arm, k, boot):
        self.arm = arm
        self.config['binary'] = self.config['binaries'][arm]
        self.idle()
        env = dict(os.environ, OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8',
                   PYTHONDONTWRITEBYTECODE='1')
        for name in list(env):
            if name.startswith(('GGML_', 'LLAMA_')) or name in ('LD_PRELOAD', 'LD_LIBRARY_PATH'):
                del env[name]
        env.update(GGML_SCHED_INPUT_STAGING='1', LD_LIBRARY_PATH=str(Path(self.config['binary']).parent))
        env['TMPDIR'] = str(self.out / 'tmp')
        env['CUDA_CACHE_PATH'] = str(self.out / 'cuda-cache')
        Path(env['TMPDIR']).mkdir(exist_ok=True)
        command = argv(self.config, arm, k, self.a.port)
        if self.a.stub:
            env['CUDA_VISIBLE_DEVICES'] = ''
            command = [sys.executable, str(DATA / 'stub.py'), '--port', str(self.a.port), '--k', str(k), '--arm', arm]
        else:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(GPUS)
        write(self.out / f'boot-{boot}.argv.json', dict(argv=command, served_argv=argv(self.config, arm, k, self.a.port),
            env={key: env[key] for key in ('CUDA_VISIBLE_DEVICES', 'GGML_SCHED_INPUT_STAGING', 'LD_LIBRARY_PATH')}))
        with (self.out / f'boot-{boot}.server.log').open('w') as log:
            self.child = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
        self.pid_start = Path(f'/proc/{self.child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        write(self.out / f'boot-{boot}.pid.json', dict(pid=self.child.pid, start=self.pid_start))
        end = time.monotonic() + self.left(180)
        while time.monotonic() < end:
            if self.child.poll() is not None:
                raise Failure(8, 'server exited before health')
            try:
                if self.http('/health', cap=2).get('status') == 'ok':
                    return
            except (OSError, Failure):
                pass
            time.sleep(0.5)
        raise Failure(7, 'boot stage cap exhausted')

    def prepare(self, prefix):
        body = dict(messages=prefix['messages'], tools=self.corpus['tools'],
                    chat_template_kwargs=dict(enable_thinking=True, reasoning_effort='xhigh', preserve_thinking=True))
        prompt = self.http('/apply-template', body)['prompt']
        ids = self.http('/tokenize', dict(content=prompt, add_special=True, parse_special=True))['tokens']
        baseline = self.bounds['prefixes'][prefix['id']]
        lower, upper = baseline['bounds_D']
        context_max = int(self.config['argv'][self.config['argv'].index('-c') + 1]) - 512
        n = len(ids) if isinstance(ids, list) else None
        types = sorted({type(t).__name__ for t in ids}) if isinstance(ids, list) else [type(ids).__name__]
        detail = (f"prefix={prefix['id']} length_M={n} element_types={types} bounds_D=[{lower},{upper}] "
                  f"context_max_exclusive={context_max} recorded_M={prefix['recorded_prompt_tokens_M']} "
                  f"cpu_rendered_M={baseline['rendered_prompt_tokens_M']}")
        if (baseline['recorded_prompt_tokens_M'] != prefix['recorded_prompt_tokens_M']
                or n is None or not all(type(t) is int for t in ids)
                or not lower <= n <= upper or n >= context_max):
            raise Failure(6, 'prefix length or token type invalid: ' + detail)
        write(self.out / f'{prefix["id"]}.tokens.json', dict(tokens=ids, prompt_sha256=object_sha(ids), n_M=n,
              recorded_prompt_tokens_M=prefix['recorded_prompt_tokens_M'],
              cpu_rendered_prompt_tokens_M=baseline['rendered_prompt_tokens_M'], bounds_D=[lower, upper],
              rendered_recorded_ratio_D=n / prefix['recorded_prompt_tokens_M']))
        return ids

    def one(self, prefix, ids, k, boot, length=256, warm=False):
        label = f'b{boot}-{prefix["id"]}' + ('-warm' if warm else '')
        body = dict(prompt=ids, seed=prefix['seed'], n_predict=length,
                    stream=True, cache_prompt=True, timings_per_token=True, **SAMPLING)
        write(self.out / f'{label}.request.json', body)
        row = dict(arm=self.arm, k=k, boot=boot, prefix=prefix['id'], session=prefix['session'], seed=prefix['seed'], prompt_tokens_M=len(ids),
                   prompt_sha256=object_sha(ids), valid=False)
        try:
            chunks = self.http('/completion', body, stream=True)
            write(self.out / f'{label}.response.json', dict(chunks=chunks))
            t = timing(chunks)
            if t is None or not any(c.get('stop') is True for c in chunks):
                raise Failure(5, 'missing final response or complete timings')
            final = next(c for c in reversed(chunks) if c.get('stop') is True)
            if not self.a.stub and final.get('model') != self.config['argv'][self.config['argv'].index('-m') + 1]:
                raise Failure(6, 'server model differs from frozen model')
            settings = final.get('generation_settings', {})
            if any(not math.isclose(settings.get(key, float('nan')), value, abs_tol=1e-6)
                   for key, value in SAMPLING.items()) or settings.get('seed') != prefix['seed']:
                raise Failure(6, 'served sampling settings differ')
            if final.get('truncated') or settings.get('ignore_eos') is not False or settings.get('n_predict') != length:
                raise Failure(6, 'length or EOS policy drift')
            if settings.get('backend_sampling') is not True or 'draft-mtp' not in settings.get('speculative.types', ''):
                raise Failure(6, 'MTP or target backend sampling disabled')
            content = ''.join(c['content'] for c in chunks if isinstance(c.get('content'), str))
            row.update(tokens_M=t['predicted_n'], tok_s_M=t['predicted_per_second'],
                       decode_ms_M=t['predicted_ms'], drafted_M=t['draft_n'], accepted_M=t['draft_n_accepted'],
                       content_sha256=hashlib.sha256(content.encode()).hexdigest(), timings=t)
            # EOS stays enabled, and real turns often end after 60-300 tokens (k-sweep-iq4-3, 2026-10-05), so the
            # floor only rejects degenerate replies; the per-prefix rate is a median over boots.
            if not warm and not MIN_DECODE <= t['predicted_n'] <= length:
                raise Failure(5, 'short replay: below the frozen prefix token minimum')
            if not warm and abs(t.get('cache_n', 0) + t.get('prompt_n', 0) - len(ids)) > 1:
                raise Failure(6, 'prompt truncated or prompt counters inconsistent')
            row['valid'] = True
        except Exception as e:
            row['error'] = str(e)
            row['code'] = e.code if isinstance(e, Failure) else 8
            write(self.out / f'{label}.error.json', dict(error=str(e)))
        if warm:
            write(self.out / f'{label}.warmup.json', row)
        else:
            self.rows.append(row)
            with (self.out / 'results.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
                f.flush()
        return row

    def run(self):
        if self.a.stub:
            if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
                raise Failure(6, 'CPU rehearsal requires hidden GPUs')
        elif not self.a.execute:
            raise Failure(6, 'GPU run requires parent --execute flag')
        else:
            verify(self.config)
        deepest = max(self.corpus['prefixes'], key=lambda p: p['recorded_prompt_tokens_M'])
        sessions = sorted({p['session'] for p in self.corpus['prefixes']})
        prefixes = [deepest if session == deepest['session'] else
                    min((p for p in self.corpus['prefixes'] if p['session'] == session),
                        key=lambda p: p['recorded_prompt_tokens_M']) for session in sessions]
        prefixes.sort(key=lambda p: p['recorded_prompt_tokens_M'])
        order = (('subset', 7),) if self.a.canary else PLAN
        write(self.out / 'identity.json', self.config)
        for boot, (arm, k) in enumerate(order):
            self.left()
            self.boot(arm, k, boot)
            try:
                prepared = {p['id']: self.prepare(p) for p in prefixes}
                if max(map(len, prepared.values())) < 75000:
                    raise Failure(6, 'no deep real-agent prefix')
                warm = self.one(prefixes[0], prepared[prefixes[0]['id']], k, boot, 8, True)
                if not warm['valid']:
                    raise Failure(warm['code'], 'warmup failed: ' + warm['error'])
                for p in prefixes:
                    for repeat in range(2 if self.a.canary else 1):
                        self.one(p, prepared[p['id']], k, boot * 2 + repeat if self.a.canary else boot)
                        self.check_hazard()
            finally:
                self.stop()
            self.check_hazard()
            invalid = [r for r in self.rows if not r['valid']]
            if invalid:
                write(self.out / 'summary.json', dict(status='INVALID', failures=invalid))
                raise Failure(max(r['code'] for r in invalid), 'request gate failed after boot')
        result = summary(self.rows, [p['id'] for p in prefixes], self.a.canary)
        write(self.out / 'summary.json', result)
        if result['status'] != 'ok':
            raise Failure(5, 'paired control or completeness gate failed')
        if not self.a.stub:
            verify(self.config)
            self.left()
        print(json.dumps(dict(status='ok', control=result['control'], ratios_D=result['ratios_D'])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=('j5q', 'iq4'), required=True)
    ap.add_argument('--port', type=int, default=int(os.environ.get('PORT', '8092')))
    ap.add_argument('--out', default=os.environ.get('OUT', str(ROOT / 'runs/manual')))
    ap.add_argument('--budget-s', type=int, default=3000)
    ap.add_argument('--canary', action='store_true')
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    args = ap.parse_args()
    runner = None
    try:
        if args.preflight:
            if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
                raise Failure(6, 'CPU preflight requires hidden GPUs')
            verify(json.loads((DATA / f'{args.model}.json').read_text()))
            tests = subprocess.run([sys.executable, str(DATA / 'test_sweep.py')], cwd=ROOT)
            return tests.returncode
        runner = Runner(args)
        def interrupted(signum, frame):
            raise Failure(7, 'queue interrupted run')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        runner.run()
        write(runner.out / 'verdict.json', dict(status='ok', rc=0))
        return 0
    except Exception as e:
        code = e.code if isinstance(e, Failure) else 8
        print(f'FAIL {code}: {e}', file=sys.stderr)
        if runner:
            write(runner.out / 'verdict.json', dict(status='INVALID', rc=code, error=str(e)))
        return code
    finally:
        if runner:
            runner.stop()


if __name__ == '__main__':
    sys.exit(main())
