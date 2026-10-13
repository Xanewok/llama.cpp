#!/usr/bin/env python3
"""Served-flag health boot of one serve.sh profile on this tree's fresh build (bw-stack11-s2 window, health boots).

Contract: argv and env are serve.sh's own (`PROFILE=P PORT=N serve.sh command`, read at run time and recorded with its sha256; not
pinned, the owner edits serve.sh), with the executable replaced by build-cuda/bin/round-trace, LD_LIBRARY_PATH on build-cuda/bin and
the log-only envs of engagement.health() added; every GGML_*/LLAMA_* variable inherited from the caller is dropped. One short chat
completion per slot (-np 2: both at once), then the server must still be healthy, its log must carry the profile's engagement lines
and no fatal line, and it must exit on SIGTERM within STOP_S.

usage: health.py PROFILE OUT_DIR     env: ROOT PORT CUDA_VISIBLE_DEVICES XID_SINCE
       health.py --parse PORT        parse every profile's launch line (CPU only, no server); prints one JSON line per profile
       health.py --selftest          the log judge on synthetic logs: a complete log passes, a dropped line or a fatal line fails
exit: 0 pass | 3 Xid | 4 GPUs or port busy | 6 serve.sh output not understood | 8 boot, request, engagement or stop failed
"""
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engagement

SERVE = Path.home() / '.claude/skills/local-qwen-server/bin/serve.sh'
PROFILES = ('swift15-j5q', 'swift15-j5q-duo', 'swift15-iq4', 'swift15-iq4-duo')
BOOT_S, REQUEST_S, STOP_S = 420, 300, 60
PROMPT = 'Write a Rust function that returns the sum of a slice of i32, with one unit test.'
ASSIGN = re.compile(r'^[A-Z_][A-Z0-9_]*=')


class Fail(Exception):
    def __init__(self, code, reason):
        super().__init__(reason)
        self.code, self.reason = code, reason


def launch(profile, port):
    """serve.sh's launch line for PROFILE: {env, exe, args, serve_sha256}."""
    env = dict(os.environ, PROFILE=profile, PORT=str(port))
    p = subprocess.run(['bash', str(SERVE), 'command'], env=env, capture_output=True, text=True, timeout=60)
    if p.returncode:
        raise Fail(6, f'serve.sh command rc={p.returncode}: {p.stderr.strip()[-300:]}')
    words = shlex.split(p.stdout.strip())
    i = 0
    out_env = {}
    while i < len(words) and ASSIGN.match(words[i]):
        k, v = words[i].split('=', 1)
        out_env[k] = v
        i += 1
    if i >= len(words) or not words[i].startswith('/') or Path(words[i]).name not in ('llama-server', 'round-trace'):
        raise Fail(6, f'no server executable in serve.sh output: {p.stdout[:200]}')
    exe, args = words[i], words[i + 1:]
    opt = lambda name: args[args.index(name) + 1] if name in args else None
    if opt('--port') != str(port):
        raise Fail(6, f'serve.sh did not take PORT={port}: --port {opt("--port")}')
    if not opt('-m') or not Path(opt('-m')).is_file():
        raise Fail(6, f'model of {profile} missing: {opt("-m")}')
    if 'CUDA_VISIBLE_DEVICES' not in out_env:
        raise Fail(6, 'serve.sh sets no CUDA_VISIBLE_DEVICES')
    return dict(profile=profile, env=out_env, exe=exe, args=args, np=int(opt('-np') or 1),
                serve=str(SERVE), serve_sha256=hashlib.sha256(SERVE.read_bytes()).hexdigest())


def server_env(spec, bin_dir):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_'))}
    env.update(spec['env'])
    env.update(engagement.HEALTH_ENV)
    env['LD_LIBRARY_PATH'] = str(bin_dir)
    return env


def port_busy(port):
    with socket.socket() as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def gpu_used():
    q = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30)
    return [int(x) for x in q.stdout.split()] if q.returncode == 0 else None


def xid(since):
    p = subprocess.run(['journalctl', '-k', '--no-pager', '--since', since], capture_output=True, text=True, timeout=30)
    if p.returncode:
        raise Fail(6, 'kernel journal unreadable')
    return bool(re.search(r'\bXid\b|fallen off the bus', p.stdout))


def get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read().decode(errors='replace')


def healthy(port):
    try:
        status, body = get(f'http://127.0.0.1:{port}/health', 3)
        return status == 200 and '"ok"' in body
    except OSError:
        return False


def chat(port, seed, out):
    body = json.dumps(dict(messages=[dict(role='user', content=PROMPT)], max_tokens=96, seed=seed, stream=False)).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', body, {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_S) as r:
            text = r.read().decode(errors='replace')
        out.write_text(text)
        j = json.loads(text)
        n = j['usage']['completion_tokens']
        reason = j['choices'][0]['finish_reason']
        return None if n >= 1 and reason in ('stop', 'length') else f'seed {seed}: {n} tokens, finish {reason}'
    except (OSError, ValueError, KeyError, IndexError) as e:
        return f'seed {seed}: {type(e).__name__}: {e}'


def start_time(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def stop(p, birth):
    """SIGTERM to the server's own process group; rc and seconds to exit, or None if it needed SIGKILL."""
    if p.poll() is not None:
        return dict(rc=p.returncode, stop_s=0.0, killed=False, already_dead=True)
    if start_time(p.pid) != birth:
        raise Fail(6, 'owned PID starttime changed')
    t0 = time.monotonic()
    os.killpg(p.pid, signal.SIGTERM)
    try:
        rc = p.wait(timeout=STOP_S)
        return dict(rc=rc, stop_s=round(time.monotonic() - t0, 1), killed=False, already_dead=False)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait(timeout=30)
        return dict(rc=p.returncode, stop_s=None, killed=True, already_dead=False)


def judge(log, profile):
    h = engagement.health(profile)
    missing = [pat for pat in h['patterns'] if not re.search(pat, log, re.M)]
    fatal = sorted(set(m.group(0) for m in re.finditer(h['fatal'], log)))
    recorded = {c: [len(re.findall(pat, log, re.M)) for pat in pats] for c, pats in h['recorded'].items()}
    return dict(missing=missing, fatal=fatal, recorded_counts_M=recorded, gating=h['gating'])


def run(profile, out):
    root = Path(os.environ['ROOT'])
    port = int(os.environ['PORT'])
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    since = os.environ.get('XID_SINCE') or time.strftime('%Y-%m-%d %H:%M:%S')
    out.mkdir(parents=True, exist_ok=True)
    bin_dir = root / 'build-cuda/bin'
    spec = launch(profile, port)
    if set(spec['env']['CUDA_VISIBLE_DEVICES'].split(',')) != set(visible.split(',')):
        raise Fail(6, f'serve.sh devices {spec["env"]["CUDA_VISIBLE_DEVICES"]} are not the window GPUs {visible}')
    argv = [str(bin_dir / 'round-trace'), *spec['args']]
    env = server_env(spec, bin_dir)
    (out / 'launch.json').write_text(json.dumps(dict(spec, argv=argv, server_env={k: v for k, v in env.items()
                                                                                    if k.startswith(('GGML_', 'LLAMA_', 'CUDA_', 'LD_'))}), indent=2))
    if xid(since):
        raise Fail(3, 'Xid before the boot')
    used = gpu_used()
    if used is None or max(used) > 1500:
        raise Fail(4, f'GPUs hold {used} MiB before the boot')
    if port_busy(port):
        raise Fail(4, f'port {port} busy')
    result = dict(profile=profile, port=port)
    with open(out / 'server.log', 'w') as log:
        p = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    birth = start_time(p.pid)
    try:
        t0 = time.monotonic()
        while not healthy(port):
            if p.poll() is not None:
                raise Fail(8, f'server exited rc={p.returncode} before /health')
            if time.monotonic() - t0 > BOOT_S:
                raise Fail(8, f'not healthy within {BOOT_S} s')
            time.sleep(1)
        result['boot_s_M'] = round(time.monotonic() - t0, 1)
        errors = [None] * spec['np']
        threads = [threading.Thread(target=lambda i=i: errors.__setitem__(i, chat(port, 1729 + i, out / f'reply-{i}.json')))
                   for i in range(spec['np'])]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        result['requests'] = spec['np']
        if any(errors):
            raise Fail(8, f'request failed: {[e for e in errors if e]}')
        time.sleep(2)
        if not healthy(port):
            raise Fail(8, 'unhealthy after the requests')
    finally:
        result['stop'] = stop(p, birth)
        for _ in range(STOP_S):
            if not port_busy(port):
                break
            time.sleep(1)
        result['port_free_after_stop'] = not port_busy(port)
        (out / 'result.json').write_text(json.dumps(result, indent=2))
    if result['stop']['killed'] or result['stop']['already_dead']:
        raise Fail(8, f'server did not stop cleanly: {result["stop"]}')
    if not result['port_free_after_stop']:
        raise Fail(8, 'port still bound after the stop')
    if xid(since):
        raise Fail(3, 'Xid during the boot')
    result.update(judge((out / 'server.log').read_text(errors='replace'), profile))
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    if result['fatal'] or result['missing']:
        raise Fail(8, f'fatal lines {result["fatal"]}, missing engagement {result["missing"]}')
    print(f'health {profile}: PASS boot {result["boot_s_M"]} s, {spec["np"]} request(s), stop {result["stop"]["stop_s"]} s')


def selftest():
    e = engagement.ENGAGEMENT
    common = [s for c in ('base', 'k7', 'k4', 'k6a', 'k10', 'p90', 'k11a', 'k11b') for s in e[c]['samples']]
    np1 = '\n'.join(common + ['MTP chain: k=5', 'mtp_fold_decode: engaged n_seq=1', 'llama_context: n_kv_pad              = 4096'])
    duo = '\n'.join(common + ['MTP chain disabled: needs one sequence (n_seq=2)', 'mtp_fold_decode: engaged n_seq=2',
                              'llama_context: n_kv_pad              = 4096'])
    for profile, log in (('swift15-j5q', np1), ('swift15-iq4-duo', duo)):
        r = judge(log, profile)
        assert not r['missing'] and not r['fatal'], (profile, r)
        assert judge(log.replace('top_k_batched', 'x'), profile)['missing'], profile
        assert judge(log + '\nCUDA error: an illegal memory access', profile)['fatal'], profile
        assert judge(log + '\nsync_elide: missed: backend busy at a skipped sync', profile)['missing'], profile
    assert judge(duo + '\nMTP chain: k=5', 'swift15-j5q-duo')['missing']
    assert judge(np1, 'swift15-j5q-duo')['missing'] and judge(duo, 'swift15-j5q')['missing']
    print('health: log judge passes complete logs and fails dropped, fatal and wrong-shape logs PASS')


def main():
    if sys.argv[1:] == ['--selftest']:
        selftest()
        return 0
    if sys.argv[1:2] == ['--parse'] and len(sys.argv) == 3:
        for profile in PROFILES:
            spec = launch(profile, int(sys.argv[2]))
            h = engagement.health(profile)
            assert spec['np'] == (2 if profile.endswith('-duo') else 1), spec['np']
            print(json.dumps(dict(profile=profile, exe=spec['exe'], np=spec['np'], env=spec['env'], gating=h['gating'],
                                  patterns=len(h['patterns']), serve_sha256=spec['serve_sha256'])))
        return 0
    if len(sys.argv) != 3 or sys.argv[1] not in PROFILES:
        raise SystemExit(__doc__)
    try:
        run(sys.argv[1], Path(sys.argv[2]))
        return 0
    except Fail as e:
        print(f'health {sys.argv[1]}: FAIL rc={e.code} {e.reason}', file=sys.stderr)
        return e.code


if __name__ == '__main__':
    sys.exit(main())
