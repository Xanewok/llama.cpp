#!/usr/bin/env python3
"""compute-sanitizer memcheck of the AllReduce-node spike (GGML_META_AR_NODE=1) on both GPUs.

--canary: control. The stack6 frozen server (host-driven AllReduce) under memcheck, Qwen3.5-4B, one short completion.
          A control that errors or hits its cap means memcheck is not applicable to a two-GPU server (exit 5, not a finding).
--run:    the spike's frozen server with GGML_META_AR_NODE=1, same request. Needs a clean memcheck log, the server's
          `meta_ar_node: ... split=N->K` plan line (engagement) and a completed request.
Preflight/selftest are CPU only: frozen builds, parsers on recorded sanitizer logs, verdict logic on synthetic stages.
Exit codes: 0 ok, 3 hazard (Xid), 5 insufficient_data, 6 integrity, 7 timeout, 8 deterministic finding.
"""
import argparse, collections, json, os, re, signal, socket, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
UP = '/home/xanewok/repos/qwen38-eval-worktrees/frozen/stack6'
PATCH = os.path.realpath(HERE + '/../frozen/graph-capture-spike')
SANITIZER = '/usr/local/cuda-13.2/bin/compute-sanitizer'
MODEL = '/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf'
DEVICES = 'GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f'
SAN_COMMON = ['--tool', 'memcheck', '--error-exitcode', '99', '--print-limit', '50', '--report-api-errors', 'no']
RC_OK, RC_HAZARD, RC_INSUFFICIENT, RC_INTEGRITY, RC_TIMEOUT, RC_FINDING = 0, 3, 5, 6, 7, 8
N_PREDICT = 32
PROMPT = 'The sequence of positive integers begins with 1, 2, 3, 4, 5. Continue the sequence:'

ANSI = re.compile(r'\x1b\[[0-9;]*m')
ERR_SUMMARY = re.compile(r'ERROR SUMMARY: (\d+) errors?')
XID = re.compile(r'NVRM: Xid \(PCI:([0-9a-fA-F:.]+)\): (\d+), pid=([^,]*), name=([^,]*),')
PLAN = re.compile(r'meta_ar_node: uid=\d+ sites=([1-9]\d*) split=(\d+)->(\d+)')
SPIN = re.compile(r'ggml_cuda_ar_node: spin timeout')
SAN_NOT_ERROR = ('COMPUTE-SANITIZER', 'ERROR SUMMARY', 'RACECHECK SUMMARY', 'Target application returned an error')


def log(msg):
    print(f'probe: {msg}', flush=True)


def events(line):
    p = os.environ.get('GPUQ_EVENTS')
    if p:
        with open(p, 'a') as f:
            f.write(line + '\n')


# ---- parsers (tested on recorded logs in fixtures/) ----

def parse_sanitizer(text):
    san = [ln for ln in text.splitlines() if ln.startswith('=========')]
    headers = [i for i, ln in enumerate(san)
               if re.match(r'========= [^ ]', ln) and not any(k in ln for k in SAN_NOT_ERROR)]
    e = ERR_SUMMARY.findall(text)
    first = headers[0] if headers else None
    return dict(banner=any('COMPUTE-SANITIZER' in ln for ln in san),
                errors=int(e[-1]) if e else None,
                headers=len(headers),
                first_lines=san[first:first + 50] if first is not None else [])


def parse_engagement(text):
    plans = [tuple(int(x) for x in m.groups()) for m in PLAN.finditer(text)]
    return dict(plan_lines=len(plans), plans=sorted(set(plans)), spin_timeouts=len(SPIN.findall(text)))


def parse_xids(text):
    return [dict(code=int(m.group(2)), pid=m.group(3), name=m.group(4), line=ln.strip())
            for ln in text.splitlines() for m in [XID.search(ln)] if m]


def journal_xids(since):
    try:
        r = subprocess.run(['journalctl', '-k', '--no-pager', '-o', 'short-unix', '--since', f'@{int(since)}'],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_xids(r.stdout) if r.returncode == 0 else None


# ---- verdict ----

def judge(stage, s):
    """(findings, integrity, inapplicable) for one finished stage. stage is 'control' or 'run'."""
    finds, integ, inapp = [], [], []
    if s.get('journal_unavailable'):
        integ.append(f'{stage}: kernel journal unavailable')
    san, eng = s['san'], s['eng']
    if s['timeout']:
        what = f'{stage}: hit its cap in phase {s.get("phase")} after {s["wall_s"]:.0f}s'
        (inapp if stage == 'control' else []).append(what)
        if stage == 'run' and (san['headers'] or san['errors'] or eng['spin_timeouts']):
            finds.append(f'{stage}: capped after sanitizer findings ({san["headers"]} headers, {eng["spin_timeouts"]} spin timeouts)')
        return finds, integ, inapp
    if stage == 'control':
        if san['errors'] or san['headers'] or s['rc'] not in (0,) or not s['completed']:
            inapp.append(f'control: memcheck on the stack6 server is not clean (rc {s["rc"]}, errors {san["errors"]}, '
                         f'headers {san["headers"]}, completed {s["completed"]})')
        if not san['banner']:
            inapp.append('control: no sanitizer banner, the control was not instrumented')
        if eng['plan_lines']:
            integ.append('control: logged a meta_ar_node plan; the control must run without node mode')
        return finds, integ, inapp
    if eng['spin_timeouts']:
        finds.append(f'run: {eng["spin_timeouts"]} AllReduce-node spin timeouts (peer never arrived)')
    if san['errors']:
        finds.append(f'run: memcheck reports {san["errors"]} errors ({san["headers"]} headers)')
    if s['rc'] not in (0,):
        finds.append(f'run: server or sanitizer exit code {s["rc"]}')
    if not san['banner']:
        integ.append('run: no sanitizer banner, the run was not instrumented')
    if san['errors'] is None:
        integ.append('run: no ERROR SUMMARY line in the sanitizer log')
    if not s['completed']:
        integ.append('run: the completion request did not return 32 tokens')
    if not eng['plan_lines']:
        integ.append('run: no `meta_ar_node: uid=.. sites=N split=M->K` line with N>=1: node mode not engaged')
    if not s['device_ok']:
        integ.append('run: the server did not report two RTX 5060 Ti devices')
    return finds, integ, inapp


def classify(stages, xids):
    finds, integ, inapp = [], [], []
    for name, s in stages.items():
        f, i, a = judge(name, s)
        finds += f
        integ += i
        inapp += a
    if xids:
        return RC_HAZARD, integ, finds + [f'kernel fault code(s) {sorted({x["code"] for x in xids})}'], inapp
    if inapp and 'control' in stages:
        return RC_INSUFFICIENT, integ, finds, inapp
    tmo = any(s['timeout'] for n, s in stages.items() if n == 'run')
    rc = RC_FINDING if finds else RC_INTEGRITY if integ else RC_TIMEOUT if tmo else RC_OK
    return rc, integ, finds, inapp


# ---- execution ----

def check_frozen(root, name):
    bad = []
    bin_ = os.path.join(root, 'build/bin')
    try:
        stamp = open(os.path.join(root, 'SOURCE')).readline().strip()[:9]
    except OSError:
        return [f'{name}: SOURCE missing']
    r = subprocess.run(['sha256sum', '-c', '--quiet', 'SHA256SUMS'], cwd=bin_, capture_output=True, text=True)
    if r.returncode != 0:
        bad.append(f'{name}: SHA256SUMS: ' + (r.stdout.strip() or r.stderr.strip() or 'missing'))
    libs = sorted(f for f in os.listdir(bin_) if re.fullmatch(r'libggml-base\.so\.[0-9.]+', f)) if os.path.isdir(bin_) else []
    if not libs or stamp.encode() + b'\0' not in open(f'{bin_}/{libs[0]}', 'rb').read():
        bad.append(f'{name}: ggml stamp {stamp} not found in libggml-base')
    exe = f'{bin_}/round-trace'
    if not os.path.isfile(exe):
        bad.append(f'{name}: {exe} missing')
    else:
        r = subprocess.run(['ldd', exe], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        if 'not found' in r.stdout:
            bad.append(f'{name}: round-trace has unresolved libraries')
        for ln in r.stdout.splitlines():
            m = re.match(r'\s*(libggml\S*) => (\S+)', ln)
            if m and os.path.dirname(os.path.realpath(m.group(2))) != os.path.realpath(bin_):
                bad.append(f'{name}: ggml libs resolve outside the frozen dir: {m.group(2)}')
    return bad


def preflight_checks():
    bad = check_frozen(UP, 'stack6') + check_frozen(PATCH, 'spike')
    if not os.access(SANITIZER, os.X_OK):
        bad.append(f'{SANITIZER} missing')
    if not os.path.isfile(MODEL):
        bad.append(f'{MODEL} missing')
    return bad


def server_argv(exe, port):
    # The depth experiment's server shape on the 4B fixture: both GPUs, MTP k=5, backend sampling; no draft vocab (the
    # 27B vocabulary file does not fit the 4B model). Plain flags, no KV padding: the point is the AllReduce path.
    return [exe, '-m', MODEL, '-c', '4096', '-np', '1', '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0',
            '-sm', 'tensor', '--tensor-split', '50,50', '-ngl', '99', '-b', '512', '-ub', '128',
            '--spec-type', 'draft-mtp', '--spec-draft-n-max', '5', '--spec-draft-n-min', '0', '--spec-draft-p-min', '0',
            '--spec-draft-sampling', 'probabilistic', '--spec-draft-backend-sampling', '--backend-sampling',
            '--cache-ram', '0', '-t', '4', '-tb', '4', '--host', '127.0.0.1', '--port', str(port)]


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def descendants(pid):
    out = []
    try:
        for t in os.listdir(f'/proc/{pid}/task'):
            kids = open(f'/proc/{pid}/task/{t}/children').read().split()
            for k in kids:
                out.append(int(k))
                out += descendants(int(k))
    except OSError:
        pass
    return out


def http(url, body=None, timeout=30):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def run_stage(name, root, extra_env, cap, raw):
    """One sanitized server run: boot, one completion, graceful stop (the summary prints at exit)."""
    sanlog, srvlog = f'{raw}/{name}-sanitizer.log', f'{raw}/{name}-server.log'
    port = free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_META', 'GGML_CUDA_DISABLE'))}
    env.update(CUDA_VISIBLE_DEVICES=DEVICES, LD_LIBRARY_PATH=f'{root}/build/bin', LLAMA_ARG_LOG_VERBOSITY='4',
               GGML_SCHED_INPUT_STAGING='1', OMP_NUM_THREADS='4')
    env.update(extra_env)
    argv = [SANITIZER] + SAN_COMMON + ['--log-file', sanlog] + server_argv(f'{root}/build/bin/round-trace', port)
    t0 = time.time()
    log(f'stage {name} start cap={cap:.0f}s port={port} env={extra_env}')
    completed, timeout, phase, rc = False, False, 'boot', None
    with open(srvlog, 'wb') as f:
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            while True:
                if p.poll() is not None:
                    break
                if time.time() - t0 > cap:
                    raise TimeoutError
                try:
                    if http(f'http://127.0.0.1:{port}/health', timeout=5).get('status') == 'ok':
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(5)
            if p.poll() is None:
                phase = 'completion'
                r = http(f'http://127.0.0.1:{port}/completion',
                         dict(prompt=PROMPT, n_predict=N_PREDICT, temperature=0, seed=1729, cache_prompt=False),
                         timeout=max(30, cap - (time.time() - t0)))
                completed = r.get('tokens_predicted') == N_PREDICT
                phase = 'stop'
                for kid in descendants(p.pid):
                    try:
                        if open(f'/proc/{kid}/comm').read().strip() == 'round-trace':
                            os.kill(kid, signal.SIGTERM)
                    except OSError:
                        pass
                rc = p.wait(timeout=max(30, min(240, cap - (time.time() - t0))))
            else:
                rc = p.returncode
        except (TimeoutError, subprocess.TimeoutExpired, OSError, ValueError):
            timeout = True
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
    time.sleep(3)
    san_text = open(sanlog, errors='replace').read() if os.path.isfile(sanlog) else ''
    srv_text = open(srvlog, errors='replace').read()
    return dict(rc=rc, timeout=timeout, phase=phase, wall_s=time.time() - t0, t0=t0, completed=completed,
                san=parse_sanitizer(san_text), eng=parse_engagement(srv_text),
                device_ok=len(re.findall(r'NVIDIA GeForce RTX 5060 Ti', srv_text)) >= 2,
                paths=dict(sanitizer=sanlog, server=srvlog))


def save(out, name, s):
    for key, path in s['paths'].items():
        if os.path.isfile(path):
            lines = open(path, errors='replace').read().splitlines()
            with open(f'{out}/{name}-{key}.tail.txt', 'w') as f:
                f.write('\n'.join(ANSI.sub('', l) for l in lines[-120:]) + '\n')
    if s['san']['first_lines']:
        with open(f'{out}/{name}-sanitizer-first50.txt', 'w') as f:
            f.write('\n'.join(s['san']['first_lines']) + '\n')


def execute(a, canary):
    os.nice(19)
    out, raw = a.out, a.raw
    for d in (out, raw):
        os.makedirs(d, exist_ok=True)
    t_start = time.time()
    bad = preflight_checks()
    if bad:
        for b in bad:
            log(f'integrity: {b}')
        return RC_INTEGRITY
    # Memcheck slows kernels ~100x: the in-kernel deadlock trap must outlive any legitimate lag between the two devices.
    name, root, env = ('control', UP, {}) if canary else ('run', PATCH, {'GGML_META_AR_NODE': '1', 'GGML_CUDA_AR_NODE_SPIN_MS': '240000'})
    cap = a.budget - (time.time() - t_start) - 60
    if cap < 120:
        log('budget: no time left')
        return RC_TIMEOUT
    events('measure_start')
    s = run_stage(name, root, env, cap, raw)
    save(out, name, s)
    jx = journal_xids(s['t0'] - 2)
    if jx is None:
        s['journal_unavailable'] = True
        log('integrity: kernel journal unavailable')
    stages = {name: s}
    rc, integ, finds, inapp = classify(stages, jx or [])
    log(f'stage {name}: rc={s["rc"]} timeout={s["timeout"]} phase={s["phase"]} wall={s["wall_s"]:.0f}s completed={s["completed"]} '
        f'sanitizer_errors={s["san"]["errors"]} engagement={s["eng"]} fault_codes={[x["code"] for x in jx or []]}')
    verdict = dict(rc=rc, integrity=integ, findings=finds, inapplicable=inapp, xids=jx,
                   stage={k: v for k, v in s.items() if k not in ('san',)} | dict(sanitizer={k: v for k, v in s['san'].items() if k != 'first_lines'}),
                   wall_s=time.time() - t_start)
    with open(f'{out}/verdict.json', 'w') as f:
        json.dump(verdict, f, indent=1)
    for tag, items in (('integrity', integ), ('finding', finds), ('sanitizer not applicable', inapp)):
        for x in items:
            log(f'{tag}: {x}')
    log(f'result rc={rc}')
    events(f'summary rc={rc} findings={len(finds)} integrity={len(integ)}')
    return rc


# ---- self-test on recorded artifacts and synthetic stages ----

def selftest():
    fx = lambda n: open(f'{HERE}/fixtures/{n}', errors='replace').read()
    ok = True

    def check(cond, msg):
        nonlocal ok
        log(f'selftest {"ok  " if cond else "FAIL"} {msg}')
        ok &= bool(cond)

    s = parse_sanitizer(fx('memcheck-fault-stack-on.log'))
    check(s['errors'] == 52 and s['banner'] and len(s['first_lines']) == 50, f'recorded memcheck faults errors={s["errors"]}')
    s = parse_sanitizer(fx('memcheck-clean.log'))
    check(s['errors'] == 0 and s['banner'] and not s['first_lines'], 'recorded clean memcheck')
    check(parse_sanitizer(fx('memcheck-0err-testfail.log'))['errors'] == 0, 'recorded 0-error log')
    check(parse_sanitizer('plain\n')['errors'] is None, 'plain log has no sanitizer summary')
    check([x['code'] for x in parse_xids(fx('xid-lines.txt'))] == [79], 'recorded Xid lines')
    # SYNTHETIC: the plan line format of ggml-backend-meta.cpp and the spin diagnostic of allreduce.cu.
    plan = 'meta_ar_node: uid=7 sites=19 split=22->1 nodes=900 rows=6\nmeta_ar_node: uid=8 sites=19 split=22->2 nodes=900 rows=1\n'
    check(parse_engagement(plan) == dict(plan_lines=2, plans=[(19, 22, 1), (19, 22, 2)], spin_timeouts=0), 'engagement parse')
    check(parse_engagement('meta_ar_node: uid=7 sites=0 split=22->22 nodes=900 rows=6\n')['plan_lines'] == 0, 'sites=0 is not engagement')
    check(parse_engagement('ggml_cuda_ar_node: spin timeout waiting for peer (arrival), block 3, want 5, got 4\n')['spin_timeouts'] == 1, 'spin diagnostic parse')

    def stage(san_text, srv_text, rc=0, to=False, completed=True, phase='stop'):
        return dict(rc=rc, timeout=to, phase=phase, wall_s=1.0, completed=completed, san=parse_sanitizer(san_text),
                    eng=parse_engagement(srv_text), device_ok=True)

    clean, fault = fx('memcheck-clean.log'), fx('memcheck-fault-stack-on.log')
    r = classify({'control': stage(clean, '')}, [])
    check(r[0] == RC_OK, f'clean control -> {r}')
    r = classify({'control': stage(clean, '', to=True, completed=False, phase='boot')}, [])
    check(r[0] == RC_INSUFFICIENT, 'control timeout -> insufficient_data, not a finding')
    r = classify({'control': stage(fault, '', rc=99)}, [])
    check(r[0] == RC_INSUFFICIENT and not r[2], 'control with errors -> insufficient_data, not a finding')
    r = classify({'run': stage(clean, plan)}, [])
    check(r[0] == RC_OK, f'clean engaged run -> {r}')
    r = classify({'run': stage(fault, plan, rc=99)}, [])
    check(r[0] == RC_FINDING, 'run with memcheck errors -> finding')
    r = classify({'run': stage(clean, '')}, [])
    check(r[0] == RC_INTEGRITY and any('not engaged' in i for i in r[1]), 'clean run without the plan line -> integrity')
    r = classify({'run': stage('no banner\n', plan)}, [])
    check(r[0] == RC_INTEGRITY, 'uninstrumented run -> integrity')
    r = classify({'run': stage(clean, plan, completed=False)}, [])
    check(r[0] == RC_INTEGRITY, 'request not completed -> integrity')
    r = classify({'run': stage(clean, plan + 'ggml_cuda_ar_node: spin timeout waiting for peer (arrival), block 0, want 2, got 1\n', rc=-6)}, [])
    check(r[0] == RC_FINDING, 'spin timeout trap -> finding')
    r = classify({'run': stage(clean, plan, to=True, completed=False, phase='completion')}, [])
    check(r[0] == RC_TIMEOUT and not r[2], 'run cap without findings -> timeout')
    r = classify({'run': stage(clean, plan)}, parse_xids(fx('xid-lines.txt')))
    check(r[0] == RC_HAZARD, 'Xid in journal -> hazard')
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ('preflight', 'canary', 'run', 'selftest'):
        g.add_argument('--' + m, action='store_true')
    ap.add_argument('--out', default=os.environ.get('OUT'))
    ap.add_argument('--raw', default=os.environ.get('RAW'))
    ap.add_argument('--budget', type=float, default=float(os.environ.get('BUDGET_S', 0)))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.preflight:
        bad = preflight_checks()
        for b in bad:
            log(f'integrity: {b}')
        if bad or selftest():
            return RC_INTEGRITY
        log('preflight ok')
        return 0
    if os.environ.get('GPUQ_ACTIVE') != '1':
        log('integrity: GPU queue required')
        return RC_INTEGRITY
    if not (a.out and a.raw and a.budget > 0):
        log('integrity: OUT, RAW and BUDGET_S are required')
        return RC_INTEGRITY
    return execute(a, a.canary)


if __name__ == '__main__':
    sys.exit(main())
