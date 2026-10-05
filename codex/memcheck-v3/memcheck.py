#!/usr/bin/env python3
"""v3 sanitizer gates for shared matmuls, residual RMS fusion and same-process exactness.
Preflight/selftest are CPU only. Canary exercises every stage on a small subset.
"""
import argparse, collections, json, os, re, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.realpath(HERE + '/../frozen/q8-dedupe-v3/build/bin')
SOURCE = HERE + '/../frozen/q8-dedupe-v3/SOURCE'
STAMP = open(SOURCE).readline().strip()[:9] if os.path.isfile(SOURCE) else ''
SANITIZER = '/usr/local/cuda-13.2/bin/compute-sanitizer'
SAN_COMMON = ['--error-exitcode', '99', '--print-limit', '50', '--report-api-errors', 'no']
RC_OK, RC_HAZARD, RC_INTEGRITY, RC_TIMEOUT, RC_FINDING = 0, 3, 6, 7, 8

FUSED_OPS = 'RMS_NORM_MUL_ADD'
# weight_broadcast=1 is what keeps the cases out of the multi-ADD fusion (needs equal src1 layouts), so the
# ADD+RMS_NORM+MUL matcher is reached; the 18 new cases plus one older post_mul case.
FUSED_P = 'weight_broadcast=1'
CANARY_P = r'ne=\[512,1,1,1\],.*alias_rms_input=0,weight_broadcast=1'
FUSED_SHAPES = {(r, c) for r in (1, 6, 8) for c in (512, 2560, 4096)}
Q8_CASES = 63
SHARED_OP = 'MUL_MAT_SHARED'
SHARED_CANARY = r'type=q4_K,rows=6,strided=0,overwrite=0'

ANSI = re.compile(r'\x1b\[[0-9;]*m')
CASE = re.compile(r'(?:^|\s)([A-Z][A-Z0-9_]+)\((.*?)\): (OK|FAIL|not supported.*)$')
SUMMARY = re.compile(r'(\d+)/(\d+) tests passed')
ERR_SUMMARY = re.compile(r'ERROR SUMMARY: (\d+) errors?')
RACE_SUMMARY = re.compile(r'RACECHECK SUMMARY: (\d+) hazards? displayed \((\d+) errors?, (\d+) warnings?\)')
XID = re.compile(r'NVRM: Xid \(PCI:([0-9a-fA-F:.]+)\): (\d+), pid=([^,]*), name=([^,]*),')
SAN_NOT_ERROR = ('COMPUTE-SANITIZER', 'ERROR SUMMARY', 'RACECHECK SUMMARY', 'Target application returned an error')
ADD_RMS = re.compile(r'add_rms_norm: device=0 rows=(\d+) columns=(\d+)')
Q8_LINE = re.compile(r'q8_reuse: device=0 graph=\d+ rows=(\d+) skipped=(\d+)')


def log(msg):
    print(f'probe: {msg}', flush=True)


def events(line):
    p = os.environ.get('GPUQ_EVENTS')
    if p:
        with open(p, 'a') as f:
            f.write(line + '\n')


# ---- parsers (tested on recorded logs in fixtures/) ----

def parse_cases(text, op=None):
    c = collections.Counter()
    for ln in text.splitlines():
        m = CASE.search(ANSI.sub('', ln).rstrip())
        if m and op in (None, m.group(1)):
            c['NS' if m.group(3).startswith('not') else m.group(3)] += 1
    return dict(ok=c['OK'], fail=c['FAIL'], ns=c['NS'])


def parse_summary(text):
    m = SUMMARY.findall(text)
    return f'{m[-1][0]}/{m[-1][1]}' if m else None


def parse_sanitizer(text):
    san = [ln for ln in text.splitlines() if ln.startswith('=========')]
    headers = [i for i, ln in enumerate(san)
               if re.match(r'========= [^ ]', ln) and not any(k in ln for k in SAN_NOT_ERROR)]
    e, r = ERR_SUMMARY.findall(text), RACE_SUMMARY.findall(text)
    first = headers[0] if headers else None
    return dict(banner=any('COMPUTE-SANITIZER' in ln for ln in san),
                errors=int(e[-1]) if e else None,
                hazards=int(r[-1][1]) if r else None,  # racecheck errors only: `all` also displays benign analysis hazards
                headers=len(headers),
                first_lines=san[first:first + 50] if first is not None else [])


def parse_engagement(text):
    shapes = {(int(m.group(1)), int(m.group(2))) for m in ADD_RMS.finditer(text)}
    q8 = [(int(m.group(1)), int(m.group(2))) for m in Q8_LINE.finditer(text)]
    return dict(add_rms_shapes=sorted(shapes), q8_evals=len(q8),
                q8_rows6_skipped=any(r == 6 and s > 0 for r, s in q8))


def parse_xids(text):
    return [dict(code=int(m.group(2)), pid=m.group(3), name=m.group(4), line=ln.strip())
            for ln in text.splitlines() for m in [XID.search(ln)] if m]


def journal_xids(since):
    r = subprocess.run(['journalctl', '-k', '--no-pager', '-o', 'short-unix', '--since', f'@{int(since)}'],
                       capture_output=True, text=True, timeout=60)
    return parse_xids(r.stdout) if r.returncode == 0 else None


# ---- stage definitions ----

def tbo_target(p, op=FUSED_OPS):
    return [BIN + '/test-backend-ops', 'test', '-b', 'CUDA0', '-o', op, '-p', p]


def sanitized(tool, target):
    return [SANITIZER, '--tool', tool] + SAN_COMMON + (['--racecheck-report', 'all'] if tool == 'racecheck' else []) + target


def stage_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith('GGML_CUDA_DISABLE')}
    env.pop('GGML_CUDA_GRAPH_OPT', None)
    env.update(LD_LIBRARY_PATH=BIN, GGML_CUDA_DEBUG_Q8_REUSE='1', GPUQ_ACTIVE='1')
    return env


def stage_list(raw, canary=False):
    fused = tbo_target(CANARY_P if canary else FUSED_P)
    shared = tbo_target(SHARED_CANARY if canary else '', SHARED_OP)
    exact = [BIN + '/q8-exact', '--canary' if canary else '--self-check']
    return [
        ('mc-fused', 'tbo', 'memcheck', sanitized('memcheck', fused), 60 if canary else 150),
        ('mc-shared', 'shared', 'memcheck', sanitized('memcheck', shared), 60 if canary else 150),
        ('mc-q8', 'q8', 'memcheck', sanitized('memcheck', exact), 120 if canary else 300),
        ('rc-fused', 'tbo', 'racecheck', sanitized('racecheck', fused), 45 if canary else 120),
        ('sc-fused', 'tbo', 'synccheck', sanitized('synccheck', fused), 45 if canary else 120),
    ]


# ---- verdict ----

def judge(name, kind, tool, s, canary=False):
    """Returns (findings, integrity) for one finished stage dict."""
    finds, integ = [], []
    if s.get("journal_unavailable"):
        integ.append(f"{name}: kernel journal unavailable")
    if s['timeout']:
        # A capped stage may already have logged real errors; they outrank the timeout.
        san = s['san']
        if san['headers'] or san['errors'] or san['hazards']:
            finds.append(f'{name}: capped after sanitizer findings ({san["headers"]} headers)')
        return finds, integ
    if s['rc'] not in (0, 99):
        finds.append(f'{name}: process died or test failed (rc {s["rc"]})')
    if s['rc'] == 99 and not (s['san']['errors'] or s['san']['hazards']):
        finds.append(f'{name}: sanitizer exit code 99 without a parsed error')
    san = s['san']
    if not san['banner']:
        integ.append(f'{name}: no sanitizer banner, the run was not instrumented')
    if san['errors'] is None and san['hazards'] is None:
        integ.append(f'{name}: no ERROR SUMMARY / RACECHECK SUMMARY line')
    if san['errors']:
        finds.append(f'{name}: {tool} reports {san["errors"]} errors ({san["headers"]} headers)')
    if san['hazards']:
        finds.append(f'{name}: racecheck reports {san["hazards"]} hazards')
    if kind == 'tbo':
        n = s['cases']
        if n['fail']:
            finds.append(f'{name}: {n["fail"]} failed cases')
        want = 1 if canary else 19
        if n['ok'] + n['fail'] != want:
            integ.append(f'{name}: {n["ok"] + n["fail"]} supported cases, expected {want}')
        if not s['device_ok']:
            integ.append(f'{name}: device 0 is not an RTX 5060 Ti')
        shapes = set(map(tuple, s['eng']['add_rms_shapes']))
        need = {(1, 512)} if canary else FUSED_SHAPES
        if not need <= shapes:
            integ.append(f'{name}: fused add_rms_norm did not run for rows/columns {sorted(need - shapes)}')
    elif kind == 'shared':
        n = s['shared_cases']
        want = 1 if canary else 60
        if n['fail']:
            finds.append(f'{name}: {n["fail"]} shared cases failed')
        if n['ok'] != want or n['ns'] or not s['device_ok']:
            integ.append(f'{name}: shared supported cases {n}, expected {want} on RTX 5060 Ti')
        if not s['eng']['q8_rows6_skipped']:
            integ.append(f'{name}: shared q8 reuse did not run at rows=6')
    else:
        want = 3 if canary else Q8_CASES
        if f'PASS: exact self-check cases={want}' not in s['text_ok'] or 'PASS: device=CUDA1' in s['text_ok']:
            integ.append(f'{name}: q8-exact did not report exactly {want} cases on CUDA0 only')
        e = s['eng']
        if e['q8_evals'] != want * 4 or not e['q8_rows6_skipped']:
            integ.append(f'{name}: q8 reuse not engaged (evaluations {e["q8_evals"]}, rows=6 skipped>0 {e["q8_rows6_skipped"]})')
        if (6, 2560) not in set(map(tuple, e['add_rms_shapes'])):
            integ.append(f'{name}: fused add_rms_norm did not run for rows=6 columns=2560')
    return finds, integ


def classify(stages, xids, canary=False):
    finds, integ, tmo = [], [], []
    for name, (kind, tool, s) in stages.items():
        if s['timeout']:
            tmo.append(f'{name}: hit its cap after {s["wall_s"]:.0f}s')
        f, i = judge(name, kind, tool, s, canary)
        finds += f
        integ += i
    if xids:
        return RC_HAZARD, integ, finds + [f'kernel fault code(s) {sorted({x["code"] for x in xids})}'], tmo
    rc = RC_FINDING if finds else RC_INTEGRITY if integ else RC_TIMEOUT if tmo else RC_OK
    return rc, integ, finds, tmo


# ---- execution ----

def check_frozen():
    bad = []
    if not STAMP:
        return ['frozen SOURCE missing']
    r = subprocess.run(['sha256sum', '-c', '--quiet', 'SHA256SUMS'], cwd=BIN, capture_output=True, text=True)
    if r.returncode != 0:
        bad.append('SHA256SUMS: ' + (r.stdout.strip() or r.stderr.strip()))
    libs = sorted(f for f in os.listdir(BIN) if re.fullmatch(r'libggml-base\.so\.[0-9.]+', f))
    if not libs or STAMP.encode() + b'\0' not in open(f'{BIN}/{libs[0]}', 'rb').read():
        bad.append(f'ggml stamp {STAMP} not found in libggml-base')
    for exe in ('test-backend-ops', 'q8-exact'):
        r = subprocess.run(['ldd', f'{BIN}/{exe}'], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        if 'not found' in r.stdout:
            bad.append(f'{exe}: unresolved libraries')
        for ln in r.stdout.splitlines():
            m = re.match(r'\s*(libggml\S*) => (\S+)', ln)
            if m and os.path.dirname(os.path.realpath(m.group(2))) != BIN:
                bad.append(f'{exe}: ggml libs resolve outside the frozen dir: {m.group(2)}')
    if not os.access(SANITIZER, os.X_OK):
        bad.append(f'{SANITIZER} missing')
    return bad


def run_stage(raw, name, argv, cap, env):
    path = f'{raw}/{name}.log'
    t0 = time.time()
    log(f'stage {name} start cap={cap:.0f}s: {" ".join(argv[-6:])}')
    to = False
    with open(path, 'wb') as f:
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            rc = p.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            to = True
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            rc = None
    time.sleep(3)
    text = open(path, errors='replace').read()
    return dict(rc=rc, timeout=to, wall_s=time.time() - t0, t0=t0, path=path, san=parse_sanitizer(text),
                cases=parse_cases(text, FUSED_OPS), shared_cases=parse_cases(text, SHARED_OP), summary=parse_summary(text), eng=parse_engagement(text),
                device_ok='RTX 5060 Ti' in text, text_ok='\n'.join(l for l in text.splitlines() if l.startswith(('PASS', 'FAIL'))))


def save(out, name, s):
    with open(f'{out}/{name}.tail.txt', 'w') as f:
        lines = open(s['path'], errors='replace').read().splitlines()
        f.write('\n'.join(ANSI.sub('', l) for l in lines[-80:]) + '\n')
    if s['san']['first_lines']:
        with open(f'{out}/{name}-sanitizer-first50.txt', 'w') as f:
            f.write('\n'.join(s['san']['first_lines']) + '\n')


def execute(a, canary):
    os.nice(19)
    out, raw = a.out, a.raw
    for d in (out, raw):
        os.makedirs(d, exist_ok=True)
    t_start = time.time()
    bad = check_frozen()
    if bad:
        for b in bad:
            log(f'integrity: {b}')
        return RC_INTEGRITY
    env = stage_env()
    plan = stage_list(raw, canary)
    events('measure_start')
    stages, xids, skipped = {}, [], []
    for name, kind, tool, argv, cap in plan:
        left = a.budget - (time.time() - t_start)
        cap = min(cap, left - 30)
        if cap < 30:
            log(f'budget: no time left for stage {name}')
            skipped.append(name)
            continue
        if tool != 'memcheck' and any(any(judge(n, k, t, s, canary)) or s["timeout"] for n, (k, t, s) in stages.items()):
            log(f'skip {name}: a memcheck stage already has findings')
            skipped.append(name)
            continue
        s = run_stage(raw, name, argv, cap, env)
        stages[name] = (kind, tool, s)
        save(out, name, s)
        jx = journal_xids(s['t0'] - 2)
        xids += jx or []
        log(f'stage {name}: rc={s["rc"]} timeout={s["timeout"]} wall={s["wall_s"]:.0f}s cases={s["cases"]} '
            f'sanitizer_errors={s["san"]["errors"]} hazards={s["san"]["hazards"]} engagement={s["eng"]} '
            f'fault_codes={[x["code"] for x in jx or []]}')
        if jx is None:
            s['journal_unavailable'] = True
            log('integrity: kernel journal unavailable')
            break
        if jx or any(judge(name, kind, tool, s, canary)) or s['timeout']:
            break
    rc, integ, finds, tmo = classify(stages, xids, canary)
    if skipped and rc == RC_OK:
        tmo.append('stages skipped for budget: ' + ','.join(skipped))
        rc = RC_TIMEOUT
    verdict = dict(rc=rc, integrity=integ, findings=finds, timeouts=tmo, skipped=skipped, raw_dir=raw,
                   stages={n: {k: v for k, v in s.items() if k not in ('text_ok', 'san')} | dict(
                       sanitizer={k: v for k, v in s['san'].items() if k != 'first_lines'}) for n, (_, _, s) in stages.items()},
                   xids=xids, stamp=STAMP, wall_s=time.time() - t_start)
    with open(f'{out}/verdict.json', 'w') as f:
        json.dump(verdict, f, indent=1)
    for tag, items in (('integrity', integ), ('finding', finds), ('budget', tmo)):
        for x in items:
            log(f'{tag}: {x}')
    log(f'result rc={rc}')
    events(f'summary rc={rc} findings={len(finds)} integrity={len(integ)}')
    return rc


# ---- self-test on recorded artifacts and synthetic engagement lines ----

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
    t = fx('memcheck-0err-testfail.log')
    check(parse_sanitizer(t)['errors'] == 0 and parse_cases(t) == dict(ok=16, fail=50, ns=0), f'0-error log with failing cases {parse_cases(t)}')
    check(parse_sanitizer('plain\n')['errors'] is None, 'plain log has no sanitizer summary')
    xl = parse_xids(fx('xid-lines.txt'))
    check([x['code'] for x in xl] == [79], 'recorded Xid lines')
    check(parse_sanitizer(fx('memcheck-clean.log'))['hazards'] is None, 'recorded memcheck is not a racecheck report')
    check(parse_engagement(fx('memcheck-clean.log')) == dict(add_rms_shapes=[], q8_evals=0, q8_rows6_skipped=False), 'recorded old kernel log has no q8/fusion engagement')
    check(parse_summary(fx('memcheck-clean.log')) is not None, 'recorded backend summary format')
    # SYNTHETIC: formats taken from ggml-cuda.cu:4524 / :4845, racecheck summary from the compute-sanitizer manual.
    syn = ('add_rms_norm: device=0 rows=1 columns=512\nadd_rms_norm: device=0 rows=6 columns=2560\n'
           'q8_reuse: device=0 graph=7 rows=6 skipped=3 quantized=2 bytes=1024\nq8_reuse: device=0 graph=8 rows=1 skipped=0 quantized=1 bytes=0\n'
           '========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n')
    e = parse_engagement(syn)
    check(e == dict(add_rms_shapes=[(1, 512), (6, 2560)], q8_evals=2, q8_rows6_skipped=True), f'engagement parse {e}')
    check(parse_sanitizer(syn)['hazards'] == 0, 'racecheck summary parse')
    cases = '  RMS_NORM_MUL_ADD(type=f32,ne=[512,1,1,1],eps=0.000001,broadcast=0,multi_add=0,post_mul=0,alias_rms_input=0,weight_broadcast=1): \x1b[1;32mOK\x1b[0m\n'
    check(parse_cases(cases)['ok'] == 1, 'SYNTHETIC RMS_NORM_MUL_ADD OK line')

    def stage(text, rc=0, to=False):
        return dict(rc=rc, timeout=to, wall_s=1.0, san=parse_sanitizer(text), cases=parse_cases(text, FUSED_OPS), shared_cases=parse_cases(text, SHARED_OP), eng=parse_engagement(text),
                    device_ok='RTX 5060 Ti' in text, text_ok='\n'.join(l for l in text.splitlines() if l.startswith('PASS')))

    good = fx('memcheck-clean.log') + cases + syn
    r = classify({'mc-canary': ('tbo', 'memcheck', stage(good))}, [], canary=True)
    check(r[0] == RC_OK, f'clean canary -> {r}')
    r = classify({'mc-canary': ('tbo', 'memcheck', stage(fx('memcheck-fault-stack-on.log') + cases + syn, 99))}, [], canary=True)
    check(r[0] == RC_FINDING, f'sanitizer errors -> rc {r[0]}')
    r = classify({'mc-canary': ('tbo', 'memcheck', stage(fx('memcheck-clean.log') + cases))}, [], canary=True)
    check(r[0] == RC_INTEGRITY and any('did not run' in i for i in r[1]), 'clean but fused kernel not engaged -> integrity')
    r = classify({'mc-canary': ('tbo', 'memcheck', stage('no banner\n' + cases + syn))}, [], canary=True)
    check(r[0] == RC_INTEGRITY, 'uninstrumented run -> integrity')
    r = classify({'mc-canary': ('tbo', 'memcheck', stage(good, None, True))}, [], canary=True)
    check(r[0] == RC_TIMEOUT, 'cap hit with no finding -> timeout')
    r = classify({'mc-canary': ('tbo', 'memcheck', stage(good))}, xl, canary=True)
    check(r[0] == RC_HAZARD, 'Xid in journal -> hazard')
    q8 = 'PASS: device=CUDA0 cases=63\nPASS: exact self-check cases=63\n' + fx('memcheck-clean.log') + ''.join(
        f'q8_reuse: device=0 graph={i} rows=6 skipped=1 quantized=1 bytes=1\n' for i in range(250)) + syn
    r = classify({'mc-q8': ('q8', 'memcheck', stage(q8))}, [])
    check(r[0] == RC_OK, f'clean q8-exact stage -> {r}')
    r = classify({'mc-q8': ('q8', 'memcheck', stage(re.sub(r'skipped=\d+', 'skipped=0', q8)))}, [])
    check(r[0] == RC_INTEGRITY, 'q8 reuse never skipped -> integrity')
    shared = fx('memcheck-clean.log') + 'RTX 5060 Ti\n' + ''.join(
        'MUL_MAT_SHARED(type=q4_K,rows=6,strided=0,overwrite=0): OK\n' for _ in range(60)) + syn
    r = classify({'mc-shared': ('shared', 'memcheck', stage(shared))}, [])
    check(r[0] == RC_OK, f'shared clean stage -> {r}')
    r = classify({'mc-shared': ('shared', 'memcheck', stage(shared.replace(': OK', ': not supported')))}, [])
    check(r[0] == RC_INTEGRITY, 'unregistered/unsupported shared cases rejected')
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
        bad = check_frozen()
        for b in bad:
            log(f'integrity: {b}')
        if bad or selftest():
            return RC_INTEGRITY
        log('preflight ok')
        return 0
    if not (a.out and a.raw and a.budget > 0):
        log('integrity: OUT, RAW and BUDGET_S are required')
        return RC_INTEGRITY
    return execute(a, a.canary)


if __name__ == '__main__':
    sys.exit(main())
