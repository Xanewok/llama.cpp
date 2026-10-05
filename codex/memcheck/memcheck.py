#!/usr/bin/env python3
"""Sanitizer gates for GATED_DELTA_NET (old op, kernel file changed) and GATED_DELTA_NET_INDEXED.
Preflight/selftest are CPU only. Canary: the smallest indexed case under memcheck.
"""
import argparse, collections, json, os, re, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.realpath(HERE + '/../frozen/gdn-gather-inplace/build/bin')
SOURCE = HERE + '/../frozen/gdn-gather-inplace/SOURCE'
STAMP = open(SOURCE).readline().strip()[:9] if os.path.isfile(SOURCE) else ''
SANITIZER = '/usr/local/cuda-13.2/bin/compute-sanitizer'
SAN_COMMON = ['--error-exitcode', '99', '--print-limit', '50', '--report-api-errors', 'no']
RC_OK, RC_HAZARD, RC_INTEGRITY, RC_TIMEOUT, RC_FINDING = 0, 3, 6, 7, 8

OLD_OPS = 'GATED_DELTA_NET,GATED_DELTA_NET_CACHE_FUSION'
IDX_OPS = 'GATED_DELTA_NET_INDEXED,GATED_DELTA_NET_INDEXED_CACHE_FUSION'
CANARY_OP = 'GATED_DELTA_NET_INDEXED'
# Smallest registered indexed case: head_size=16, n_seqs=1, T=6.
CANARY_P = r'head_size=16,n_seq_tokens=6,n_seqs=1,'
# Registered cases on CUDA0 (tests/test-backend-ops.cpp @38c59b368, recorded CPU run in fixtures/gdn-cpu-cases.log). The CUDA
# capability check (ggml-cuda.cu:6037) rejects indexed n_seqs>1, so only the n_seqs=1 indexed cases are supported there.
EXPECT = {
    'old': dict(ok=41, ns=0, ops={'GATED_DELTA_NET': 36, 'GATED_DELTA_NET_CACHE_FUSION': 5}),
    'idx': dict(ok=5, ns=10, ops={'GATED_DELTA_NET_INDEXED': 4, 'GATED_DELTA_NET_INDEXED_CACHE_FUSION': 1}),
}
CANARY_EXPECT = {'idx': dict(ok=1, ns=0, ops={'GATED_DELTA_NET_INDEXED': 1})}
IDX_TOKENS = [6]  # every supported indexed case has n_seq_tokens=6

ANSI = re.compile(r'\x1b\[[0-9;]*m')
CASE = re.compile(r'(?:^|\s)([A-Z][A-Z0-9_]+)\((.*?)\): (OK|FAIL|not supported.*)$')
SUMMARY = re.compile(r'(\d+)/(\d+) tests passed')
ERR_SUMMARY = re.compile(r'ERROR SUMMARY: (\d+) errors?')
RACE_SUMMARY = re.compile(r'RACECHECK SUMMARY: (\d+) hazards? displayed \((\d+) errors?, (\d+) warnings?\)')
XID = re.compile(r'NVRM: Xid \(PCI:([0-9a-fA-F:.]+)\): (\d+), pid=([^,]*), name=([^,]*),')
SAN_NOT_ERROR = ('COMPUTE-SANITIZER', 'ERROR SUMMARY', 'RACECHECK SUMMARY', 'Target application returned an error')
GDN_LINE = re.compile(r'gdn_state_indexed: device=0 graph=\d+ tokens=(\d+) gathers_skipped=(\d+)')


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


def parse_gdn_cases(text):
    ops = {o: parse_cases(text, o) for o in EXPECT['old']['ops'] | EXPECT['idx']['ops']}
    return dict(ops=ops, cases={k: sum(c[k] for c in ops.values()) for k in ('ok', 'fail', 'ns')})


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
    g = [(int(m.group(1)), int(m.group(2))) for m in GDN_LINE.finditer(text)]
    return dict(gdn_indexed_lines=len(g), tokens=sorted({t for t, _ in g}), all_skipped=bool(g) and all(k >= 1 for _, k in g))


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



# ---- stage definitions ----

def tbo_target(p, op):
    return [BIN + '/test-backend-ops', 'test', '-b', 'CUDA0', '-o', op] + (['-p', p] if p else [])


def sanitized(tool, target):
    return [SANITIZER, '--tool', tool] + SAN_COMMON + (['--racecheck-report', 'all'] if tool == 'racecheck' else []) + target


def stage_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith('GGML_CUDA_DISABLE')}
    env.pop('GGML_CUDA_GRAPH_OPT', None)
    env.update(LD_LIBRARY_PATH=BIN, GGML_CUDA_DEBUG_GDN_STATE='1', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8')
    return env


def stage_list(raw, canary=False):
    old = tbo_target('', OLD_OPS)
    idx = tbo_target(CANARY_P if canary else '', CANARY_OP if canary else IDX_OPS)
    if canary:
        return [
            ('mc-idx', 'idx', 'memcheck', sanitized('memcheck', idx), 90),
            ('rc-idx', 'idx', 'racecheck', sanitized('racecheck', idx), 45),
            ('sc-idx', 'idx', 'synccheck', sanitized('synccheck', idx), 45),
        ]
    return [
        ('mc-old', 'old', 'memcheck', sanitized('memcheck', old), 240),
        ('mc-idx', 'idx', 'memcheck', sanitized('memcheck', idx), 90),
        ('rc-idx', 'idx', 'racecheck', sanitized('racecheck', idx), 90),
        ('sc-idx', 'idx', 'synccheck', sanitized('synccheck', idx), 90),
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
    ex = (CANARY_EXPECT if canary else EXPECT)[kind]
    n, e = s['cases'], s['eng']
    if n['fail']:
        finds.append(f'{name}: {n["fail"]} failed cases')
    if n['ok'] + n['fail'] != ex['ok']:
        integ.append(f'{name}: {n["ok"] + n["fail"]} supported cases, expected {ex["ok"]}')
    if n['ns'] != ex['ns']:
        integ.append(f'{name}: {n["ns"]} unsupported cases, expected {ex["ns"]}')
    for op, want in ex['ops'].items():
        got = s['ops'][op]
        if got['ok'] + got['fail'] != want:
            integ.append(f'{name}: {op} ran {got["ok"] + got["fail"]} supported cases, expected {want}')
    if not s['device_ok']:
        integ.append(f'{name}: device 0 is not an RTX 5060 Ti')
    if kind == 'idx':
        if e['gdn_indexed_lines'] < ex['ok'] or e['tokens'] != IDX_TOKENS or not e['all_skipped']:
            integ.append(f'{name}: indexed GDN path not engaged (lines {e["gdn_indexed_lines"]} of {ex["ok"]}, tokens {e["tokens"]}, skipped {e["all_skipped"]})')
    elif e['gdn_indexed_lines']:
        integ.append(f'{name}: indexed path logged {e["gdn_indexed_lines"]} times in a stage that must not run it')
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
    for exe in ('test-backend-ops',):
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
                **parse_gdn_cases(text), summary=parse_summary(text), eng=parse_engagement(text),
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
    check(parse_engagement(fx('memcheck-clean.log')) == dict(gdn_indexed_lines=0, tokens=[], all_skipped=False), 'recorded old kernel log has no indexed GDN engagement')
    check(parse_summary(fx('memcheck-clean.log')) is not None, 'recorded backend summary format')
    # Registered cases: recorded CPU run of this branch's test-backend-ops (same case list, CPU supports every case).
    rec = fx('gdn-cpu-cases.log')
    allops = EXPECT['old']['ops'] | EXPECT['idx']['ops']
    ops = {o: parse_cases(rec, o) for o in allops}
    check({o: c['ok'] for o, c in ops.items()} == {'GATED_DELTA_NET': 36, 'GATED_DELTA_NET_CACHE_FUSION': 5,
                                                   'GATED_DELTA_NET_INDEXED': 12, 'GATED_DELTA_NET_INDEXED_CACHE_FUSION': 3}, f'registered case counts {ops}')
    one = re.compile(r'n_seqs=1[,)]')
    sup = {o: sum(1 for ln in rec.splitlines() if ln.strip().startswith(o + '(') and one.search(ln)) for o in EXPECT['idx']['ops']}
    check(sup == EXPECT['idx']['ops'] and sum(ops[o]['ok'] for o in sup) - sum(sup.values()) == EXPECT['idx']['ns'],
          f'indexed n_seqs=1 cases {sup} are the CUDA-supported ones, the remaining {EXPECT["idx"]["ns"]} unsupported')
    check(sum(ops[o]['ok'] for o in EXPECT['old']['ops']) == EXPECT['old']['ok'], 'old-op case total')
    check(sum(1 for ln in rec.splitlines() if ln.strip().startswith('GATED_DELTA_NET_INDEXED(') and re.search(CANARY_P, ln)) == 1,
          'canary filter selects exactly one indexed case')
    # SYNTHETIC: formats taken from ggml-cuda.cu:4598, test-backend-ops.cpp print_test_console, racecheck summary from the compute-sanitizer manual.
    syn1 = 'gdn_state_indexed: device=0 graph=0 tokens=6 gathers_skipped=1\n'
    syn = syn1 * 5 + '========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n'
    e = parse_engagement(syn)
    check(e == dict(gdn_indexed_lines=5, tokens=[6], all_skipped=True), f'engagement parse {e}')
    check(parse_sanitizer(syn)['hazards'] == 0, 'racecheck summary parse')
    oks = lambda op, n: ''.join(f'  {op}(type=f32,head_count=4,head_size={16 << i},n_seq_tokens=6,n_seqs=1,v_repeat=2,permuted=0,kda=0,K=6): \x1b[1;32mOK\x1b[0m\n' for i in range(n))
    nss = lambda op, n: ''.join(f'  {op}(type=f32,head_count=4,head_size={16 << i},n_seq_tokens=6,n_seqs=2,v_repeat=2,permuted=0,kda=0,K=6): not supported [CUDA0] \n' for i in range(n))
    check(parse_cases(oks('GATED_DELTA_NET_INDEXED', 1))['ok'] == 1 and parse_cases(nss('GATED_DELTA_NET_INDEXED', 2))['ns'] == 2, 'SYNTHETIC case lines')

    def stage(text, rc=0, to=False):
        return dict(rc=rc, timeout=to, wall_s=1.0, san=parse_sanitizer(text), **parse_gdn_cases(text), eng=parse_engagement(text),
                    device_ok='RTX 5060 Ti' in text, text_ok='\n'.join(l for l in text.splitlines() if l.startswith('PASS')))

    clean = fx('memcheck-clean.log')
    cases = oks('GATED_DELTA_NET_INDEXED', 1)
    good = clean + cases + syn1
    r = classify({'mc-idx': ('idx', 'memcheck', stage(good))}, [], canary=True)
    check(r[0] == RC_OK, f'clean canary -> {r}')
    r = classify({'mc-idx': ('idx', 'memcheck', stage(fx('memcheck-fault-stack-on.log') + cases + syn1, 99))}, [], canary=True)
    check(r[0] == RC_FINDING, f'sanitizer errors -> rc {r[0]}')
    r = classify({'mc-idx': ('idx', 'memcheck', stage(clean + cases))}, [], canary=True)
    check(r[0] == RC_INTEGRITY and any('not engaged' in i for i in r[1]), 'clean but indexed path not engaged -> integrity')
    r = classify({'mc-idx': ('idx', 'memcheck', stage('no banner\n' + cases + syn1))}, [], canary=True)
    check(r[0] == RC_INTEGRITY, 'uninstrumented run -> integrity')
    r = classify({'mc-idx': ('idx', 'memcheck', stage(good, None, True))}, [], canary=True)
    check(r[0] == RC_TIMEOUT, 'cap hit with no finding -> timeout')
    r = classify({'mc-idx': ('idx', 'memcheck', stage(good))}, xl, canary=True)
    check(r[0] == RC_HAZARD, 'Xid in journal -> hazard')
    full_idx = clean + oks('GATED_DELTA_NET_INDEXED', 4) + oks('GATED_DELTA_NET_INDEXED_CACHE_FUSION', 1) + \
        nss('GATED_DELTA_NET_INDEXED', 8) + nss('GATED_DELTA_NET_INDEXED_CACHE_FUSION', 2) + syn
    r = classify({'mc-idx': ('idx', 'memcheck', stage(full_idx))}, [])
    check(r[0] == RC_OK, f'clean indexed stage -> {r}')
    r = classify({'mc-idx': ('idx', 'memcheck', stage(full_idx.replace(': not supported [CUDA0] ', ': \x1b[1;32mOK\x1b[0m', 1)))}, [])
    check(r[0] == RC_INTEGRITY, 'multi-sequence indexed case unexpectedly supported -> integrity')
    r = classify({'mc-idx': ('idx', 'memcheck', stage(full_idx.replace('gathers_skipped=1', 'gathers_skipped=0')))}, [])
    check(r[0] == RC_INTEGRITY, 'indexed lines with no gather skipped -> integrity')
    r = classify({'mc-idx': ('idx', 'memcheck', stage(full_idx.replace('GATED_DELTA_NET_INDEXED_CACHE_FUSION(', 'GATED_DELTA_NET_INDEXED(')))}, [])
    check(r[0] == RC_INTEGRITY, 'per-op case count mismatch -> integrity')
    old_cases = oks('GATED_DELTA_NET', 36) + oks('GATED_DELTA_NET_CACHE_FUSION', 5)
    full_old = clean + old_cases
    r = classify({'mc-old': ('old', 'memcheck', stage(full_old))}, [])
    check(r[0] == RC_OK, f'clean old-op stage -> {r}')
    r = classify({'mc-old': ('old', 'memcheck', stage(full_old + syn1))}, [])
    check(r[0] == RC_INTEGRITY, 'old-op stage that logs the indexed path -> integrity')
    r = classify({'mc-old': ('old', 'memcheck', stage(clean + old_cases.replace('OK', 'FAIL', 1)))}, [])
    check(r[0] == RC_FINDING, 'failed case -> finding')
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
    if os.environ.get('GPUQ_ACTIVE') != '1':
        log('integrity: GPU queue required')
        return RC_INTEGRITY
    if not (a.out and a.raw and a.budget > 0):
        log('integrity: OUT, RAW and BUDGET_S are required')
        return RC_INTEGRITY
    return execute(a, a.canary)


if __name__ == '__main__':
    sys.exit(main())
