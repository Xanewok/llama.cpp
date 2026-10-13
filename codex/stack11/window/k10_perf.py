#!/usr/bin/env python3
"""Log parsers for the K10 window: engagement, negative controls, perf tables."""
import json
import re
import sys
from pathlib import Path

ANSI = re.compile(r'\x1b\[[0-9;]*m')
# a failing case is printed after its '[TOP_K] ERR = ...' message on the same line
TEST_LINE = re.compile(r'(?:^|\s)TOP_K\((.*?)\): (.*)$')
PERF_LINE = re.compile(r'^\s+TOP_K\((.*?)\):\s+(\d+) runs -\s+([\d.]+) us/run')


def lines(path):
    return [ANSI.sub('', l) for l in Path(path).read_text(errors='replace').splitlines()]


def results(path):
    rows = []
    for l in lines(path):
        m = TEST_LINE.search(l)
        if m:
            status = 'FAIL' if 'FAIL' in m.group(2) else 'OK' if m.group(2).endswith('OK') else 'UNSUPPORTED' if 'not supported' in m.group(2) else '?'
            rows.append((m.group(1), status))
    return rows


def engaged(log):
    rows = results(log)
    canon = [r for r in rows if 'canonical=1' in r[0]]
    assert len(canon) >= 100, f'only {len(canon)} canonical cases ran'
    bad = [r for r in rows if r[1] != 'OK']
    assert not bad, f'non-OK cases: {bad[:5]}'
    print(f'canonical cases OK: {len(canon)} of {len(rows)} TOP_K cases')


def failures(log, arm):
    # a crashed run stops before most cases: its failures prove nothing about the cases that never ran
    assert any(re.search(r'\d+/\d+ tests passed', l) for l in lines(log)), f'control {arm}: run did not complete'
    rows = results(log)
    failed = [r for r in rows if r[1] == 'FAIL']
    if arm == 'm5':
        # `key >= prefix` makes the tie order racy: its canonical failures are luck (0-3 per run), its 53 ties=1 set failures are not
        assert failed, f'control {arm}: no case failed'
    else:
        assert any('canonical=1' in r[0] for r in failed), f'control {arm}: no canonical case failed'
    if arm == 'off':
        # per-row CUB path: the set/value cases stay exact, only the order promise breaks
        other = [r for r in failed if 'canonical=1' not in r[0]]
        assert not other, f'control off failed non-canonical cases: {other[:5]}'
    print(f'control {arm}: {len(failed)} failing cases')


def perf(outdir, dev):
    table = {}
    for p in sorted(Path(outdir).glob(f'perf-*-{dev}-*.log')):
        arm = p.name.split('-')[1]
        for l in lines(p):
            m = PERF_LINE.match(l)
            if m:
                table.setdefault(m.group(1), {}).setdefault(arm, []).append(float(m.group(3)))
    out = {}
    for shape, arms in sorted(table.items()):
        row = {a: dict(us_min=min(v), us_all=v) for a, v in arms.items()}
        if 'auto' in arms and 'off' in arms:
            row['auto_over_off'] = min(arms['auto']) / min(arms['off'])
        out[shape] = row
    assert out and all({'auto', 'off', 'force'} <= set(r) for r in out.values()), 'missing perf arms'
    print(json.dumps(out, indent=1))


def selftest():
    import tempfile
    ok = '  TOP_K(type=f32,ne=[{0},1,1,1],k=1,{1}): \x1b[1;32mOK\x1b[0m'
    bad = '[TOP_K] ERR = 2.5 > 0.0000001   TOP_K(type=f32,ne=[{0},1,1,1],k=1,{1}): \x1b[1;31mFAIL\x1b[0m'
    done = '  3/4 tests passed'
    cases = {
        # (arm, lines, must pass)
        'm5 ties only': ('m5', [ok.format(1, 'canonical=1'), bad.format(2, 'ties=1'), done], True),
        'm5 nothing failed': ('m5', [ok.format(1, 'canonical=1'), ok.format(2, 'ties=1'), done], False),
        'm5 crashed': ('m5', [bad.format(2, 'ties=1')], False),
        'm1 canonical': ('m1', [bad.format(1, 'canonical=1'), done], True),
        'm1 ties only': ('m1', [ok.format(1, 'canonical=1'), bad.format(2, 'ties=1'), done], False),
        'off non-canonical': ('off', [bad.format(1, 'canonical=1'), bad.format(2, 'ties=1'), done], False),
    }
    with tempfile.TemporaryDirectory() as d:
        for name, (arm, text, want) in cases.items():
            p = Path(d)/'log'
            p.write_text('\n'.join(text) + '\n')
            try:
                failures(p, arm)
                got = True
            except AssertionError:
                got = False
            assert got == want, name


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'selftest':
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            selftest()
        print('k10 control judge selftest PASS')
    elif cmd == 'engaged':
        engaged(sys.argv[2])
    elif cmd == 'failures':
        failures(sys.argv[2], sys.argv[3])
    elif cmd == 'perf':
        perf(sys.argv[2], sys.argv[3])
    else:
        sys.exit(2)
