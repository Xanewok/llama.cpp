"""Gates on fa-q8-identity `prefill` output. usage: fa_q8_prefill_check.py NEW.txt REF.txt [--strict-bytes]

NEW is the library under test, REF the base library (the knobs are ignored there: its four arms are one path).
Hard gates: no UNSUPPORTED/non-finite output; B (cap 0, convert + f16) equals REF bytes for every case; engagement of A, M and D
equals the dispatch expectation, B never engages; NMSE(A, B) <= 1e-6; M (K/V swapped) differs from A wherever A engages.
Byte identity A == B is counted and printed; with --strict-bytes it is also a gate.
"""
import re
import sys
from pathlib import Path

NMSE_MAX = 1e-6


def parse(path):
    arms, nmse, expect, bad = {}, {}, {}, []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if not line.startswith('prefill '):
            continue
        if f[-1] == 'UNSUPPORTED':
            bad.append(f'UNSUPPORTED: {line}')
        elif f[1] == 'nmse':
            nmse[f[2]] = float(f[3])
        elif f[1] == 'expect':
            expect[f[2]] = dict(kv.split('=') for kv in f[3:])
        else:
            m = re.fullmatch(r'nonfinite=(\d+)', f[5])
            e = re.fullmatch(r'engaged=(\d+)', f[6])
            assert m and e, line
            if int(m.group(1)):
                bad.append(f'non-finite output: {line}')
            arms[(f[1], f[2])] = dict(digest=f[3], nbytes=int(f[4]), engaged=int(e.group(1)))
    return arms, nmse, expect, bad


def main():
    new, ref = parse(sys.argv[1]), parse(sys.argv[2])
    strict = '--strict-bytes' in sys.argv
    arms, nmse, expect, bad = new
    rarms, _, _, rbad = ref
    bad += [f'ref {b}' for b in rbad]
    if bad:
        print('\n'.join(f'VIOLATION {b}' for b in bad[:30]))
        print('prefill check FAIL')
        return 1
    cases = sorted({c for _, c in arms})
    if len(cases) < 40 or len(expect) != len(cases):
        bad.append(f'vacuous: {len(cases)} cases, {len(expect)} expectations')
    ident = engaged_cases = 0
    for c in cases:
        a, b, m, d = (arms[(x, c)] for x in 'ABMD')
        if b['digest'] != rarms[('B', c)]['digest'] or any(rarms[(x, c)]['digest'] != rarms[('B', c)]['digest'] for x in 'AMD'):
            bad.append(f'{c}: B differs from REF, or REF arms differ')
        if any(rarms[(x, c)]['engaged'] for x in 'ABMD'):
            bad.append(f'{c}: REF engaged')
        if b['engaged']:
            bad.append(f'{c}: B engaged')
        e = expect[c]
        if a['engaged'] != int(e['uncapped']) or m['engaged'] != int(e['uncapped']) or d['engaged'] != int(e['default']):
            bad.append(f'{c}: engagement A={a["engaged"]} M={m["engaged"]} D={d["engaged"]} expected uncapped={e["uncapped"]} default={e["default"]}')
        if int(e['uncapped']):
            engaged_cases += 1
            if a['digest'] == m['digest']:
                bad.append(f'{c}: mutant M equals A (gate cannot see a K/V swap)')
            if nmse[c] > NMSE_MAX:
                bad.append(f'{c}: NMSE(A,B) {nmse[c]:.3e} > {NMSE_MAX}')
            ident += a['digest'] == b['digest']
            if strict and a['digest'] != b['digest']:
                bad.append(f'{c}: A bytes differ from B (nmse {nmse[c]:.3e})')
        elif a['digest'] != b['digest']:
            bad.append(f'{c}: not engaged but A differs from B')
        if d['engaged'] and d['digest'] != a['digest']:
            bad.append(f'{c}: default arm engaged but differs from A')
        if not d['engaged'] and d['digest'] != b['digest']:
            bad.append(f'{c}: default arm not engaged but differs from B')
    worst = max((nmse[c] for c in cases if int(expect[c]['uncapped'])), default=0)
    print(f'cases={len(cases)} engaged={engaged_cases} A==B byte-identical: {ident}/{engaged_cases} worst NMSE(A,B)={worst:.3e}')
    for b in bad[:30]:
        print('VIOLATION', b)
    print('prefill check', 'FAIL' if bad else 'PASS')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
