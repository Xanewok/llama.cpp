"""SASS gate for the multi-query q8_0 verify FA kernels of the integrated stack10 build (copy of codex/sass_gate.py).
Differences: the common-kernel resource check covers the flash_attn_ext instances only (the other components legitimately add or
change non-FA kernels); every other changed kernel is listed, not judged.

usage: sass_gate.py OUT_DIR STACK9_LIB NEW_LIB [--control CAND_LIB]
Exit 0: NEW_LIB's <256,256,8|4,8,Q8_0> kernels stay within stack9 (instructions <= +1 %, STL/LDL/HMMA/REG/STACK/LOCAL/SHARED
not above or different), no common kernel changed its resource usage, and the one-row decode kernels exist.
--control: the same gate on a library that has the regression must FAIL, otherwise the gate cannot see it.
Counting is the one used for the measured regression: instructions are lines starting with /*hhhh*/, STL/LDL lines containing ' STL'.
"""
import re
import subprocess
import sys
from pathlib import Path

INSTR = re.compile(r'^\s+/\*[0-9a-f]{4,}\*/')
RES = re.compile(r'REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)')
VERIFY = {n: rf'_Z18flash_attn_ext_f16ILi256ELi256ELi{n}ELi8ELb0ELb0ELb0EL9ggml_type8E' for n in (8, 4)}
FA_PREFIX = '_Z18flash_attn_ext_'
DECODE = rf'_Z18flash_attn_ext_f16ILi256ELi256ELi1ELi8ELb0ELb0ELb0EL9ggml_type8E'


def run(*cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def resources(lib, out):
    text = run('cuobjdump', '--dump-resource-usage', str(lib))
    out.write_text(text)
    usage, name = {}, None
    for line in text.splitlines():
        m = re.match(r'\s*Function (\S+):', line)
        if m:
            name = m.group(1)
            continue
        m = RES.search(line)
        if m and name:
            usage[name] = tuple(int(x) for x in m.groups())
            name = None
    return usage


def find(usage, prefix):
    hits = [n for n in usage if n.startswith(prefix)]
    assert len(hits) == 1, f'{prefix}: {len(hits)} kernels'
    return hits[0]


def sass(lib, name, path):
    text = run('cuobjdump', '-sass', '-fun', name, str(lib))
    path.write_text(text)
    lines = text.splitlines()
    return dict(instr=sum(bool(INSTR.match(l)) for l in lines), STL=sum(' STL' in l for l in lines),
                LDL=sum(' LDL' in l for l in lines), HMMA=sum(' HMMA' in l for l in lines),
                BRA=sum(' BRA' in l for l in lines))


def measure(lib, out, tag):
    out.mkdir(parents=True, exist_ok=True)
    usage = resources(lib, out/f'{tag}.res')
    rows = {}
    for n, prefix in VERIFY.items():
        name = find(usage, prefix)
        rows[n] = dict(name=name, REG=usage[name][0], STACK=usage[name][1], SHARED=usage[name][2], LOCAL=usage[name][3],
                       **sass(lib, name, out/f'{tag}.{n}.sass'))
    decode = [n for n in usage if n.startswith(DECODE)]
    return usage, rows, decode


def violations(ref, new):
    bad = []
    for n in (8, 4):
        r, c = ref[n], new[n]
        if c['instr'] > r['instr']*1.01:
            bad.append(f'ncols1={n}: instructions {c["instr"]} > stack9 {r["instr"]} + 1 %')
        for key in ('STL', 'LDL'):
            if c[key] > r[key]:
                bad.append(f'ncols1={n}: {key} {c[key]} > stack9 {r[key]}')
        for key in ('HMMA', 'REG', 'STACK', 'LOCAL', 'SHARED'):
            if c[key] != r[key]:
                bad.append(f'ncols1={n}: {key} {c[key]} != stack9 {r[key]}')
    return bad


def main():
    out, s9, new = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    control = Path(sys.argv[5]) if len(sys.argv) == 6 and sys.argv[4] == '--control' else None
    assert len(sys.argv) in (4, 6) and (len(sys.argv) == 4 or control), __doc__
    usage9, ref, _ = measure(s9, out, 'stack9')
    usage_new, cur, decode = measure(new, out, 'new')
    for n in (8, 4):
        print(f'ncols1={n} stack9 {ref[n]}')
        print(f'ncols1={n} new    {cur[n]}')
    bad = violations(ref, cur)
    common = usage9.keys() & usage_new.keys()
    changed = sorted(k for k in common if usage9[k] != usage_new[k] and k.startswith(FA_PREFIX))
    other = sorted(k for k in common if usage9[k] != usage_new[k] and not k.startswith(FA_PREFIX))
    print(f'common kernels: {len(common)}, common flash_attn_ext instances: {sum(k.startswith(FA_PREFIX) for k in common)}, '
          f'FA resource usage changed: {len(changed)}, non-FA changed (informational): {len(other)}')
    for k in other:
        print(f'non-FA kernel changed: {k} {usage9[k]} -> {usage_new[k]}')
    bad += [f'resource usage of common flash_attn_ext instance changed: {k} {usage9[k]} -> {usage_new[k]}' for k in changed]
    if not decode:
        bad.append('no one-row decode kernel <256,256,1,8,Q8_0> in the new library')
    else:
        print(f'one-row decode kernels (256): {len(decode)}, resource usage {[usage_new[k] for k in decode]}')
    if control:
        _, reg, _ = measure(control, out, 'control')
        for n in (8, 4):
            print(f'ncols1={n} control {reg[n]}')
        seen = violations(ref, reg)
        print(f'control violations: {seen}')
        if not seen:
            bad.append(f'negative control {control} passes the gate: it cannot see the regression')
    for b in bad:
        print('GATE VIOLATION:', b)
    print('sass gate', 'FAIL' if bad else 'PASS')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
