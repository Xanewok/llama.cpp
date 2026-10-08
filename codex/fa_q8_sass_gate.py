"""SASS gate for the q8_0 prefill FA instances.

usage: fa_q8_sass_gate.py OUT_DIR REF_LIB NEW_LIB [--control CTRL_LIB]
REF_LIB is the base build (frozen/stack10-cand2). Exit 0 when
  - NEW_LIB's <256,256,8|4,8,Q8_0> verify kernels stay within REF (instructions <= +1 %, STL/LDL not above, HMMA/REG/STACK/
    LOCAL/SHARED equal),
  - no kernel common to both libraries changed its resource usage,
  - the new kernels are exactly the expected instances (each present once, nothing else new),
  - the one-row decode kernels still exist.
Resources, STL/LDL and instruction counts of every new instance are printed (informational: spills decide the tile choice
together with the timing). --control: the same verify gate on a library carrying the stack10-cand regression must FAIL.
Instructions are lines starting with /*hhhh*/, STL/LDL lines containing ' STL'/' LDL'.
"""
import re
import subprocess
import sys
from pathlib import Path

INSTR = re.compile(r'^\s+/\*[0-9a-f]{4,}\*/')
RES = re.compile(r'REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)')


def prefix(dk, ncols1, ncols2):
    return rf'_Z18flash_attn_ext_f16ILi{dk}ELi{dk}ELi{ncols1}ELi{ncols2}ELb0ELb0ELb0EL9ggml_type8E'


VERIFY = {n: prefix(256, n, 8) for n in (8, 4)}
DECODE = prefix(256, 1, 8)
EXPECTED_NEW = [(256, 16, 2), (256, 32, 2), (256, 8, 4), (256, 16, 4),
                (128, 16, 2), (128, 32, 2), (128, 8, 4), (128, 16, 4), (128, 4, 8), (128, 8, 8)]


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


def find(usage, pre):
    hits = [n for n in usage if n.startswith(pre)]
    assert len(hits) == 1, f'{pre}: {len(hits)} kernels'
    return hits[0]


def sass(lib, name, path):
    text = run('cuobjdump', '-sass', '-fun', name, str(lib))
    path.write_text(text)
    lines = text.splitlines()
    return dict(instr=sum(bool(INSTR.match(l)) for l in lines), STL=sum(' STL' in l for l in lines),
                LDL=sum(' LDL' in l for l in lines), HMMA=sum(' HMMA' in l for l in lines),
                BRA=sum(' BRA' in l for l in lines))


def row(lib, usage, name, path):
    return dict(name=name, REG=usage[name][0], STACK=usage[name][1], SHARED=usage[name][2], LOCAL=usage[name][3],
                **sass(lib, name, path))


def measure(lib, out, tag):
    out.mkdir(parents=True, exist_ok=True)
    usage = resources(lib, out/f'{tag}.res')
    rows = {n: row(lib, usage, find(usage, pre), out/f'{tag}.{n}.sass') for n, pre in VERIFY.items()}
    decode = [n for n in usage if n.startswith(DECODE)]
    return usage, rows, decode


def violations(ref, new):
    bad = []
    for n in (8, 4):
        r, c = ref[n], new[n]
        if c['instr'] > r['instr']*1.01:
            bad.append(f'ncols1={n}: instructions {c["instr"]} > ref {r["instr"]} + 1 %')
        for key in ('STL', 'LDL'):
            if c[key] > r[key]:
                bad.append(f'ncols1={n}: {key} {c[key]} > ref {r[key]}')
        for key in ('HMMA', 'REG', 'STACK', 'LOCAL', 'SHARED'):
            if c[key] != r[key]:
                bad.append(f'ncols1={n}: {key} {c[key]} != ref {r[key]}')
    return bad


def main():
    args = sys.argv[1:]
    assert len(args) in (3, 5) and (len(args) == 3 or args[3] == '--control'), __doc__
    out, ref_lib, new_lib = Path(args[0]), Path(args[1]), Path(args[2])
    control = Path(args[4]) if len(args) == 5 else None
    usage_ref, ref, _ = measure(ref_lib, out, 'ref')
    usage_new, cur, decode = measure(new_lib, out, 'new')
    for n in (8, 4):
        print(f'ncols1={n} ref {ref[n]}')
        print(f'ncols1={n} new {cur[n]}')
    bad = violations(ref, cur)
    changed = sorted(k for k in usage_ref.keys() & usage_new.keys() if usage_ref[k] != usage_new[k])
    print(f'common kernels: {len(usage_ref.keys() & usage_new.keys())}, resource usage changed: {len(changed)}')
    bad += [f'resource usage of common kernel changed: {k} {usage_ref[k]} -> {usage_new[k]}' for k in changed]
    if not decode:
        bad.append('no one-row decode kernel <256,256,1,8,Q8_0> in the new library')
    removed = sorted(usage_ref.keys() - usage_new.keys())
    bad += [f'kernel missing from the new library: {k}' for k in removed]
    new_names = sorted(usage_new.keys() - usage_ref.keys())
    expected = {}
    for dk, n1, n2 in EXPECTED_NEW:
        pre = prefix(dk, n1, n2)
        hits = [k for k in new_names if k.startswith(pre)]
        if len(hits) != 1:
            bad.append(f'expected new instance <{dk},{dk},{n1},{n2},Q8_0>: {len(hits)} kernels')
        else:
            expected[(dk, n1, n2)] = hits[0]
    extra = sorted(set(new_names) - set(expected.values()))
    bad += [f'unexpected new kernel: {k}' for k in extra]
    print(f'new kernels: {len(new_names)} (expected {len(EXPECTED_NEW)})')
    for (dk, n1, n2), name in sorted(expected.items()):
        r = row(new_lib, usage_new, name, out/f'new.{dk}-{n1}-{n2}.sass')
        print(f'new instance <{dk},{dk},{n1},{n2},Q8_0>: REG={r["REG"]} STACK={r["STACK"]} SHARED={r["SHARED"]} LOCAL={r["LOCAL"]} '
              f'instr={r["instr"]} STL={r["STL"]} LDL={r["LDL"]} HMMA={r["HMMA"]}')
        if r['REG'] > 255:
            bad.append(f'<{dk},{dk},{n1},{n2}> REG {r["REG"]} > 255')
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
