"""SASS gate for the windowed-K/V flash attention change: no existing kernel may change, the new instances are dumped.

usage: sass_gate.py OUT_DIR STACK9_LIB BASE_LIB NEW_LIB [--control CAND_LIB]
Exit 0 when NEW_LIB
  - keeps the <256,256,8|4,8,Q8_0> verify kernels within stack9 (instructions <= +1 %, STL/LDL/HMMA/REG/STACK/LOCAL/SHARED
    not above or different) and identical to BASE_LIB (the pkg54-fixed stack10-cand2),
  - has no common kernel (same mangled name in BASE_LIB) whose resource usage changed,
  - differs from BASE_LIB only in the instances of flash_attn_stream_k_fixup_general (new template parameter): the removed and
    the added ones pair up one to one by <DV,ncols1,ncols2>, and the windowed (Lb1) ones exist. The MMA kernels have no new instance.
The windowed instances' resource usage and STL/LDL are printed (new-instances.json); the unwindowed ones are compared informationally.
--control: the verify check on a library that has the stack10 regression must FAIL, otherwise the gate cannot see it.
Counting is the one used for the measured regression: instructions are lines starting with /*hhhh*/, STL/LDL lines containing ' STL'.
"""
import re
import subprocess
import sys
from pathlib import Path

INSTR = re.compile(r'^\s+/\*[0-9a-f]{4,}\*/')
RES = re.compile(r'REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)')
VERIFY = {n: rf'_Z18flash_attn_ext_f16ILi256ELi256ELi{n}ELi8ELb0ELb0ELb0EL9ggml_type8E' for n in (8, 4)}
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


FIXUP = '_Z33flash_attn_stream_k_fixup_general'
FIXUP_KEY = re.compile(r'ILi(\d+)ELi(\d+)ELi(\d+)E(?:Lb(\d))?E')


def fixup_key(name):
    m = FIXUP_KEY.search(name)
    assert m, name
    return m.group(1, 2, 3), m.group(4)


def main():
    import json
    args = sys.argv[1:]
    control = None
    if len(args) == 6 and args[4] == '--control':
        control = Path(args[5])
        args = args[:4]
    assert len(args) == 4, __doc__
    out, s9, base, new = Path(args[0]), Path(args[1]), Path(args[2]), Path(args[3])
    usage9, ref, _ = measure(s9, out, 'stack9')
    usage_base, cur_base, _ = measure(base, out, 'base')
    usage_new, cur, decode = measure(new, out, 'new')
    bad = []
    for n in (8, 4):
        print(f'ncols1={n} stack9 {ref[n]}')
        print(f'ncols1={n} base   {cur_base[n]}')
        print(f'ncols1={n} new    {cur[n]}')
    bad += violations(ref, cur)
    for n in (8, 4):
        keys = ('instr', 'STL', 'LDL', 'HMMA', 'BRA', 'REG', 'STACK', 'LOCAL', 'SHARED')
        if any(cur[n][k] != cur_base[n][k] for k in keys):
            bad.append(f'ncols1={n}: verify instance differs from base {cur_base[n]} -> {cur[n]}')
    common = usage_base.keys() & usage_new.keys()
    changed = sorted(k for k in common if usage_base[k] != usage_new[k])
    print(f'common kernels: {len(common)}, resource usage changed: {len(changed)}')
    bad += [f'resource usage of common kernel changed: {k} {usage_base[k]} -> {usage_new[k]}' for k in changed]
    removed = sorted(usage_base.keys() - usage_new.keys())
    added = sorted(usage_new.keys() - usage_base.keys())
    print(f'removed {len(removed)} added {len(added)} kernels')
    bad += [f'kernel other than the fixup disappeared: {k}' for k in removed if not k.startswith(FIXUP)]
    bad += [f'kernel other than the fixup is new: {k}' for k in added if not k.startswith(FIXUP)]
    # base instances <DV,n1,n2> have no 4th parameter
    base_keys = sorted(fixup_key(k)[0] for k in removed if k.startswith(FIXUP))
    stock_keys = sorted(fixup_key(k)[0] for k in added if k.startswith(FIXUP) and fixup_key(k)[1] == '0')
    windowed = [k for k in added if k.startswith(FIXUP) and fixup_key(k)[1] == '1']
    if base_keys != stock_keys:
        bad.append(f'unwindowed fixup instances {stock_keys} do not replace the base ones {base_keys}')
    if not windowed:
        bad.append('no windowed fixup instance in the new library')
    rows = {}
    for k in windowed + [k for k in added if k.startswith(FIXUP) and fixup_key(k)[1] == '0']:
        row = sass(new, k, out/f'fixup-{len(rows)}.sass')
        rows[k] = dict(res=dict(zip(('REG', 'STACK', 'SHARED', 'LOCAL'), usage_new[k])), **row)
        print(f'{"windowed " if k in windowed else "stock    "}{fixup_key(k)}: {rows[k]}')
    old = {fixup_key(k)[0]: usage_base[k] for k in removed if k.startswith(FIXUP)}
    for k in added:
        if k.startswith(FIXUP) and fixup_key(k)[1] == '0':
            print(f'stock fixup {fixup_key(k)[0]}: base {old.get(fixup_key(k)[0])} new {usage_new[k]} (informational)')
    (out/'new-instances.json').write_text(json.dumps(rows, indent=1))
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
