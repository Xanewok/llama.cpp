"""SASS gate for the multi-query q8_0 verify FA kernels of the stack11 build against frozen/stack10.

Spill stays judged, with a master-anchored reference (owner 10-13): master abee0c847's FA rewrite raised STL/LDL/STACK of the whole
<256,256,8,8> family and master has no q8_0 verify kernel, so a verify kernel's spill is judged by its excess over its f16 sibling
in the same library. The sibling is flash_attn_ext_f16<256,256,ncols1,8,softcap=0,V_is_K_view=0,sparse=0,F16>: every template
argument equal but type_KV, i.e. the same tile loop without the in-kernel q8_0 K/V dequant (the softcap and sparse siblings run other
code). Per ncols1 in (8, 4):
  STL, LDL, STACK, LOCAL:  new(q8) - new(f16) <= ref(q8) - ref(f16)
  instructions <= ref(q8) + 1 %, BRA <= ref(q8), HMMA and SHARED == ref(q8); REG recorded, not judged
  anchor: new(f16) equals MASTER_LIB's kernel of the same arguments (master has no type_KV parameter) in every counted field;
  otherwise a stack change moved the sibling and the excess bound means nothing.
The one-row decode kernel <256,256,1,8,Q8_0> must exist. Resource changes of other common kernels are listed, not judged.
Negative controls, each must fail: CAND_LIB (frozen/stack10-cand, the measured regression) on a spill key; mutants of NEW's dumped
resource text: q8 STACK one step past the bound at ncols1=8 and at 4, and the ncols1=8 f16 sibling's STACK raised (anchor).

usage: sass_gate.py OUT_DIR REF_LIB NEW_LIB --master MASTER_LIB --control CAND_LIB
       sass_gate.py --replay DIR     judge the texts a previous run dumped into DIR (no cuobjdump)
       sass_gate.py --selftest       the judge on synthetic texts
Counting is the one used for the measured regression: instructions are lines starting with /*hhhh*/, STL/LDL/HMMA/BRA are lines
containing ' STL' etc.
"""
import contextlib
import io
import re
import subprocess
import sys
from pathlib import Path

INSTR = re.compile(r'^\s+/\*[0-9a-f]{4,}\*/')
RES = re.compile(r'REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)')
NCOLS1 = (8, 4)
SPILL = ('STL', 'LDL', 'STACK', 'LOCAL')
FIELDS = ('REG', 'STACK', 'SHARED', 'LOCAL', 'instr', 'STL', 'LDL', 'HMMA', 'BRA')
FA_PREFIX = '_Z18flash_attn_ext_'
DECODE = '_Z18flash_attn_ext_f16ILi256ELi256ELi1ELi8ELb0ELb0ELb0EL9ggml_type8E'
STEP = 8  # STACK grows in 8-byte units
TAGS = ('ref', 'new', 'master', 'control')


def prefix(n, kind):
    head = f'_Z18flash_attn_ext_f16ILi256ELi256ELi{n}ELi8ELb0ELb0ELb0E'
    return head + {'q8': 'L9ggml_type8E', 'f16': 'L9ggml_type1E', 'master': 'Ev'}[kind]


def kinds_of(tag):
    return ('master',) if tag == 'master' else ('q8', 'f16')


def run(*cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def parse_res(text):
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


def counts(text):
    lines = text.splitlines()
    return dict(instr=sum(bool(INSTR.match(l)) for l in lines), STL=sum(' STL' in l for l in lines),
                LDL=sum(' LDL' in l for l in lines), HMMA=sum(' HMMA' in l for l in lines), BRA=sum(' BRA' in l for l in lines))


def dump(lib, out, tag):
    out.mkdir(parents=True, exist_ok=True)
    text = run('cuobjdump', '--dump-resource-usage', str(lib))
    (out/f'{tag}.res').write_text(text)
    usage = parse_res(text)
    for n in NCOLS1:
        for kind in kinds_of(tag):
            (out/f'{tag}.{n}.{kind}.sass').write_text(run('cuobjdump', '-sass', '-fun', find(usage, prefix(n, kind)), str(lib)))


def texts(out, tag):
    return (out/f'{tag}.res').read_text(), {(n, k): (out/f'{tag}.{n}.{k}.sass').read_text() for n in NCOLS1 for k in kinds_of(tag)}


def rows(res_text, sass, tag):
    """{(ncols1, kind): fields}; master's kernel is filed as f16, beside the sibling it anchors."""
    usage = parse_res(res_text)
    out = {}
    for n in NCOLS1:
        for kind in kinds_of(tag):
            name = find(usage, prefix(n, kind))
            reg, stack, shared, local = usage[name]
            out[n, 'f16' if kind == 'master' else kind] = dict(name=name, REG=reg, STACK=stack, SHARED=shared, LOCAL=local,
                                                               **counts(sass[n, kind]))
    return usage, out


def violations(ref, new, master=None):
    bad = []
    for n in NCOLS1:
        r, c = ref[n, 'q8'], new[n, 'q8']
        if c['instr'] > r['instr']*1.01:
            bad.append(f'ncols1={n}: instructions {c["instr"]} > reference {r["instr"]} + 1 %')
        if c['BRA'] > r['BRA']:
            bad.append(f'ncols1={n}: BRA {c["BRA"]} > reference {r["BRA"]}')
        for key in ('HMMA', 'SHARED'):
            if c[key] != r[key]:
                bad.append(f'ncols1={n}: {key} {c[key]} != reference {r[key]}')
        for key in SPILL:
            got, lim = c[key] - new[n, 'f16'][key], r[key] - ref[n, 'f16'][key]
            if got > lim:
                bad.append(f'ncols1={n}: spill {key} excess over the f16 sibling {got} ({c[key]} - {new[n, "f16"][key]}) > '
                           f'reference {lim} ({r[key]} - {ref[n, "f16"][key]})')
        if master is not None:
            moved = {k: (new[n, 'f16'][k], master[n, 'f16'][k]) for k in FIELDS if new[n, 'f16'][k] != master[n, 'f16'][k]}
            if moved:
                bad.append(f'ncols1={n}: anchor: the f16 sibling differs from master, (new, master): {moved}')
    return bad


def set_stack(res_text, name, value):
    lines = res_text.splitlines(keepends=True)
    i = next(i for i, l in enumerate(lines) if re.match(rf'\s*Function {re.escape(name)}:', l))
    lines[i + 1], k = re.subn(r'STACK:\d+', f'STACK:{value}', lines[i + 1])
    assert k == 1, name
    return ''.join(lines)


def mutants(ref, new, new_res):
    """(label, mutated copy of NEW's resource text, prefix of the violation it must raise)."""
    out = []
    for n in NCOLS1:
        bound = new[n, 'f16']['STACK'] + ref[n, 'q8']['STACK'] - ref[n, 'f16']['STACK']
        out.append((f'q8 ncols1={n} STACK {new[n, "q8"]["STACK"]} -> {bound + STEP} (bound {bound})',
                    set_stack(new_res, new[n, 'q8']['name'], bound + STEP), f'ncols1={n}: spill STACK'))
    f = new[8, 'f16']
    out.append((f'f16 sibling ncols1=8 STACK {f["STACK"]} -> {f["STACK"] + STEP}', set_stack(new_res, f['name'], f['STACK'] + STEP),
                'ncols1=8: anchor'))
    return out


def judge(t):
    """t: tag -> (resource text, sass texts). Returns the violations; empty is PASS."""
    rr = {tag: rows(*t[tag], tag) for tag in TAGS}
    ref, new, master, control = (rr[tag][1] for tag in TAGS)
    for tag in TAGS:
        for (n, kind), v in sorted(rr[tag][1].items()):
            print(f'{tag:8} ncols1={n} {kind:4}', {k: v[k] for k in FIELDS})
    for n in NCOLS1:
        for key in SPILL:
            got, lim = new[n, 'q8'][key] - new[n, 'f16'][key], ref[n, 'q8'][key] - ref[n, 'f16'][key]
            print(f'ncols1={n} {key:5} q8 excess over f16: new {got:5} <= ref {lim:5}, slack {lim - got}')
    bad = violations(ref, new, master)
    if not any(n.startswith(DECODE) for n in rr['new'][0]):
        bad.append('no one-row decode kernel <256,256,1,8,Q8_0> in the new library')
    seen = violations(ref, control)
    print(f'control violations: {seen}')
    if not any(': spill ' in s for s in seen):
        bad.append('negative control frozen/stack10-cand shows no spill violation: the spill bound cannot see the regression')
    for label, res, want in mutants(ref, new, t['new'][0]):
        seen = violations(ref, rows(res, t['new'][1], 'new')[1], master)
        print(f'mutant {label}: {seen}')
        if not any(s.startswith(want) for s in seen):
            bad.append(f'mutant escaped: {label}')
    return bad


def report(bad):
    for b in bad:
        print('GATE VIOLATION:', b)
    print('sass gate', 'FAIL' if bad else 'PASS')
    return 1 if bad else 0


def changed_kernels(out):
    usage_ref, usage_new = parse_res((out/'ref.res').read_text()), parse_res((out/'new.res').read_text())
    common = usage_ref.keys() & usage_new.keys()
    changed = sorted(k for k in common if usage_ref[k] != usage_new[k])
    print(f'common kernels: {len(common)}, common flash_attn_ext instances: {sum(k.startswith(FA_PREFIX) for k in common)}, '
          f'resource usage changed (informational): {len(changed)} ({sum(k.startswith(FA_PREFIX) for k in changed)} FA)')
    for k in changed:
        print(f'kernel changed (informational): {k} {usage_ref[k]} -> {usage_new[k]}')


def fake(tag, spec):
    """A synthetic dump: spec (ncols1, kind) -> fields."""
    res, sass = [], {}
    for (n, kind), v in spec.items():
        res += [f' Function {prefix(n, kind)}vx:', f'  REG:{v["REG"]} STACK:{v["STACK"]} SHARED:{v["SHARED"]} LOCAL:{v["LOCAL"]}']
        body = [' STL'] * v['STL'] + [' LDL'] * v['LDL'] + [' HMMA'] * v['HMMA'] + [' BRA'] * v['BRA']
        body += [' NOP'] * (v['instr'] - len(body))
        sass[n, kind] = '\n'.join(f'        /*{i*16:04x}*/ {op} ;' for i, op in enumerate(body))
    if tag == 'new':
        res += [f' Function {DECODE}Evx:', '  REG:255 STACK:16 SHARED:1024 LOCAL:0']
    return '\n'.join(res) + '\n', sass


def selftest():
    # bw-stack11-base a2's libraries as measured: REG STACK SHARED LOCAL instr STL LDL HMMA BRA
    m = lambda *v: dict(zip(FIELDS, v))
    s10 = {(8, 'q8'): m(255, 128, 1024, 0, 10056, 92, 80, 768, 36), (8, 'f16'): m(254, 56, 1024, 0, 7768, 23, 11, 768, 32),
           (4, 'q8'): m(255, 40, 1024, 0, 7848, 29, 13, 384, 50), (4, 'f16'): m(252, 16, 1024, 0, 5616, 18, 6, 384, 48)}
    s11 = {(8, 'q8'): m(255, 168, 1024, 0, 10128, 158, 132, 768, 35), (8, 'f16'): m(255, 248, 1024, 0, 8112, 190, 150, 768, 31),
           (4, 'q8'): m(255, 40, 1024, 0, 7768, 29, 13, 384, 50), (4, 'f16'): m(254, 16, 1024, 0, 5576, 18, 6, 384, 48)}
    cand = dict(s10)
    cand[8, 'q8'] = m(255, 128, 1024, 0, 10536, 151, 108, 768, 94)
    cand[4, 'q8'] = m(255, 48, 1024, 0, 8216, 42, 21, 384, 112)
    mas = {(n, 'master'): s11[n, 'f16'] for n in NCOLS1}
    t = dict(ref=fake('ref', s10), new=fake('new', s11), master=fake('master', mas), control=fake('control', cand))

    def bumped(n, kind, **kw):
        s = dict(s11)
        s[n, kind] = dict(s11[n, kind], **kw)
        return dict(t, new=fake('new', s))

    assert judge(t) == [], 'stack11 must pass'
    assert any('negative control' in b for b in judge(dict(t, control=fake('control', s10)))), 'a control without spill growth'
    for key in ('STL', 'LDL', 'LOCAL'):
        bad = judge(bumped(4, 'q8', **{key: s11[4, 'q8'][key] + 1}))
        assert any(b.startswith(f'ncols1=4: spill {key}') for b in bad), (key, bad)
    for key, v in (('BRA', 51), ('instr', 7927), ('HMMA', 383)):
        assert judge(bumped(4, 'q8', **{key: v})), key
    # spill raised in both kernels keeps the excess: only the anchor sees it
    s = dict(s11)
    s[8, 'q8'] = dict(s11[8, 'q8'], STL=s11[8, 'q8']['STL'] + 40)
    s[8, 'f16'] = dict(s11[8, 'f16'], STL=s11[8, 'f16']['STL'] + 40)
    bad = judge(dict(t, new=fake('new', s)))
    assert bad and all('anchor' in b for b in bad), bad


def main(argv):
    if argv == ['--selftest']:
        with contextlib.redirect_stdout(io.StringIO()):
            selftest()
        print('sass gate selftest PASS')
        return 0
    if len(argv) == 2 and argv[0] == '--replay':
        out = Path(argv[1])
        return report(judge({tag: texts(out, tag) for tag in TAGS}))
    if len(argv) != 7 or argv[3] != '--master' or argv[5] != '--control':
        raise SystemExit(__doc__)
    out = Path(argv[0])
    for tag, lib in zip(TAGS, (argv[1], argv[2], argv[4], argv[6])):
        dump(Path(lib), out, tag)
    changed_kernels(out)
    return report(judge({tag: texts(out, tag) for tag in TAGS}))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
