"""k9fix_sass.py BASE.sass CAND.sass KERNELS... -> per-kernel SASS comparison on stdout, rc 0 when CAND matches BASE.

Inputs are `cuobjdump -sass` listings of the same object (an MMQ instance or quantize.cu.o) from the stock build (BASE) and
a PDL build (CAND). KERNELS are mangled-name prefixes up to the template arguments (e.g. `_Z9mul_mat_qIL9ggml_type23ELi16ELb0`);
each must match at least one function in both listings. Kernels are paired by the mangled name up to `EEv`, so the extra
`pdl_mutant` parameter of the local commit does not break the pairing.

CAND matches BASE when, per paired kernel:
- every memory and MMA opcode, with all modifiers (LDG.E.CONSTANT vs LDG.E, LDS, STG, LDSM, LDL/STL, IMMA, ...), has the same count;
- the remaining opcodes differ by at most SLACK in total, not counting ACQBULK/PREEXIT (the PDL wait/trigger) and NANOSLEEP
  (the local mutant sleeps);
- CAND has a griddepcontrol.wait (ACQBULK) and no global memory access before the first one in program order.
"""
import collections
import re
import sys

SLACK = 32
PDL_OPS = {'ACQBULK', 'PREEXIT', 'NANOSLEEP'}
STRICT = re.compile(r'^(LDG|STG|LDS|STS|LDSM|LDGSTS|LDL|STL|ATOM|ATOMG|ATOMS|RED|REDG|IMMA|HMMA|QMMA|OMMA|UTMALDG)\b')
GLOBAL = re.compile(r'^(LDG|STG|LDGSTS|ATOMG|RED|REDG)\b')
INSN = re.compile(r'^\s+/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;')


def parse(path):
    funcs, cur = {}, None
    with open(path) as f:
        for line in f:
            m = re.search(r'Function : (\S+)', line)
            if m:
                cur = m.group(1)
                funcs[cur] = []
                continue
            m = INSN.match(line)
            if m and cur is not None:
                op = re.sub(r'^@!?U?P[T0-9]+\s+', '', m.group(1)).split()[0]
                funcs[cur].append(op)
    return funcs


def pair_key(name):
    return name.split('EEv', 1)[0]


def compare(b_ops, c_ops):
    errs = []
    bs = collections.Counter(o for o in b_ops if STRICT.match(o))
    cs = collections.Counter(o for o in c_ops if STRICT.match(o))
    for op in sorted(set(bs) | set(cs)):
        if bs[op] != cs[op]:
            errs.append(f'{op} base={bs[op]} cand={cs[op]}')
    bo = collections.Counter(o.split('.')[0] for o in b_ops if not STRICT.match(o) and o.split('.')[0] not in PDL_OPS)
    co = collections.Counter(o.split('.')[0] for o in c_ops if not STRICT.match(o) and o.split('.')[0] not in PDL_OPS)
    other = sum(abs(bo[k] - co[k]) for k in set(bo) | set(co))
    if other > SLACK:
        errs.append(f'other opcodes differ by {other} > {SLACK}')
    waits = [i for i, o in enumerate(c_ops) if o.startswith('ACQBULK')]
    glob = [i for i, o in enumerate(c_ops) if GLOBAL.match(o)]
    if not waits:
        errs.append('no ACQBULK (griddepcontrol.wait)')
    elif glob and glob[0] < waits[0]:
        errs.append(f'global access at insn {glob[0]} before the first ACQBULK at {waits[0]}')
    summary = (f"n_base={len(b_ops)} n_cand={len(c_ops)} other_diff={other} "
               f"ldg_const_base={sum(v for k, v in bs.items() if k.startswith('LDG') and 'CONSTANT' in k)} "
               f"ldg_const_cand={sum(v for k, v in cs.items() if k.startswith('LDG') and 'CONSTANT' in k)} "
               f"acqbulk={len(waits)}")
    return errs, summary


def main():
    base, cand = parse(sys.argv[1]), parse(sys.argv[2])
    bad = 0
    for prefix in sys.argv[3:]:
        bk = {pair_key(n): n for n in base if n.startswith(prefix)}
        ck = {pair_key(n): n for n in cand if n.startswith(prefix)}
        if not bk or set(bk) != set(ck):
            print(f'FAIL {prefix}: unpaired kernels base={sorted(bk)} cand={sorted(ck)}')
            bad += 1
            continue
        for k in sorted(bk):
            errs, summary = compare(base[bk[k]], cand[ck[k]])
            print(f"{'FAIL' if errs else 'PASS'} {k} {summary}" + ''.join(f'\n    {e}' for e in errs))
            bad += bool(errs)
    print(f'k9fix_sass: {"FAIL" if bad else "PASS"} {bad} kernel(s) differ')
    return 1 if bad else 0


sys.exit(main())
