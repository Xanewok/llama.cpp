"""Judge the bw-ext-driver CPU matrix. Usage: bw-ext-compare.py OUT_DIR.

Run files: {meta}-{mode}-np{np}-{variant}.json; variant ref/ref2 = stack6 base build, off = this tree with the extension
off, on = extension on. Exit: 0 pass | 2 off differs from the stack6 base | 5 base not repeatable | 6 no engagement |
7 OOV gate (off counted anything, or on accepted no out-of-head draft token on the copy task) | 8 missing run or crash.
Greedy tokens ext on vs off are reported (common prefix), not gated: the arms verify different batch shapes.
"""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
CASES = [(meta, mode, 1) for meta in (0, 1) for mode in ('greedy', 'prob')] + [(meta, 'prob', 2) for meta in (0, 1)]


def load(meta, mode, np, variant):
    path = OUT / f'{meta}-{mode}-np{np}-{variant}.json'
    if not path.exists():
        print(f'FAIL missing {path}')
        sys.exit(8)
    return json.loads(path.read_text())


def core(run):
    return [(p['tokens'], [(s['draft'], s['accepted'], s['rounds']) for s in p['stats']]) for p in run['phases']]


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    rc = 0
    for meta, mode, np in CASES:
        tag = f'meta={meta} {mode} np{np}'
        ref, off, on = (load(meta, mode, np, v) for v in ('ref', 'off', 'on'))
        if core(ref) != core(load(meta, mode, np, 'ref2')):
            print(f'FAIL {tag}: stack6 base is not repeatable')
            sys.exit(5)
        if core(off) != core(ref):
            print(f'FAIL {tag}: extension off differs from the stack6 base (tokens or draft counters)')
            rc = rc or 2
        ext_off, ext_on = off['ext_stats'], on['ext_stats']
        if ext_off['drafted_oov'] or ext_off['accepted_oov'] or ext_off['requests']:
            print(f'FAIL {tag}: extension off is not inert (N = 0 must not count): {ext_off}')
            rc = rc or 7
        if ext_on['requests'] != 2 * np or ext_on['placed'] <= 0:
            print(f'FAIL {tag}: no engagement: {ext_on}')
            rc = rc or 6
        if np == 1 and mode == 'greedy' and ext_on['accepted_oov'] <= 0:
            print(f'FAIL {tag}: extension on accepted no out-of-head draft token on the copy task: {ext_on}')
            rc = rc or 7
        for p_off, p_on in zip(off['phases'], on['phases']):
            for s_off, s_on, t_off, t_on in zip(p_off['stats'], p_on['stats'], p_off['tokens'], p_on['tokens']):
                print(f'M {tag} phase {p_off["phase"]} slot {s_off["slot"]}: accept/draft off {s_off["accepted"]}/{s_off["draft"]} '
                      f'on {s_on["accepted"]}/{s_on["draft"]}; tokens/round off {len(t_off) / max(1, s_off["rounds"]):.3f} '
                      f'on {len(t_on) / max(1, s_on["rounds"]):.3f}; common token prefix {common_prefix(t_off, t_on)}/{len(t_off)}')
        print(f'M {tag}: ext on {ext_on}; ext off {ext_off}')
    print('PASS' if rc == 0 else f'rc={rc}')
    sys.exit(rc)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # a malformed run file is a CPU-stage failure, not a verdict
        print(f'FAIL compare crashed: {type(e).__name__}: {e}')
        sys.exit(8)
