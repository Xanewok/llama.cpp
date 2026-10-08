"""Which function of which component window is each component's CUDA gate (single source for gates.sh, shfn selftest, preflight).

usage: gates_table.py code COMP    shell text defining COMP's extracted gate functions
       gates_table.py comps        selectable component names in run order
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shfn

K6_SPLIT_A = ['--cut', r'run "\$OUT/addrms-\$dev\.log" -- -b', r'^    done$']
K6_SPLIT_B = ['--cut', r'run "\$OUT/shared-\$dev\.log" -- -b', r'run "\$OUT/addrms-\$dev\.log" -- -b']

# (component, script relative to the tree root, shfn emit arguments); the last NAME of a row takes --cut.
EXTRACTIONS = [
    ('base', 'codex/bw-stack10-cand.sh', ['ops']),
    ('k7', 'codex/k7/bw-k7.sh', ['run_pair', 'ops']),
    ('k4', 'codex/k4/bw-k4.sh', ['--var', 'MODEL', 'tbo', 'arm', 'model', 'ops']),
    ('k6a', 'codex/k6/bw-s10-k6.sh', ['run', 'count', 'ops', *K6_SPLIT_A]),
    ('k6b', 'codex/k6/bw-s10-k6.sh', ['run', 'count', 'ops', *K6_SPLIT_B]),
    ('k6', 'codex/k6/bw-s10-k6.sh', ['run', 'memcheck']),
    ('k2', 'codex/k2/bw-s10-k2.sh', ['--var', 'PASS_LINE', 'tbo', 'identity', 'ops']),
    ('k10', 'codex/k10/bw-k10.sh', ['tbo', 'ops', 'controls']),
    ('k9', 'codex/k9/bw-k9.sh', ['--var', 'RC', 'chain', 'ops']),
    ('k11b', 'codex/k11b/bw-s10-k11b.sh', ['--var', 't', 'run_test', 'ops']),
    ('fafix', 'codex/bw-s10c-fafix.sh', ['runid', 'same', 'identity']),
]
ORDER = ['p55', 'k7', 'k4', 'k6a', 'k6b', 'k2', 'k10', 'k9', 'k8', 'k11b', 'k11a']


def code(comp):
    parts = [shfn.emit([str(shfn.ROOT / script), *args]) for c, script, args in EXTRACTIONS if c == comp]
    if not parts:
        raise SystemExit(f'gates_table: no extraction for {comp}')
    return ''.join(parts)


if __name__ == '__main__':
    if sys.argv[1:2] == ['code'] and len(sys.argv) == 3:
        sys.stdout.write(code(sys.argv[2]))
    elif sys.argv[1:] == ['comps']:
        print(' '.join(ORDER))
    else:
        raise SystemExit(__doc__)
