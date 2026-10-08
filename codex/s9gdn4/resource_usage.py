"""resource_usage.py CAND_LIB BASE_LIB: per-instance registers of gated_delta_net_cuda from cuobjdump --dump-resource-usage.

Gate (rc 1): any candidate instance with a stack frame or local memory (register spills), or an instance count that
differs from the base, or no 128-wide four-column instances. A *.txt argument is read as saved cuobjdump output.
Also prints the resident blocks per SM implied by the registers (65536 regs, 128 threads, allocation unit 256 per warp).
"""
import re
import subprocess
import sys

FUNC = re.compile(r'^\s*Function (_Z\d+gated_delta_net_cuda\S+):')
NAME = re.compile(r'gated_delta_net_cudaILi(\d+)ELb([01])ELb([01])ELi(\d+)(?:ELi(\d+))?EEv')
RES = re.compile(r'^\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)')


def stream(path):
    if path.endswith('.txt'):
        with open(path, errors='replace') as f:
            yield from f
        return
    p = subprocess.Popen(['cuobjdump', '--dump-resource-usage', path], stdout=subprocess.PIPE, text=True, errors='replace')
    yield from p.stdout
    assert p.wait() == 0, f'cuobjdump failed on {path}'


def parse(path):
    rows, cur = {}, None
    for line in stream(path):
        m = FUNC.match(line)
        if m:
            cur = m[1]
            continue
        m = RES.match(line)
        if m and cur:
            n = NAME.search(cur)
            assert n, cur
            key = (int(n[1]), int(n[2]), int(n[3]), int(n[4]), int(n[5] or 1))
            assert key not in rows, f'duplicate instance {key}'
            rows[key] = tuple(int(x) for x in m.groups())
            cur = None
    return rows


def blocks_per_sm(reg):
    per_warp = -(-reg * 32 // 256) * 256
    return 65536 // (per_warp * 4)


def main():
    cand, base = parse(sys.argv[1]), parse(sys.argv[2])
    ok = True
    print('S_v\tKDA\tkeep_rs\tNS\tbase_reg\tcand_reg\tcand_stack\tcand_local\tcand_blocks_per_sm\tbase_blocks_per_sm')
    for (s, kda, keep, ns, cols), (reg, stack, shared, local) in sorted(cand.items()):
        b = base.get((s, kda, keep, ns, 1))
        breg = b[0] if b else -1
        print(f'{s}\t{kda}\t{keep}\t{ns}\t{breg}\t{reg}\t{stack}\t{local}\t{blocks_per_sm(reg)}\t{blocks_per_sm(breg) if b else -1}')
        if cols != 4 or stack or local or shared:
            ok = False
            print(f'  ^ FAIL cols={cols} stack={stack} shared={shared} local={local}')
    if {k[:4] for k in cand} != {k[:4] for k in base}:
        ok = False
        print('FAIL instance set differs from the base')
    if not any(k[0] == 128 and k[4] == 4 for k in cand):
        ok = False
        print('FAIL no 128-wide four-column instance')
    assert all(k[4] == 1 for k in base), 'base already has four-column instances'
    print('resource gate:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


main()
