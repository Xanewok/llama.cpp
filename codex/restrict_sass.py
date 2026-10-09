"""restrict_sass.py STOCK_DIR FORCED_DIR CUOBJDUMP > sass-counts.tsv: global-load census of the #24030 objects, stock vs forced restrict.

STOCK_DIR/FORCED_DIR are ggml-cuda.dir object dirs of the two builds. Per object: LDG with .CONSTANT (read-only path), plain LDG,
and generic LD, counted from `cuobjdump -sass`. Contract: the forced build must show more .CONSTANT and fewer plain LDG in mmvq.cu.o
(the switch engaged); otherwise rc 5 (negative control void). Object sets must match (rc 6).
"""
import re
import subprocess
import sys
from pathlib import Path

TOUCHED = re.compile(r'^(fattn.*|getrows|mmvf|mmvq|quantize|sumrows|mean|set-rows|ssm-conv|ssm-scan)\.cu\.o$')
LDG = re.compile(r'(?:^|\s)LDG(\.[A-Za-z0-9_]+)*\s')
LD = re.compile(r'(?:^|\s)LD(\.[A-Za-z0-9_]+)*\s')


def objects(root):
    return {str(p.relative_to(root)): p for p in sorted(root.rglob('*.cu.o')) if TOUCHED.match(p.name)}


def census(cuobjdump, obj):
    const = plain = generic = 0
    proc = subprocess.Popen([cuobjdump, '-sass', str(obj)], stdout=subprocess.PIPE, text=True, errors='replace')
    for line in proc.stdout:
        if 'LD' not in line or '/*' not in line:
            continue
        m = LDG.search(line)
        if m:
            if 'CONSTANT' in m.group(0):
                const += 1
            else:
                plain += 1
        elif LD.search(line):
            generic += 1
    if proc.wait() != 0:
        raise SystemExit(f'restrict_sass: cuobjdump failed on {obj}')
    return const, plain, generic


def main():
    stock, forced, cuobjdump = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    so, fo = objects(stock), objects(forced)
    if not so or set(so) != set(fo):
        print(f'restrict_sass: object sets differ or are empty (stock {len(so)}, forced {len(fo)})', file=sys.stderr)
        return 6
    print('object\tstock_const\tstock_plain\tstock_generic\tforced_const\tforced_plain\tforced_generic\tconst_delta')
    rows = {}
    for rel in sorted(so):
        s, f = census(cuobjdump, so[rel]), census(cuobjdump, fo[rel])
        rows[rel] = (s, f)
        print(f'{rel}\t{s[0]}\t{s[1]}\t{s[2]}\t{f[0]}\t{f[1]}\t{f[2]}\t{f[0] - s[0]:+d}')
    s, f = rows['mmvq.cu.o']
    tot_s = [sum(r[0][i] for r in rows.values()) for i in range(3)]
    tot_f = [sum(r[1][i] for r in rows.values()) for i in range(3)]
    changed = sum(1 for r in rows.values() if r[0] != r[1])
    print(f'restrict_sass: objects={len(rows)} changed={changed} total const {tot_s[0]}->{tot_f[0]} plain {tot_s[1]}->{tot_f[1]}',
          file=sys.stderr)
    if not (f[0] > s[0] and f[1] < s[1]):
        print(f'restrict_sass: negative control void: mmvq.cu.o const {s[0]}->{f[0]} plain {s[1]}->{f[1]} (switch not engaged)',
              file=sys.stderr)
        return 5
    return 0


sys.exit(main())
