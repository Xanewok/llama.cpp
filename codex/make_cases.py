"""Export TP2 MUL_MAT operators in test-export-graph-ops format."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
TYPES = {'iq4_xs': (23, 136, 256), 'q5_K': (13, 176, 256), 'q6_K': (14, 210, 256),
         'q8_0': (8, 34, 32), 'q4_K': (12, 144, 256)}
SHAPES = [(8704, 5120), (5120, 8704), (6144, 5120), (512, 5120),
          (5120, 3072), (5120, 5120), (3072, 5120)]

def row(quant, m, k, n):
    type_id, size, block = TYPES[quant]
    row_bytes = k // block * size
    name = f'mmvq-wide:{quant}:{m}:{k}:{n}'
    return ' '.join(map(str, [29, 0, m, n, 1, 1, 0, 2,
                              type_id, k, m, 1, 1, size, row_bytes, row_bytes*m, row_bytes*m,
                              0, k, n, 1, 1, 4, 4*k, 4*k*n, 4*k*n, name]))

def write(name, widths, shapes):
    (HERE / 'cases' / (name + '.txt')).write_text('\n'.join(
        row(t, m, k, n) for t in TYPES for m, k in shapes for n in widths) + '\n')

if __name__ == '__main__':
    write('eval', range(9, 17), SHAPES)
    write('perf', [6, 12], SHAPES)
    write('canary', [9, 12, 16], [(4, 256)])
    write('cpu', range(9, 17), [(512, 5120), (1023, 256)])
