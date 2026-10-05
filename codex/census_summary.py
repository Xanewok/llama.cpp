import collections
import json
from pathlib import Path

QUANT = {'q4_K', 'q5_K', 'q6_K', 'iq4_xs', 'q8_0', 'q3_K', 'q2_K'}
LIMIT = {'q2_K': 5, 'q3_K': 5, 'q4_K': 5, 'q5_K': 6, 'q6_K': 7}
DS = {'q4_K': 'DS4', 'q5_K': 'DS4'}

def summarize(path):
    nodes = [line.rstrip().split('\t') for line in path.read_text().splitlines() if line.startswith('node\t')]
    tensors = {r[1]: r for r in nodes}
    groups = collections.defaultdict(list)
    for i, r in enumerate(nodes):
        if r[3] not in ('MUL_MAT', 'MUL_MAT_ID') or r[6] not in QUANT:
            continue
        src = tensors.get(r[5])
        if src is None:
            continue
        rows = int(src[8])
        variant = 'MMVQ' if rows <= LIMIT.get(r[6], 8) else 'MMQ_' + DS.get(r[6], 'D4')
        groups[r[5], variant].append(dict(index=i, name=r[2], weight=r[6], rows=rows))
    shared = [g for g in groups.values() if len(g) > 1]
    residuals = sum(r[3] == 'RMS_NORM' and tensors.get(r[4], [None] * 4)[3] == 'ADD' for r in nodes)
    return dict(nodes_M=len(nodes), quantized_matmuls_M=sum(map(len, groups.values())),
                shared_groups_D=len(shared), duplicate_quantizations_before_fusion_D=sum(len(g) - 1 for g in shared),
                residual_norm_pairs_M=residuals, shared=shared)

if __name__ == '__main__':
    result = {p.stem: summarize(p) for p in sorted(Path('codex').glob('census-*.tsv'))}
    Path('codex/census.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, data in result.items():
        print(name, {k: v for k, v in data.items() if k != 'shared'})
