"""Compare tensor bytes in callback order; never infer a causal law from hashes alone."""
import argparse
import json
from pathlib import Path
import re


def named_layer(name):
    match = re.search(r'(?:_l|-|\.)(\d+)(?:$|\D)', name)
    return int(match[1]) if match else None


def read_trace(path, steps=4, draft=True, draft_steps=5):
    path = Path(path)
    if not 0 < path.stat().st_size <= 200_000_000:
        raise ValueError('invalid trace file size')
    rows = []
    expected = {}
    layers = {}
    with path.open() as stream:
        for line in stream:
            if len(line) > 1_000_000:
                raise ValueError('trace row is too large')
            row = json.loads(line)
            if row['kind'] != 'node' or (row['domain'] not in ('target', 'catchup') and not re.fullmatch(r'draft-[0-7]', row['domain'])):
                raise ValueError('unknown trace row')
            key = row['phase'], row['domain']
            if row['index'] != expected.get(key, 0):
                raise ValueError('node order is not contiguous')
            expected[key] = row['index'] + 1
            for tensor in [row['output'], *row['inputs']]:
                if len(tensor['shape']) != 4 or len(tensor['strides']) != 4 or any(n < 0 for n in tensor['shape']):
                    raise ValueError('invalid tensor shape')
                if tensor['status'] not in ('ok', 'frozen_weight', 'unavailable_before_plan'):
                    raise ValueError('unknown tensor status')
                if tensor['status'] == 'ok':
                    if not 1 <= len(tensor['slices']) <= 2:
                        raise ValueError('invalid slice count')
                    for rank, part in enumerate(tensor['slices']):
                        data = part['data']
                        if (part['rank'] != rank or len(part['shape']) != 4 or len(part['strides']) != 4
                                or not re.fullmatch('[0-9a-f]{64}', data['sha256']) or data['bytes'] < 0
                                or not 0 <= data['absmax_bytes'] <= 255 or not 0 <= data['sum_bytes'] <= 255*data['bytes']):
                            raise ValueError('invalid tensor fingerprint')
            if row['output']['status'] != 'ok':
                raise ValueError('missing graph node output')
            known = layers.setdefault(key, {})
            layer = named_layer(row['output']['name'])
            if layer is None:
                candidates = [named_layer(t['name']) if named_layer(t['name']) is not None else known.get(t['name'])
                              for t in row['inputs']]
                layer = max((n for n in candidates if n is not None), default=None)
            row['layer_D'] = layer
            known[row['output']['name']] = layer
            rows.append(row)
    phases = ['prefill', *(f'decode-{i}' for i in range(steps))]
    for phase in phases:
        if (phase, 'target') not in expected:
            raise ValueError('missing target capture phase')
    if draft:
        if ('prefill', 'catchup') not in expected:
            raise ValueError('missing prefill draft catchup')
        for phase in phases[1:]:
            for i in range(draft_steps):
                if (phase, f'draft-{i}') not in expected:
                    raise ValueError('missing forced draft step')
    return rows


def layout(tensor):
    return {key: tensor[key] for key in ('name', 'op', 'type', 'shape', 'strides', 'op_params', 'status')} | {
        'split_axis': tensor.get('split_axis'),
        'slices': [{key: part[key] for key in ('rank', 'shape', 'strides', 'flags')}
                   for part in tensor.get('slices', [])]}


def values(tensor):
    return [part['data'] for part in tensor.get('slices', [])]


def first_difference(a, b):
    for left, right in zip(a, b):
        key = 'phase', 'domain', 'index'
        if any(left[k] != right[k] for k in key) or layout(left['output']) != layout(right['output']):
            return {'class': 'graph_or_layout_difference', 'left': left, 'right': right}
        if values(left['output']) != values(right['output']):
            src_a, src_b = left['inputs'], right['inputs']
            complete = bool(src_a) and bool(src_b) and all(t['status'] in ('ok', 'frozen_weight') for t in src_a + src_b)
            same = complete and src_a == src_b
            op = left['output']['op']
            state_copy = op in ('NONE', 'VIEW', 'RESHAPE', 'PERMUTE', 'TRANSPOSE', 'CPY', 'GET_ROWS', 'SET_ROWS')
            layer = named_layer(left['output']['name'])
            if layer is None:
                layer = left.get('layer_D')
            return {'class': 'state_or_view_bytes' if state_copy else
                    'arithmetic_identical_observed_inputs' if same else 'arithmetic_inputs_differ_or_unobserved',
                    **{k: left[k] for k in key}, 'name': left['output']['name'], 'op': op,
                    'layer_D': layer, 'inputs_complete': complete,
                    'inputs_equal': same, 'left': left, 'right': right}
    if len(a) != len(b):
        return {'class': 'graph_length_difference', 'left_nodes_D': len(a), 'right_nodes_D': len(b)}
    return None


def compare(a, b):
    domains = dict.fromkeys(row['domain'] for row in a + b)
    storage = {'NONE', 'VIEW', 'RESHAPE', 'PERMUTE', 'TRANSPOSE', 'CPY', 'SET_ROWS'}
    def produced(rows):
        return [r for r in rows if r['output']['op'] not in storage]
    return {'first': first_difference(a, b), 'by_domain': {
        domain: first_difference([r for r in a if r['domain'] == domain], [r for r in b if r['domain'] == domain])
        for domain in domains}, 'first_produced_by_domain': {
        domain: first_difference(produced([r for r in a if r['domain'] == domain]),
                                 produced([r for r in b if r['domain'] == domain])) for domain in domains}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('short', type=Path)
    p.add_argument('long', type=Path)
    p.add_argument('--target-only', action='store_true')
    args = p.parse_args()
    print(json.dumps(compare(read_trace(args.short, draft=not args.target_only),
                             read_trace(args.long, draft=not args.target_only)), indent=2))


if __name__ == '__main__':
    main()
