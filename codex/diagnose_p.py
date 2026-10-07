#!/usr/bin/env python3
import json
from pathlib import Path
import sys

from check_rows import distance
from spec_tf import array


def main():
    target = Path(sys.argv[1])
    trace = Path(sys.argv[2])
    h = json.loads((target / 'header.json').read_text())
    rows = [json.loads(x) for x in trace.read_text().splitlines()]
    output = []
    for i, stream in enumerate(h['streams']):
        n, pk = stream['n_tokens'], h['p_width']
        ids = array(target / str(i), 'p_ids', '<i4', [n, pk])
        probs = array(target / str(i), 'p_probs', '<f4', [n, pk])
        t = stream['n_prompt']
        errors = []
        for row in rows:
            if row['r'] != i:
                continue
            for j, p in enumerate(row['pall']):
                if t + j < n - 1:
                    errors.append(distance(ids[t + j], probs[t + j], p)[0])
            t += row['L'] + 1
        output.append({'stream': i, 'p_max_abs_error_M': max(errors), 'p_mean_abs_error_M': sum(errors) / len(errors)})
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
