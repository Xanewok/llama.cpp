#!/usr/bin/env python3
"""Break-even analysis; MTP skip savings and lookup overhead are guesses, not measurements."""
import json
from pathlib import Path

from replay import expected_core, summarize


def main():
    here = Path(__file__).resolve().parent
    data = [json.loads(p.read_text()) for p in sorted((here / 'cache').glob('campaign-hq15-ckptab-g*.json'))]
    # Initializes arrays and calibrates both acceptance sensitivities.
    summarize(data)
    output = []
    for marginal_ms in (1.4, 2.5):
        for overhead_ms in (0., .1, .3):
            for saved_draft_ms in (0., 2., 4., 6., 8.):
                base_s = hybrid_s = 0.
                per_campaign = {}
                for task in data:
                    alpha = task['alphas'][True]
                    step = task['step_ms']
                    ratio = 1 + (12-task['k']) * marginal_ms / step - saved_draft_ms / step
                    b = h = 0.
                    for row, arrays in zip(task['rows'], task['arrays']):
                        rb, rh, _, _ = expected_core(*arrays, task['k'], alpha, 12, 12, ratio, True, overhead_ms/step)
                        missing = max(0, row['usage_out']-len(row['labs'])) / task['tau']
                        b += rb + missing
                        # Charge lookup overhead on missing tokens too, without any copy benefit.
                        h += rh + missing*(1 + overhead_ms/step)
                    missing = task.get('excluded_tokens', 0) / task['tau']
                    b += missing
                    h += missing*(1 + overhead_ms/step)
                    base_s += task['decode_s']
                    hybrid_s += task['decode_s'] * h/b
                    cb, ch = per_campaign.get(task['campaign'], (0., 0.))
                    per_campaign[task['campaign']] = cb+task['decode_s'], ch+task['decode_s']*h/b
                row = {'marginal_ms': marginal_ms, 'overhead_ms_G': overhead_ms,
                       'saved_draft_ms_G': saved_draft_ms,
                       'speed_pct_D': 100*(base_s/hybrid_s-1),
                       'campaigns': {c:100*(b/h-1) for c,(b,h) in per_campaign.items()}}
                output.append(row)
                print(marginal_ms, overhead_ms, saved_draft_ms, round(row['speed_pct_D'], 2))
    (here / 'sensitivity.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
