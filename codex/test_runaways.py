#!/usr/bin/env python3
import argparse

import runaways


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--broken-detector', action='store_true')
    args = ap.parse_args()
    if args.broken_detector:
        runaways.spans = lambda _: []
    letters = 'prefix\n' + 'S, F, O, R, T, H, E, ' * 400 + '\nend'
    zeros = '{"cmd":"' + '[0, 0, 0, 0],\\n' * 800 + '"}'
    lines = 'same repeated phrase\n' * 300
    for output in [letters, zeros, lines]:
        assert runaways.spans(output), 'detector missed positive fixture'
    assert not runaways.spans('S, F, O, R, T, H, E, ' * 20)
    assert not runaways.spans('\n'.join(f'row {i}: result = value_{i} + {i * 17}' for i in range(3000)))
    assert not list(runaways.generated(dict(type='function_call_output', output=letters)))
    assert not list(runaways.generated(dict(type='message', role='user', content=[dict(text=letters)])))
    assert list(runaways.generated(dict(type='function_call', arguments=zeros)))
    assert runaways.ci(0, 16)[0] == 0
    assert abs(runaways.ci(0, 16)[1] - 0.2059072142078227) < 1e-12
    rows = [dict(exposure_id='same', complete=True, completion_tokens=10, generated_chars=100,
                 attempt=p, **{k: 'value' for k in runaways.FACTORS}) for p in ['a', 'b']]
    unique = runaways.deduplicate(rows)
    assert len(unique) == 1 and unique[0]['aliases'] == ['b']
    print('M: detector positives, ordinary output controls, generated-only filtering, deduplication and exact CI pass')


if __name__ == '__main__':
    main()
