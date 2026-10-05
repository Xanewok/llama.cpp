import json
from pathlib import Path

results = []
for np in (1, 2):
    records = {}
    for arm in ('base', 'patch'):
        text = Path(f'codex/cpu-state-v2-{arm}-np{np}.log').read_text()
        records[arm] = [line for line in text.splitlines() if line.startswith('STATE_MOVERS trial=')]
        assert len(records[arm]) == 8 * np, (arm, np, 'missing trials')
        assert 'STATE_MOVERS negative_controls=PASS' in text, (arm, np, 'negative controls')
    assert records['base'] == records['patch'], (np, 'CPU state/logit/token fingerprints differ')
    changed = records['patch'].copy()
    changed[0] = changed[0].split('token=')[0] + 'token=-1'
    assert changed != records['base'], 'identity negative control escaped'
    results.append({'np': np, 'trials_M': 8, 'fingerprints_M': len(records['patch']),
                    'base_patch_identity': 'PASS', 'wrong_snapshot_index': 'DETECTED',
                    'dropped_snapshot': 'DETECTED', 'identity_negative_control': 'DETECTED', 'fingerprints': records['patch']})
print(json.dumps({'label': 'M', 'baseline': '590912a1d Qwen3.5 graph order; shared CPU kernels',
                  'sampling': 'greedy probe tokens plus bitwise selected state/logits', 'results': results}, indent=2))
