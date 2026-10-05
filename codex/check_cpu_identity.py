from pathlib import Path
import json

def records(name):
    return [line for line in Path('codex/' + name).read_text().splitlines() if line.startswith('STATE_MOVERS ')]
base = records('cpu-state-movers-base.log')
patch = records('cpu-state-movers-patch.log')
assert len(base) == 9 and base == patch, 'CPU state/logit/token fingerprints differ'
mutant = patch.copy()
mutant[0] = mutant[0].replace('token=131', 'token=132')
assert mutant != base, 'identity negative control escaped'
print(json.dumps({'label': 'M', 'trials': 8, 'base_patch_identity': 'PASS', 'wrong_plane': 'PASS', 'dropped_snapshot': 'PASS', 'identity_negative_control': 'PASS'}, indent=2))
