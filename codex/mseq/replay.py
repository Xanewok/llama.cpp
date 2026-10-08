"""Mac-only replay of the GPU harness parsers, using recorded GPU artifacts copied with their relative paths."""
import os
from pathlib import Path
import sys
import tomllib

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'codex'))
import gpu_stack7
from harness import lib

gpu_stack7.fixture_checks()
record = gpu_stack7.REAL / 'server.log'
for np in (1, 2):
    name = f'fold-mseq-np{np}-' + ('real' if np == 2 else 'depth') + '-j5q'
    x = tomllib.loads((root / f'codex/mseq/items/{name}.toml.in').read_text())
    markers = x['arms']['patch']['canary_log']
    assert markers == [rf'mtp_fold_decode: engaged n_seq={np}\b']
    try:
        lib.verify_log_patterns(record.read_text(), markers, 'recorded base without multi-sequence fold')
    except lib.Fail:
        print(f'PASS recorded artifact rejects absent n_seq={np} fold proof')
    else:
        raise AssertionError('missing fold proof escaped')
print('PASS parser replay; positive fold engagement remains a GPU canary gate')
