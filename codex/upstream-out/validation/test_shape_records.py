import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

recorded = Path(sys.argv[1]).resolve()
checker = Path(__file__).with_name('check_model_results.py')
pattern = r'PACK_SHAPE type=\d+ tokens=\d+ seqs=\d+ outputs=\d+'
logs = [(recorded / f'np2-{arm}.log').read_text() for arm in (0, 1)]
records = [re.findall(pattern, log) for log in logs]
assert records[0] and records[0] == records[1], 'recorded schedules differ'
assert any('PACK_SHAPE ' in line and not line.startswith('PACK_SHAPE ') for log in logs for line in log.splitlines()), 'recorded prefixed shape absent'

with tempfile.TemporaryDirectory(prefix='shape-controls-', dir=recorded.parent) as tmp:
    out = Path(tmp)
    for np in (1, 2):
        for arm in (0, 1):
            for suffix in ('json', 'dump', 'log'):
                name = f'np{np}-{arm}.{suffix}'
                shutil.copyfile(recorded / name, out / name)
    prefixed = logs[1].replace('PACK_SHAPE ', 'parser text.PACK_SHAPE ')
    match = re.search(r'PACK_SHAPE type=1 tokens=1 seqs=1 outputs=1', prefixed)
    assert match, 'recorded single-slot draft absent'
    cases = {
        'prefixed': (prefixed, None),
        'missing': (prefixed[:match.start()] + prefixed[match.end():], 'np2 decode shapes differ'),
        'changed': (prefixed[:match.start()] + match[0].replace('tokens=1', 'tokens=2') + prefixed[match.end():], 'np2 decode shapes differ'),
        'malformed': (prefixed[:match.start()] + match[0].replace('outputs=1', 'outputs=?') + prefixed[match.end():], 'malformed decode shape record'),
    }
    for name, (log, error) in cases.items():
        (out / 'np2-1.log').write_text(log)
        result = subprocess.run([sys.executable, str(checker), 'chain', str(out)], capture_output=True, text=True)
        passes = error is None
        assert (result.returncode == 0) == passes, (name, result.stdout, result.stderr)
        if error:
            assert error in result.stderr, (name, result.stderr)
        print(f'M: shape parser {name}: rc={result.returncode}, expected_pass={passes}')
