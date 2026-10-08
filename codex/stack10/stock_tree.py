"""Source of the stock-MMQ reference build: HEAD with the K8 and K9 ggml commits reverted, so the MMQ code is that of the base.

usage: stock_tree.py list            commits to revert, newest first (each subject must match exactly one commit of BASE..HEAD)
       stock_tree.py make DEST       git archive HEAD into DEST (new directory), revert them in order, verify the MMQ files
The K8 and K9 gates compare the merged build with this one: K9's byte identity "to the stock-MMQ baseline binary", K8's "default ==
stock". It keeps HEAD's tests/test-backend-ops.cpp (the hash and chain drivers) and every other component, so the only difference
is the MMQ launch code. The frozen stack8 libraries the K8 script used cannot be mixed with this tree's test binary.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = 'def4a4e20a689047e68e8edd81c97790b69a1de7'
SUBJECTS = [
    'cuda : launch the MMQ q8_1 quantize kernel with PDL',
    'cuda : launch the MMQ kernel and its stream-k fixup with PDL',
    'local: K9 MMQ PDL off switch',
    'cuda : take SMALL_I MMQ tiles on stream-k launches too',
    'cuda : MMQ stream-k partitions in default-tile units for narrow tiles',
    'cuda : shape-gated small-I MMQ tiles for 16 and 24 columns on Blackwell',
    'local: K8/L1 MMQ tile A/B switch',
]
# Files the reverted series owns: after the reverts they equal the base.
OWNED = ['ggml/src/ggml-cuda/mmq.cuh', 'ggml/src/ggml-cuda/mmq-load-tiles.cuh', 'ggml/src/ggml-cuda/mmq-vec-dot.cuh',
         'ggml/src/ggml-cuda/mmq-config-blackwell.cuh', 'ggml/src/ggml-cuda/quantize.cu', 'ggml/src/ggml-cuda/quantize.cuh']
ENV = dict(os.environ, GIT_NO_LAZY_FETCH='1')


def git(*args, cwd=ROOT, **kw):
    return subprocess.run(['git', *args], cwd=cwd, env=ENV, check=True, capture_output=True, **kw)


def commits():
    log = git('log', '--format=%H %s', f'{BASE}..HEAD', text=True).stdout.splitlines()
    picked = []
    for subject in SUBJECTS:
        hits = [l.split(' ', 1)[0] for l in log if l.split(' ', 1)[1].startswith(subject)]
        if len(hits) != 1:
            raise SystemExit(f'stock_tree: "{subject}" matches {len(hits)} commits')
        picked.append(hits[0])
    order = [l.split(' ', 1)[0] for l in log]
    return sorted(picked, key=order.index)


def make(dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if {q.name for q in dest.iterdir()} - {'PURPOSE'}:
        raise SystemExit(f'stock_tree: {dest} is not empty')
    archive = subprocess.Popen(['git', 'archive', 'HEAD'], cwd=ROOT, env=ENV, stdout=subprocess.PIPE)
    subprocess.run(['tar', '-x', '-C', str(dest)], stdin=archive.stdout, check=True)
    if archive.wait():
        raise SystemExit('stock_tree: git archive failed')
    for c in commits():
        patch = git('diff', f'{c}^', c, '--', 'ggml', text=False).stdout
        if patch:
            subprocess.run(['git', 'apply', '-R', '--whitespace=nowarn', '-'], cwd=dest, input=patch, check=True, capture_output=True)
        print(f'reverted {c[:9]} ({len(patch)} bytes of ggml diff)')
    for name in OWNED:
        have = (dest / name).read_bytes() if (dest / name).exists() else None
        try:
            want = git('show', f'{BASE}:{name}').stdout
        except subprocess.CalledProcessError:
            want = None
        if have != want:
            raise SystemExit(f'stock_tree: {name} differs from the base after the reverts')
    print(f'stock tree PASS: {len(OWNED)} MMQ files equal {BASE[:9]}')


if __name__ == '__main__':
    if sys.argv[1:] == ['list']:
        print('\n'.join(commits()))
    elif sys.argv[1:2] == ['make'] and len(sys.argv) == 3:
        make(sys.argv[2])
    else:
        raise SystemExit(__doc__)
