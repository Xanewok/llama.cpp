"""The window's pin set: what bw-stack11-s2.toml must pin (every file the window reads) and what it must not (anything it writes).

Contract: required() is the single definition used by mk_window_toml.py to emit the pins and by this check to verify the toml. A pinned
file the window writes would park the queued item, so no pin may lie under WRITES. Not pinned on purpose: serve.sh (the owner edits
it; the health stage records its sha256 and argv) and the Mac evidence (lands after this commit; preflight checks it by content hash).

usage: check_pins.py [TOML]    exit 0: every required file is pinned and exists, no pin is written by the window
       check_pins.py list      print the required pins
"""
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
W = ROOT.parent.parent
KROOT = W / 'wt/stack11-s2-k'
HOME = Path.home()
BASE = 'abee0c8476b585b0d081d580b5ce21d0addc22c4'

# The window writes only here: OUT, the rendered items and their snapshots, both CUDA builds, the frozen results, the bin/wt index,
# the build temp dir.
WRITES = [str(ROOT / 'codex/window'), str(HERE / 'rendered'), str(HERE / 'snapshots'), str(ROOT / 'build-cuda'), str(KROOT / 'build-cuda'),
          str(W / 'frozen/stack11-s2'), str(W / 'frozen/stack11-s2-nok'), str(W / 'INDEX.md'), '/mnt/scratch/tmp-build']
# Window-owned files that are not inputs: dry renders (DRY_RUN only) and caches.
NOT_INPUTS = ('rendered-dry', 'rendered', 'snapshots', '__pycache__')
WT_FILES = ['bin/buildgate', 'bin/wt', 'bin/wt-index.py', 'work/aa-load/ccx01.sh',
            'frozen/stack11-base/SOURCE', 'frozen/stack11-base/build/bin/SHA256SUMS', 'frozen/stack11-base/build/bin/test-backend-ops',
            'frozen/stack11-base/build/bin/round-trace', 'frozen/stack11-base/build/bin/libggml-cuda.so']
# Inputs of the gates that are not in the tree: np2 smoke and health boots.
EXTERNAL = [str(HOME / 'models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf'), str(HOME / 'models/Swift-1.5-Qwen3.8-27B-J5q-bf16.gguf'),
            str(HOME / 'models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json'),
            '/home/xanewok/repos/attrib-pack/gpu/inputs/chat-template.jinja']


def written(path):
    p = str(path)
    return any(p == w or p.startswith(w + '/') for w in WRITES)


def own_files():
    files = {str(p) for p in HERE.rglob('*') if p.is_file() and not set(p.relative_to(HERE).parts) & set(NOT_INPUTS) and p.suffix != '.pyc'
             and not p.name.startswith('.gpuq-')}  # gpuq writes its run wrapper beside the script
    return sorted(files | {str(HERE / 'bw-stack11-s2.toml')})


def source_files(tree):
    out = subprocess.run(['git', 'diff', '--name-only', BASE, 'HEAD', '--', '.', ':!codex'], cwd=tree, text=True, capture_output=True, check=True)
    return sorted(str(tree / f) for f in out.stdout.split() if (tree / f).is_file())


def required():
    paths = set(own_files())
    paths |= {str(W / n) for n in WT_FILES}
    paths |= set(source_files(ROOT)) | set(source_files(KROOT))
    paths |= set(EXTERNAL)
    return sorted(paths)


def check(toml_path):
    pins = tomllib.loads(Path(toml_path).read_text())['pins']['files']
    problems = []
    for p in pins:
        if not Path(p).exists():
            problems.append(f'pinned file missing: {p}')
        if written(p):
            problems.append(f'pinned file is written by the window: {p}')
    problems += [f'not pinned: {p}' for p in required() if p not in set(pins)]
    problems += [f'pinned twice: {p}' for p in set(pins) if pins.count(p) > 1]
    return problems


if __name__ == '__main__':
    args = sys.argv[1:]
    if args == ['list']:
        print('\n'.join(required()))
        sys.exit(0)
    problems = check(args[0] if args else HERE / 'bw-stack11-s2.toml')
    print('\n'.join(problems) if problems else f'pins: {len(required())} required files pinned, exist, none written by the window PASS')
    sys.exit(1 if problems else 0)
