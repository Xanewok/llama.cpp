"""The window's pin set: what bw-stack10.toml must pin (every file the window reads) and what it must not (anything it writes).

Contract: required() is the single definition used by mk_window_toml.py to emit the pins and by this check to verify the toml.
A pinned file that the window writes would park the queued item, so no pin may lie under WRITES.

usage: check_pins.py [TOML]    exit 0: every required file is pinned and exists, no pin is written by the window
       check_pins.py list      print the required pins
"""
import re
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gates_table

ROOT = Path(__file__).resolve().parents[2]
W = ROOT.parent.parent
EVAL = Path('/home/xanewok/repos/qwen38-eval')
STACK9 = 'cd776e104dbd516c7b74506992d2e8efd7600155'
HOME = Path.home()

# The window writes only here (OUT, the rendered items and their snapshots, the build, the frozen result, the scratch work dir, the
# bin/wt index) and in the k11a dry run directory; none of these may be pinned.
WRITES = [str(ROOT / 'codex/window'), str(ROOT / 'codex/stack10/rendered'), str(ROOT / 'codex/stack10/snapshots'), str(ROOT / 'build-cuda'),
          str(W / 'frozen/stack10'), str(W / 'work/s10-stock-a'), str(W / 'INDEX.md')]


def written(path):
    """WRITES entries are directories, files, or (the stock work dir) a name prefix ending in -a."""
    p = str(path)
    return any(p == w or p.startswith(w + '/') or (w.endswith('-a') and p.startswith(w)) for w in WRITES)
OWN = ['bw-stack10.toml', 'bw-stack10.sh', 'gates.sh', 'gates_table.py', 'shfn.py', 'full_checks.py', 'sass_gate.py', 'k8_gpu_patch.py', 'stock_tree.py',
       'engagement.py', 's10items.py', 'render_stack10_items.py', 'check_pins.py', 'mk_window_toml.py', 'gpu/gpu_ab.sh', 'gpu/gpu_s10.py']
# codex-root files the window or its items import or read (the wave-1 runner stack, the fafix digest driver, the evidence).
CODEX_ROOT = ['gpu_stack9.py', 'gpu_stack10.py', 'fa-identity.cpp', 'stack10-evidence/summary.json']
WT_FILES = ['bin/buildgate', 'bin/cpugate', 'bin/wt', 'bin/wt-index.py', 'work/aa-load/ccx01.sh', 'frozen/stack6/RECIPE.txt',
            'frozen/stack9/SOURCE', 'frozen/stack9-p55/SOURCE', 'frozen/stack10-cand/SOURCE',
            'frozen/stack9/build/bin/libggml-cuda.so', 'frozen/stack10-cand/build/bin/libggml-cuda.so',
            'frozen/stack9-p55/build/bin/test-backend-ops', 'frozen/stack9-p55/build/bin/libggml-cuda.so']
# Inputs of the component gates that are not in the tree (K4 model logits, K11a capture).
EXTERNAL = ['/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf',
            str(HOME / 'models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf'),
            str(HOME / 'models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json'),
            *(f'/home/xanewok/repos/attrib-pack/{n}' for n in (
                'attrib_pack.py', 'runs/hqgap8-analysis/an.py', 'gpu/llama_drive.py', 'gpu/inputs.sha256', 'gpu/inputs/agg.py',
                'gpu/inputs/corpus.txt', 'gpu/inputs/chat-template.jinja', 'models/llama-swift15-iq4xs.json'))]
# Scripts scanned for further codex/... files they call.
SCANNED = ['gates.sh', 'bw-stack10.sh', 'gates_table.py']
TOKEN = re.compile(r'codex/[A-Za-z0-9_./-]+\.(?:sh|py|cpp|toml|json|jinja)\b')


def scanned_tokens():
    texts = [(ROOT / 'codex/stack10' / n).read_text() for n in SCANNED]
    scripts = {s for _, s, _ in gates_table.EXTRACTIONS} | {'codex/k8/gpu/k8-gpu.sh', 'codex/k11a/sync_count.sh'}
    texts += [(ROOT / s).read_text() for s in scripts]
    found = set(scripts)
    for text in texts:
        found |= set(TOKEN.findall(text))
    return sorted(f for f in found if '$' not in f and (ROOT / f).is_file() and not written(ROOT / f))


def source_files():
    out = subprocess.run(['git', 'diff', '--name-only', STACK9, 'HEAD', '--', '.', ':!codex'], cwd=ROOT, text=True, capture_output=True, check=True)
    return sorted(f for f in out.stdout.split() if (ROOT / f).is_file())


def required():
    paths = {str(ROOT / 'codex/stack10' / n) for n in OWN}
    paths |= {str(ROOT / 'codex/stack10/items' / p.name) for p in (ROOT / 'codex/stack10/items').glob('*.toml.in')}
    paths |= {str(ROOT / 'codex' / n) for n in CODEX_ROOT}
    paths |= {str(ROOT / n) for n in scanned_tokens()}
    paths |= {str(W / n) for n in WT_FILES}
    paths |= {str(ROOT / n) for n in source_files()}
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
    if sys.argv[1:] == ['list']:
        print('\n'.join(required()))
        sys.exit(0)
    problems = check(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'codex/stack10/bw-stack10.toml')
    print('\n'.join(problems) if problems else f'pins: {len(required())} required files pinned, exist, none written by the window PASS')
    sys.exit(1 if problems else 0)
