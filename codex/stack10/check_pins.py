"""The window's pin set: what bw-stack10b.toml must pin (every file the window reads) and what it must not (anything it writes).

Contract: required() is the single definition used by mk_window_toml.py to emit the pins and by this check to verify the toml.
A pinned file that the window writes would park the queued item, so no pin may lie under WRITES.

usage: check_pins.py [TOML]    exit 0: every required file is pinned and exists, no pin is written by the window
       check_pins.py list      print the required pins
       check_pins.py --tail [TOML | list]   the same for a tail item (bw-stack10b-tail.toml): the full window's pins plus the reused
                               attempt's evidence and build-cuda/bin, which the tail reads and never writes
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
          str(W / 'frozen/stack10'), str(W / 'work/s10b-stock-a'), str(W / 'INDEX.md')]


# The tail writes only its own attempt dirs, the k11a dry run directory, the rendered items and the frozen result; build-cuda and the
# reused attempt (TAIL_FROM) are inputs.
WRITES_TAIL = [str(ROOT / 'codex/window/stack10b-tail-a'), str(ROOT / 'codex/window/preflight'), str(ROOT / 'codex/stack10/rendered'),
               str(ROOT / 'codex/stack10/snapshots'), str(W / 'frozen/stack10'), str(W / 'INDEX.md')]
TAIL_FROM = 'stack10b-a1'


def written(path, tail=False):
    """WRITES entries are directories, files, or (the stock work dir) a name prefix ending in -a."""
    p = str(path)
    return any(p == w or p.startswith(w + '/') or (w.endswith('-a') and p.startswith(w)) for w in (WRITES_TAIL if tail else WRITES))
OWN = ['bw-stack10b.toml', 'bw-stack10b.sh', 'np2_smoke.sh', 'np2_smoke.py', 'tail_check.py', 'gates.sh', 'gates_table.py', 'shfn.py', 'full_checks.py', 'sass_gate.py', 'k8_gpu_patch.py', 'stock_tree.py',
       'engagement.py', 's10items.py', 'render_stack10_items.py', 'check_pins.py', 'mk_window_toml.py', 'gpu/gpu_ab.sh', 'gpu/gpu_s10.py']
# codex-root files the window or its items import or read (the wave-1 runner stack, the fafix digest driver, the evidence).
CODEX_ROOT = ['gpu_stack9.py', 'gpu_stack10.py', 'fa-identity.cpp', 'stack10/mac-evidence/summary.json']
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
SCANNED = ['gates.sh', 'bw-stack10b.sh', 'gates_table.py']
TOKEN = re.compile(r'codex/[A-Za-z0-9_./-]+\.(?:sh|py|cpp|toml|json|jinja)\b')


def scanned_tokens(tail=False):
    texts = [(ROOT / 'codex/stack10' / n).read_text() for n in SCANNED]
    scripts = {s for _, s, _ in gates_table.EXTRACTIONS} | {'codex/k8/gpu/k8-gpu.sh', 'codex/k11a/sync_count.sh', 'codex/k11a/drive_np2.py'}
    texts += [(ROOT / s).read_text() for s in scripts]
    found = set(scripts)
    for text in texts:
        found |= set(TOKEN.findall(text))
    return sorted(f for f in found if '$' not in f and (ROOT / f).is_file() and not written(ROOT / f, tail) and not f.startswith('codex/stack10-evidence/'))  # stale cand evidence, named only in the cand preflight


def source_files():
    out = subprocess.run(['git', 'diff', '--name-only', STACK9, 'HEAD', '--', '.', ':!codex'], cwd=ROOT, text=True, capture_output=True, check=True)
    return sorted(f for f in out.stdout.split() if (ROOT / f).is_file())


def tail_inputs():
    """What the tail reads of the reused attempt and of its build (stage 41 recorded no hashes; the build files are pinned instead)."""
    a1 = ROOT / 'codex/window' / TAIL_FROM
    stock = re.search(r'^STOCK_DIR=(.*)$', (a1 / 'stock-dir.txt').read_text(), re.M).group(1)
    files = [a1 / n for n in ('stages.tsv', 'summary.json', 'payload.log', 'stock-dir.txt', 'cuda-stamps.log')] + [Path(stock) / 'k8-bin/SHA256SUMS']
    files += sorted(p for p in (ROOT / 'build-cuda/bin').iterdir() if not p.is_symlink())
    return {str(f) for f in files} | {str(ROOT / 'codex/stack10/bw-stack10b-tail.toml')}


def required(tail=False):
    paths = {str(ROOT / 'codex/stack10' / n) for n in OWN}
    paths |= {str(ROOT / 'codex/stack10/items' / p.name) for p in (ROOT / 'codex/stack10/items').glob('*.toml.in')}
    paths |= {str(ROOT / 'codex' / n) for n in CODEX_ROOT}
    paths |= {str(ROOT / n) for n in scanned_tokens(tail)}
    paths |= {str(W / n) for n in WT_FILES}
    paths |= {str(ROOT / n) for n in source_files()}
    paths |= set(EXTERNAL)
    if tail:
        paths |= tail_inputs()
    return sorted(paths)


def check(toml_path, tail=False):
    pins = tomllib.loads(Path(toml_path).read_text())['pins']['files']
    problems = []
    for p in pins:
        if not Path(p).exists():
            problems.append(f'pinned file missing: {p}')
        if written(p, tail):
            problems.append(f'pinned file is written by the window: {p}')
    problems += [f'not pinned: {p}' for p in required(tail) if p not in set(pins)]
    problems += [f'pinned twice: {p}' for p in set(pins) if pins.count(p) > 1]
    return problems


if __name__ == '__main__':
    args = sys.argv[1:]
    tail = args[:1] == ['--tail']
    if tail:
        args = args[1:]
    if args == ['list']:
        print('\n'.join(required(tail)))
        sys.exit(0)
    default = ROOT / 'codex/stack10' / ('bw-stack10b-tail.toml' if tail else 'bw-stack10b.toml')
    problems = check(args[0] if args else default, tail)
    print('\n'.join(problems) if problems else f'pins: {len(required(tail))} required files pinned, exist, none written by the window PASS')
    sys.exit(1 if problems else 0)
