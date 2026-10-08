"""Derive the combined window's copy of codex/k8/gpu/k8-gpu.sh without editing the component's gate logic.

usage: k8_gpu_patch.py SRC DST BIN STOCK_BIN
Replaced, each anchor exactly once (otherwise the derivation fails closed):
  HERE      absolute directory of SRC, since DST lives elsewhere
  BIN       the merged build assembled outside the tree (a `k8-bin` at the tree root would block bin/wt freeze)
  STOCK     the stock-MMQ build (HEAD with the K8/K9 ggml commits reverted) instead of frozen/stack8, whose libggml
            cannot be mixed with this tree's test binary
  stock_setup  runs the stock build's own test-backend-ops, not a copy of the K8 binary beside stack8 libraries
  step 3    the full MUL_MAT / MUL_MAT_ID suites; the combined window runs the whole suite once with flake handling (full_checks.py)
"""
import re
import sys
from pathlib import Path

STOCK_SETUP = '''stock_setup() { # the stock-MMQ build assembled like k8-bin: same test harness source, MMQ code of def4a4e20
    local libs stray
    [[ -x $STOCK/test-backend-ops ]] || die 6 "no stock test-backend-ops in $STOCK"
    (cd "$STOCK" && sha256sum -c --quiet SHA256SUMS) || die 6 "stock build bytes changed"
    grep -aq 'GGML_CUDA_MMQ_SK_MUTANT' "$STOCK/libggml-cuda.so" && die 6 "the stock build contains K8 code"
    libs=$(ldd "$STOCK/test-backend-ops")
    [[ -z $(grep 'not found' <<< "$libs" || true) ]] || die 5 "stock arm: unresolved libraries in $STOCK"
    stray=$(awk '/libggml/ {print $3}' <<< "$libs" | grep -v "^$STOCK/" || true)
    [[ -z $stray ]] || die 5 "stock arm: ggml libs resolve outside $STOCK: $stray"
    STOCK_EXE=$STOCK
}'''


def once(text, pattern, repl, flags=0):
    new, n = re.subn(pattern, lambda m: repl, text, flags=flags)
    if n != 1:
        raise SystemExit(f'k8_gpu_patch: /{pattern}/ matched {n} times')
    return new


def main(src, dst, bin_dir, stock_dir):
    here = str(Path(src).resolve().parent)
    t = Path(src).read_text()
    t = once(t, r'^HERE=.*$', f'HERE={here}', re.M)
    t = once(t, r'^BIN=\$ROOT/k8-bin.*$', f'BIN={bin_dir}', re.M)
    t = once(t, r'^STOCK=.*$', f'STOCK={stock_dir}', re.M)
    t = once(t, r'^stock_setup\(\) \{.*?^\}$', STOCK_SETUP, re.M | re.S)
    t = once(t, r'^    # 3\. the whole existing suites.*?(?=^    guard$)', '', re.M | re.S)
    Path(dst).write_text(t)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
