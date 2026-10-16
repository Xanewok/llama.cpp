#!/usr/bin/env bash
# Run inside a gpuq probe/build window. Both arms must use the same test cases.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EVAL=$HOME/repos/qwen38-eval
REF=$HOME/repos/qwen38-eval-worktrees/frozen/stack11-fa-s2/build/bin
BIN=$ROOT/build-cuda/bin/test-backend-ops
OUT=${1:?output directory}
DEV=${2:?CUDA0 or CUDA1}
[[ $DEV == CUDA0 || $DEV == CUDA1 ]]
[[ -n ${CUDA_VISIBLE_DEVICES:-} ]]
mkdir -p "$OUT"
PARAMS='type_a=f32,type_b=f32,m=(24|31|32|33|48),n=(1|3|6|7|8|12),k=5120,'

for arm in new ref; do
    LIB=$ROOT/build-cuda/bin
    [[ $arm == new ]] || LIB=$REF
    LD_LIBRARY_PATH=$LIB ldd "$BIN" > "$OUT/libs-$arm-$DEV.log"
    awk -v prefix="$LIB/" '
        /libggml.* =>/ { count++; if (index($3, prefix) != 1) bad = 1; if ($1 ~ /^libggml-cuda/) cuda = 1 }
        END { exit !(count >= 4 && cuda && !bad) }
    ' "$OUT/libs-$arm-$DEV.log"
    LD_LIBRARY_PATH=$LIB "$BIN" perf -o MUL_MAT -b "$DEV" -p "$PARAMS" > "$OUT/perf-$arm-$DEV.log" 2>&1
done

python3 - "$OUT" "$DEV" "$EVAL" <<'PY'
import sys
from pathlib import Path

out, dev, repo = sys.argv[1:]
sys.path.insert(0, repo)
from harness.perf_gate import parse

logs = [parse(Path(out) / f"perf-{arm}-{dev}.log") for arm in ("new", "ref")]
for cases in logs:
    assert len(cases) == 30, f"expected 30 focused perf cases, got {len(cases)}"
assert logs[0].keys() == logs[1].keys(), "candidate/reference cases differ"
for case, latency in logs[0].items():
    if any(f",m={m},n=12," in case for m in (24, 31)):
        ratio = latency / logs[1][case]
        assert ratio <= 1.05, f"width-12 fallback regressed: {ratio:.3f} {case}"
print("perf: all 30 cases matched, including both thin shapes at widths 6 and 12")
PY
python3 "$EVAL/harness/perf_gate.py" "$OUT/perf-new-$DEV.log" "$OUT/perf-ref-$DEV.log"
