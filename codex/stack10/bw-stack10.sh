#!/usr/bin/env bash
# Combined stack10 CUDA window (parent queue only, no A/B timing): CUDA build, every selected component's own CUDA gates, the
# fafix SASS gate, test-backend-ops in full on both GPUs, freeze as frozen/stack10, render and check the A/B items.
#   env COMPONENTS (default: p55 k7 k4 k6a k6b k2 k10 k9 k8 k11a k11b) selects the gates and the canaries of the rendered items;
#   the base ops, the fafix gates and the full suite always run. gdn4 is not part of this window.
# rc: 0 pass | 3 Xid | 4 transient (GPU busy/missing) | 5 void negative control or too little data | 6 integrity
#     7 timeout | 8 deterministic failure. A failed gate does not stop the other gates (the drop map needs every failing component),
#     but nothing is frozen or rendered after one; the exit code is the worst of them (3 > 6 > 5 > 8 > 7 > 4), an Xid stops at once.
set -euo pipefail
VISIBLE=${CUDA_VISIBLE_DEVICES:-}
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BUILDGATE_MIN_GIB=10 CPUGATE_MIN_GIB=10
export PATH="$HOME/.local/bin:/usr/local/cuda-13.2/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
cd "$ROOT"
W=$HOME/repos/qwen38-eval-worktrees
EVAL=$HOME/repos/qwen38-eval
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
GPUS=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
BUILDGATE=$W/bin/buildgate
CPUS=0-3,12-15
STACK9=cd776e104dbd516c7b74506992d2e8efd7600155
CAND2=def4a4e20a689047e68e8edd81c97790b69a1de7
S9_LIB=$W/frozen/stack9/build/bin
CAND_LIB=$W/frozen/stack10-cand/build/bin
P55_LIB=$W/frozen/stack9-p55/build/bin
EVIDENCE=codex/stack10/mac-evidence/summary.json
ALL_COMPONENTS='p55 k7 k4 k6a k6b k2 k10 k9 k8 k11a k11b'
COMPONENTS=$(echo "${COMPONENTS:-$ALL_COMPONENTS}" | xargs)
# shellcheck source=gates.sh
source "$ROOT/codex/stack10/gates.sh"

# Same flags as bw-stack10-cand.sh (LLAMA_BUILD_TESTS=ON for the test binaries).
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)

preflight() {
    local tool word p script
    for word in $COMPONENTS; do
        [[ " $ALL_COMPONENTS " == *" $word "* ]] || { echo "unknown component $word"; return 1; }
    done
    case $(git branch --show-current) in topic/stack10|topic/stack10-*) ;; *) echo 'branch is not topic/stack10[-*]'; return 1 ;; esac
    [[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'tracked tree is dirty'; return 1; }
    [[ ! -e $W/frozen/stack10 ]] || { echo 'frozen/stack10 exists'; return 1; }
    for tool in cmake git python3 nice gpuq ccache strings timeout patchelf flock taskset g++ cuobjdump nsys compute-sanitizer nvidia-smi \
                journalctl curl bc ss awk sed xargs; do
        command -v "$tool" > /dev/null || { echo "missing tool $tool"; return 1; }
    done
    git merge-base --is-ancestor "$STACK9" HEAD || { echo 'stack9 is not an ancestor'; return 1; }
    git merge-base --is-ancestor "$CAND2" HEAD || { echo 'stack10-cand2 is not an ancestor'; return 1; }
    local log
    log=$(git log --format=%s "$CAND2..HEAD")
    [[ $log != *'process 4 GDN columns per warp'* && $log != *'time the fused GDN+CPY pair'* ]] || { echo 'gdn4 commits are still in HEAD'; return 1; }
    for p in "$CCCL" "$BUILDGATE" "$W/bin/cpugate" "$W/bin/wt" "$W/frozen/stack6/RECIPE.txt" "$W/work/aa-load/ccx01.sh" /usr/local/cuda-13.2/bin/nvcc \
             "$EVAL/harness/mkitem.py" "$W/frozen/stack9/SOURCE" "$W/frozen/stack9-p55/SOURCE" "$W/frozen/stack10-cand/SOURCE" \
             "$S9_LIB/libggml-cuda.so" "$CAND_LIB/libggml-cuda.so" "$P55_LIB/test-backend-ops" "$P55_LIB/round-trace" "$EVIDENCE" /mnt/scratch; do
        [[ -e $p ]] || { echo "missing $p"; return 1; }
    done
    [[ $(head -n 1 "$W/frozen/stack9/SOURCE") == "$STACK9" ]] || { echo 'frozen/stack9 is not cd776e104'; return 1; }
    [[ -w $W/work ]] || { echo 'work dir not writable'; return 1; }
    for script in codex/stack10/bw-stack10.sh codex/stack10/gates.sh codex/stack10/gpu/gpu_ab.sh codex/k8/gpu/k8-gpu.sh codex/k11a/sync_count.sh; do
        bash -n "$script" || { echo "bash -n $script"; return 1; }
    done
    python3 codex/stack10/shfn.py selftest > /dev/null || { echo 'shfn selftest'; return 1; }
    python3 codex/stack10/full_checks.py --fixtures > /dev/null || { echo 'full_checks fixtures'; return 1; }
    python3 codex/stack10/engagement.py selftest > /dev/null || { echo 'engagement selftest'; return 1; }
    python3 codex/stack10/stock_tree.py list > /dev/null || { echo 'stock tree commits'; return 1; }
    python3 codex/stack10/check_pins.py > /dev/null || { echo 'pins'; return 1; }
    local tmp
    tmp=$(mktemp -d)
    python3 codex/stack10/k8_gpu_patch.py codex/k8/gpu/k8-gpu.sh "$tmp/k8-gpu.sh" "$tmp/k8-bin" "$tmp/stock-bin" && bash -n "$tmp/k8-gpu.sh" || { rm -rf "$tmp"; echo 'k8 derivation'; return 1; }
    rm -rf "$tmp"
    HARNESS_CODE_ROOT=$EVAL PYTHONPATH=$EVAL python3 -P codex/stack10/gpu/gpu_s10.py --fixtures > /dev/null || { echo 'item fixtures'; return 1; }
    python3 - "$EVIDENCE" <<'PY_GATE' || { echo 'Mac evidence'; return 1; }
import hashlib, json, subprocess, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
assert r['result'] == 'PASS', r['result']
files = subprocess.check_output(['git', 'ls-files', 'src', 'ggml', 'tests', 'common'], text=True).splitlines()
assert files == r['source_files'], 'source file list differs from the Mac gate'
h = hashlib.sha256()
for name in files:
    h.update(name.encode() + b'\0' + Path(name).read_bytes())
assert h.hexdigest() == r['source_sha256'], 'source differs from the Mac CPU gate'
PY_GATE
    if has k11a; then
        DRY_RUN=1 ROOT="$ROOT" OUT="$ROOT/codex/window/preflight" bash codex/k11a/sync_count.sh > /dev/null || { echo 'k11a dry run'; return 1; }
    fi
    return 0
}

if [[ ${1:-} == --preflight ]]; then
    if preflight; then echo 'bw-stack10: PASS preflight'; exit 0; fi
    echo 'bw-stack10: FAIL preflight'; exit 8
fi
[[ ${GPUQ_ITEM:-} == bw-stack10 && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/window/stack10-a[0-9]*) ;; *) exit 8 ;; esac
ATTEMPT=${OUT##*stack10-a}
mkdir -p "$OUT"
exec > >(tee "$OUT/payload.log") 2>&1
preflight || { echo 'bw-stack10: FAIL preflight'; exit 8; }
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short=9 HEAD)
P55_SHA=$(head -n 1 "$P55_LIB/../../SOURCE")
STARTED=$(date '+%Y-%m-%d %H:%M:%S')
echo "bw-stack10: components=[$COMPONENTS] commit=$SHA visible=$VISIBLE"
[[ $VISIBLE == GPU-*,GPU-* ]] || { echo 'bw-stack10: FAIL no two GPU UUIDs in CUDA_VISIBLE_DEVICES'; exit 4; }

fatal 40 cuda-build build
fatal 41 cuda-stamps stamps
if has k8 || has k9; then
    fatal 42 stock-mmq-build stock_build
    STOCK_DIR=$(sed -n 's/^STOCK_DIR=//p' "$OUT/stock-dir.txt")
    [[ -d $STOCK_DIR/stock-bin && -d $STOCK_DIR/k8-bin ]]
fi

gate 50 base-ops g_base
if has k7; then gate 52 k7-pair g_k7; fi
if has k4; then gate 53 k4-conv-rs g_k4; fi
if has k6a; then gate 55 k6a-q8-reuse g_k6a; fi
if has k6b; then gate 56 k6b-add-rms g_k6b; fi
if has k6a || has k6b; then gate 57 k6-memcheck g_k6_memcheck; fi
if has k2; then gate 58 k2-shared-mask g_k2; fi
if has k10; then gate 59 k10-topk g_k10; fi
if has k9; then gate 60 k9-mmq-pdl g_k9; fi
if has k8; then gate 61 k8-mmq-tiles g_k8; fi
if has k11b; then gate 62 k11b-input-stage g_k11b; fi
gate 64 fafix-sass g_sass
gate 65 fafix-identity g_fafix
gate 66 full-suite g_full
# The server-launching capture goes last and only on a clean sheet: it costs the most GPU time and adds no drop-map information
# once another gate has failed.
if has k11a; then
    if [[ ${#FAILS[@]} == 0 ]]; then gate 63 k11a-sync-count g_k11a; else echo 'bw-stack10: k11a-sync-count skipped, an earlier gate failed'; fi
fi
if [[ ${#FAILS[@]} != 0 ]]; then finish_fail; fi

fatal 70 freeze-stack10 freeze
fatal 71 render-and-check-items items
python3 - "$OUT" "$SHA" "$COMPONENTS" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
stages = [dict(code=int(c), name=n, rc=int(rc), wall_s_M=int(t)) for c, n, rc, t in (l.split('\t') for l in (p/'stages.tsv').read_text().splitlines())]
(p/'summary.json').write_text(json.dumps(dict(result='PASS', commit=sys.argv[2], components=sys.argv[3].split(), GPU_execution=True, stages=stages,
    frozen='frozen/stack10', items='codex/stack10/rendered; checked, NOT submitted'), indent=2)+'\n')
PY_RESULT
echo 'bw-stack10: PASS; frozen as stack10 and items checked, NOT submitted'
