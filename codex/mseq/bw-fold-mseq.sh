#!/usr/bin/env bash
# Parent build window only: verify Mac gates, CUDA compile/freeze, render/check GPU items. No tests or GPU execution here.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$HOME/.local/bin:$PATH"
W=$HOME/repos/qwen38-eval-worktrees
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
cd "$ROOT"
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
MAC=${BW_MAC_SUMMARY:-$ROOT/codex/mseq/mac-out/a2/matrix/summary.json}
if [[ ${1:-} == --preflight ]]; then
    for f in "$W/bin/buildgate" "$W/bin/wt" "$CCCL" "$W/work/aa-load/ccx01.sh" \
             "$W/work/build-window-1007/mmq.sh" "$W/frozen/stack7/build/bin/round-trace" "$MAC"; do
        [[ -e $f ]] || { echo "bw-fold-mseq: missing $f"; exit 8; }
    done
    [[ ! -e $W/frozen/fold-mseq ]] || { echo 'bw-fold-mseq: freeze already exists'; exit 8; }
    exit 0
fi
OUT=${1:?output directory}
mkdir -p "$OUT"
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'bw-fold-mseq: tracked tree is dirty'; exit 8; }
SHA=$(git rev-parse HEAD)
nice -n 19 python3 - "$MAC" <<'PY'
import hashlib, json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
assert r['rc'] == 0, 'Mac gates must pass'
h = hashlib.sha256()
for name in ('common/speculative.cpp', 'src/llama-context.cpp', 'src/llama-context.h', 'src/llama-ext.h',
             'src/llama-graph.cpp', 'src/llama-graph.h', 'src/models/qwen35.cpp'):
    h.update(name.encode() + b'\0' + Path(name).read_bytes())
assert h.hexdigest() == r['source_sha256'], 'Mac-tested fold sources changed'
PY
BUILDGATE=${BW_BUILDGATE:-$W/bin/buildgate}
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL=$CCCL
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
{
    export CUDA_HOME=/usr/local/cuda-13.2
    "$BUILDGATE" cmake -B build-cuda "${flags[@]}"
    "$BUILDGATE" cmake --build build-cuda --target llama-server -j4
    [[ $(git rev-parse HEAD) == "$SHA" ]]
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace
    n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    [[ $n == 25 ]]
    strings build-cuda/bin/libllama.so | grep -q 'mtp_fold_decode: engaged n_seq='
    "$W/bin/wt" freeze "$ROOT" fold-mseq --build build-cuda --desc 'stack7 + multi-sequence MTP catch-up fold; default on; exact layout fallback; local A/B controls'
    {
        printf 'commit %s\n' "$SHA"
        printf 'configure: cmake -B build-cuda '; printf '%q ' "${flags[@]}"; printf '\n'
        printf 'build: bin/buildgate cmake --build build-cuda --target llama-server -j4\n'
        printf 'Mac CPU evidence %s; no CUDA binary executed\n' "$MAC"
    } > "$W/frozen/fold-mseq/RECIPE.txt"
} > "$OUT/cuda.log" 2>&1
nice -n 19 python3 codex/mseq/make_items.py "$SHA" "$MAC" "$ROOT/codex/mseq/rendered" > "$OUT/items.log" 2>&1
printf 'bw-fold-mseq: frozen; items checked, NOT submitted\n'
