#!/usr/bin/env bash
set -euo pipefail
VISIBLE=${CUDA_VISIBLE_DEVICES:-}
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BUILDGATE_MIN_GIB=10 CPUGATE_MIN_GIB=10
export PATH="$HOME/.local/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
W=$HOME/repos/qwen38-eval-worktrees
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
preflight() {
    [[ $(git branch --show-current) == topic/s10-k2 ]] || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    git merge-base --is-ancestor 0cfc0723e HEAD || return 1
    for p in "$CCCL" "$W/bin/buildgate" /usr/local/cuda-13.2/bin/nvcc codex/s10-k2-evidence/summary.json; do
        [[ -e $p ]] || return 1
    done
    python3 - <<'PY_GATE' || return 1
import hashlib, json
from pathlib import Path
r = json.loads(Path('codex/s10-k2-evidence/summary.json').read_text())
assert r['result'] == 'PASS'
assert {20,21,22,31} <= {s['code'] for s in r['stages'] if s['rc'] == 0}
h = hashlib.sha256()
for name in r['source_files']:
    h.update(name.encode()+b'\0'+Path(name).read_bytes())
assert h.hexdigest() == r['source_sha256'], 'source differs from the Mac CPU gate'
PY_GATE
    return 0
}
if [[ ${1:-} == --preflight ]]; then
    preflight || { echo 'bw-s10-k2: FAIL preflight'; exit 8; }
    echo 'bw-s10-k2: PASS preflight'; exit 0
fi
[[ ${GPUQ_ITEM:-} == bw-s10-k2 && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in "$ROOT"/codex/window/s10-k2-a[0-9]*) ;; *) exit 8 ;; esac
mkdir -p "$OUT"
exec > >(tee "$OUT/payload.log") 2>&1
preflight || exit 8
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-s10-k2: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit 8
}
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
build() {
    nice -n 19 "$W/bin/buildgate" cmake -B build-cuda "${flags[@]}"
    nice -n 19 "$W/bin/buildgate" cmake --build build-cuda --target test-backend-ops test-fattn-kv-max-cache -j4
    grep -F 'set(CMAKE_CXX_COMPILER_ID "GNU")' build-cuda/CMakeFiles/*/CMakeCXXCompiler.cmake >/dev/null
    grep -E 'set\(CMAKE_CXX_COMPILER_VERSION "13\.' build-cuda/CMakeFiles/*/CMakeCXXCompiler.cmake >/dev/null
}
stamps() {
    local n
    n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    echo "DeviceTopK_count_M=$n"
    [[ $n == 25 ]]
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" >/dev/null
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
}
tbo() {
    local dev=$1 log=$2
    shift 2
    env CUDA_VISIBLE_DEVICES="$VISIBLE" "$@" timeout 1800 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -j 1 > "$log" 2>&1
}
# Runs only the same-backend identity check: the filter matches no eval case.
identity() {
    local dev=$1 log=$2
    shift 2
    env CUDA_VISIBLE_DEVICES="$VISIBLE" "$@" timeout 900 nice -n 19 build-cuda/bin/test-backend-ops test -b "$dev" -o FLASH_ATTN_EXT -p 'no-such-case' -j 1 > "$log" 2>&1
}
PASS_LINE='FLASH_ATTN_EXT shared mask: same-backend byte identity, replay and overwrite PASS cases=5'
ops() {
    [[ $VISIBLE == GPU-* ]] || return 1
    nice -n 19 build-cuda/bin/test-fattn-kv-max-cache > "$OUT/unit.log" 2>&1
    local dev
    for dev in CUDA0 CUDA1; do
        # every attention case of the suite: a mask used by one layer must take the unchanged path
        tbo "$dev" "$OUT/fa-all-$dev.log" -o FLASH_ATTN_EXT
        grep -aE '[1-9][0-9]*/[1-9][0-9]* tests passed' "$OUT/fa-all-$dev.log" >/dev/null
        if grep -aq 'Failing tests' "$OUT/fa-all-$dev.log"; then echo "FAIL FLASH_ATTN_EXT regression on $dev"; return 1; fi
        tbo "$dev" "$OUT/shared-$dev.log" -o FLASH_ATTN_EXT -p 'shared_mask=1'
        grep -aq "$PASS_LINE" "$OUT/shared-$dev.log"
        grep -aE '19/19 tests passed' "$OUT/shared-$dev.log" >/dev/null
        identity "$dev" "$OUT/identity-$dev.log"
        grep -aq "$PASS_LINE" "$OUT/identity-$dev.log"
        identity "$dev" "$OUT/identity-nograph-$dev.log" GGML_CUDA_DISABLE_GRAPHS=1
        grep -aq "$PASS_LINE" "$OUT/identity-nograph-$dev.log"
        # negative controls: each must fail the identity check with its own message
        if identity "$dev" "$OUT/mutant-off-$dev.log" GGML_CUDA_FATTN_KV_MAX_SHARE=0; then echo 'FAIL share-off control escaped'; return 1; fi
        grep -aq 'FAIL FLASH_ATTN_EXT shared mask shared scan did not engage' "$OUT/mutant-off-$dev.log"
        if identity "$dev" "$OUT/mutant-1-$dev.log" GGML_CUDA_FATTN_KV_MAX_MUTANT=1; then echo 'FAIL invalidation mutant escaped'; return 1; fi
        grep -aE 'FAIL FLASH_ATTN_EXT shared mask (mask overwrite did not force a rescan|mask overwritten between layers)' "$OUT/mutant-1-$dev.log" >/dev/null
        if identity "$dev" "$OUT/mutant-2-$dev.log" GGML_CUDA_FATTN_KV_MAX_MUTANT=2; then echo 'FAIL stale-scan mutant escaped'; return 1; fi
        grep -aq 'FAIL FLASH_ATTN_EXT shared mask' "$OUT/mutant-2-$dev.log"
    done
}
# ABBA pairs of the same binary: sharing on, then off by GGML_CUDA_FATTN_KV_MAX_SHARE=0.
perf() {
    [[ $VISIBLE == GPU-* ]] || return 1
    local dev i arm args=()
    for dev in CUDA0 CUDA1; do
        args=()
        i=0
        for arm in 1 0 0 1 1 0 0 1; do
            env CUDA_VISIBLE_DEVICES="$VISIBLE" GGML_CUDA_FATTN_KV_MAX_SHARE=$arm timeout 900 nice -n 19 build-cuda/bin/test-backend-ops perf -b "$dev" -o FLASH_ATTN_EXT -p 'shared_mask=1' > "$OUT/perf-$dev-$i-share$arm.log" 2>&1
            args+=("share$arm=$OUT/perf-$dev-$i-share$arm.log")
            i=$((i + 1))
        done
        python3 codex/s10_k2_perf.py "$OUT/perf-$dev.json" "${args[@]}"
    done
}
stage 40 cuda-build build
stage 41 cuda-stamps stamps
stage 42 cuda-shared-scan-identity-and-controls ops
stage 43 cuda-shared-scan-perf perf
python3 - "$OUT" "$SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
(p/'summary.json').write_text(json.dumps(dict(result='PASS',commit=sys.argv[2],GPU_execution=True,perf={d: json.loads((p/f'perf-{d}.json').read_text()) for d in ('CUDA0','CUDA1')}),indent=2)+'\n')
PY_RESULT
echo 'bw-s10-k2: PASS; not frozen (the parent combines components)'
