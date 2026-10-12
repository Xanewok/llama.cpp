#!/usr/bin/env bash
# Mac CPU gate for master-cands (master abee0c847 + pkgs 61/87/88/04/91/90), run through bin/bw-mac; argument = OUT dir.
# Each package's own CPU test on the combined tree: 91 test-alloc (meta subgraph reuse), 87 test-llama-archs qwen35 MTP reset,
# 88 test-speculative, 04 test-backend-sampler/test-sampling/--tp-sampling seeds 42 and 1, 90 test-host-mapped (SKIP on CPU).
# 61 has no model-free test (default pad = master's; covered only by the builds).
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd "$(dirname "$0")/../.."
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 8
OUT=${1:?output}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'FAIL dirty tracked tree'; exit 9; }
git rev-parse HEAD > "$OUT/source-commit.txt"
: > "$OUT/stages.tsv"
fails=0
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE="${BW_CCACHE:-ON}" -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
TARGETS=(llama-server test-alloc test-backend-sampler test-sampling test-llama-archs test-speculative test-host-mapped)
# RAM gate: <= 4 GiB for this payload; refuse to start below 4 GiB available
free_gib() { vm_stat | awk -v ps="$(pagesize)" '/Pages (free|inactive|speculative)/ { gsub("\\.", "", $NF); n += $NF } END { printf "%d", n*ps/1073741824 }'; }
stage() { # NAME VERDICT RC T0
    printf '%s\trc=%s\tgate=%s\twall_s=%s\n' "$1" "$3" "$2" "$((SECONDS-$4))" | tee -a "$OUT/stages.tsv"
    [[ $2 != FAIL ]] || fails=$((fails+1))
}
# NAME PATTERN CMD...: rc 0 and PATTERN in stdout or stderr
run_gate() {
    local name=$1 pat=$2 rc=0 t0=$SECONDS v
    shift 2
    "$BW_GATE" "$@" > "$OUT/$name.txt" 2> "$OUT/$name.log" || rc=$?
    if [[ $rc == 0 ]] && { [[ -z $pat ]] || grep -aEq -- "$pat" "$OUT/$name.txt" "$OUT/$name.log"; }; then v=PASS; else v=FAIL; fi
    stage "$name" "$v" "$rc" "$t0"
}
g=$(free_gib); echo "free_gib=$g" > "$OUT/ram-gate.txt"
[[ $g -ge 4 ]] || { stage ram-gate FAIL "$g" "$SECONDS"; exit 8; }
t0=$SECONDS rc=0
{ "$BW_BUILDGATE" cmake -S . -B build-cpu "${flags[@]}" "${extra[@]}" &&
  "$BW_BUILDGATE" cmake --build build-cpu --target "${TARGETS[@]}" -j"${BW_JOBS:-4}"; } > "$OUT/build.log" 2>&1 || rc=$?
stage build "$([[ $rc == 0 ]] && echo PASS || echo FAIL)" "$rc" "$t0"
[[ $rc == 0 ]] || exit 8
run_gate server-help-kv-pad 'kv-pad N' build-cpu/bin/llama-server --help
run_gate test-alloc-91 '^test_meta_subgraph_reuse_same_graph PASSED' build-cpu/bin/test-alloc
t0=$SECONDS; n=$(grep -acE '^test_meta_subgraph_reuse_(two_splits|ten_splits|same_graph) PASSED' "$OUT/test-alloc-91.txt" || true)
stage test-alloc-91-reuse-3 "$([[ $n == 3 ]] && echo PASS || echo FAIL)" "$n" "$t0"
run_gate mtp-reset-87 'qwen35: MTP draft state reset: .*OK' build-cpu/bin/test-llama-archs -a qwen35 -b CPU -s 1
run_gate test-speculative-88 'uses_prompt, draft size' build-cpu/bin/test-speculative
run_gate test-backend-sampler-04 'backend top-p on top-p candidates test PASSED' build-cpu/bin/test-backend-sampler --model "$BW_MODEL" --device cpu
run_gate test-sampling-04 '' build-cpu/bin/test-sampling
for s in 42 1; do run_gate "tp-sampling-s$s-04" 'test_tp_sampling: OK' build-cpu/bin/test-llama-archs --tp-sampling -s "$s"; done
run_gate test-host-mapped-90 '^test-host-mapped: SKIP' build-cpu/bin/test-host-mapped
echo "fails=$fails"
[[ $fails == 0 ]] || exit 8
