#!/usr/bin/env bash
# Mac CPU gate of stack11-s2 (run through bin/bw-mac; argument = OUT dir). Stage filter: words in the bw-mac OUT_NAME after the first
# "__" (stack11-s2-mac__p87__p90 runs p87 and p90). Fail-before = the production hunk reverted from codex/stack11-s2/revert/*.patch
# (test kept) or a one-line mutant; pass-after = the tree as committed. Every job <= 600 s (alarm) and waits for >= 3 GiB available.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 ASAN_OPTIONS=detect_leaks=0
cd "$(dirname "$0")/../.."
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 8
OUT=${1:?output}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'FAIL dirty tracked tree'; exit 9; }
git rev-parse HEAD > "$OUT/source-commit.txt"
: > "$OUT/stages.tsv"
fails=0
G=codex/stack11-s2
B=build-cpu

only=${COMPONENTS:-}
if [[ -z $only ]]; then
    run_name=$(basename "$(dirname "$(dirname "$(dirname "$OUT")")")")
    [[ $run_name != *__* ]] || { only=${run_name#*__}; only=${only//__/ }; }
fi
want() { [[ -z $only || " $only " == *" $1 "* ]]; }

flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE="${BW_CCACHE:-ON}" -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"

stage() { # NAME VERDICT(PASS|FAIL|INFO) RC T0
    printf '%s\trc=%s\tgate=%s\twall_s=%s\n' "$1" "$3" "$2" "$((SECONDS-$4))" | tee -a "$OUT/stages.tsv"
    [[ $2 != FAIL ]] || fails=$((fails+1))
}
free_gib() { vm_stat | awk -v ps="$(pagesize)" '/Pages (free|inactive|speculative)/ { gsub("\\.", "", $NF); n += $NF } END { printf "%d", n*ps/1073741824 }'; }
# model-free jobs use < 2 GiB; wait up to 10 min for 3 GiB of available memory (the owner's Halmos runs share this host), then fail
t0=$SECONDS
while g=$(free_gib); [[ $g -lt 3 ]]; do
    (( SECONDS - t0 < 600 )) || { echo "free_gib=$g" > "$OUT/ram-gate.txt"; stage ram-gate FAIL "$g" "$t0"; exit 8; }
    sleep 10
done
echo "free_gib=$g waited_s=$((SECONDS-t0))" > "$OUT/ram-gate.txt"

configure() { "$BW_BUILDGATE" cmake -S . -B "$B" "${flags[@]}" "${extra[@]}" "$@"; }
do_build() { # NAME TARGETS...
    local name=$1 t0=$SECONDS rc=0
    shift
    "$BW_BUILDGATE" cmake --build "$B" --target "$@" -j"${BW_JOBS:-4}" > "$OUT/$name.log" 2>&1 || rc=$?
    stage "$name" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)" "$rc" "$t0"
    [[ $rc == 0 ]] || exit 8
}
# NAME EXPECT(0|nz|any) [GREP-STRING...] -- CMD...; strings must be present in stdout+stderr
run() {
    local name=$1 expect=$2 rc=0 t0=$SECONDS ok=1 s
    shift 2
    local greps=()
    while [[ $1 != -- ]]; do greps+=("$1"); shift; done
    shift
    "$BW_GATE" perl -e 'alarm 600; exec @ARGV' "$@" > "$OUT/$name.txt" 2>&1 || rc=$?
    if [[ $expect == 0 ]]; then [[ $rc == 0 ]] || ok=0; elif [[ $expect == nz ]]; then [[ $rc != 0 ]] || ok=0; fi
    for s in ${greps[@]+"${greps[@]}"}; do grep -aq -- "$s" "$OUT/$name.txt" || ok=0; done
    stage "$name" "$([[ $ok == 1 ]] && echo PASS || echo FAIL)" "$rc" "$t0"
}
absent() { # NAME FILE STRING
    local t0=$SECONDS
    stage "$1" "$(grep -aq -- "$3" "$OUT/$2.txt" && echo FAIL || echo PASS)" - "$t0"
}
revert() { git apply -R "$G/revert/$1"; }
restore() { git checkout -q -- . ; [[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'FAIL tree not restored'; exit 9; }; }

configure
do_build build test-llama-archs test-alloc test-speculative test-backend-ops test-host-mapped

if want p91; then
    # 91 is not in the tree: its tests must pass on the plan-retaining Meta backend without the package's production change
    run p91-test-alloc 0 'test_meta_subgraph_reuse_two_splits' 'test_meta_subgraph_reuse_ten_splits' 'test_meta_subgraph_reuse_same_graph' -- "$B/bin/test-alloc"
fi

echo "stack11-s2 mac: fails=$fails"
[[ $fails == 0 ]]
