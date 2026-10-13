#!/usr/bin/env bash
# Mac CPU gate of one pre-PR package on master (run through bin/bw-mac; argument = OUT dir; package name in codex/gate/PKG).
# Fail-before = production hunks reverted from codex/gate/prod*.patch (or a mutant), test kept; pass-after = the tree as committed.
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
G=codex/gate
PKG=pr87

flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE="${BW_CCACHE:-ON}" -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
if false; then
    flags=("${flags[@]/-DLLAMA_BUILD_TOOLS=OFF/-DLLAMA_BUILD_TOOLS=ON}")
    flags=("${flags[@]/-DLLAMA_BUILD_SERVER=OFF/-DLLAMA_BUILD_SERVER=ON}")
fi
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
B=build-cpu

stage() { # NAME VERDICT(PASS|FAIL|INFO) RC T0
    printf '%s\trc=%s\tgate=%s\twall_s=%s\n' "$1" "$3" "$2" "$((SECONDS-$4))" | tee -a "$OUT/stages.tsv"
    [[ $2 != FAIL ]] || fails=$((fails+1))
}
# RAM gate: refuse to start below 4 GiB available (3 GiB for the model-free gates, whose own use is < 2 GiB)
need=4; [[ $PKG == 01-03 ]] || need=3
free_gib() { vm_stat | awk -v ps="$(pagesize)" '/Pages (free|inactive|speculative)/ { gsub("\\.", "", $NF); n += $NF } END { printf "%d", n*ps/1073741824 }'; }
g=$(free_gib); echo "free_gib=$g" > "$OUT/ram-gate.txt"
[[ $g -ge $need ]] || { stage ram-gate FAIL "$g" "$SECONDS"; exit 8; }
configure() { "$BW_BUILDGATE" cmake -S . -B "$B" "${flags[@]}" "${extra[@]}" "$@"; }
build() { "$BW_BUILDGATE" cmake --build "$B" --target "$@" -j"${BW_JOBS:-4}"; }
# NAME TARGETS...
do_build() {
    local name=$1 t0=$SECONDS rc=0
    shift
    build "$@" > "$OUT/$name.log" 2>&1 || rc=$?
    stage "$name" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)" "$rc" "$t0"
    [[ $rc == 0 ]] || exit 8
}
# NAME EXPECT(0|nz|any) [GREP-STRING...] -- CMD...; 600 s cap; EXPECT nz = must exit nonzero; strings must be present (EXPECT 0) in stdout+stderr
run() {
    local name=$1 expect=$2 rc=0 t0=$SECONDS v ok=1 g
    shift 2
    local greps=()
    while [[ $1 != -- ]]; do greps+=("$1"); shift; done
    shift
    "$BW_GATE" perl -e 'alarm 600; exec @ARGV' "$@" > "$OUT/$name.txt" 2>&1 || rc=$?
    if [[ $expect == 0 ]]; then [[ $rc == 0 ]] || ok=0; elif [[ $expect == nz ]]; then [[ $rc != 0 ]] || ok=0; fi
    for g in ${greps[@]+"${greps[@]}"}; do grep -aq -- "$g" "$OUT/$name.txt" || ok=0; done
    stage "$name" "$([[ $ok == 1 ]] && echo PASS || echo FAIL)" "$rc" "$t0"
}
# stage the absence of a string (negative control: the fixed marker must not appear)
absent() { # NAME FILE STRING
    local t0=$SECONDS
    stage "$1" "$(grep -aq -- "$3" "$OUT/$2.txt" && echo FAIL || echo PASS)" - "$t0"
}
revert() { git apply -R "$G/$1"; }
restore() { git checkout -q -- . ; [[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'FAIL tree not restored'; exit 9; }; }

case $PKG in
pr87)
    configure
    do_build build test-llama-archs
    for s in 1 2 3 4 5; do run "fixed-s$s" 0 'MTP draft state reset: .*OK' -- "$B/bin/test-llama-archs" -a qwen35 -b CPU -s "$s"; done
    run fixed-all-archs 0 -- "$B/bin/test-llama-archs"
    # reverted: production hunk only, test kept
    revert prod.patch
    do_build reverted-build test-llama-archs
    for s in 1 2 3 4 5; do
        run "reverted-s$s" nz 'MTP draft state reset: .*FAIL' -- "$B/bin/test-llama-archs" -a qwen35 -b CPU -s "$s"
        absent "reverted-s$s-no-ok-line" "reverted-s$s" 'MTP draft state reset: .*OK'
    done
    restore
    # overreset: the carry is zeroed at every batch start, not only on an empty draft memory
    python3 - common/speculative.cpp <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = ' && llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id) < 0'
assert s.count(old) == 1
open(p, 'w').write(s.replace(old, ''))
PY
    git diff --stat > "$OUT/overreset.diffstat"
    do_build overreset-build test-llama-archs
    for s in 1 2 3 4 5; do run "overreset-s$s" any -- "$B/bin/test-llama-archs" -a qwen35 -b CPU -s "$s"; done
    restore
    ;;
*) echo "unknown PKG $PKG"; exit 8 ;;
esac

echo "pkg$PKG: fails=$fails"
[[ $fails == 0 ]]
