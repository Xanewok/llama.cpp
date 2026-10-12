#!/usr/bin/env bash
# Mac CPU gate for pkg 04 (TP backend sampling) on master abee0c847, run through bin/bw-mac; argument = OUT dir.
# Gated: build; test-backend-sampler (CPU, all tests incl. test_top_p_on_top_p); test-sampling; test-llama-archs --tp-sampling
# on Meta(CPU,CPU) seeds 42/1/2/3; mutant ids (vocab id i^1) must fail; fail-before: 0002's src reverted must fail
# test_top_p_on_top_p; real-model Meta(CPU,CPU) driver: backend top-k(20)+greedy tokens and candidate sets equal the host's
# on Meta (same numerics, candidate sets included) unless the model is refused (tied/replicated head: recorded, not a failure);
# plain CPU vs Meta host tokens equal (control).
# Informational: mutant-no-empty-guard (0006 disabled) tells whether 0006 is still needed after #30223.
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
P=codex/tp-bsamp

flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE="${BW_CCACHE:-ON}" -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
TARGETS=(llama test-backend-sampler test-sampling test-llama-archs)

stage() { # NAME VERDICT(PASS|FAIL|INFO) RC T0
    printf '%s\trc=%s\tgate=%s\twall_s=%s\n' "$1" "$3" "$2" "$((SECONDS-$4))" | tee -a "$OUT/stages.tsv"
    [[ $2 != FAIL ]] || fails=$((fails+1))
}
build() { # LOG
    "$BW_BUILDGATE" cmake --build build-cpu --target "${TARGETS[@]}" -j"${BW_JOBS:-4}" >> "$1" 2>&1
}
# NAME EXPECT(0|nz) CMD...: stdout and stderr apart; EXPECT nz = must fail (mutant / fail-before)
run_expect() {
    local name=$1 expect=$2 rc=0 t0=$SECONDS v
    shift 2
    "$BW_GATE" "$@" > "$OUT/$name.txt" 2> "$OUT/$name.log" || rc=$?
    if [[ $expect == 0 ]]; then v=$([[ $rc == 0 ]] && echo PASS || echo FAIL); else v=$([[ $rc != 0 ]] && echo PASS || echo FAIL); fi
    stage "$name" "$v" "$rc" "$t0"
}

t0=$SECONDS rc=0
{ "$BW_BUILDGATE" cmake -S . -B build-cpu "${flags[@]}" "${extra[@]}" && build "$OUT/build.log"; } > "$OUT/build.log" 2>&1 || rc=$?
stage build "$([[ $rc == 0 ]] && echo PASS || echo FAIL)" "$rc" "$t0"
[[ $rc == 0 ]] || exit 8
t0=$SECONDS rc=0
"$BW_GATE" clang++ -std=c++17 -O1 -Iinclude -Iggml/include $P/driver.cpp -o build-cpu/bin/tp-bsamp-driver \
    -Lbuild-cpu/bin -lllama -lggml -lggml-base -Wl,-rpath,"$PWD/build-cpu/bin" > "$OUT/driver-build.log" 2>&1 || rc=$?
stage driver-build "$([[ $rc == 0 ]] && echo PASS || echo FAIL)" "$rc" "$t0"

run_expect test-backend-sampler 0 build-cpu/bin/test-backend-sampler --model "$BW_MODEL" --device cpu
t0=$SECONDS
grep -aq 'backend top-p on top-p candidates test PASSED' "$OUT/test-backend-sampler.txt" && v=PASS || v=FAIL
stage test-backend-sampler-top-p-on-top-p "$v" - "$t0"
run_expect test-sampling 0 build-cpu/bin/test-sampling
for s in 42 1 2 3; do
    run_expect "tp-sampling-s$s" 0 build-cpu/bin/test-llama-archs --tp-sampling -s "$s"
    t0=$SECONDS
    grep -aq 'test_tp_sampling: OK' "$OUT/tp-sampling-s$s.txt" "$OUT/tp-sampling-s$s.log" && v=PASS || v=FAIL
    stage "tp-sampling-s$s-ok-line" "$v" - "$t0"
done

# real model: host argmax/top-20 on plain CPU and on Meta(CPU,CPU) vs the backend chain on Meta(CPU,CPU)
for m in cpu meta meta-bs; do
    t0=$SECONDS rc=0
    "$BW_GATE" build-cpu/bin/tp-bsamp-driver "$BW_MODEL" "$m" 24 > "$OUT/model-$m.txt" 2> "$OUT/model-$m.log" || rc=$?
    if [[ $m == meta-bs && $rc == 7 ]]; then
        stage "model-$m" INFO "$rc" "$t0"   # refused: the 4B model's LM head is not vocab-split (tied/replicated)
    else
        stage "model-$m" "$([[ $rc == 0 ]] && grep -aq '^step24 seq1 ' "$OUT/model-$m.txt" && echo PASS || echo FAIL)" "$rc" "$t0"
    fi
done
t0=$SECONDS
# tokens only: the split matmul reorders sums, so near-equal logits at rank 20 may swap between plain CPU and Meta
cmp <(grep -a '^step' "$OUT/model-cpu.txt" | cut -d' ' -f1-4) <(grep -a '^step' "$OUT/model-meta.txt" | cut -d' ' -f1-4) > /dev/null && v=PASS || v=FAIL
stage model-token-identity-cpu-meta "$v" - "$t0"
if grep -aq '^backend sampler accepted' "$OUT/model-meta-bs.txt"; then
    t0=$SECONDS
    cmp <(grep -a '^step' "$OUT/model-meta.txt") <(grep -a '^step' "$OUT/model-meta-bs.txt") > /dev/null && v=PASS || v=FAIL
    stage model-identity-meta-backend "$v" - "$t0"
fi
grep -a 'SPLIT_MODE_TENSOR' "$OUT"/model-*.log > "$OUT/model-refusals.txt" || true

# mutants and fail-before: patch, rebuild, expect failure, restore
mutate() { # NAME PATCH(R=reverse) EXPECT CMD...
    local name=$1 patch=$2 expect=$3 t0=$SECONDS rc=0
    shift 3
    if [[ $patch == R:* ]]; then git apply -R "${patch#R:}"; else git apply "$patch"; fi
    build "$OUT/$name.build.log" || rc=$?
    if [[ $rc != 0 ]]; then stage "$name-build" FAIL "$rc" "$t0"; else run_expect "$name" "$expect" "$@"; fi
    if [[ $patch == R:* ]]; then git apply "${patch#R:}"; else git apply -R "$patch"; fi
    [[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'FAIL tree not restored'; exit 9; }
}
mutate mutant-ids $P/mutant-ids.patch nz build-cpu/bin/test-llama-archs --tp-sampling -s 42
mutate fail-before-0002 R:$P/fail-before-0002.patch nz build-cpu/bin/test-backend-sampler --model "$BW_MODEL" --device cpu --test test_top_p_on_top_p
# informational: does the TP sampling test still need 0006 on master?
t0=$SECONDS rc=0
git apply $P/mutant-no-empty-guard.patch
build "$OUT/mutant-no-empty-guard.build.log" || rc=$?
[[ $rc == 0 ]] && { "$BW_GATE" build-cpu/bin/test-llama-archs --tp-sampling -s 42 > "$OUT/mutant-no-empty-guard.txt" 2> "$OUT/mutant-no-empty-guard.log" || rc=$?; }
git apply -R $P/mutant-no-empty-guard.patch
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'FAIL tree not restored'; exit 9; }
stage "mutant-no-empty-guard(rc0=0006-not-needed)" INFO "$rc" "$t0"
build "$OUT/restore.build.log" || true

echo "tp-bsamp mac: fails=$fails"
[[ $fails == 0 ]]
