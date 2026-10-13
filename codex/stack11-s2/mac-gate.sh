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
T="$B/bin/test-backend-ops"
HAVE_K=0
[[ -f tests/test-topk-batched.cpp ]] && HAVE_K=1

if want p87; then
    # LLAMA_MTP_FOLD_C=0 selects the catch-up decode path of common/speculative.cpp, the default takes the folded path of llama-context.cpp:
    # each hunk of 87 is reached by one of the two
    for sd in 1 2 3 4 5; do run "p87-fixed-s$sd" 0 'MTP draft state reset: .*OK' -- "$B/bin/test-llama-archs" -a qwen35 -b CPU -s "$sd"; done
    for sd in 1 2 3; do run "p87-fixed-nofoldc-s$sd" 0 'MTP draft state reset: .*OK' -- env LLAMA_MTP_FOLD_C=0 "$B/bin/test-llama-archs" -a qwen35 -b CPU -s "$sd"; done
    # fail-before: all production hunks of 87 reverted, test kept
    revert 87-prod.patch
    do_build p87-reverted-build test-llama-archs
    for sd in 1 2 3 4 5; do
        run "p87-reverted-s$sd" nz 'MTP draft state reset: .*FAIL' -- "$B/bin/test-llama-archs" -a qwen35 -b CPU -s "$sd"
        absent "p87-reverted-s$sd-no-ok" "p87-reverted-s$sd" 'MTP draft state reset: .*OK'
    done
    for sd in 1 2 3; do run "p87-reverted-nofoldc-s$sd" nz 'MTP draft state reset: .*FAIL' -- env LLAMA_MTP_FOLD_C=0 "$B/bin/test-llama-archs" -a qwen35 -b CPU -s "$sd"; done
    restore
    # the fold refusal alone reverted: the folded path fails, the catch-up path (fold off) still passes
    revert 87-fold.patch
    do_build p87-nofold-build test-llama-archs
    for sd in 1 2 3; do run "p87-nofold-s$sd" nz 'MTP draft state reset: .*FAIL' -- "$B/bin/test-llama-archs" -a qwen35 -b CPU -s "$sd"; done
    for sd in 1 2 3; do run "p87-nofold-nofoldc-s$sd" 0 'MTP draft state reset: .*OK' -- env LLAMA_MTP_FOLD_C=0 "$B/bin/test-llama-archs" -a qwen35 -b CPU -s "$sd"; done
    restore
    # the speculative.cpp reset alone reverted: fails with fold off, and with the default too (batches whose size is not the folded
    # decode's size take the catch-up path)
    revert 87-spec.patch
    do_build p87-nospec-build test-llama-archs
    for sd in 1 2 3; do run "p87-nospec-nofoldc-s$sd" nz 'MTP draft state reset: .*FAIL' -- env LLAMA_MTP_FOLD_C=0 "$B/bin/test-llama-archs" -a qwen35 -b CPU -s "$sd"; done
    for sd in 1 2 3; do run "p87-nospec-s$sd" nz 'MTP draft state reset: .*FAIL' -- "$B/bin/test-llama-archs" -a qwen35 -b CPU -s "$sd"; done
    restore
    # overreset: the carry is zeroed at every batch start, not only on an empty draft memory
    python3 - common/speculative.cpp <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = ' && llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id) < 0) {\n                    std::fill_n'
assert s.count(old) == 1
open(p, 'w').write(s.replace(old, ') {\n                    std::fill_n'))
PY
    git diff --stat > "$OUT/p87-overreset.diffstat"
    do_build p87-overreset-build test-llama-archs
    for sd in 1 2 3; do run "p87-overreset-nofoldc-s$sd" nz 'MTP draft state reset: .*FAIL' -- env LLAMA_MTP_FOLD_C=0 "$B/bin/test-llama-archs" -a qwen35 -b CPU -s "$sd"; done
    restore
    do_build p87-rebuild test-llama-archs
fi

if want p88; then
    run p88-fixed 0 'uses_prompt' -- "$B/bin/test-speculative"
    # mutant: the ngram-simple drafter stops declaring that it reads the prompt
    python3 - common/speculative.cpp <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
i = s.index('struct common_speculative_impl_ngram_simple')
old = 'bool uses_prompt() const override { return true; }'
j = s.index(old, i)
open(p, 'w').write(s[:j] + 'bool uses_prompt() const override { return false; }' + s[j + len(old):])
PY
    git diff --stat > "$OUT/p88-mutant.diffstat"
    do_build p88-mutant-build test-speculative
    run p88-mutant nz 'CHECK failed: common_speculative_uses_prompt' -- "$B/bin/test-speculative"
    restore
    do_build p88-rebuild test-speculative
fi

if want p91; then
    # 91 has no production commit here: its tests must pass on the plan-retaining Meta backend of stack11-base
    run p91-test-alloc 0 'test_meta_subgraph_reuse_two_splits PASSED' 'test_meta_subgraph_reuse_ten_splits PASSED' 'test_meta_subgraph_reuse_same_graph PASSED' -- "$B/bin/test-alloc"
    # control: a plan that forgets its cached uid is rebuilt on every compute, so the simple backend sees new subgraph uids each time
    python3 - ggml/src/ggml-backend-meta.cpp <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = '        backend_ctx->uid         = cgraph->uid;\n'
assert s.count(old) == 1
open(p, 'w').write(s.replace(old, '        backend_ctx->uid         = 0;\n'))
PY
    git diff --stat > "$OUT/p91-uidchurn.diffstat"
    do_build p91-uidchurn-build test-alloc
    run p91-uidchurn nz 'GGML_ASSERT' -- "$B/bin/test-alloc"
    restore
    do_build p91-rebuild test-alloc
fi

if want p90; then
    run p90-test-alloc 0 'test_meta_host_mapped_weight PASSED' -- "$B/bin/test-alloc"
    run p90-host-mapped-cpu 0 'SKIP' -- "$B/bin/test-host-mapped"
    run p90-get-rows-cpu 0 -- "$T" test -b CPU -o GET_ROWS -j 1
    absent p90-get-rows-no-fail p90-get-rows-cpu 'FAIL'
    # fail-before: the iterative split-state driver without the host-source branch reads the foreign buffer context as a Meta one
    revert 90-meta-driver.patch
    do_build p90-nodriver-build test-alloc
    run p90-nodriver nz -- "$B/bin/test-alloc"
    absent p90-nodriver-no-pass p90-nodriver 'test_meta_host_mapped_weight PASSED'
    restore
    # fail-before: the Meta device no longer accepts the foreign host buffer type
    revert 90-meta-all.patch
    do_build p90-nometa-build test-alloc
    run p90-nometa nz -- "$B/bin/test-alloc"
    absent p90-nometa-no-pass p90-nometa 'test_meta_host_mapped_weight PASSED'
    restore
    do_build p90-rebuild test-alloc
fi

if want whole; then
    run whole-llama-archs 0 -- "$B/bin/test-llama-archs"
    run whole-test-alloc 0 -- "$B/bin/test-alloc"
    run whole-speculative 0 -- "$B/bin/test-speculative"
    run whole-get-rows 0 -- "$T" test -b CPU -o GET_ROWS -j 1
fi

if want drops; then
    if [[ $HAVE_K == 1 ]]; then
        run k7-pair-identity 0 'RMS_NORM_SCALE_PAIR: same-backend byte identity' -- "$T" test -b CPU -o RMS_NORM_SCALE_PAIR -j 1
        do_build k10-topk-unit test-topk-batched
        run k10-topk-unit-run 0 -- "$B/bin/test-topk-batched"
    else
        [[ ! -e ggml/src/ggml-rms-norm-scale-pair.h && ! -e ggml/src/ggml-cuda/top-k-batched.cuh ]] || { echo 'FAIL K7/K10 files still present'; exit 8; }
        run drop-top-k 0 -- "$T" test -b CPU -o TOP_K -j 1
        run drop-rms-norm 0 -- "$T" test -b CPU -o RMS_NORM -j 1
        run drop-scale 0 -- "$T" test -b CPU -o SCALE -j 1
        stage drop-files-absent PASS - "$SECONDS"
    fi
fi

echo "stack11-s2 mac: fails=$fails"
[[ $fails == 0 ]]
