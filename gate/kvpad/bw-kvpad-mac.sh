#!/usr/bin/env bash
# Mac CPU gate for pkg 61 (n_kv padding as a context parameter) on master. Run through bin/bw-mac.
#  build, plumbing (flag/env/default, n_ctx padded, non-multiple rounded up), model identity across pads on Qwen3.5-4B (hybrid GDN):
#  plain CPU, 1 thread: logits bitwise identical for pad 0 vs 4096; Meta(CPU,CPU), 4 threads: tokens identical (CPU FA split-KV depends on n_kv).
#  Negative controls: a perturbed prompt must change the logits hash; a non-engaged pad must show the same n_ctx as default.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 8
OUT=${1:?output}
MODEL=${BW_MODEL:?}
mkdir -p "$OUT"
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-kvpad-mac: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit "$code"
}
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
build() {
    "$BW_BUILDGATE" cmake -S . -B build-cpu "${flags[@]}" "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-cpu --target llama llama-completion -j4
    nice -n 19 clang++ -std=c++17 -O1 -Iinclude -Iggml/include gate/kvpad/driver.cpp -o build-cpu/bin/kvpad-driver \
        -Lbuild-cpu/bin -lllama -lggml -lggml-base -Wl,-rpath,"$PWD/build-cpu/bin"
}
# one boot per case; stdout and stderr are separate files, the pattern gates read the stderr log
completion() {  # NAME EXPECT_PAD EXPECT_NCTX WARN(0|1) [args...]
    local name=$1 pad=$2 nctx=$3 warn=$4 rc=0
    shift 4
    env LLAMA_ARG_LOG_VERBOSITY=4 "$BW_GATE" build-cpu/bin/llama-completion -m "$MODEL" -c 1500 -n 2 -p hi -t 4 -ngl 0 -fa on \
        --no-warmup -no-cnv "$@" > "$OUT/cli-$name.out" 2> "$OUT/cli-$name.err" || rc=$?
    [[ $rc == 0 ]] || { echo "cli $name rc=$rc"; return 1; }
    grep -aEq "llama_context: n_kv_pad += $pad\$" "$OUT/cli-$name.err" || { echo "cli $name: n_kv_pad != $pad"; return 1; }
    grep -aEq "llama_context: n_ctx += $nctx\$" "$OUT/cli-$name.err" || { echo "cli $name: n_ctx != $nctx"; return 1; }
    if [[ $warn == 1 ]]; then
        grep -aq 'is not a multiple of 256, rounded up' "$OUT/cli-$name.err" || { echo "cli $name: no round-up warning"; return 1; }
    elif grep -aq 'is not a multiple of 256' "$OUT/cli-$name.err"; then
        echo "cli $name: unexpected warning"; return 1
    fi
    echo "cli $name PASS (n_kv_pad $pad, n_ctx $nctx)"
}
plumbing() {
    "$BW_GATE" build-cpu/bin/llama-completion --help > "$OUT/help.out" 2> "$OUT/help.err" || true
    [[ $(grep -ac -e '--kv-pad' "$OUT/help.out" "$OUT/help.err" | awk -F: '{s+=$2} END {print s+0}') -ge 1 ]] || { echo 'no --kv-pad in --help'; return 1; }
    completion default 256 1536 0
    completion flag256 256 1536 0 --kv-pad 256
    completion flag4096 4096 4096 0 --kv-pad 4096
    LLAMA_ARG_KV_PAD=1024 completion env1024 1024 2048 0
    completion flag300 512 1536 1 --kv-pad 300
}
run() {  # NAME META PAD VARIANT THREADS
    local rc=0
    "$BW_GATE" build-cpu/bin/kvpad-driver "$MODEL" "$2" "$3" 1500 80 "$5" "$4" > "$OUT/run-$1.out" 2> "$OUT/run-$1.err" || rc=$?
    [[ $rc == 0 ]] || { echo "run $1 rc=$rc"; return 1; }
    echo "run $1: $(grep -a '^n_ctx' "$OUT/run-$1.out") $(grep -a '^logits_fnv' "$OUT/run-$1.out")"
}
field() { grep -a "^$2" "$OUT/run-$1.out"; }
identity() {
    local p np
    # plain CPU at one thread: CPU FA splits the KV range across threads at small q, so only 1 thread is n_kv independent
    for p in 0 4096; do run "m0-pad$p" 0 "$p" 0 1; done
    run m0-pad0-variant 0 0 1 1
    for p in 0 256 4096; do run "m1-pad$p" 1 "$p" 0 4; done
    # the decode must cross an n_kv step (multiple of 256) after the prefill
    np=$(field m0-pad0 n_prompt | awk '{print $2}')
    [[ $np -lt 768 && $((np + 80)) -gt 768 ]] || { echo "decode does not cross 768: n_prompt=$np"; return 1; }
    [[ $np -gt 256 ]] || return 1
    # engagement: n_ctx as created differs between pads
    [[ $(field m0-pad0 n_ctx) == 'n_ctx 1536' && $(field m1-pad256 n_ctx) == 'n_ctx 1536' ]] || { echo 'default/256 n_ctx'; return 1; }
    [[ $(field m0-pad4096 n_ctx) == 'n_ctx 4096' ]] || { echo 'pad 4096 not engaged'; return 1; }
    # plain CPU: bitwise logits and tokens
    [[ $(field m0-pad0 logits_fnv) == $(field m0-pad4096 logits_fnv) ]] || { echo 'plain logits differ at pad 4096'; return 1; }
    [[ $(field m0-pad0 gen:) == $(field m0-pad4096 gen:) ]] || { echo 'plain tokens differ at pad 4096'; return 1; }
    # Meta: tokens only
    for p in 256 4096; do
        [[ $(field m1-pad0 gen:) == $(field "m1-pad$p" gen:) ]] || { echo "Meta tokens differ at pad $p"; return 1; }
    done
    # negative control: the same comparison on a perturbed prompt must fail
    if [[ $(field m0-pad0 logits_fnv) == $(field m0-pad0-variant logits_fnv) ]]; then echo 'negative control: perturbed prompt kept the hash'; return 1; fi
    echo "identity PASS: plain 1-thread logits_fnv equal for pad 0/4096, Meta tokens equal, negative control differs"
    echo "info: Meta logits_fnv pad0/256/4096: $(field m1-pad0 logits_fnv | awk '{print $2}') $(field m1-pad256 logits_fnv | awk '{print $2}') $(field m1-pad4096 logits_fnv | awk '{print $2}')"
}
stage 10 build build
stage 11 plumbing plumbing
stage 12 identity identity
echo 'bw-kvpad-mac: PASS'
