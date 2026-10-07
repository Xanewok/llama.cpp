#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 10
OUT=${1:?output}
mkdir -p "$OUT"
export S7_MODEL=${BW_MODEL:?} S7_VOCAB=$HOME/qwen38-stack7/models/swift_draft_vocab_ids.json
export S7_GATE=$ROOT/codex/mac-model-gate.sh
unset LLAMA_MTP_CHAIN LLAMA_MTP_FOLD_C LLAMA_MTP_FOLD_DEBUG_MUTANT LLAMA_MTP_CHAIN_DEBUG_REVERSE_U LLAMA_MTP_CHAIN_DEBUG_SHIFT_EMBD
unset LLAMA_ROUND_TRACE LLAMA_MTP_CHAIN_DUMP LLAMA_MTP_DRAFT_DUMP GGML_ROUND_TRACE_SYNC
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
S7_ONLY=${S7_ONLY:-'20 21 23 29 32 33 34'}
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    [[ " $S7_ONLY " == *" $code "* ]] || return 0
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-stack8: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit "$code"
}
[[ -z $(git status --porcelain --untracked-files=no) ]]
git rev-parse HEAD > "$OUT/source-commit.txt"
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE="${BW_CCACHE:-ON}" -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
build() {
    "$BW_BUILDGATE" cmake -S codex/stack7-test -B build-cpu "${flags[@]}" "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-cpu --target test-sampling test-llama-archs test-alloc test-recurrent-state-rollback bw-np2-driver chain-cpu bw-cand-driver -j4
}
units() {
    "$S7_GATE" build-cpu/bin/test-sampling
    "$S7_GATE" build-cpu/bin/test-alloc
    "$S7_GATE" build-cpu/bin/test-recurrent-state-rollback -m "$S7_MODEL" -c 4096 -t 4 -ngl 0
    for seed in 42 1729; do
        "$S7_GATE" build-cpu/bin/test-llama-archs --tp-sampling -s "$seed" > "$OUT/tp-sampling-$seed.log" 2>&1
        grep -q 'test_banned_candidates: .*: OK' "$OUT/tp-sampling-$seed.log"
        grep -q 'unextended chain caught.*: OK' "$OUT/tp-sampling-$seed.log"
    done
}
chain() {
    "$BW_GATE" python3 codex/chain_cpu_identity.py --out "$OUT/chain"
    "$BW_GATE" python3 codex/check-chain.py "$OUT/chain"
}
np2() {
    bash codex/check-np2-interactions.sh "$OUT/np2"
    grep -q 'MTP chain disabled: needs one sequence (n_seq=2)' "$OUT/np2/0-1.log"
    grep -q 'MTP chain disabled: needs one sequence (n_seq=2)' "$OUT/np2/1-1.log"
}
old_control() {
    mkdir -p build-np2-old-source
    git archive HEAD | tar -x -C build-np2-old-source
    (cd build-np2-old-source && patch -p1 < "$ROOT/codex/np2-old.patch")
    "$BW_BUILDGATE" cmake -S codex/stack7-test -B build-np2-old-cpu -DLLAMA_SOURCE="$ROOT/build-np2-old-source" "${flags[@]}" "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-np2-old-cpu --target bw-np2-driver -j4
    mkdir -p "$OUT/np2-old"
    for meta in 0 1; do
        for chain in 0 1; do
            env LLAMA_MTP_CHAIN="$chain" LLAMA_MTP_FOLD_C=1 GGML_SCHED_INPUT_STAGING=1 HOST_KV_Q8=1 \
                LLAMA_ROUND_TRACE="$OUT/np2-old/$meta-$chain.trace.jsonl" LLAMA_MTP_CHAIN_DUMP="$OUT/np2-old/$meta-$chain.dump.txt" \
                "$S7_GATE" build-np2-old-cpu/bin/bw-np2-driver "$S7_MODEL" "$OUT/np2-old/$meta-$chain.json" "$meta" 2 1729 64 \
                > "$OUT/np2-old/$meta-$chain.log" 2>&1
        done
        grep -q 'MTP chain: k=3' "$OUT/np2-old/$meta-1.log"
    done
    local rc=0
    bash codex/check-np2-interactions.sh "$OUT/np2-old" --check-only > "$OUT/np2-old/control.log" 2>&1 || rc=$?
    [[ $rc != 0 ]]
    grep -q 'np2 chain request changed tokens or counters' "$OUT/np2-old/control.log"
    echo "old np2 binary caught rc=$rc"
}
candidates() {
    mkdir -p "$OUT/cand"
    for spec in 1 0; do
        for temp in 0 1; do
            for arm in cand host refused; do
                "$S7_GATE" build-cpu/bin/bw-cand-driver "$S7_MODEL" "$OUT/cand/1-$spec-$temp-$arm.json" 1 "$spec" "$temp" "$arm" 1729 96 \
                    > "$OUT/cand/1-$spec-$temp-$arm.log" 2>&1
            done
        done
    done
    for temp in 0 1; do
        for arm in cand host; do
            "$S7_GATE" build-cpu/bin/bw-cand-driver "$S7_MODEL" "$OUT/cand/0-1-$temp-$arm.json" 0 1 "$temp" "$arm" 1729 96 \
                > "$OUT/cand/0-1-$temp-$arm.log" 2>&1
        done
    done
    python3 codex/bw-cand-compare.py "$OUT/cand"
}
parsers() {
    local primary=${S8_PRIMARY:?} data=$S8_PRIMARY/src/build-parser-inputs
    mkdir -p "$OUT/parser-evidence"
    python3 - "$primary/out" "$OUT/parser-evidence" <<'PY'
from pathlib import Path
import hashlib, json, sys
source, out = map(Path, sys.argv[1:])
records = {}
for name, path in [('np2-on.log', source/'np2/1-1.log'), ('np2-old-on.log', source/'np2-old/1-1.log')]:
    text = path.read_text()
    lines = [line for line in text.splitlines() if 'MTP chain' in line]
    assert lines, path
    (out/name).write_text('\n'.join(lines)+'\n')
    records[name] = {'source':str(path), 'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(out/'provenance.json').write_text(json.dumps(records, indent=2)+'\n')
PY
    HARNESS_CODE_ROOT="$data" S8_EVAL="$data" S8_MAC_EVIDENCE="$OUT/parser-evidence" \
        "$BW_GATE" python3 codex/gpu_stack8.py --fixtures
}
stage 20 cpu-build build
stage 21 unit-tests units
stage 23 np1-chain-identity chain
stage 29 np2-chain-identity np2
stage 31 recorded-parser-controls parsers
stage 32 np2-old-negative-control old_control
stage 33 cand-cpu-identity candidates
stage 34 cand-mutant bash codex/check-cand-mutant.sh "$OUT/cand-mutant"
echo 'bw-stack8: PASS Mac CPU stages'
