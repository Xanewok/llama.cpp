#!/usr/bin/env bash
# Combined Mac CPU payload of topk/stack10 (run through bin/bw-mac, argument = OUT dir). One configure+build for the union of
# targets, then per-component libs (codex/stack10/mac/<comp>.sh), each sourced in its own subshell. Cheap stages first, long
# model runs last. S10_ONLY="k7 k4" reruns components only (the build always runs); S10_KEEP_GOING=0 stops at the first failure.
# bw-mac forwards no environment, so the filter may also ride in the bw-mac OUT_NAME: `stack10-mac__k7__k4` selects k7 and k4.
# Libs define <comp>_cheap and/or <comp>_model, which call `stage CODE NAME FN`.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 8
OUT=${1:?output}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
MODEL=${BW_MODEL:?}
if [[ -z ${S10_ONLY:-} ]]; then
    run_name=$(basename "$(dirname "$(dirname "$(dirname "$OUT")")")")
    [[ $run_name != *__* ]] || { S10_ONLY=${run_name#*__}; S10_ONLY=${S10_ONLY//__/ }; }
fi
LIBDIR=$ROOT/codex/stack10/mac
MGATE=$LIBDIR/model-gate.sh
export ROOT OUT MODEL LIBDIR MGATE
# envs that switch a component off or arm a mutant must not leak in from the caller
unset LLAMA_KV_ROT_PERSIST LLAMA_KV_ROT_ALLOC_FAIL LLAMA_KV_ROT_MUTANT GGML_SSM_CONV_RS_OFF GGML_SSM_CONV_RS_MUTANT GGML_SSM_CONV_RS_LOG
unset LLAMA_MTP_CHAIN LLAMA_MTP_FOLD_C LLAMA_MTP_FOLD_DEBUG_MUTANT LLAMA_MTP_CHAIN_DEBUG_REVERSE_U LLAMA_MTP_CHAIN_DEBUG_SHIFT_EMBD
unset LLAMA_ROUND_TRACE LLAMA_MTP_CHAIN_DUMP LLAMA_MTP_DRAFT_DUMP GGML_ROUND_TRACE_SYNC LLAMA_SYNC_ELIDE LLAMA_SYNC_ELIDE_DEBUG LLAMA_SYNC_ELIDE_MUTANT
unset GGML_RMS_NORM_SCALE_PAIR_ALIAS_MUTANT GGML_META_INPUT_STAGE_MUTANT TOPK_MUTANT TOPK_FAILFAST TOPK_CANONICAL_ANY HOST_DEFER HOST_KV_Q8
export S7_MODEL=$MODEL S7_GATE=$MGATE S7_VOCAB=$HOME/qwen38-stack7/models/swift_draft_vocab_ids.json

[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'FAIL dirty tracked tree'; exit 9; }
HEAD_SHA=$(git rev-parse HEAD)
printf '%s\n' "$HEAD_SHA" > "$OUT/source-commit.txt"
: > "$OUT/stages.tsv"

# CODE NAME FN [ARGS]: FN runs in a subshell with set -e; log $OUT/<comp>/NAME.log; the stage row carries the component prefix.
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$D/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$COMP-$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-stack10: $COMP-$name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 || ${S10_KEEP_GOING:-1} == 1 ]] || exit "$code"
    return 0
}

# One configure and one build for every component; -S codex/stack7-test adds the np2/chain/cand drivers to the llama.cpp tree.
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE="${BW_CCACHE:-ON}" -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
targets=(test-backend-ops test-sampling test-alloc test-llama-archs test-recurrent-state-rollback bw-np2-driver chain-cpu
    bw-cand-driver test-mmvq-params test-fattn-kv-max-cache test-meta-input-stage test-topk-batched llama-completion)
build() {
    "$BW_BUILDGATE" cmake -S codex/stack7-test -B build-cpu "${flags[@]}" "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-cpu --target "${targets[@]}" -j"${BW_JOBS:-4}"
    local t
    for t in "${targets[@]}"; do [[ -x build-cpu/bin/$t ]] || { echo "FAIL missing build-cpu/bin/$t"; return 1; }; done
}
COMP=build D=$OUT/build
mkdir -p "$D"
S10_KEEP_GOING=0 stage 10 cpu-build build

CHEAP_ORDER=(ops p55 k7 k6 k2 k10 k9 k8 k11b base gdn4)
MODEL_ORDER=(p55 k7 k4 base k11a)
want() { [[ -z ${S10_ONLY:-} || " $S10_ONLY " == *" $1 "* ]]; }
run_phase() {
    local phase=$1 comp
    shift
    for comp in "$@"; do
        want "$comp" || continue
        mkdir -p "$OUT/$comp"
        ( COMP=$comp D=$OUT/$comp; source "$LIBDIR/$comp.sh"; if declare -F "${comp}_$phase" > /dev/null; then "${comp}_$phase"; fi )
    done
}
run_phase cheap "${CHEAP_ORDER[@]}"
run_phase model "${MODEL_ORDER[@]}"

# anything in the stage table with rc != 0 fails the payload with that stage's code
python3 - "$OUT" "$HEAD_SHA" <<'PY_SUMMARY'
import hashlib, json, subprocess, sys
from pathlib import Path
p, head = Path(sys.argv[1]), sys.argv[2]
files = subprocess.check_output(['git', 'ls-files', 'src', 'ggml', 'tests', 'common'], text=True).splitlines()
h = hashlib.sha256()
for name in files:
    h.update(name.encode() + b'\0' + Path(name).read_bytes())
stages = [dict(code=int(c), name=n, rc=int(rc), wall_s_M=int(t)) for c, n, rc, t in (l.split('\t') for l in (p/'stages.tsv').read_text().splitlines())]
dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
failed = [s for s in stages if s['rc'] != 0]
result = 'PASS' if not failed and not dirty else 'FAIL'
(p/'summary.json').write_text(json.dumps(dict(result=result, commit=head, source_files=files, source_sha256=h.hexdigest(), tree_dirty=dirty,
    stages=stages, total_wall_s_M=sum(s['wall_s_M'] for s in stages)), indent=2) + '\n')
print('bw-stack10:', result, f'{len(stages)} stages', 'failed:', [s['name'] for s in failed])
sys.exit(0 if result == 'PASS' else (failed[0]['code'] if failed else 9))
PY_SUMMARY
