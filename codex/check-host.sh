#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${1:?output}
mkdir -p "$OUT"
CPUGATE=${S7_GATE:-$HOME/repos/qwen38-eval-worktrees/bin/cpugate}
MODEL=${S7_MODEL:-/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf}
VARIANTS=(
    'ref:LLAMA_HOST_TRIMS_SEQRM=0 LLAMA_HOST_TRIMS_BULKH=0'
    'ref2:LLAMA_HOST_TRIMS_SEQRM=0 LLAMA_HOST_TRIMS_BULKH=0'
    'seqrm:LLAMA_HOST_TRIMS_SEQRM=1 LLAMA_HOST_TRIMS_BULKH=0'
    'bulk:LLAMA_HOST_TRIMS_SEQRM=0 LLAMA_HOST_TRIMS_BULKH=1'
    'fix:' 'fixv:LLAMA_HOST_TRIMS_VERIFY=1'
    'legacy:LLAMA_HOST_TRIMS_SEQRM=1 LLAMA_HOST_TRIMS_BULKH=2'
    'mut-bulk:LLAMA_HOST_TRIMS_MUTANT=bulk_off'
    'mut-seqrm:LLAMA_HOST_TRIMS_MUTANT=seqrm_skip'
)
for meta in 0 1; do
    for spec in "${VARIANTS[@]}"; do
        name=${spec%%:*}
        read -r -a assigns <<< "${spec#*:}"
        rc=0
        env -u LLAMA_HOST_TRIMS_SEQRM -u LLAMA_HOST_TRIMS_BULKH -u LLAMA_HOST_TRIMS_VERIFY -u LLAMA_HOST_TRIMS_MUTANT \
            ${assigns[@]+"${assigns[@]}"} LD_LIBRARY_PATH="$ROOT/build-cpu/bin" GGML_SCHED_INPUT_STAGING=1 HOST_KV_Q8=1 \
            "$CPUGATE" build-cpu/bin/bw-np2-driver "$MODEL" "$OUT/$meta-$name.json" "$meta" 2 1729 64 \
            > "$OUT/$meta-$name.log" 2>&1 || rc=$?
        if [[ $rc != 0 ]]; then
            [[ $name == mut-* ]] || exit 1
            echo "$rc" > "$OUT/$meta-$name.rc"
        fi
    done
done
rc=0
python3 codex/bw-np2-compare.py "$OUT" 0,1 || rc=$?
# The historical hoisted-pointer defect can be GPU-only. Both intentional mutants must still be caught.
[[ $rc == 0 || $rc == 3 ]]
