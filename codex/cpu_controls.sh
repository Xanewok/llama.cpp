#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
OUT=${1:-codex/cpu-validation}
MODEL=/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf
GATE=("$HOME/repos/qwen38-eval-worktrees/bin/cpugate" ./codex/gate.sh)
FLAGS=(-m "$MODEL" -c 4096 -np 1 -ngl 0 -fa on --kv-pad 4096 -b 2048 -ub 512 --temp 1 \
       --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-type-k f16 --spec-draft-type-v f16 -t 8 -tb 8 --tf-carry-streams)
for arm in rerun misaligned permuted; do
    EXTRA=()
    [[ "$arm" == misaligned ]] && EXTRA=(--tf-hidden-offset -2)
    [[ "$arm" == permuted ]] && EXTRA=(--tf-permute)
    "${GATE[@]}" build-cpu/bin/llama-spec-tf "${FLAGS[@]}" --tf-stage B --tf-target "$OUT/tool/target" \
        --tf-out "$OUT/tool/$arm" "${EXTRA[@]}"
done
"${GATE[@]}" python3 codex/spec_tf.py --target "$OUT/tool/target" --arm "$OUT/tool/full" --arm "$OUT/tool/rerun" \
    --arm "$OUT/tool/misaligned" --arm "$OUT/tool/permuted" --output "$OUT/controls"
"${GATE[@]}" python3 codex/check_controls.py "$OUT"
