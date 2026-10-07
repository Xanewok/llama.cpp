#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
OUT=${1:-codex/cpu-exact}
PROMPTS=${2:-1}
GATE=(./codex/gate.sh "$HOME/repos/qwen38-eval-worktrees/bin/cpugate")
MODEL=/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf
"${GATE[@]}" python3 codex/cpu_validate.py --prompts "$PROMPTS" --predict 256 --out "$OUT"
"${GATE[@]}" build-cpu/bin/llama-spec-tf -m "$MODEL" -c 4096 -np 1 -ngl 0 -fa on -ctk q8_0 -ctv q8_0 \
    --kv-pad 4096 -b 2048 -ub 512 --temp 1 --top-k 20 --top-p 0.95 --min-p 0.05 --ignore-eos \
    --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-type-k f16 --spec-draft-type-v f16 -t 8 -tb 8 \
    --tf-stage AB --tf-streams "$OUT/streams.json" --tf-out "$OUT/tool"
"${GATE[@]}" python3 codex/check_rows.py --target "$OUT/tool/target" --arm "$OUT/tool/full" --trace "$OUT/trace.jsonl" \
    --streams "$OUT/streams.json" --output "$OUT/rows.json"
"${GATE[@]}" python3 codex/spec_tf.py --target "$OUT/tool/target" --arm "$OUT/tool/full" \
    --bootstrap 2000 --output "$OUT/estimate"
