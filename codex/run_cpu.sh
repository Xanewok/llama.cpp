#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODEL=${MODEL:-$HOME/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
nice -n 19 g++ -std=c++17 -O2 -Iinclude -Isrc -Iggml/include codex/check-inputs.cpp -Lbuild-cpu/bin \
    -Wl,-rpath,'$ORIGIN/../build-cpu/bin' -lllama -lggml -lggml-base -o codex/check-inputs
for mode in 0 1; do
    for run in 1 2 3; do
        for arm in base patch; do
            tag=$mode-$arm-$run
            if [[ $arm = base ]]; then export GGML_INPUT_BATCH_DISABLE=1; else unset GGML_INPUT_BATCH_DISABLE; fi
            GGML_INPUT_UPLOAD_PROFILE=1 nice -n 19 codex/check-inputs "$MODEL" "$mode" "codex/$tag.bin" > "codex/$tag.log" 2>&1
            cmp "codex/$mode-base-1.bin" "codex/$tag.bin"
        done
    done
    unset GGML_INPUT_BATCH_DISABLE
    LLAMA_GRAPH_REUSE_DISABLE=1 nice -n 19 codex/check-inputs "$MODEL" "$mode" "codex/$mode-no-reuse.bin" > "codex/$mode-no-reuse.log" 2>&1
    cmp "codex/$mode-base-1.bin" "codex/$mode-no-reuse.bin"
    nice -n 19 codex/check-inputs "$MODEL" "$mode" "codex/$mode-control.bin" perturb > "codex/$mode-control.log" 2>&1
    if cmp -s "codex/$mode-base-1.bin" "codex/$mode-control.bin"; then
        printf 'Negative control did not change logits\n' >&2
        exit 1
    fi
done
nice -n 19 python3 codex/summarize_cpu.py > codex/cpu-summary.txt
cat codex/cpu-summary.txt
