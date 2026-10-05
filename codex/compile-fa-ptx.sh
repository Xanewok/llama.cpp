#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
mkdir -p build-audit/v4
for file in fattn-common.cuh fattn-vec.cuh; do
    git show "400e24962:ggml/src/ggml-cuda/$file" > "build-audit/v4/$file"
done
printf '#include "fattn-vec.cuh"\nDECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);\n' > build-audit/v4/probe.cu
printf '#include "fattn-vec.cuh"\nDECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);\n' > build-audit/vec.cu
printf '#include "fattn-tile.cuh"\nDECL_FATTN_TILE_CASE(256, 256);\n' > build-audit/tile.cu
printf '#include "fattn-mma-f16.cuh"\nDECL_FATTN_MMA_F16_CASE(256, 256, 4, 8);\nDECL_FATTN_MMA_Q8_0_CASE(256, 256, 4, 8);\n' > build-audit/mma.cu
gate=/home/xanewok/repos/qwen38-eval-worktrees/bin/buildgate
cccl=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
flags=(-ptx -O3 -std=c++17 -arch=compute_120a -use_fast_math -extended-lambda -DGGML_CUDA_USE_GRAPHS -Iggml/include -Iggml/src -Iggml/src/ggml-cuda -I"$cccl/thrust" -I"$cccl/libcudacxx/include" -I"$cccl/cub")
for pair in 'v4-q8 v4/probe' 'v5-q8 vec' 'v5-tile tile' 'v5-mma mma'; do
    read -r name src <<< "$pair"
    nice -n 19 "$gate" /usr/local/cuda-13.2/bin/nvcc "${flags[@]}" "build-audit/$src.cu" -o "build-audit/$name.ptx"
done
for tag in v4 v5; do
    nice -n 19 "$gate" /usr/local/cuda-13.2/bin/ptxas -arch=sm_120a -g "build-audit/$tag-q8.ptx" -o "build-audit/$tag-q8-debug.cubin"
done
nice -n 19 python3 codex/check-fa-ptx.py
