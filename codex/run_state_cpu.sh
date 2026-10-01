#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
unset E_PREFIX E_K E_R E_FOLLOW E_EXACT E_BROKEN_SLOT E_SERIALIZED E_FA E_F16 E_UBATCH E_THREADS E_LOGITS_DIR GGML_CPU_TILED_MM_FORCE
export GGML_CPU_TILED_MM=1
MODEL=${MODEL:-/home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
NATIVE_BIN=${NATIVE_BIN:-/home/xanewok/repos/ra-gpu-paths/stack2/build-cpu/bin}
mkdir -p build-cpu/tmp build-probe
export TMPDIR="$PWD/build-cpu/tmp"
nice -n 19 cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DGGML_CPU_REPACK=OFF \
    -DGGML_LLAMAFILE=OFF -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF
nice -n 19 cmake --build build-cpu -j8 --target llama-common
nice -n 19 g++ -std=c++17 -O2 -Icommon -Iinclude -Isrc -Iggml/include -Ivendor \
    codex/state_split_probe.cpp build-cpu/bin/libllama-common.so build-cpu/bin/libllama.so \
    build-cpu/bin/libggml.so build-cpu/bin/libggml-cpu.so build-cpu/bin/libggml-base.so \
    build-cpu/common/libllama-common-base.a -Wl,-rpath,"$PWD/build-cpu/bin" -o build-probe/state-split-probe
nice -n 19 python3 codex/state_summary.py --self-test
E_EXACT=1 nice -n 19 build-probe/state-split-probe "$MODEL" > codex/sweep-exact.log 2>&1
E_PREFIX=257 E_SERIALIZED=1 E_EXACT=1 \
    nice -n 19 build-probe/state-split-probe "$MODEL" > codex/state-bytes-exact.log 2>&1
E_PREFIX=257 E_K=5 E_R=5 E_FOLLOW=200 E_EXACT=1 E_SERIALIZED=1 \
    nice -n 19 build-probe/state-split-probe "$MODEL" > codex/follow-exact.log 2>&1
E_PREFIX=257 E_K=5 E_R=5 E_FOLLOW=200 LD_LIBRARY_PATH="$NATIVE_BIN" \
    nice -n 19 build-probe/state-split-probe "$MODEL" > codex/follow-native.log 2>&1
E_PREFIX=257 E_K=5 E_R=5 E_UBATCH=64 LD_LIBRARY_PATH="$NATIVE_BIN" \
    nice -n 19 build-probe/state-split-probe "$MODEL" > codex/ubatch-native-64.log 2>&1
E_PREFIX=257 LD_LIBRARY_PATH="$NATIVE_BIN" \
    nice -n 19 build-probe/state-split-probe "$MODEL" > codex/depth-native.log 2>&1
E_K=5 E_R=5 LD_LIBRARY_PATH="$NATIVE_BIN" \
    nice -n 19 build-probe/state-split-probe "$MODEL" > codex/prefix-native.log 2>&1
rc=0
E_PREFIX=257 E_K=5 E_R=5 E_BROKEN_SLOT=1 \
    nice -n 19 build-probe/state-split-probe "$MODEL" > codex/wrong-slot-negative.log 2>&1 || rc=$?
[[ $rc == 1 ]] && rg -q 'FAIL: negative control: wrong snapshot detected' codex/wrong-slot-negative.log
nice -n 19 python3 codex/state_summary.py --exact --sweep codex/sweep-exact.log codex/state-bytes-exact.log codex/follow-exact.log > codex/exact-summary.json
nice -n 19 python3 codex/state_summary.py codex/follow-native.log codex/depth-native.log codex/prefix-native.log codex/ubatch-native-64.log > codex/native-summary.json
nice -n 19 bash -n codex/gpu_ab.sh
nice -n 19 bash codex/gpu_ab.sh
