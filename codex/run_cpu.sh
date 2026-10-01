#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TMPDIR="$PWD/build-cpu/tmp"
mkdir -p "$TMPDIR"
MODEL=${MODEL:-/home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
nice -n 19 cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DGGML_CPU_REPACK=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF
nice -n 19 cmake --build build-cpu -j8 --target test-sampling test-sampling-prefilter test-llama-archs
for probe in state draft; do
    nice -n 19 g++ -std=c++17 -O2 -Icommon -Iinclude -Isrc -Iggml/include -Ivendor \
        "codex/${probe}_probe.cpp" build-cpu/bin/libllama-common.so build-cpu/bin/libllama.so \
        build-cpu/bin/libggml.so build-cpu/bin/libggml-cpu.so build-cpu/bin/libggml-base.so \
        build-cpu/common/libllama-common-base.a -Wl,-rpath,"$PWD/build-cpu/bin" -o "build-cpu/bin/audit-$probe"
done
nice -n 19 python3 codex/audit_controls.py > codex/controls.log 2>&1
status=0
nice -n 19 python3 codex/audit_controls.py --broken-mask-bound > codex/controls-negative.log 2>&1 || status=$?
[[ $status != 0 ]] && rg -q 'FAIL: padded MMA mask address' codex/controls-negative.log
nice -n 19 build-cpu/bin/audit-state "$MODEL" > codex/law.log 2>&1
status=0
S4_CORRUPT_Q=1 nice -n 19 build-cpu/bin/audit-state "$MODEL" > codex/law-negative.log 2>&1 || status=$?
[[ $status == 1 ]] && rg -q 'FAIL: law oracle failed' codex/law-negative.log
nice -n 19 build-cpu/bin/test-sampling > codex/test-sampling.log 2>&1
nice -n 19 build-cpu/bin/test-sampling-prefilter "$MODEL" > codex/prefilter.log 2>&1
nice -n 19 build-cpu/bin/test-llama-archs --tp-sampling -s 42 > codex/tp-rows.log 2>&1
nice -n 19 build-cpu/bin/test-llama-archs --input-upload > codex/input-upload.log 2>&1
nice -n 19 build-cpu/bin/audit-state "$MODEL" tp > codex/state-short-cpu.log 2>&1
for rows in 4 5 6; do
    S4_ROWS=$rows nice -n 19 build-cpu/bin/audit-draft "$MODEL" tp 3 deferred > "codex/draft-cpu-$rows.log" 2>&1
done
# Preserve the extended fixture result, including a failure; it is not a passing rollback certificate.
status=0
D_AUDIT_PREFIX_N=257 nice -n 19 build-cpu/bin/audit-state "$MODEL" tp > codex/state-cpu.log 2>&1 || status=$?
printf '%s\n' "$status" > codex/state-extended.status
failures=$status
for attention in on off; do
    settings=(); [[ $attention == off ]] && settings=(D_AUDIT_FA_OFF=1)
    rc=0
    env "${settings[@]}" D_AUDIT_KV_F16=1 D_AUDIT_K=5 D_AUDIT_FULL_ONLY=1 D_AUDIT_PREFIX_N=257 \
        nice -n 19 build-cpu/bin/audit-state "$MODEL" decode > "codex/state-single-f16-fa-$attention.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" > "codex/state-single-f16-fa-$attention.status"
    [[ $rc == 0 ]] || failures=1
done
nice -n 19 bash -n codex/gpu_ab.sh
nice -n 19 bash codex/gpu_ab.sh > codex/gpu-plan.log
exit "$failures"
