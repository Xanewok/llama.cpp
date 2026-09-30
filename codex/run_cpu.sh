#!/usr/bin/env bash
set -euo pipefail
renice 19 $$ >/dev/null
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
MODEL=${MODEL:-$HOME/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
export CPU_OUT=${CPU_OUT:-codex/cpu-results-$(date +%Y%m%d-%H%M%S)}
case "$(realpath -m "$CPU_OUT")" in "$PWD"/codex/*) ;; *) echo "CPU_OUT must be inside codex" >&2; exit 1;; esac
mkdir -p "$CPU_OUT"
cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF > codex/cmake-cpu.log 2>&1
cmake --build build-cpu --target llama-server -j8 > codex/build-cpu.log 2>&1
cmake -S ../up -B build-base -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF > codex/cmake-base.log 2>&1
cmake --build build-base --target llama-server -j8 > codex/build-base.log 2>&1
compile() {
    local source=$1 build=$2 output=$3
    g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED \
        -I"$source/tools/server" -I"$source/common" -I"$source/include" -I"$source/ggml/include" \
        -I"$source/vendor" -I"$source/tools/mtmd" -I"$source" codex/cpu-test.cpp \
        "$build/tools/server/libserver-context.a" "$build/common/libllama-common-base.a" \
        -L"$build/bin" -Wl,-rpath,"$PWD/$build/bin" -lllama-common -lmtmd -lllama -lggml -lggml-cpu -lggml-base -o "$output"
}
compile . build-cpu build-cpu/bin/round-trace-cpu
compile ../up build-base build-base/bin/round-trace-cpu
for meta in 0 1; do
    for arm in up off on; do
        exe=build-cpu/bin/round-trace-cpu
        [[ $arm != up ]] || exe=build-base/bin/round-trace-cpu
        trace=''
        [[ $arm != on ]] || trace="$(realpath "$CPU_OUT")/$meta.jsonl"
        [[ -z $trace || ! -e $trace ]] || { echo 'fresh CPU trace required' >&2; exit 1; }
        LLAMA_ROUND_TRACE="$trace" "$exe" "$MODEL" "$CPU_OUT/$meta-$arm.bin" "$meta" > "$CPU_OUT/$meta-$arm.log" 2>&1
    done
done
python3 - <<'PY'
from pathlib import Path
import hashlib, os
out = Path(os.environ['CPU_OUT'])
for meta in (0, 1):
    ref = (out / f'{meta}-up.bin').read_bytes()
    for arm in ('off', 'on'):
        data = (out / f'{meta}-{arm}.bin').read_bytes()
        if data != ref:
            raise SystemExit(f'byte mismatch: meta={meta} arm={arm}')
    wrong = bytearray(ref)
    wrong[8] ^= 1
    if wrong == ref:
        raise SystemExit('changed token escaped negative control')
    wrong = bytearray(ref)
    wrong[-1] ^= 1
    if wrong == ref:
        raise SystemExit('changed logit escaped negative control')
    print(f'M: meta={meta}, {len(ref)} identical bytes, SHA256={hashlib.sha256(ref).hexdigest()}, token/logit negative controls detected')
PY
python3 codex/attrib.py "$CPU_OUT/0.jsonl" --skip 0 --trim 1 --min-rounds 1 --json > "$CPU_OUT/0-attrib.json"
python3 codex/attrib.py "$CPU_OUT/1.jsonl" --skip 0 --trim 1 --min-rounds 1 --json > "$CPU_OUT/1-attrib.json"
python3 codex/test_attrib.py
python3 codex/test_harness.py
