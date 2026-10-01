#!/usr/bin/env bash
set -euo pipefail
renice 19 $$ >/dev/null
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/build-tmp"
mkdir -p "$TMPDIR"
MODEL=${MODEL:-$HOME/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
export CPU_OUT=${CPU_OUT:-codex/ar-pool-cpu-results}
case "$(realpath -m "$CPU_OUT")" in "$PWD"/codex/*) ;; *) echo 'CPU_OUT must be inside codex' >&2; exit 1;; esac
mkdir -p "$CPU_OUT" build-ar-pool-reference/src
git archive 2b645a655 | tar -x -C build-ar-pool-reference/src
for build in build-ar-pool-reference build-cpu; do
    source=.
    [[ $build != build-ar-pool-reference ]] || source=$build/src
    cmake -S "$source" -B "$build" -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF > "$CPU_OUT/$build-cmake.log" 2>&1
    cmake --build "$build" --target server-context -j8 > "$CPU_OUT/$build-build.log" 2>&1
    g++ -std=c++17 -O2 -pthread -DGGML_SHARED -DLLAMA_SHARED \
        -I"$source/tools/server" -I"$source/common" -I"$source/include" -I"$source/ggml/include" \
        -I"$source/vendor" -I"$source/tools/mtmd" -I"$source" "$source/codex/cpu-test.cpp" \
        "$build/tools/server/libserver-context.a" "$build/common/libllama-common-base.a" \
        -L"$build/bin" -Wl,-rpath,"$PWD/$build/bin" -lllama-common -lmtmd -lllama -lggml -lggml-cpu -lggml-base -o "$build/bin/round-trace-cpu"
done
python3 codex/test_ar_pool.py
for meta in 0 1; do
    for arm in reference default pool2 pool8 trace8; do
        exe=build-cpu/bin/round-trace-cpu
        [[ $arm != reference ]] || exe=build-ar-pool-reference/bin/round-trace-cpu
        trace=''
        unset GGML_CUDA_AR_POOL
        [[ $arm != pool2 ]] || export GGML_CUDA_AR_POOL=2
        [[ $arm != pool8 && $arm != trace8 ]] || export GGML_CUDA_AR_POOL=8
        if [[ $arm == trace8 ]]; then
            trace="$(realpath "$CPU_OUT")/$meta.jsonl"
            [[ ! -e $trace ]] || { echo 'fresh trace required; set CPU_OUT' >&2; exit 1; }
        fi
        LLAMA_ROUND_TRACE="$trace" "$exe" "$MODEL" "$CPU_OUT/$meta-$arm.bin" "$meta" > "$CPU_OUT/$meta-$arm.log" 2>&1
    done
done
python3 - <<'PY'
from pathlib import Path
import hashlib, json, os, re
out = Path(os.environ['CPU_OUT'])
results = []
for meta in (0, 1):
    reference = (out / f'{meta}-reference.bin').read_bytes()
    for arm in ('reference', 'default', 'pool2', 'pool8', 'trace8'):
        data = (out / f'{meta}-{arm}.bin').read_bytes()
        if data != reference:
            raise SystemExit(f'identity failed: {meta}/{arm}')
        log = (out / f'{meta}-{arm}.log').read_text()
        results.append(dict(meta=meta, arm=arm, bytes_M=len(data), sha256=hashlib.sha256(data).hexdigest(),
                            wall_s_M=float(re.search(r'CPU wall seconds=([0-9.]+)', log)[1])))
    for index in (8, len(reference) - 1):
        broken = bytearray(reference)
        broken[index] ^= 1
        if broken == reference:
            raise SystemExit('token/logit negative control escaped')
    print(f'M: meta={meta}: {len(reference)} identical bytes across all arms; token/logit negative controls detected')
(out / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
PY
for meta in 0 1; do
    python3 codex/attrib.py "$CPU_OUT/$meta.jsonl" --skip 0 --trim 1 --min-rounds 1 --json > "$CPU_OUT/$meta-attrib.json"
done
python3 codex/test_attrib.py
