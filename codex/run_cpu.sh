#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$PWD/build-cpu/tmp"
mkdir -p "$TMPDIR" build-cpu/common
MODEL=${MODEL:-/home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf}
CPU_BUILD=${CPU_BUILD:-/home/xanewok/repos/s4-runaway/stack2/build-cpu}
OUT=${OUT:-$PWD/build-cpu-check}
[[ $OUT == "$PWD"/build-* && ! -e $OUT ]] || { echo 'Use a fresh build-* OUT inside this clone'; exit 2; }
if [[ $CPU_BUILD != "$PWD/build-cpu" ]]; then
    rg -q '^GGML_CUDA:BOOL=OFF$' "$CPU_BUILD/CMakeCache.txt"
    rg -q '^GGML_CPU_REPACK:BOOL=OFF$' "$CPU_BUILD/CMakeCache.txt"
    cp "$CPU_BUILD/common/libllama-common-base.a" build-cpu/common/
else
    nice -n 19 cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CPU_REPACK=OFF -DGGML_CCACHE=OFF \
        -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF
    nice -n 19 cmake --build build-cpu -j8 --target llama-server-impl
fi
nice -n 19 bash codex/build_probe.sh "$CPU_BUILD" "$PWD/build-cpu/verify-probe"
mkdir -p "$OUT/history"
nice -n 19 python3 codex/reconstruct.py --out "$OUT/history" > "$OUT/extraction.json"
nice -n 19 build-cpu/verify-probe prepare "$MODEL" "$OUT/history/S4-pair1.history.json" "$OUT/history.tokens.json" > "$OUT/prepare.log" 2>&1
nice -n 19 python3 - "$OUT" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads((p / 'history.tokens.json').read_text())
d.update(tokens=d['tokens'][:80], points=[32, 48], name='cpu-fixture', onset_token_D=32, context_size=256)
(p / 'fixture.tokens.json').write_text(json.dumps(d))
(p / 'subset.json').write_text(json.dumps(list(range(248319, -1, -10))))
PY
nice -n 19 python3 codex/compare.py --self-test > "$OUT/controls.log"
for kv in q8_0 f16; do
    for subset in full subset; do
        ids=-; [[ $subset == full ]] || ids="$OUT/subset.json"
        run="$OUT/$kv-$subset"
        VERIFY_INDEPENDENT=1 nice -n 19 build-cpu/verify-probe run "$MODEL" "$OUT/fixture.tokens.json" "$run" cpu "$kv" "$ids" 5 > "$run.log" 2>&1
        nice -n 19 python3 codex/compare.py "$run" > "$run.summary.json"
        nice -n 19 python3 codex/compare.py "$run" --validate-fixture > "$run.validation.log"
    done
done
VERIFY_INDEPENDENT=1 VERIFY_SWAP_ROWS=1 nice -n 19 build-cpu/verify-probe run "$MODEL" "$OUT/fixture.tokens.json" "$OUT/negative" cpu q8_0 - 5 > "$OUT/negative.log" 2>&1
status=0
nice -n 19 python3 codex/compare.py "$OUT/negative" --validate-fixture > "$OUT/negative.summary.json" 2> "$OUT/negative.error" || status=$?
[[ $status == 1 ]] && rg -q 'independent context oracle failed' "$OUT/negative.error"
echo 'M: CPU harness validation and swapped-row negative control PASS; decode-width differences are in comparison.json'
