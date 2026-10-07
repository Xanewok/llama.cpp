#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
export PATH="$HOME/qwen38-stack7/tools/local/bin:$HOME/qwen38-tools/bin:/opt/homebrew/bin:$PATH"
TASK=$HOME/qwen38-chain-np2
while [[ ! -f "$TASK/verify.rc" ]]; do sleep 20; done
[[ $(cat "$TASK/verify.rc") == 0 ]]
cd "$TASK/src"
OUT=$PWD/codex/chain-np2-evidence/graph-proof
mkdir -p "$OUT"
git diff 48b82b0d1 HEAD -- common/speculative.cpp > "$OUT/tested-source.patch"
git apply -R "$OUT/tested-source.patch"
git apply ../round-instrumentation.patch
git apply ../graph-instrumentation.patch
restore() {
    git apply -R ../graph-instrumentation.patch
    git apply -R ../round-instrumentation.patch
    git apply "$OUT/tested-source.patch"
}
trap restore EXIT
nice -n 19 cmake --build build-cpu --target bw-np2-driver -j10
env LLAMA_MTP_CHAIN=1 LLAMA_MTP_FOLD_C=1 GGML_SCHED_INPUT_STAGING=1 HOST_KV_Q8=1 \
    "$TASK/mac-model-gate.sh" build-cpu/bin/bw-np2-driver \
    "$HOME/qwen38-stack7/models/Qwen3.5-4B-UD-Q4_K_XL.gguf" "$OUT/out.json" 0 2 1729 64 > "$OUT/log.txt" 2>&1
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
log = (p / 'log.txt').read_text()
assert 'NP2_GRAPH requested=3 path=plain samplers=2 tokens=3 outputs=3' in log
assert 'NP2_ROUND round=1 path=chain slots=2 active=1' in log
assert json.loads((p / 'out.json').read_text()) == json.loads((p.parent / 'baseline/0-1.json').read_text())
print('PASS: requested chain ran the plain graph with two attached samplers; baseline output reproduced')
PY
restore
trap - EXIT
nice -n 19 cmake --build build-cpu --target bw-np2-driver chain-cpu -j10
printf 'PASS: graph proof completed and fixed source restored\n'
