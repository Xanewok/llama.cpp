#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
export PATH="$HOME/qwen38-stack7/tools/local/bin:$HOME/qwen38-tools/bin:/opt/homebrew/bin:$PATH"
TASK=$HOME/qwen38-chain-np2
cd "$TASK/src"
OUT=$PWD/codex/chain-np2-evidence
trap 'printf "%s\n" "$?" > "$TASK/verify.rc"' EXIT
export S7_GATE=$TASK/mac-model-gate.sh GPUQ_BUILD_WINDOW=1
export S7_MODEL=$HOME/qwen38-stack7/models/Qwen3.5-4B-UD-Q4_K_XL.gguf
export S7_VOCAB=$HOME/qwen38-stack7/models/swift_draft_vocab_ids.json
chmod +x "$S7_GATE"
: > "$OUT/baseline/1-1.trace.jsonl"
env LLAMA_MTP_CHAIN=1 LLAMA_MTP_FOLD_C=1 GGML_SCHED_INPUT_STAGING=1 HOST_KV_Q8=1 \
    LLAMA_ROUND_TRACE="$OUT/baseline/1-1.trace.jsonl" LLAMA_MTP_CHAIN_DUMP="$OUT/baseline/1-1.dump.txt" \
    "$S7_GATE" "$OUT/baseline/bin/bw-np2-driver" "$S7_MODEL" "$OUT/baseline/1-1.json" 1 2 1729 64 \
    > "$OUT/baseline/1-1.log" 2>&1
if bash "$TASK/check-np2-interactions.sh" "$OUT/baseline" --check-only > "$OUT/negative-control.log" 2>&1; then
    printf 'FAIL: old binary escaped the np2 identity check\n'
    exit 1
fi
python3 - "$OUT/baseline" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
for meta in (0, 1):
    off = json.loads((p / f'{meta}-0.json').read_text())
    on = json.loads((p / f'{meta}-1.json').read_text())
    assert off != on
    assert 'MTP chain: k=3' in (p / f'{meta}-1.log').read_text()
    assert 'path=chain slots=2 active=1' in (p / f'{meta}-1.log').read_text()
print('PASS: old binary control engaged and failed identity for both Meta modes')
PY
git apply -R ../round-instrumentation.patch
git fetch ../chain-np2.bundle topic/chain-np2
git merge --ff-only FETCH_HEAD
cp "$TASK/check-np2-interactions.sh" codex/check-np2-interactions.sh
nice -n 19 cmake --build build-cpu --target bw-np2-driver chain-cpu -j10
bash codex/check-np2-interactions.sh "$OUT/fixed"
nice -n 19 python3 codex/chain_cpu_identity.py --out "$OUT/np1"
python3 codex/check-chain.py "$OUT/np1"
printf 'PASS: fixed np2 and stage 23\n'
