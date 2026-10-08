#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${1:?output}
mkdir -p "$OUT"
CPUGATE=${S7_GATE:-$HOME/repos/qwen38-eval-worktrees/bin/cpugate}
MODEL=${S7_MODEL:-/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf}
BIN=${S7_NP2_BIN:-build-cpu/bin/bw-np2-driver}
MODE=${2:-run}
[[ "$MODE" == run || "$MODE" == --check-only ]] || exit 2
if [[ "$MODE" == run ]]; then
    for meta in 0 1; do
        for chain in 0 1; do
            : > "$OUT/$meta-$chain.trace.jsonl"
            env LLAMA_MTP_CHAIN="$chain" LLAMA_MTP_FOLD_C=1 GGML_SCHED_INPUT_STAGING=1 HOST_KV_Q8=1 \
                LLAMA_ROUND_TRACE="$OUT/$meta-$chain.trace.jsonl" LLAMA_MTP_CHAIN_DUMP="$OUT/$meta-$chain.dump.txt" \
                "$CPUGATE" "$BIN" "$MODEL" "$OUT/$meta-$chain.json" "$meta" 2 1729 64 \
                > "$OUT/$meta-$chain.log" 2>&1
        done
    done
fi
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
for meta in (0,1):
    off=json.loads((p/f'{meta}-0.json').read_text())
    on=json.loads((p/f'{meta}-1.json').read_text())
    assert off == on, (meta, 'np2 chain request changed tokens or counters')
    shapes = []
    for chain in (0,1):
        log=(p/f'{meta}-{chain}.log').read_text()
        assert 'mtp_fold_decode: engaged n_seq=2' in log, (meta,chain,'joint fold did not engage')
        assert 'MTP chain: k=' not in log, (meta,chain,'chain engaged at np2')
        phases = [ph for line in (p/f'{meta}-{chain}.trace.jsonl').read_text().splitlines()
                  for ph in json.loads(line)['phases']]
        assert phases and all(ph['result'] == 0 for ph in phases), (meta,chain,'decode failed or trace empty')
        assert not any(ph['phase'] == 'DC' for ph in phases), (meta,chain,'chain decode at np2')
        active = {ph['n_tokens'] for ph in phases if ph['phase'] == 'D1'}
        assert active == {1, 2}, (meta,chain,'missing one- or two-sequence drafting')
        shapes.append([(ph['phase'], ph['n_tokens']) for ph in phases])
    assert shapes[0] == shapes[1], (meta,'verify/catch-up/draft shapes changed')
    off_dump = (p/f'{meta}-0.dump.txt').read_text()
    on_dump = (p/f'{meta}-1.dump.txt').read_text()
    assert off_dump and off_dump == on_dump, (meta,'draft IDs, candidates or q changed')
print('np2 chain fallback, joint fold, shapes and draft identity PASS')
PY
