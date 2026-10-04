#!/usr/bin/env bash
# Judge checks on top of codex/check-mask-upload.sh (run that first). CPU only.
set -euo pipefail
if [[ ${JUDGE_NICED:-} != 1 ]]; then export JUDGE_NICED=1; exec nice -n 19 bash "$0" "$@"; fi
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 TMPDIR="$PWD/build-tmp"
OUT=build-cpu/judge; mkdir -p "$OUT"
link() { # $1 src $2 lib $3 out
  c++ -O3 -UNDEBUG -std=c++17 -Iinclude -Isrc -Icommon -Iggml/include -Iggml/src -Ivendor -Ibuild-cpu/common "$1" -o "$3" \
    -Wl,--start-group "$2" build-cpu/common/libllama-common.a build-cpu/ggml/src/libggml.a build-cpu/ggml/src/libggml-cpu.a \
    build-cpu/ggml/src/libggml-base.a build-cpu/common/libllama-common-base.a build-cpu/vendor/cpp-httplib/libcpp-httplib.a \
    -Wl,--end-group -pthread -ldl -lm -lssl -lcrypto
}
link judge/host-depth-judge.cpp build-cpu/src/libllama.a "$OUT/hd-good"
link judge/host-depth-judge.cpp build-cpu/mask-controls/llama-graph.a "$OUT/hd-full"
"$OUT/hd-good" "$OUT/good-masks.bin" "$OUT/good-logits.bin" > "$OUT/good.csv" 2> "$OUT/good.log"
MASK_UPLOAD_META=1 "$OUT/hd-good" "$OUT/good-mmasks.bin" "$OUT/good-mlogits.bin" > "$OUT/good-meta.csv" 2> "$OUT/good-meta.log"
MASK_UPLOAD_META=1 "$OUT/hd-full" "$OUT/full-mmasks.bin" "$OUT/full-mlogits.bin" > "$OUT/full-meta.csv" 2> "$OUT/full-meta.log"
cmp "$OUT/good-mlogits.bin" "$OUT/full-mlogits.bin"
echo "PASS: judge test (hole/future/ownership reuse steps) compact Meta logits == forced-full Meta logits ($(stat -c %s "$OUT/good-mlogits.bin") B)"
# which mask bytes cross into Meta: split inputs at graph build time
JUDGE_SCHED_LOG=1 GGML_SCHED_DEBUG=1 MASK_UPLOAD_META=1 "$OUT/hd-good" "$OUT/dbg-m.bin" "$OUT/dbg-l.bin" > /dev/null 2> "$OUT/sched-good.log" || true
JUDGE_SCHED_LOG=1 GGML_SCHED_DEBUG=1 MASK_UPLOAD_META=1 "$OUT/hd-full" "$OUT/dbg-m.bin" "$OUT/dbg-l.bin" > /dev/null 2> "$OUT/sched-full.log" || true
for arm in good full; do
  echo "== $arm: Meta split inputs naming the mask (unique)"
  grep -o 'SPLIT #[0-9]*: Meta[^#]*# [0-9]* inputs.*' "$OUT/sched-$arm.log" | grep -o '\[[^]]*kq_mask[^]]*\]' | sort | uniq -c
done
