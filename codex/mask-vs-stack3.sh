#!/usr/bin/env bash
# Byte identity of the patched libllama against unpatched stack3 (efd30e041, static build in build-mask-base):
#   host-depth-logits: tiny Qwen35 target+MTP logits, CPU and Meta(CPU,CPU), depths 8192/90112, n_pos 4, 6-row verify + 1-row drafts
#   host-depth-orig:   raw masks+logits of the pre-prefix host-step fixture (non-meta)
#   mask-fuzz:         random KV cell states x random ubatches, raw mask bytes
set -euo pipefail
if [[ ${MASKCMP_NICED:-} != 1 ]]; then export MASKCMP_NICED=1; exec nice -n 19 bash "$0" "$@"; fi
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 TMPDIR="$PWD/build-tmp"
BASE_SRC=build-tmp/base3-src; OUT=build-mask-cpu/vs-stack3; mkdir -p "$OUT"
# logits-only variant: drop compact_test/cache_test (they need get_kq_mask_prefix, absent from unpatched stack3)
awk 'NR<51 || NR>242' codex/host-depth.cpp | sed '/cache_test(\*model, depth, draft, masks);/d' > codex/host-depth-logits.cpp
git show 407bc8f25:codex/host-depth.cpp > codex/host-depth-orig.cpp
mkdir -p "$BASE_SRC/codex"
cp codex/host-depth-logits.cpp codex/host-depth-orig.cpp "$BASE_SRC/codex/"
# the guard-row statistics need llama-kv-cells seq_pos_get (59c721c98); they do not touch the RNG or the mask bytes
awk 'NR<146 || NR>161' codex/mask-fuzz.cpp > "$BASE_SRC/codex/mask-fuzz.cpp"
link() { # tree src lib out
  c++ -O3 -UNDEBUG -std=c++17 -I"$1/include" -I"$1/src" -I"$1/common" -I"$1/ggml/include" -I"$1/ggml/src" -I"$1/vendor" -I"$5/common" "$1/codex/$2" -o "$4" \
    -Wl,--start-group "$3/src/libllama.a" "$5/common/libllama-common.a" "$5/ggml/src/libggml.a" "$5/ggml/src/libggml-cpu.a" "$5/ggml/src/libggml-base.a" \
    "$5/common/libllama-common-base.a" "$5/vendor/cpp-httplib/libcpp-httplib.a" -Wl,--end-group -pthread -ldl -lm -lssl -lcrypto
}
for src in host-depth-logits host-depth-orig mask-fuzz; do
  link "$BASE_SRC" "$src.cpp" build-mask-base "$OUT/$src-base" build-mask-base
  link . "$src.cpp" build-mask-cpu "$OUT/$src-patch" build-mask-cpu
done
for arm in base patch; do
  "$OUT/host-depth-logits-$arm" "$OUT/hdl-$arm-masks.bin" "$OUT/hdl-$arm-logits.bin" > "$OUT/hdl-$arm.csv" 2> "$OUT/hdl-$arm.log"
  MASK_UPLOAD_META=1 "$OUT/host-depth-logits-$arm" "$OUT/hdl-$arm-mmasks.bin" "$OUT/hdl-$arm-mlogits.bin" > "$OUT/hdl-$arm-meta.csv" 2> "$OUT/hdl-$arm-meta.log"
  "$OUT/host-depth-orig-$arm" "$OUT/hdo-$arm-masks.bin" "$OUT/hdo-$arm-logits.bin" > "$OUT/hdo-$arm.csv" 2> "$OUT/hdo-$arm.log"
  "$OUT/mask-fuzz-$arm" "$OUT/fuzz-$arm.bin" "${FUZZ_CASES:-600}" > "$OUT/fuzz-$arm.txt" 2> "$OUT/fuzz-$arm.log"
done
for pair in hdl-%s-logits hdl-%s-mlogits hdo-%s-masks hdo-%s-logits fuzz-%s; do
  a=$(printf "$OUT/$pair.bin" base); b=$(printf "$OUT/$pair.bin" patch)
  [[ -s $a ]] || { echo "empty reference: $a" >&2; exit 1; }
  cmp "$a" "$b"; echo "IDENTICAL $(basename "$b") $(stat -c %s "$b") B sha256 $(sha256sum < "$b" | cut -c1-16)"
done
cat "$OUT/fuzz-patch.txt"
echo "PASS: patched == unpatched stack3 byte-for-byte"
