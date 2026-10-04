#!/usr/bin/env bash
# Byte identity of this tree's static libllama against topic/stack3-mask (~/repos/stack3-mask, its build-mask-cpu; read-only):
#   host-depth: compact/full masks + logits, CPU and Meta(CPU,CPU); host-depth-orig: pre-prefix fixture; mask-fuzz: raw mask bytes
# Run codex/check-mask-upload.sh first (this tree's build-mask-cpu).
set -euo pipefail
if [[ ${MASKCMP_NICED:-} != 1 ]]; then export MASKCMP_NICED=1; exec nice -n 19 bash "$0" "$@"; fi
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 TMPDIR="$PWD/build-tmp"
REF=$HOME/repos/stack3-mask; OUT=build-mask-cpu/vs-mask; mkdir -p "$OUT"
# the fixtures include ../tests and ../src relative to themselves, so each arm compiles its own tree's copy
link() { # tree build src out
  c++ -O3 -UNDEBUG -std=c++17 -I"$1/include" -I"$1/src" -I"$1/common" -I"$1/ggml/include" -I"$1/ggml/src" -I"$1/vendor" -I"$2/common" "$1/codex/$3" -o "$4" \
    -Wl,--start-group "$2/src/libllama.a" "$2/common/libllama-common.a" "$2/ggml/src/libggml.a" "$2/ggml/src/libggml-cpu.a" "$2/ggml/src/libggml-base.a" \
    "$2/common/libllama-common-base.a" "$2/vendor/cpp-httplib/libcpp-httplib.a" -Wl,--end-group -pthread -ldl -lm -lssl -lcrypto
}
for src in host-depth host-depth-orig mask-fuzz; do
  cmp "codex/$src.cpp" "$REF/codex/$src.cpp"
  link "$REF" "$REF/build-mask-cpu" "$src.cpp" "$OUT/$src-ref"
  link . build-mask-cpu "$src.cpp" "$OUT/$src-new"
done
for arm in ref new; do
  "$OUT/host-depth-$arm" "$OUT/hd-$arm-masks.bin" "$OUT/hd-$arm-logits.bin" > "$OUT/hd-$arm.csv" 2> "$OUT/hd-$arm.log"
  MASK_UPLOAD_META=1 "$OUT/host-depth-$arm" "$OUT/hd-$arm-mmasks.bin" "$OUT/hd-$arm-mlogits.bin" > "$OUT/hd-$arm-meta.csv" 2> "$OUT/hd-$arm-meta.log"
  "$OUT/host-depth-orig-$arm" "$OUT/hdo-$arm-masks.bin" "$OUT/hdo-$arm-logits.bin" > "$OUT/hdo-$arm.csv" 2> "$OUT/hdo-$arm.log"
  "$OUT/mask-fuzz-$arm" "$OUT/fuzz-$arm.bin" "${FUZZ_CASES:-600}" > "$OUT/fuzz-$arm.txt" 2> "$OUT/fuzz-$arm.log"
done
for pair in hd-%s-masks hd-%s-logits hd-%s-mmasks hd-%s-mlogits hdo-%s-masks hdo-%s-logits fuzz-%s; do
  a=$(printf "$OUT/$pair.bin" ref); b=$(printf "$OUT/$pair.bin" new)
  [[ -s $a ]] || { echo "empty reference: $a" >&2; exit 1; }
  cmp "$a" "$b"; echo "IDENTICAL $(basename "$b") $(stat -c %s "$b") B sha256 $(sha256sum < "$b" | cut -c1-16)"
done
# negative control: one flipped byte must be caught
cp "$OUT/hd-new-mlogits.bin" "$OUT/control.bin"; printf '\x01' | dd of="$OUT/control.bin" bs=1 seek=100 conv=notrunc status=none
if cmp -s "$OUT/hd-ref-mlogits.bin" "$OUT/control.bin"; then echo "negative control escaped" >&2; exit 1; fi
echo "PASS: rebased == topic/stack3-mask byte-for-byte (flipped-byte control detected)"
