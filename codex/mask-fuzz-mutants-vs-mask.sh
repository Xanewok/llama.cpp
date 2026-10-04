#!/usr/bin/env bash
# Negative controls for the dense-fill fix: each mutant of src/llama-kv-cache.cpp must change mask-fuzz or host-depth-orig bytes vs topic/stack3-mask.
# Run codex/mask-vs-mask.sh first (reference outputs and linked fixtures).
set -uo pipefail
if [[ ${MASKCMP_NICED:-} != 1 ]]; then export MASKCMP_NICED=1; exec nice -n 19 bash "$0" "$@"; fi
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 TMPDIR="$PWD/build-tmp"
OUT=build-mask-cpu/vs-mask; M=$OUT/fill-mutants; mkdir -p "$M"
link() { # src lib out
  c++ -O3 -UNDEBUG -std=c++17 -Iinclude -Isrc -Icommon -Iggml/include -Iggml/src -Ivendor -Ibuild-mask-cpu/common "codex/$1" -o "$3" \
    -Wl,--start-group "$2" build-mask-cpu/common/libllama-common.a build-mask-cpu/ggml/src/libggml.a build-mask-cpu/ggml/src/libggml-cpu.a build-mask-cpu/ggml/src/libggml-base.a \
    build-mask-cpu/common/libllama-common-base.a build-mask-cpu/vendor/cpp-httplib/libcpp-httplib.a -Wl,--end-group -pthread -ldl -lm -lssl -lcrypto
}
python3 - <<'PY'
from pathlib import Path
s = Path("src/llama-kv-cache.cpp").read_text()
def mut(name, old, new):
    assert s.count(old) == 1, name
    d = Path("build-mask-cpu/vs-mask/fill-mutants") / name; d.mkdir(parents=True, exist_ok=True)
    (d / "llama-kv-cache.cpp").write_text(s.replace(old, new))
mut("no_dup_guard", "cells.used_max_p1() == n_used && positions.size() == n_used", "cells.used_max_p1() == n_used")
mut("no_used_max_guard", "cells.used_max_p1() == n_used && positions.size() == n_used", "positions.size() == n_used")
mut("no_2d_tie", "cells.ext_get(jt->second).is_2d_gt(p1_x, p1_y)", "false")
mut("no_future", "for (; it != positions.end(); ++it) {", "for (; false && it != positions.end(); ++it) {")
PY
rc=0
for d in "$M"/*/; do
  n=$(basename "$d"); cp build-mask-cpu/src/libllama.a "$d/libllama.a"
  c++ -O3 -DNDEBUG -DGGML_USE_CPU -std=c++17 -Isrc -Ibuild-mask-cpu/src -Iinclude -Iggml/include -c "$d/llama-kv-cache.cpp" -o "$d/llama-kv-cache.cpp.o" && (cd "$d" && ar r libllama.a llama-kv-cache.cpp.o)
  link mask-fuzz.cpp "$d/libllama.a" "$d/fuzz"; link host-depth-orig.cpp "$d/libllama.a" "$d/hdo"
  "$d/fuzz" "$d/fuzz.bin" "${FUZZ_CASES:-600}" > /dev/null 2>&1; "$d/hdo" "$d/hdo-masks.bin" "$d/hdo-logits.bin" > /dev/null 2> "$d/hdo.log"; hrc=$?
  verdict=SURVIVED
  cmp -s "$d/fuzz.bin" "$OUT/fuzz-ref.bin" || verdict="killed(fuzz bytes differ)"
  [[ $hrc != 0 ]] && verdict="$verdict+hdo-failed($(tail -n1 "$d/hdo.log"))"
  cmp -s "$d/hdo-masks.bin" "$OUT/hdo-ref-masks.bin" || verdict="$verdict+hdo-masks-differ"
  echo "$n: $verdict"
  [[ $verdict == SURVIVED ]] && rc=1
done
exit $rc
