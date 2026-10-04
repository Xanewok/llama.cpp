#!/usr/bin/env bash
set -euo pipefail
if [[ ${MASK_UPLOAD_NICED:-} != 1 ]]; then
    export MASK_UPLOAD_NICED=1
    exec nice -n 19 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CCACHE_DISABLE=1
mkdir -p build-tmp build-cpu/mask-controls
export TMPDIR="$PWD/build-tmp"
cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --target llama-common -j8
bash codex/build_fixture.sh
cp build-cpu/host-depth build-cpu/mask-controls/host-depth-good
trap 'cp build-cpu/mask-controls/host-depth-good build-cpu/host-depth' EXIT
build-cpu/host-depth build-cpu/patch-masks.bin build-cpu/patch-logits.bin > codex/mask-upload-patch.csv 2> codex/mask-upload-patch.log
MASK_UPLOAD_META=1 build-cpu/host-depth build-cpu/meta-masks.bin build-cpu/meta-logits.bin > codex/mask-upload-meta.csv 2> codex/mask-upload-meta.log
for control in MASK_UPLOAD_WRONG_CONTROL HOST_DEPTH_STALE_CONTROL; do
    if env "$control=1" build-cpu/host-depth build-cpu/control-masks.bin build-cpu/control-logits.bin > /dev/null 2> "codex/mask-upload-$control.txt"; then
        echo "negative control escaped: $control" >&2
        exit 1
    fi
    case "$control" in
        MASK_UPLOAD_WRONG_CONTROL) rg -q 'compact mask differs from host mask' "codex/mask-upload-$control.txt" ;;
        HOST_DEPTH_STALE_CONTROL) rg -q 'sequence mutation retained stale mask' "codex/mask-upload-$control.txt" ;;
    esac
done
python3 - <<'PY_INNER'
from pathlib import Path
s = Path('src/llama-graph.cpp').read_text()
needle = 'const uint32_t prefix = candidate >= 32768 ? candidate : 0;'
assert s.count(needle) == 1
Path('build-cpu/mask-controls/llama-graph.cpp').write_text(s.replace(needle, 'const uint32_t prefix = 0;'))
s = Path('src/llama-kv-cache.cpp').read_text()
needle = 'cells.used_max_p1() != n_used || positions.size() != n_used'
assert s.count(needle) == 1
Path('build-cpu/mask-controls/llama-kv-cache.cpp').write_text(s.replace(needle, 'cells.used_max_p1() != n_used'))
PY_INNER
# Keep the production source and archive unchanged; replace one object in each test archive.
for source in llama-graph llama-kv-cache; do
    c++ -O3 -DNDEBUG -DGGML_USE_CPU -Isrc -Ibuild-cpu/src -Iinclude -Iggml/include -c "build-cpu/mask-controls/$source.cpp" -o "build-cpu/mask-controls/$source.cpp.o"
    cp build-cpu/src/libllama.a "build-cpu/mask-controls/$source.a"
    ar r "build-cpu/mask-controls/$source.a" "build-cpu/mask-controls/$source.cpp.o"
    LLAMA_LIB="build-cpu/mask-controls/$source.a" bash codex/build_fixture.sh
    if [[ $source == llama-graph ]]; then
        MASK_UPLOAD_META=1 build-cpu/host-depth build-cpu/full-meta-masks.bin build-cpu/full-meta-logits.bin > codex/mask-upload-full-meta.csv 2> codex/mask-upload-full-meta.log
        cmp build-cpu/meta-masks.bin build-cpu/full-meta-masks.bin
        cmp build-cpu/meta-logits.bin build-cpu/full-meta-logits.bin
    else
        if build-cpu/host-depth build-cpu/control-masks.bin build-cpu/control-logits.bin > /dev/null 2> codex/mask-upload-broken-cert.txt; then
            echo "broken certificate escaped" >&2
            exit 1
        fi
        rg -q 'sequence mutation did not invalidate prefix' codex/mask-upload-broken-cert.txt
    fi
done
cp build-cpu/mask-controls/host-depth-good build-cpu/host-depth
if [[ $# -gt 0 ]]; then
    build-cpu/host-depth build-cpu/real-model-unused.bin build-cpu/real-model-logits.bin "$1" > codex/mask-upload-real-model.txt 2> codex/mask-upload-real-model.log
fi
echo 'PASS: exact compact masks, Meta logits against full host masks, wrong/stale bytes and broken certificate detected'
