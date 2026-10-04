#!/usr/bin/env bash
set -euo pipefail
if [[ ${MASK_UPLOAD_NICED:-} != 1 ]]; then
    export MASK_UPLOAD_NICED=1
    exec nice -n 19 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CCACHE_DISABLE=1
mkdir -p build-tmp build-mask-cpu/mask-controls
export TMPDIR="$PWD/build-tmp"
cmake -B build-mask-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-mask-cpu --target llama-common -j8
bash codex/build_fixture.sh
cp build-mask-cpu/host-depth build-mask-cpu/mask-controls/host-depth-good
trap 'cp build-mask-cpu/mask-controls/host-depth-good build-mask-cpu/host-depth' EXIT
build-mask-cpu/host-depth build-mask-cpu/patch-masks.bin build-mask-cpu/patch-logits.bin > codex/mask-upload-patch.csv 2> codex/mask-upload-patch.log
MASK_UPLOAD_META=1 build-mask-cpu/host-depth build-mask-cpu/meta-masks.bin build-mask-cpu/meta-logits.bin > codex/mask-upload-meta.csv 2> codex/mask-upload-meta.log
for control in MASK_UPLOAD_WRONG_CONTROL HOST_DEPTH_STALE_CONTROL; do
    if env "$control=1" build-mask-cpu/host-depth build-mask-cpu/control-masks.bin build-mask-cpu/control-logits.bin > /dev/null 2> "codex/mask-upload-$control.txt"; then
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
def mutate(src, dst, needle, replacement):
    s = Path(src).read_text()
    assert s.count(needle) == 1, needle
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(s.replace(needle, replacement))
c = 'build-mask-cpu/mask-controls'
mutate('src/llama-graph.cpp', f'{c}/llama-graph/llama-graph.cpp',
       'const uint32_t prefix = candidate >= 32768 ? candidate : 0;', 'const uint32_t prefix = 0;')
cert = 'cells.used_max_p1() != n_used || positions.size() != n_used'
mutate('src/llama-kv-cache.cpp', f'{c}/owner-cert/llama-kv-cache.cpp', cert, 'cells.used_max_p1() != n_used')
mutate('src/llama-kv-cache.cpp', f'{c}/hole-cert/llama-kv-cache.cpp', cert, 'positions.size() != n_used')
mutate('src/llama-graph.cpp', f'{c}/stale-reuse/llama-graph.cpp',
       '        res &= mctx->get_kq_mask_prefix(ubatch, cparams.causal_attn) >= kq_mask->src[0]->ne[0];\n', '')
mutate('src/llama-kv-cache.cpp', f'{c}/stale-reuse/llama-kv-cache.cpp',
       'GGML_ASSERT(prefix == 0 || get_kq_mask_prefix(*ubatch, n_kv, causal_attn) >= prefix);', '')
PY_INNER
# Keep the production source and archive unchanged; replace mutated objects in each test archive.
for control in llama-graph owner-cert hole-cert stale-reuse; do
    dir=build-mask-cpu/mask-controls/$control
    cp build-mask-cpu/src/libllama.a "$dir/libllama.a"
    for source in "$dir"/*.cpp; do
        c++ -O3 -DNDEBUG -DGGML_USE_CPU -Isrc -Ibuild-mask-cpu/src -Iinclude -Iggml/include -c "$source" -o "$source.o"
        (cd "$dir" && ar r libllama.a "$(basename "$source").o")
    done
    LLAMA_LIB="$dir/libllama.a" bash codex/build_fixture.sh
    case $control in
        llama-graph)
            MASK_UPLOAD_META=1 build-mask-cpu/host-depth build-mask-cpu/full-meta-masks.bin build-mask-cpu/full-meta-logits.bin > codex/mask-upload-full-meta.csv 2> codex/mask-upload-full-meta.log
            cmp build-mask-cpu/meta-masks.bin build-mask-cpu/full-meta-masks.bin
            cmp build-mask-cpu/meta-logits.bin build-mask-cpu/full-meta-logits.bin ;;
        owner-cert|hole-cert)
            if build-mask-cpu/host-depth build-mask-cpu/control-masks.bin build-mask-cpu/control-logits.bin > /dev/null 2> "codex/mask-upload-$control.txt"; then
                echo "broken certificate escaped: $control" >&2
                exit 1
            fi
            [[ $control == owner-cert ]] && expected='sequence mutation did not invalidate prefix' || expected='hole inside prefix did not invalidate prefix'
            rg -q "$expected" "codex/mask-upload-$control.txt" ;;
        stale-reuse)
            # Both reuse guards removed: a compact graph reused after a mutation inside its prefix must change logits.
            if MASK_UPLOAD_META=1 build-mask-cpu/host-depth build-mask-cpu/control-masks.bin build-mask-cpu/control-logits.bin > /dev/null 2> codex/mask-upload-stale-reuse.txt \
                    && cmp -s build-mask-cpu/control-logits.bin build-mask-cpu/full-meta-logits.bin; then
                echo "stale compact graph reuse escaped" >&2
                exit 1
            fi ;;
    esac
done
cp build-mask-cpu/mask-controls/host-depth-good build-mask-cpu/host-depth
if [[ $# -gt 0 ]]; then
    build-mask-cpu/host-depth build-mask-cpu/real-model-unused.bin build-mask-cpu/real-model-logits.bin "$1" > codex/mask-upload-real-model.txt 2> codex/mask-upload-real-model.log
fi
echo 'PASS: exact compact masks, Meta logits against full host masks, wrong/stale bytes, broken owner/hole certificate and stale graph reuse detected'
