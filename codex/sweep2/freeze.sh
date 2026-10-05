#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
if [[ $(ps -o ni= -p $$ | tr -d ' ') != 19 ]]; then exec nice -n 19 bash "$0" "$@"; fi
cd "$(dirname "$0")/../.."
git diff --quiet -- tests/bench-mmvq-shapes.cpp tests/CMakeLists.txt ggml
bash codex/sweep2/build.sh
dest=codex/sweep2/frozen
[[ ! -e $dest ]]
mkdir -p "$dest"
cp build-cuda/bin/bench-mmvq-shapes "$dest/"
for path in build-cuda/bin/libggml*.so*; do cp -L "$path" "$dest/$(basename "$path")"; done
for path in "$dest"/bench-mmvq-shapes "$dest"/*.so*; do
    case $(basename "$path") in
        libggml-cuda.so*) patchelf --set-rpath '$ORIGIN:/usr/local/cuda-13.2/lib64' "$path" ;;
        *) patchelf --set-rpath '$ORIGIN' "$path" ;;
    esac
done
{ git rev-parse HEAD; echo 'CUDA 13.2; CCCL v3.4.3 local; arch 120a; NCCL OFF; Release; bench target only'; sha256sum tests/bench-mmvq-shapes.cpp tests/CMakeLists.txt; } > "$dest/SOURCE"
(cd "$dest"; sha256sum SOURCE > SOURCE.sha256; sha256sum SOURCE SOURCE.sha256 bench-mmvq-shapes *.so* > SHA256SUMS)
