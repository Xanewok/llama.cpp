#!/usr/bin/env bash
# kld_dump for each member of the stack10/stack11 pair, built against the headers of the libraries it loads (common_params and
# llama_context_params differ between the two, so one binary is not ABI-compatible with both), then the per-model configs.
# usage: build_dump.sh GEN S11_FROZEN S11_TREE S10_FROZEN S10_REPO     (CPU only; run by bw-stack11-base stage 71 after the freeze)
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 OMP_NUM_THREADS=8
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
GEN=${1:?} S11=${2:?} TREE=${3:?} S10=${4:?} S10_REPO=${5:?}
CPUS=${CPUS:-0-3,12-15}
s11_sha=$(head -n 1 "$S11/SOURCE")
s10_sha=$(head -n 1 "$S10/SOURCE")
[[ $(git -C "$TREE" rev-parse HEAD) == "$s11_sha" ]] || { echo "tree HEAD is not frozen stack11 $s11_sha" >&2; exit 6; }
[[ -z $(git -C "$TREE" diff --name-only "$s11_sha" -- include common ggml/include) ]] || { echo 'tree headers differ from the freeze' >&2; exit 6; }
rm -rf "$GEN"
mkdir -p "$GEN/bin" "$GEN/s10-src"
git -C "$S10_REPO" archive --format=tar "$s10_sha" include common ggml/include | tar -x -C "$GEN/s10-src"
dump() { # OUT HEADER_ROOT LIBDIR
    nice -n 19 taskset -c "$CPUS" c++ -std=c++17 -O2 -pthread \
        -I"$2/include" -I"$2/common" -I"$2/ggml/include" -I"$HERE/src" \
        "$HERE/src/kld_dump.cpp" -L"$3" -Wl,-rpath-link,"$3" \
        -lllama-common -lllama -lggml -lggml-base -o "$1"
    [[ -z $(env LD_LIBRARY_PATH="$3" ldd "$1" | grep 'not found' || true) ]]
}
dump "$GEN/bin/kld_dump-s11" "$TREE" "$S11/build/bin"
dump "$GEN/bin/kld_dump-s10" "$GEN/s10-src" "$S10/build/bin"
for model in j5q iq4; do
    python3 "$HERE/mkconfig.py" --gen "$GEN" --model "$model" --s10 "$S10" --s11 "$S11"
done
echo "kld_dump built: s11 against $s11_sha headers, s10 against $s10_sha headers; configs in $GEN"
