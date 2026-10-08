#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
export PATH="$HOME/qwen38-tools/bin:/opt/homebrew/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4
run=$(cd "${1:?run directory}" && pwd)
mkdir -p "$run/logs"

gate() {
    local gate_root=$run
    [[ -d $gate_root/validation ]] || gate_root=$(dirname "$run")
    bash "$gate_root/validation/gate.sh" "$@"
}

cd "$run"
pkg=${2:?package slug}
if [[ ${PACK_TEST_ONLY:-0} != 1 ]]; then
[[ -d src/.git ]] || {
    git init src
    git -C src config gc.auto 0
    git -C src config maintenance.auto false
    git -C src config user.name 'Igor Matuszewski'
    git -C src config user.email 'xanewok@gmail.com'
}
base=7049ff0cbeb1f5ead231de4522af6b75d8d773c0
if ! git -C src cat-file -e "$base^{commit}" 2>/dev/null; then
    printf '%s\n' "$base" >> src/.git/shallow
    git -C src fetch "$run/base.bundle" refs/heads/base
fi
pkg=${2:?package slug}
cd src
git checkout --detach "$base"
if [[ -d "$run/patches/upstream/llama.cpp/$pkg/prerequisites" ]]; then
    git am "$run/patches/upstream/llama.cpp/$pkg/prerequisites/"*.patch > "$run/logs/$pkg-prerequisites.log" 2>&1
fi
git am "$run/patches/upstream/llama.cpp/$pkg/"*.patch > "$run/logs/$pkg-am.log" 2>&1
git rev-parse HEAD > "$run/logs/$pkg-tip.txt"
gate cmake -S . -B "$run/build-cpu" -G Ninja -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF \
    -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod -DGGML_CCACHE=ON \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DFETCHCONTENT_UPDATES_DISCONNECTED=ON > "$run/logs/$pkg-configure.log" 2>&1
gate cmake --build "$run/build-cpu" -j4 --target test-llama-archs test-sampling test-alloc \
    > "$run/logs/$pkg-build.log" 2>&1
printf 'PASS\n' > "$run/logs/$pkg-build-result"
[[ ${PACK_BUILD_ONLY:-0} != 1 ]] || exit 0
else
    run=$(cd "${1:?run directory}" && pwd)
    pkg=${2:?package slug}
    cd "$run/src"
    [[ -f "$run/logs/$pkg-build-result" ]]
fi
gate "$run/build-cpu/bin/test-sampling" > "$run/logs/$pkg-sampling.log" 2>&1
case "$pkg" in
    68-*)
        gate "$run/build-cpu/bin/test-llama-archs" --tp-sampling -s 42 > "$run/logs/$pkg-tp42.log" 2>&1
        gate "$run/build-cpu/bin/test-llama-archs" --tp-sampling -s 1729 > "$run/logs/$pkg-tp1729.log" 2>&1
        cp common/sampling.cpp "$run/sampling-positive.cpp"
        python3 - <<'PY'
from pathlib import Path
p = Path('common/sampling.cpp')
t = p.read_text()
old = 'candidate_k + 1 + n_ban'
assert t.count(old) == 1
p.write_text(t.replace(old, 'candidate_k + 1'))
PY
        gate cmake --build "$run/build-cpu" -j4 --target test-llama-archs > "$run/logs/$pkg-mutant-build.log" 2>&1
        set +e
        gate "$run/build-cpu/bin/test-llama-archs" --tp-sampling -s 42 > "$run/logs/$pkg-mutant.log" 2>&1
        rc=$?
        set -e
        printf '%s\n' "$rc" > "$run/logs/$pkg-mutant.rc"
        cp "$run/sampling-positive.cpp" common/sampling.cpp
        [[ $rc -ne 0 ]] || { echo 'negative control escaped'; exit 1; }
        gate cmake --build "$run/build-cpu" -j4 --target test-llama-archs > "$run/logs/$pkg-restored-build.log" 2>&1
        ;;
    69-*) gate "$run/build-cpu/bin/test-llama-archs" -s 42 > "$run/logs/$pkg-archs.log" 2>&1 ;;
    70-*) gate "$run/build-cpu/bin/test-llama-archs" --seq-rm > "$run/logs/$pkg-seq-rm.log" 2>&1 ;;
    73-*) gate "$run/build-cpu/bin/test-alloc" > "$run/logs/$pkg-alloc.log" 2>&1 ;;
esac
git diff --exit-code > "$run/logs/$pkg-clean.log"
printf 'PASS\n' > "$run/logs/$pkg-result"
