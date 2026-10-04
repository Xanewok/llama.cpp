#!/usr/bin/env bash
set -euo pipefail
if [[ ${HOST_DEPTH_NICED:-} != 1 ]]; then
    export HOST_DEPTH_NICED=1
    exec nice -n 19 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
mkdir -p build-tmp
export CUDA_VISIBLE_DEVICES="" TMPDIR="$PWD/build-tmp" CCACHE_DISABLE=1
saved="build-tmp/llama-kv-cache-guard-$$.cpp"
cp src/llama-kv-cache.cpp "$saved"
restore() {
    cp "$saved" src/llama-kv-cache.cpp
    cmake --build build-cpu --target llama -j8 > build-cpu/restored-build.log 2>&1
    bash codex/build_fixture.sh
}
trap restore EXIT
python3 - <<'PY'
from pathlib import Path
path = Path("src/llama-kv-cache.cpp")
source = path.read_text()
guard = "cells.used_max_p1() == n_used && positions.size() == n_used"
assert source.count(guard) == 1
path.write_text(source.replace(guard, "cells.used_max_p1() == n_used"))
PY
cmake --build build-cpu --target llama -j8 > build-cpu/broken-build.log 2>&1
bash codex/build_fixture.sh
cp build-cpu/host-depth build-cpu/host-depth-broken
if build-cpu/host-depth-broken build-cpu/broken-masks.bin build-cpu/broken-logits.bin > build-cpu/broken-cpu.csv 2> build-cpu/broken-cpu.log; then
    echo "broken production guard passed" >&2
    exit 1
else
    test "$?" -eq 1
    test "$(cat build-cpu/broken-cpu.log)" = "negative control: sequence mutation retained stale mask"
fi
