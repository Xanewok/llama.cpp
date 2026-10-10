#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 CCACHE_READONLY=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../.." && pwd)
WT=$(cd "$TREE/../.." && pwd)
cd "$TREE"
gate() {
    nice -n 19 taskset -c 6-11,18-23 "$WT/bin/cpugate" "$@"
}
for name in cpu stack9-p55 stack10; do
    build="$TREE/build-history-$name"
    mkdir -p "$build"
    export CCACHE_TEMPDIR="$build" TMPDIR="$build"
    case "$name" in
        cpu) frozen="$TREE/build-cpu/bin"; commit=6473647b7 ;;
        stack9-p55) frozen="$WT/frozen/$name/build/bin"; commit=45d90d3e4ec1b713bef3804bac5bfc93cbff94c1 ;;
        stack10) frozen="$WT/frozen/$name/build/bin"; commit=a588f2a44382a361df3e2a456be0c80ca10dd1d6 ;;
    esac
    gate "$WT/bin/buildgate" cmake -S "$HERE" -B "$build" -DENGINE="$TREE" -DCOMMIT="$commit" -DFROZEN="$frozen" \
        -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DCMAKE_BUILD_TYPE=Release > "$HERE/evidence/configure-$name.log" 2>&1
    targets=(history-root)
    if [[ $name == cpu ]]; then targets+=(history-fingerprint-test); fi
    gate "$WT/bin/buildgate" cmake --build "$build" --target "${targets[@]}" -j4 > "$HERE/evidence/build-$name.log" 2>&1
    readelf -d "$build/history-root" > "$HERE/evidence/readelf-$name.txt"
done
gate "$TREE/build-history-cpu/history-fingerprint-test" > "$HERE/evidence/fingerprint-test.log" 2>&1
gate python3 "$HERE/test_history.py" > "$HERE/evidence/python-tests-final.log" 2>&1
gate python3 "$HERE/cpu_check.py" --binary "$TREE/build-history-cpu/history-root" --out "$TREE/build-history-cpu/cpu-model" \
    > "$HERE/evidence/cpu-model-check.log" 2>&1
cp "$TREE/build-history-cpu/cpu-model/SUMMARY.json" "$HERE/evidence/cpu-model-summary.json"
gate python3 "$HERE/render.py"
