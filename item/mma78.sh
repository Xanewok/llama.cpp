#!/usr/bin/env bash
set -euo pipefail
export GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
export TMPDIR=/mnt/scratch/tmp-build
ROOT=/home/xanewok/repos/qwen38-eval-worktrees/wt/mma78
GATE=/home/xanewok/repos/qwen38-eval-worktrees/bin/buildgate
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
BUILD="$ROOT/build-mma78"
OUT=${OUT:?gpuq must set OUT}
mkdir -p "$OUT"

identity() {
    if [[ $(git -C "$ROOT" branch --show-current) != topic/mma78 ]] || ! git -C "$ROOT" diff --quiet HEAD; then
        echo 'mma78 integrity: require clean tracked sources on topic/mma78; restore the queued source pin' >&2
        exit 6
    fi
}

case "${1:-}" in
    --preflight)
        export CUDA_VISIBLE_DEVICES=''
        identity
        git -C "$ROOT" rev-parse HEAD > "$OUT/source-tip.txt"
        for path in "$GATE" "$CCCL/CMakeLists.txt" /usr/local/cuda-13.2/bin/nvcc; do
            if [[ ! -f "$path" ]]; then
                echo "mma78 integrity: missing $path; restore the local build dependency" >&2
                exit 6
            fi
        done
        if [[ -e "$BUILD" ]]; then
            echo 'mma78 integrity: build-mma78 already exists; move it aside for a fresh ptxas log' >&2
            exit 6
        fi
        mkdir -p "$TMPDIR"
        nice -n 19 python3 "$ROOT/bench.py" --prepare --out "$OUT/metadata"
        "$GATE" cmake -S "$ROOT" -B "$BUILD" \
            -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
            -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF \
            -DCMAKE_CUDA_COMPILER_FORCED=ON \
            -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
            -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DCMAKE_CUDA_ARCHITECTURES=120a \
            -DCMAKE_CUDA_FLAGS=--ptxas-options=-v \
            -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
            -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
            -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL" \
            -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON \
            > "$OUT/configure.log" 2>&1
        "$GATE" cmake --build "$BUILD" -j4 --target test-backend-ops > "$OUT/build.log" 2>&1
        nice -n 19 /usr/local/cuda-13.2/bin/cuobjdump --dump-sass "$BUILD/bin/libggml-cuda.so" > "$OUT/sass.txt"
        nice -n 19 python3 "$ROOT/item/check_mma78_build.py" "$OUT/build.log" "$OUT/sass.txt" > "$OUT/registers.txt"
        ;;
    --canary|--run)
        identity
        if [[ ${GPUQ_ACTIVE:-} != 1 || ${CUDA_VISIBLE_DEVICES:-} != 0 ]]; then
            echo 'mma78 integrity: run through gpuq with GPUQ_ACTIVE=1 and CUDA_VISIBLE_DEVICES=0' >&2
            exit 6
        fi
        if [[ ! -s "$OUT/registers.txt" || ! -s "$OUT/source-tip.txt" ||
                $(git -C "$ROOT" rev-parse HEAD) != "$(cat "$OUT/source-tip.txt")" ]]; then
            echo 'mma78 integrity: missing build gate or changed source; rerun fresh preflight' >&2
            exit 6
        fi
        export LD_LIBRARY_PATH="$BUILD/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        set +e
        if [[ $1 == --canary ]]; then
            timeout --signal=TERM --kill-after=10s 350s nice -n 19 \
                python3 "$ROOT/bench.py" --correctness --arms default mma \
                --binary "$BUILD/bin/test-backend-ops" --out "$OUT/correctness"
        else
            if [[ ! -s "$OUT/correctness/correctness.json" ]]; then
                echo 'mma78 integrity: missing correctness result; run the canary before perf' >&2
                exit 6
            fi
            timeout --signal=TERM --kill-after=10s 1430s nice -n 19 \
                python3 "$ROOT/bench.py" --run --arms default mma --repeats 4 \
                --binary "$BUILD/bin/test-backend-ops" --out "$OUT/perf"
        fi
        rc=$?
        if [[ $rc == 124 || $rc == 137 ]]; then exit 7; fi
        exit "$rc"
        ;;
    *)
        echo 'usage: mma78.sh --preflight|--canary|--run' >&2
        exit 8
        ;;
esac
