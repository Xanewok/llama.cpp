#!/usr/bin/env bash
set -euo pipefail
[[ ${BW_CPU_ONLY:-} == 1 ]]
out=$1
export BW_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 - <<'GATE'
import re, subprocess, time
for attempt in range(120):
    stat = subprocess.check_output(['vm_stat'], text=True)
    page = int(re.search(r'page size of (\d+) bytes', stat)[1])
    free = sum(int(re.search(r'Pages '+key+r':\s+(\d+)', stat)[1]) for key in ['free', 'inactive', 'speculative']) * page
    processes = subprocess.check_output(['ps', '-axo', 'comm='], text=True).splitlines()
    busy = any(p.rsplit('/', 1)[-1] in {'clang', 'clang++', 'cc1', 'ninja', 'xcodebuild', 'llama-server', 'llama-cli'} for p in processes)
    if free >= 4*1024**3 and not busy:
        print(f'RAM gate PASS: {free} bytes available; one build job', flush=True)
        break
    if attempt % 12 == 0:
        print(f'waiting: available={free}, other build/model={busy}', flush=True)
    time.sleep(5)
else:
    print('skipped: RAM/shared builder busy after 10 minutes', flush=True)
    raise SystemExit(75)
GATE
shared="$HOME/qwen38-bw/s12b-drop-incremental"
mkdir -p "$shared"
mkdir "$shared/lock" || exit 75
trap 'rmdir "$shared/lock"' EXIT
mkdir -p "$shared/source"
rsync -rc --delete --exclude '.git' --exclude 'build*' ./ "$shared/source/"
printf '%s\n' "$PWD" > "$out/source-snapshot.txt"
cd "$shared/source"
read -ra extra <<< "${BW_CPU_EXTRA:-}"
"$BW_BUILDGATE" cmake -S . -B build-stack12b-cpu \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_OPENMP=OFF \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF \
    -DGGML_CCACHE=OFF "${extra[@]}" > "$out/configure.log" 2>&1
targets=(llama-server)
rc=0
for target in "${targets[@]}"; do
    if "$BW_BUILDGATE" cmake --build build-stack12b-cpu -j1 --target "$target" > "$out/$target.log" 2>&1; then
        printf '%s\t0\n' "$target" >> "$out/targets.tsv"
    else
        code=$?
        printf '%s\t%s\n' "$target" "$code" >> "$out/targets.tsv"
        rc=8
    fi
done
if [[ $rc == 0 ]]; then
    cp build-stack12b-cpu/bin/llama-server build-stack12b-cpu/bin/round-trace
    cmp build-stack12b-cpu/bin/llama-server build-stack12b-cpu/bin/round-trace
    printf 'round-trace\t0\n' >> "$out/targets.tsv"
fi
cat "$out/targets.tsv"
exit "$rc"
