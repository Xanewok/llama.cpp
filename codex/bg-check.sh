#!/usr/bin/env bash
# bw-mac payload: CPU llama-server, tool grammar + reasoning budget on the backend candidate path vs the host path.
set -euo pipefail
OUT=$1
export CUDA_VISIBLE_DEVICES=""
read -r -a extra <<< "${BW_CPU_EXTRA:-}"
B=build-bg
"${BW_BUILDGATE:-nice}" cmake -S . -B "$B" -G "${CMAKE_GENERATOR:-Ninja}" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF \
    -DGGML_CCACHE="${BW_CCACHE:-OFF}" -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF \
    "${extra[@]}" > "$OUT/cmake.log" 2>&1
"${BW_BUILDGATE:-nice}" cmake --build "$B" --target llama-server -j3 > "$OUT/build.log" 2>&1

free_gib() {
    if command -v vm_stat > /dev/null; then
        vm_stat | awk -v ps="$(sysctl -n hw.pagesize)" '/Pages free|Pages inactive|Pages speculative/ {gsub(/\./, "", $NF); n += $NF} END {print int(n * ps / 2^30)}'
    else
        awk '/MemAvailable/ {print int($2 / 2^20)}' /proc/meminfo
    fi
}
for _ in $(seq 120); do
    others=$(ps -axo comm | awk '/llama-server|llama-cli|test-llama/ {n++} END {print n + 0}')
    [[ $(free_gib) -ge 4 && $others == 0 ]] && break
    sleep 30
done
[[ $(free_gib) -ge 4 ]] || { echo "less than 4 GiB free" >&2; exit 125; }

python3 codex/bg_server_check.py "$B/bin/llama-server" "${BW_MODEL:?}" "$OUT/server" | tee "$OUT/check.log"
