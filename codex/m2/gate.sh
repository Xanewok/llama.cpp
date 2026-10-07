#!/usr/bin/env bash
# Mac model gate: waits <= 10 min for >= 4 GiB available and no other fixture model process, then runs "$@" at nice 19.
# Exits 75 with a "skipped: RAM" line when the wait runs out.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
avail_gib() {
    vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} END{printf "%d", (f+i+s)*ps/1073741824}'
}
busy() {
    ps -axo comm | awk -F/ '{print $NF}' | grep -qxE 'bw-np2-driver|chain-cpu|fold-cpu|outproj-cpu|llama-server|llama-cli'
}
t=0
while [[ $(avail_gib) -lt 4 ]] || busy; do
    if [[ $t -ge 600 ]]; then
        echo "skipped: RAM ($(avail_gib) GiB available, other model process: $(busy && echo yes || echo no)) for: $*"
        exit 75
    fi
    sleep 15
    t=$((t + 15))
done
exec nice -n 19 "$@"
