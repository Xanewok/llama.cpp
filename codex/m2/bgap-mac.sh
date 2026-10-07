#!/bin/bash
# Mac: run bgap.py on the four sqlite exports, one at a time, each after a bounded RAM wait (>= 4 GiB, <= 10 min).
cd ~/qwen38-chain-m2/bgap || exit 2
avail_gib() { vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} END{printf "%d", (f+i+s)*ps/1073741824}'; }
for x in np1-8k np1-90k np2-8k np2-90k; do
  t=0
  while [ "$(avail_gib)" -lt 4 ]; do
    [ $t -ge 600 ] && { echo "skipped: RAM ($x, $(avail_gib) GiB)"; exit 75; }
    sleep 15; t=$((t+15))
  done
  nice -n 19 python3 bgap.py . $x.sqlite 14 $x || echo "$x failed rc=$?"
done
