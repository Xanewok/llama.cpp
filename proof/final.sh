#!/usr/bin/env bash
cd /home/xanewok/repos/swe-fresh
python3 stub_responses_server.py --port 18304 --out proof/stub-final-a --timings & A=$!
python3 stub_responses_server.py --port 18305 --out proof/stub-final-b & B=$!
sleep 1
python3 run_swe_fresh.py run --campaign proof/final-a-canary --tasks pallets__click-3239 --canary --upstream-port 18304 --engine stub --provider local --harbor-version 0.23.0 --dataset-path data/dl/swe-rebench-leaderboard --hard-timeout-s 1500 > proof/final-a.console.log 2>&1
python3 run_swe_fresh.py run --campaign proof/final-b-repeat2 --tasks pallets__click-3239 --repeat 2 --upstream-port 18305 --metrics-url http://127.0.0.1:18305 --engine stub --provider local --harbor-version 0.23.0 --dataset-path data/dl/swe-rebench-leaderboard --hard-timeout-s 1500 > proof/final-b.console.log 2>&1
kill $A $B
echo done > proof/final.done
