#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1
while python3 - <<'PY'
import json, sys
s=json.load(open('/home/xanewok/.local/state/gpuq/status.json'))
r=[i['id'] for i in s['items'] if i['state']=='running']
sys.exit(0 if any('depth' in i or 'perf' in i for i in r) else 1)
PY
do
    sleep 30
done
exec /home/xanewok/repos/qwen38-eval-worktrees/bin/buildgate "$@"
