#!/usr/bin/env bash
# Destroy any qwen38 instance past its TTL, forever, in its own process.
#
# The only thing standing between an unattended run and an open-ended bill: every other
# script may die mid-flight and leave an instance billing, and this one only has to
# survive to make that finite.
#
# TTL comes from the label -- "qwen38-5070ti-ttl4" reaps after 4h -- so a box doing a
# 20-minute sweep is not given the same rope as one mid-experiment. Labels without a
# suffix fall back to TTL_HOURS, which must therefore be generous, not tight.
#
# DRY_RUN=1 prints what it would destroy and touches nothing.
set -uo pipefail
TTL_HOURS=${TTL_HOURS:-12}
PREFIX=${PREFIX:-qwen38-}
INTERVAL=${INTERVAL:-300}
DRY_RUN=${DRY_RUN:-0}

while true; do
  vastai show instances --raw 2>/dev/null | TTL_HOURS="$TTL_HOURS" PREFIX="$PREFIX" python3 -c "
import sys, json, os, re, time
now = time.time()
default_ttl = float(os.environ['TTL_HOURS']) * 3600
prefix = os.environ['PREFIX']
try:
    rows = json.load(sys.stdin)
except Exception:
    print('QUERY-FAILED: treating as nothing-to-reap', file=sys.stderr)
    rows = []          # a failed query must never be read as 'no instances'
for i in rows:
    label = i.get('label') or ''
    if not label.startswith(prefix):
        continue       # never touch an instance created for something else
    m = re.search(r'-ttl(\d+(?:\.\d+)?)\b', label)
    ttl = float(m.group(1)) * 3600 if m else default_ttl
    age = now - (i.get('start_date') or now)
    if age > ttl:
        print(f\"{i['id']} {label} {age/3600:.1f} {ttl/3600:.1f}\")
" | while read -r id label age ttl; do
    if [ "$DRY_RUN" = 1 ]; then
      echo "$(date '+%F %T') WOULD REAP $id ($label, age ${age}h > ttl ${ttl}h)"
    else
      echo "$(date '+%F %T') REAPING $id ($label, age ${age}h > ttl ${ttl}h)"
      vastai destroy instance "$id" -y </dev/null
    fi
  done
  [ "${ONCE:-0}" = 1 ] && break
  sleep "$INTERVAL"
done
