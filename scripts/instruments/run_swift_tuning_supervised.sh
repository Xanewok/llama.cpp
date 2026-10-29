#!/usr/bin/env bash
# Persistent campaign wrapper; inference and safety logic stay in hillclimb.
set -u

campaign=${1:?campaign path required}
budget_hours=${2:-12}
repo=/home/xanewok/repos/qwen38-eval
case "$campaign" in
  /*) campaign_path=$campaign ;;
  *) campaign_path="$repo/$campaign" ;;
esac
runner_log="$campaign_path/runner.log"
restore_log="$campaign_path/restore.log"

cd "$repo" || exit 125
python3 -m scripts.instruments.hillclimb run \
  --campaign "$campaign_path" --budget-hours "$budget_hours" \
  >>"$runner_log" 2>&1
status=$?

# The selected interactive profile was stopped for this campaign. Restore it even
# when the runner stops on a capacity, thermal, or instrumentation gate.
restore_status=1
health=$(curl -s --max-time 3 http://127.0.0.1:8080/health 2>/dev/null || true)
if [[ -n "$health" && "$health" != *'"status":"ok"'* && "$health" != *'"status": "ok"'* ]]; then
  echo "restore skipped: port 8080 has a non-Qwen listener" >>"$restore_log"
else
  for delay in 0 5 15; do
    if (( delay > 0 )); then sleep "$delay"; fi
    if ~/.codex/skills/local-qwen-server/bin/serve.sh start \
        >>"$restore_log" 2>&1; then
      restore_status=0
      break
    fi
  done
fi
printf 'runner_exit=%s restore_exit=%s finished_epoch=%s\n' \
  "$status" "$restore_status" "$(date +%s)" >>"$restore_log"
exit "$status"
