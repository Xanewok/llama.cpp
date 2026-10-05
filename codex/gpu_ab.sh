#!/usr/bin/env bash
set -euo pipefail
exec nice -n 19 bash "$(dirname "$0")/draft-attn-probe/run.sh" "$@"
