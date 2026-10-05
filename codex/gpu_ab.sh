#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/draft-attn-cause-probe/run.sh" "$@"
