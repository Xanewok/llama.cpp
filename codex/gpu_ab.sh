#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/w2r4-general-probe/run.sh" "$@"
