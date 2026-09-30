#!/usr/bin/env bash
# Real-traffic A/B: baseline S0 vs subset S3/S4/S5, three interleaved passes.
# Bytes are frozen and checked on every boot. Only the parent may set DRY_RUN=0.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PHASE1_ONLY=1 DRY_RUN=${DRY_RUN:-1}
exec bash "$ROOT/run_weekend.sh"
