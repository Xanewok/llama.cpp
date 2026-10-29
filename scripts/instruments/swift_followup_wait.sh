#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/xanewok/repos/qwen38-eval
MODEL_DIR=$ROOT/.cache/hillclimb-models/swift-iq4-xs
PARTIAL=$MODEL_DIR/Swift-Qwen3.8-27B-IQ4_XS.gguf.partial
MODEL=$MODEL_DIR/Swift-Qwen3.8-27B-IQ4_XS.gguf
CAMPAIGN=$ROOT/results/2026-09-12-mixed-pair/campaign-swift-iq4-xs-longtail-20260914
ASSETS=$ROOT/results/2026-09-12-mixed-pair/campaign-confirmed-fullsuite-20260914-r8/snapshot/agentic-assets

mkdir -p "$MODEL_DIR"
while [[ ! -f "$MODEL" ]]; do
    if systemctl --user is-failed --quiet swift-q4-m-download.service; then
        echo "Swift download service failed" >&2
        exit 1
    fi
    if [[ -f "$PARTIAL" && "$(stat -c %s "$PARTIAL")" -eq 15688288352 ]]; then
        # X-Linked-Etag for IQ4_XS at pinned HF commit b12fc8f44af7bdf31a3c3309891acb345d3106a2.
        expected=27f5531a012a4bb3c3e379318dc62216f057131d34347498fabfc23186a4ffda
        actual=$(sha256sum "$PARTIAL" | awk '{print $1}')
        [[ "$actual" == "$expected" ]] || { echo "Swift checksum mismatch" >&2; exit 1; }
        mv "$PARTIAL" "$MODEL"
        break
    fi
    sleep 30
done

exec env TMPDIR=$ROOT/.cache/benchmark-tmp PYTHONPATH=$ROOT \
    python3 -m scripts.instruments.swift_longtail \
    --campaign "$CAMPAIGN" --assets "$ASSETS" \
    --deadline-epoch "$(($(date +%s) + 43200))"
