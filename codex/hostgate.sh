#!/usr/bin/env bash
# S7_GATE / S7_BUILDGATE for a host without gpuq, /proc/meminfo or the shared buildgate (the Mac mini): no queue or RAM gate, just nice.
exec nice -n 19 "$@"
