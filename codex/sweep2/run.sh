#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
if [[ $(ps -o ni= -p $$ | tr -d " ") != 19 ]]; then exec nice -n 19 bash "$0" "$@"; fi
ROOT=/home/xanewok/repos/qwen38-eval/results/2026-10-05-mmvq-sweep2
MODE=${1:?usage: run.sh --preflight|--canary|--run}
# Local rehearsal must not redirect a GPU run away from the pinned package.
if [[ -n ${SWEEP2_PREFLIGHT_ROOT:-} ]]; then
  [[ $MODE == --preflight && ${CUDA_VISIBLE_DEVICES-unset} == "" ]] || exit 6
  ROOT=$SWEEP2_PREFLIGHT_ROOT
fi
FROZEN=$ROOT/frozen
BIN=$FROZEN/bench-mmvq-shapes
TOOLS=$ROOT/bench_tools.py
OUT=${OUT:?OUT required}
BUDGET_S=${BUDGET_S:-900}
[[ $BUDGET_S =~ ^[0-9]+$ && $BUDGET_S -ge 60 && $BUDGET_S -le 900 ]] || exit 6
START=$(date +%s)
RAW=$OUT/raw
mkdir -p "$RAW"
export TMPDIR="$OUT/tmp"
mkdir -p "$TMPDIR"
die() { local rc=$1; shift; echo "run.sh: $*" >&2; exit "$rc"; }
event() { if [[ -n ${GPUQ_EVENTS:-} ]]; then echo "$*" >>"$GPUQ_EVENTS"; fi; }
remaining() { echo $(( BUDGET_S - ($(date +%s) - START) )); }
cap() { local r; r=$(remaining); echo $(( r - 30 < $1 ? r - 30 : $1 )); }
(cd "$FROZEN" && sha256sum -c --quiet SHA256SUMS) || die 6 "frozen bytes changed"
# SOURCE.sha256 pins the source identity independently of the payload manifest.
(cd "$FROZEN" && sha256sum -c --quiet SOURCE.sha256) || die 6 "frozen bytes changed: SOURCE"
[[ -z $(ldd "$BIN" | rg "not found" || true) ]] || die 6 "unresolved libraries"
stray=$(ldd "$BIN" | awk "/libggml/ {print \$3}" | rg -v "^$FROZEN/" || true)
[[ -z $stray ]] || die 6 "ggml libs resolve outside frozen dir: $stray"
python3 "$TOOLS" selftest
if [[ $MODE == --preflight ]]; then echo "preflight ok"; exit 0; fi
[[ $MODE == --canary || $MODE == --run ]] || die 8 "unknown mode $MODE"
shopt -s nullglob
old_rows=("$RAW"/*.jsonl)
[[ ${#old_rows[@]} -eq 0 && ! -e $OUT/results.md && ! -e $OUT/summary.json ]] || die 6 "output directory already contains measurement data"
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
# The queue handles the memory idle gate and also monitors kernel hazards while a process runs.
hazard_gate() {
  journalctl -k --since "@$START" --no-pager >"$OUT/kernel-since-start.txt" 2>"$OUT/kernel-read.err" || die 6 "kernel hazard journal unavailable"
  python3 "$TOOLS" hazard "$OUT/kernel-since-start.txt" || exit $?
}
hazard_gate
nvidia-smi --query-gpu=index,name,driver_version,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,power.limit,pcie.link.gen.current,temperature.gpu --format=csv >"$OUT/env.txt" 2>"$OUT/device.err" || die 4 "no CUDA device"
python3 "$TOOLS" device "$OUT/env.txt" || exit $?
cat "$FROZEN/SOURCE" >>"$OUT/env.txt"
run_bench() {
  local tag=$1 limit=$2 rc=0; shift 2
  [[ $limit -gt 5 ]] || die 7 "budget exhausted"
  python3 "$TOOLS" execute --cap "$limit" --prefix "$RAW/$tag" -- "$BIN" "$@" || rc=$?
  hazard_gate
  case $rc in
    0) ;;
    7) die 7 "$tag hit its cap (budget)" ;;
    *)
      if rg -q "baseline disagrees with CPU dequantized dot|variant disagrees with baseline" "$RAW/$tag.err"; then
        die 6 "$tag correctness failure"
      fi
      if rg -q "no CUDA-capable device|CUDA error.*no device" "$RAW/$tag.err"; then die 4 "no CUDA device"; fi
      tail -5 "$RAW/$tag.err" >&2
      die 8 "$tag bench binary failed: rc $rc"
      ;;
  esac
}
negative_control() {
  local rc=0
  python3 "$TOOLS" execute --cap 60 --prefix "$RAW/negative" -- "$BIN" --variant w2r6 --type q5_K --shape qkv --ncols 6 --repeats 1 --trials 1 --warmup-ms 0 --negative-control || rc=$?
  hazard_gate
  [[ $rc -eq 1 ]] && rg -q "variant disagrees with baseline" "$RAW/negative.err" || die 6 "negative control did not fail at the comparison gate (rc $rc)"
  [[ ! -s $RAW/negative.jsonl ]] || die 6 "negative control emitted a performance row"
}
parse() {
  local expect=$1; shift
  python3 "$TOOLS" parse "$@" --expect "$expect" --md "$OUT/results.md" --json "$OUT/summary.json"
}
negative_control
event measure_start
if [[ $MODE == --canary ]]; then
  # One positive boot tests every type, variant and column on the smallest common shape.
  run_bench r1-canary "$(cap 160)" --type all --shape qkv --ncols all --variant all --repeats 5 --trials 1 --warmup-ms 0
  parse canary "$RAW/r1-canary.jsonl"
  python3 "$TOOLS" select --canary "$RAW/r1-canary.jsonl" >"$OUT/best-two.txt"
  event "summary canary ok"
  exit 0
fi
declare -A DUR
timed() {
  local tag=$1 t0; t0=$(date +%s)
  run_bench "$@"
  DUR[$tag]=$(( $(date +%s) - t0 ))
}
for t in q6_K q5_K iq4_xs q4_K q8_0; do
  timed "r1-n6-$t" "$(cap 240)" --type "$t" --ncols 6 --variant all --repeats 100 --trials 7 --warmup-ms 1000
done
python3 "$TOOLS" select "$RAW"/r1-n6-*.jsonl >"$OUT/best-two.txt"
for t in q6_K q5_K iq4_xs q4_K q8_0; do
  timed "r1-wide-$t" "$(cap 300)" --type "$t" --ncols 5,7,8 --variant all --repeats 100 --trials 7 --warmup-ms 300
done
# Use the larger all-variant boot as the conservative measured estimate for each optional boot.
while read -r t variants; do
  estimate=${DUR[r1-wide-$t]}
  [[ $(remaining) -gt $(( estimate + 60 )) ]] || { echo "budget: skipped optional $t"; continue; }
  timed "r3-small-$t" "$(cap "$((estimate + 30))")" --type "$t" --ncols 1,2,4 --variant "$variants" --repeats 100 --trials 7 --warmup-ms 300
done <"$OUT/best-two.txt"
shopt -s nullglob
parse full "$RAW"/r1-*.jsonl "$RAW"/r3-*.jsonl
hazard_gate
event "summary full ok"
echo "done in $(( $(date +%s) - START )) s (M)"
