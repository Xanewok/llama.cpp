#!/usr/bin/env bash
# Diagnostic for a getter-gate failure on x86 (nmse 5e-4 > 1e-4, two-slot verify batch): is the by-id read wrong, or does the CPU
# compute batch-of-12 vs batch-of-6 differently? Usage: gate-diag.sh OUT   (reads cpu-gate.sh's CPU flags; never submits)
# Builds NEW (HEAD) and CTL (cpu-gate.sh's control: getter for one sequence only, i.e. the gather path), runs the gate test on each
# with the test patched to OBSERVE (log every logits nmse, no bound, structure checks non-fatal), writes OUT/diag.txt.
# CTL nmse > 1e-4 on the same phases => numeric batch-shape difference, not the getter. CTL <= 1e-4 < NEW => getter bug on x86.
# Variants per build: t4 = gate as run, t1 = one thread (thread partitioning), fa0 = -fa off (CPU flash-attn path), t4b = repeat of
# t4 on NEW (run-to-run determinism). Repack is already OFF in cpu-gate.sh's flags (-DGGML_CPU_REPACK=OFF, llamafile OFF), so that
# hypothesis is excluded by construction; to test repack ON: BW_CPU_EXTRA=-DGGML_CPU_REPACK=ON.
# rc: 0 all runs completed (judge diag.txt; test rc is data) | 8 setup/build failure.
set -uo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
cd "$ROOT" || exit 8
if [[ ${1:-} == --preflight ]]; then
    for p in "${BW_BUILDGATE:-$HOME/repos/qwen38-eval-worktrees/bin/buildgate}" "${BW_GATE:-$HOME/repos/qwen38-eval-worktrees/bin/cpugate}" \
             "${BW_MODEL:-/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf}" codex/np2/cpu-gate.sh; do
        [[ -e $p ]] || { echo "gate-diag: preflight missing $p"; exit 8; }
    done
    [[ -z $(git status --porcelain --untracked-files=no) ]] || { echo "gate-diag: preflight: tracked changes present"; exit 8; }
    exit 0
fi
OUT=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${1:?output}")
mkdir -p "$OUT"
W=$HOME/repos/qwen38-eval-worktrees
MODEL=${BW_MODEL:-/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf}
GATE=${BW_GATE:-$W/bin/cpugate}
BUILDGATE=${BW_BUILDGATE:-$W/bin/buildgate}
JOBS=${BW_JOBS:-3}
(( JOBS <= 3 )) || JOBS=3
SHA=$(git rev-parse HEAD)
TEST_SRC=tests/test-recurrent-state-rollback.cpp
GATE_SRC=src/models/delta-net-base.cpp
log() { echo "gate-diag: $(date '+%H:%M:%S') $*" | tee -a "$OUT/summary.txt"; }
die() { local rc=$1; shift; log "FAIL rc=$rc $*"; exit "$rc"; }
restore() { git restore --source=HEAD -- "$TEST_SRC" "$GATE_SRC"; }
trap restore EXIT
[[ -f $MODEL ]] || die 8 "model missing: $MODEL"
[[ -z $(git status --porcelain --untracked-files=no) ]] || die 8 "dirty tracked tree"

# Same flags as cpu-gate.sh by construction: take its array literally.
eval "$(awk '/^cpu_flags=\(/ {p=1} p {print} p && /\)$/ {exit}' codex/np2/cpu-gate.sh)"
(( ${#cpu_flags[@]} > 10 )) || die 8 "could not read cpu_flags from cpu-gate.sh"
# shellcheck disable=SC2206
[[ -z ${BW_CPU_EXTRA:-} ]] || cpu_flags+=($BW_CPU_EXTRA)
log "HEAD $SHA flags: ${cpu_flags[*]}"

build() { # NAME
    timeout 1100 "$BUILDGATE" cmake --build build-cpu --target test-recurrent-state-rollback -j"$JOBS" \
        > "$OUT/build-$1.log" 2>&1 || die 8 "build $1 (see $OUT/build-$1.log)"
}
timeout 300 "$BUILDGATE" cmake -S codex/stack7-test -B build-cpu "${cpu_flags[@]}" > "$OUT/configure.log" 2>&1 || die 8 "configure"

# Observation patch (both builds): log each logits nmse, lift the bound, make the by-id/gathered structure checks non-fatal.
sed -i \
    -e 's/^\( *\)if (err > nmse_eps) {$/\1LOG_INF("%s: DIAG %s: %s: seq %u pos %d nmse %g\\n", __func__, sc.name, ph.name, s, (int) pos, err);\n&/' \
    -e 's/^\( *constexpr double nmse_eps = \)1e-4;$/\11e30;/' \
    -e 's/if (ph\.expect == form::BY_ID && (/if (false \&\& ph.expect == form::BY_ID \&\& (/' \
    -e 's/if (ph\.expect == form::GATHERED && (/if (false \&\& ph.expect == form::GATHERED \&\& (/' \
    "$TEST_SRC"
git diff -- "$TEST_SRC" > "$OUT/observe.diff"
[[ $(grep -c '^+' "$OUT/observe.diff") -ge 5 ]] || die 8 "observation patch did not apply (see $OUT/observe.diff)"
grep -q 'nmse_eps = 1e30' "$TEST_SRC" && grep -q 'DIAG %s' "$TEST_SRC" || die 8 "observation patch incomplete"

: > "$OUT/runs.tsv"
RAN_EXTRA=()
run() { # BUILD VARIANT ARGS...   (timeout is data: rc 124 recorded)
    local b=$1 v=$2 name=$1-$2; shift 2
    local t0=$SECONDS rc=0
    timeout 600 "$GATE" env LD_LIBRARY_PATH="$ROOT/build-cpu/bin" \
        build-cpu/bin/test-recurrent-state-rollback --getter-gate -m "$MODEL" -c 4096 "$@" > "$OUT/$name.log" 2>&1 || rc=$?
    printf '%s\t%s\t%ss\n' "$name" "$rc" "$(( SECONDS - t0 ))" >> "$OUT/runs.tsv"
    log "$name rc=$rc ($(( SECONDS - t0 )) s)"
}
variants() { # BUILD   t4 always; extras only while the 30 min bound is safe (NEW decides, CTL mirrors)
    local b=$1
    run "$b" t4 -t 4
    if [[ $b == new ]]; then
        if (( SECONDS < 600 )); then
            RAN_EXTRA=(t1 fa0)
            run new t4b -t 4
            run new t1 -t 1
            run new fa0 -t 4 -fa off
        else
            log "extras skipped on new: $SECONDS s elapsed"
        fi
    else
        for v in "${RAN_EXTRA[@]}"; do
            (( SECONDS < 1500 )) || { log "ctl $v skipped: $SECONDS s elapsed"; continue; }
            case $v in
                t1) run ctl t1 -t 1 ;;
                fa0) run ctl fa0 -t 4 -fa off ;;
            esac
        done
    fi
}

build new; log "built new"
variants new

# control = cpu-gate.sh's stack7 gate: the getter only for one sequence, so a two-slot verify batch gathers
sed -i 's/mctx_cur->get_n_rs() == (uint32_t) n_seqs) {/n_seqs == 1) {/' "$GATE_SRC"
git diff --quiet -- "$GATE_SRC" && die 8 "control mutation did not change $GATE_SRC"
git diff -- "$GATE_SRC" > "$OUT/ctl.diff"
build ctl; log "built ctl"
variants ctl
restore
[[ -z $(git status --porcelain --untracked-files=no) && $(git rev-parse HEAD) == "$SHA" ]] || die 8 "tree not clean after restore"

python3 - "$OUT" > "$OUT/diag.txt" <<'PY'
import re, sys
from pathlib import Path
out = Path(sys.argv[1])
rx = re.compile(r'DIAG (.*?): (.*?): seq (\d+) pos (\d+) nmse (\S+)')
runs = {}
for line in (out / 'runs.tsv').read_text().splitlines():
    n, rc, t = line.split('\t'); runs[n] = rc
print(f'HEAD / bound 1e-4; run rc (test rc is data; 124 = timeout): {runs}\n')
data = {}
for n in runs:
    d = {}
    for l in (out / f'{n}.log').read_text(errors='replace').splitlines():
        m = rx.search(l)
        if m: d[(m[1], m[2], int(m[3]), int(m[4]))] = float(m[5])
    data[n] = d
print('== per run: n values, n > 1e-4, max nmse')
for n, d in data.items():
    v = list(d.values())
    print(f'{n:10s} n={len(v):3d} over={sum(x > 1e-4 for x in v):3d} max={max(v) if v else float("nan"):.3g}')
print('\n== max nmse per (scenario, phase): columns = runs')
names = list(data)
keys = list(dict.fromkeys(k[:2] for d in data.values() for k in d))
print(f'{"scenario / phase":58s}' + ''.join(f'{n:>11s}' for n in names))
for k in keys:
    row = ''
    for n in names:
        v = [x for kk, x in data[n].items() if kk[:2] == k]
        row += f'{max(v):11.2e}' if v else f'{"-":>11s}'
    print(f'{k[0] + " / " + k[1]:58s}' + row)
print('\n== determinism: new-t4 vs new-t4b identical values:', data.get('new-t4') == data.get('new-t4b') if 'new-t4b' in data else 'n/a')
print('\n== raw lines (nmse / differ / by id / result) per run')
for n in names:
    print(f'--- {n} rc={runs[n]}')
    for l in (out / f'{n}.log').read_text(errors='replace').splitlines():
        if re.search(r'DIAG|logits differ|by id|all scenarios|FAIL|error|ERR', l):
            print(l)
PY
log "done: $OUT/diag.txt"
exit 0
