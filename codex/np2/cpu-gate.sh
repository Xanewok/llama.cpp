#!/usr/bin/env bash
# CPU gate for the multi-sequence GDN getter. Usage: cpu-gate.sh OUT (bin/bw-mac payload; also runs on Linux).
# rc: 0 pass, 8 setup/build, 21 new build fails its own gate, 22 a mutant escaped, 23 restore failed,
#     24 np2 driver output differs from the stack7 gate (control), 75 skipped: RAM (bounded wait, never a pass).
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
OUT=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${1:?output}")
mkdir -p "$OUT"
W=$HOME/repos/qwen38-eval-worktrees
MODEL=${BW_MODEL:-/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf}
GATE=${BW_GATE:-$W/bin/cpugate}
BUILDGATE=${BW_BUILDGATE:-$W/bin/buildgate}
JOBS=${BW_JOBS:-3}
(( JOBS <= 3 )) || JOBS=3
SHA=$(git rev-parse HEAD)
GATE_SRC=src/models/delta-net-base.cpp
RS_SRC=src/llama-graph.cpp
OPS_SRC=ggml/src/ggml-cpu/ops.cpp
log() { echo "np2-gate: $*" | tee -a "$OUT/summary.txt"; }
die() { local rc=$1; shift; log "FAIL rc=$rc $*"; exit "$rc"; }
restore() { git restore --source=HEAD -- "$GATE_SRC" "$RS_SRC" "$OPS_SRC"; }
trap restore EXIT
[[ -f $MODEL ]] || die 8 "model missing: $MODEL"
[[ -z $(git status --porcelain --untracked-files=no) ]] || die 8 "dirty tracked tree"

avail_bytes() {
    if [[ $(uname) == Darwin ]]; then
        vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|inactive|speculative):/ {gsub("\\.","",$NF); n+=$NF} END {print n*ps}'
    else
        awk '/^MemAvailable:/ {print $2*1024}' /proc/meminfo
    fi
}
# One model job at a time; wait at most 10 min for 4 GiB, then stop with rc 75.
ram_wait() {
    local t0=$SECONDS
    until (( $(avail_bytes) >= 4*1024*1024*1024 )); do
        (( SECONDS - t0 < 600 )) || { log "skipped: RAM before $1 ($(( $(avail_bytes) >> 20 )) MiB free after 10 min)"; exit 75; }
        sleep 20
    done
}

cpu_flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE="${BW_CCACHE:-ON}" -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON)
[[ $(uname) != Darwin ]] || cpu_flags+=(-DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF)
# shellcheck disable=SC2206
[[ -z ${BW_CPU_EXTRA:-} ]] || cpu_flags+=($BW_CPU_EXTRA)
build() { # NAME
    "$BUILDGATE" cmake --build build-cpu --target test-recurrent-state-rollback bw-np2-driver -j"$JOBS" \
        > "$OUT/build-$1.log" 2>&1 || die 8 "build $1 (see $OUT/build-$1.log)"
}
"$BUILDGATE" cmake -S codex/stack7-test -B build-cpu "${cpu_flags[@]}" > "$OUT/configure.log" 2>&1 || die 8 "configure"
build new
log "built $SHA"

RC=0
run() { # NAME CMD...
    local name=$1; shift
    ram_wait "$name"
    RC=0
    "$GATE" env LD_LIBRARY_PATH="$ROOT/build-cpu/bin" "$@" > "$OUT/$name.log" 2>&1 || RC=$?
    printf '%s\t%s\n' "$name" "$RC" >> "$OUT/runs.tsv"
}
gate_test() { run "$1" build-cpu/bin/test-recurrent-state-rollback --getter-gate -m "$MODEL" -c 4096 -t 4; }
# serving np2 shape: two slots, MTP, chain/fold off, q8 KV, staged inputs
driver() { # NAME META
    run "$1" env NP2_DRIVER_THREADS=$(( $2 ? 2 : 4 )) LLAMA_MTP_CHAIN=0 LLAMA_MTP_FOLD_C=0 GGML_SCHED_INPUT_STAGING=1 HOST_KV_Q8=1 \
        build-cpu/bin/bw-np2-driver "$MODEL" "$OUT/$1.json" "$2" 2 1729 64
}
mutate() { # FILE SED_EXPR NAME
    sed "$2" "$1" > "$1.np2tmp" && cat "$1.np2tmp" > "$1" && rm -f "$1.np2tmp"
    git diff --quiet -- "$1" && die 8 "mutant $3 did not change $1"
    git diff -- "$1" > "$OUT/$3.diff"
    build "$3"
}
expect_fail() { # NAME REASON ENGAGED_REGEX
    grep -E "$3" "$OUT/$1.log" >/dev/null || die 22 "mutant $1 not engaged (no /$3/ in $OUT/$1.log)"
    [[ $RC != 0 ]] && grep -E "$2" "$OUT/$1.log" >/dev/null || die 22 "mutant $1 escaped (rc=$RC, expected /$2/)"
    log "mutant $1 caught (rc=$RC)"
}
MULTI_BY_ID='two slots: verify batch, both seqs: gdn nodes [1-9][0-9]* by id, 0 gathered; state rows gathered 0'

# 1. new build: the gate scenarios, the full rollback suite, and np2 driver outputs (plain + Meta)
gate_test gate-new
[[ $RC == 0 ]] && grep -E "$MULTI_BY_ID" "$OUT/gate-new.log" >/dev/null || die 21 "gate test on the new build (see $OUT/gate-new.log)"
run full-new build-cpu/bin/test-recurrent-state-rollback -m "$MODEL" -c 4096 -t 4
[[ $RC == 0 ]] || die 21 "full rollback suite on the new build (see $OUT/full-new.log)"
driver drv-new-plain 0; [[ $RC == 0 ]] || die 21 "driver plain (see $OUT/drv-new-plain.log)"
driver drv-new-meta 1;  [[ $RC == 0 ]] || die 21 "driver Meta (see $OUT/drv-new-meta.log)"
log "new build: gate scenarios, full suite and np2 driver pass"

# 2. control = the stack7 gate (getter for one sequence only): the gate test must flag it, the np2 driver must agree
mutate "$GATE_SRC" 's/mctx_cur->get_n_rs() == (uint32_t) n_seqs) {/n_seqs == 1) {/' ctl
gate_test ctl
expect_fail ctl 'ubatches without extra states must read the state by id' 'two slots: verify batch, both seqs: gdn nodes 0 by id, [1-9][0-9]* gathered'
driver drv-ctl-plain 0; [[ $RC == 0 ]] || die 24 "control driver plain (see $OUT/drv-ctl-plain.log)"
driver drv-ctl-meta 1;  [[ $RC == 0 ]] || die 24 "control driver Meta (see $OUT/drv-ctl-meta.log)"
python3 - "$OUT" <<'PY' || die 24 "np2 driver output differs between the getter and the gather"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
for meta in ('plain', 'meta'):
    new = json.loads((p / f'drv-new-{meta}.json').read_text())
    ctl = json.loads((p / f'drv-ctl-{meta}.json').read_text())
    assert new == ctl, f'{meta}: tokens or draft counters differ'
    n = sum(len(row) for ph in new['phases'] for row in ph['tokens'])
    print(f'np2 identity {meta}: {n} tokens and all draft counters equal')
PY
restore

# 3. mutants on the new build
mutate "$GATE_SRC" 's/mctx_cur->get_n_rs() == (uint32_t) n_seqs) {/true) {/' mut-extras
gate_test mut-extras
expect_fail mut-extras 'ubatches with extra states must keep the gathered state' 'extra state: verify batch, seqs 0 and 1: gdn nodes [1-9][0-9]* by id'
restore

mutate "$RS_SRC" 's/? inp->s_copy_none : inp->s_copy_tail;/? inp->s_copy_tail : inp->s_copy_tail;/' mut-tail
gate_test mut-tail
expect_fail mut-tail 'must read the state by id and gather none' 'two slots: verify batch, both seqs: gdn nodes [1-9][0-9]* by id, 0 gathered; state rows gathered [1-9]'
restore

# every sequence reads sequence 0's row: invisible at one sequence, so a catch also shows the np2 getter ran
mutate "$OPS_SRC" 's/((const int32_t \*) dst->src\[6\]->data)\[iv3\]/((const int32_t *) dst->src[6]->data)[0]/' mut-row
gate_test mut-row
expect_fail mut-row 'logits differ from the single-sequence context' 'two slots: verify batch, both seqs: gdn nodes [1-9][0-9]* by id'
driver drv-mut-row 0
if [[ $RC == 0 ]] && python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])) != json.load(open(sys.argv[2])) else 1)' \
        "$OUT/drv-mut-row.json" "$OUT/drv-ctl-plain.json"; then
    log "mutant drv-mut-row caught by the np2 driver comparison"
elif [[ $RC != 0 ]]; then
    log "mutant drv-mut-row caught (driver rc=$RC)"
else
    die 22 "mutant drv-mut-row escaped the np2 driver comparison"
fi
restore

# 4. restore and recheck
build restore
[[ -z $(git status --porcelain --untracked-files=no) && $(git rev-parse HEAD) == "$SHA" ]] || die 23 "tree not clean or HEAD moved"
gate_test gate-restored
[[ $RC == 0 ]] && grep -E "$MULTI_BY_ID" "$OUT/gate-restored.log" >/dev/null || die 23 "gate test after restore"
log "PASS $SHA"
