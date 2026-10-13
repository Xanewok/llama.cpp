#!/usr/bin/env bash
# stack11-s2 CUDA window (parent queue only, no serving A/B): CUDA build of both variants (frozen/stack11-s2 = stack11-base + 87/88/91/90
# with K7 and K10, built in wt/stack11-s2-k; frozen/stack11-s2-nok = the same without K7 and K10, built in wt/stack11-s2), the np2
# joint-replay smoke, CPU-device unit tests, test-host-mapped (bitwise per type, scheduler placement, NO_PINNED fallback) and the touched
# ops on both GPUs against frozen/stack11-base's binary, served-flag health boots, freeze, then render and `gpuq submit --check` the
# A/B items (never submitted), then the pkg 90 placement proof on the frozen builds (stage 73, vs frozen/stack11-base as the negative
# control; meta_rows.cpp, stage 54, compares the mapped-weight gather through Meta(CUDA0,CUDA1) bitwise with the device-resident one). Not repeated from the stack11-base window: its kernel gates and full test-backend-ops (87/88/91 touch no
# kernel; 90 touches GET_ROWS dispatch and buffer types only, covered here).
#   --preflight, --selftest (gpuq stages, CPU only), --list-stages.
#   DRY_RUN=1 (CPU only, no gpuq): preflight (Mac evidence reported), stage list, selftests, dry renders against labelled stand-ins into
#   rendered-dry/ with `gpuq submit --check`, and the window toml check.
# rc: 0 pass | 3 Xid | 4 transient (GPU busy/missing) | 5 void negative control or too little data | 6 integrity
#     7 timeout | 8 deterministic failure. A failed gate does not stop the other gates; nothing is frozen or rendered after one.
set -euo pipefail
VISIBLE=${CUDA_VISIBLE_DEVICES:-}
DRY=${DRY_RUN:-0}
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BUILDGATE_MIN_GIB=10 CPUGATE_MIN_GIB=10
export PATH="$HOME/.local/bin:/usr/local/cuda-13.2/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2
WIN=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
ROOT=$(cd "$WIN/../../.." && pwd)
cd "$ROOT"
W=$HOME/repos/qwen38-eval-worktrees
KROOT=$W/wt/stack11-s2-k
EVAL=$HOME/repos/qwen38-eval
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
BUILDGATE=$W/bin/buildgate
CPUS=0-3,12-15
BASE=abee0c8476b585b0d081d580b5ce21d0addc22c4
BASE_FROZEN=$W/frozen/stack11-base
BASE_SHA=7e7b039628ada54d3b978dda866a45804388fcee
SERVE=$HOME/.claude/skills/local-qwen-server/bin/serve.sh
MACEV=$ROOT/codex/stack11-s2/mac-evidence
ALL_COMPONENTS='p55 k7 k4 k6a k6b k10 k8 k11a k11b p90'
COMPONENTS=$(echo "${COMPONENTS:-$ALL_COMPONENTS}" | xargs)
ITEM=bw-stack11-s2
OUT_GLOB=$ROOT/codex/window/stack11-s2-a[0-9]*
# shellcheck source=gates.sh
source "$WIN/gates.sh"

flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)

# kind code name function: the single definition of what runs.
STAGES=(
    'fatal 40 cuda-build-k build_k'
    'fatal 41 cuda-build-nok build_nok'
    'fatal 42 stamps stamps'
    'fatal 43 np2-smoke np2_smoke'
    'gate 50 unit-cpu g_unit'
    'gate 51 host-mapped g_mapped'
    'gate 52 ops-k g_ops_k'
    'gate 53 ops-nok g_ops_nok'
    'gate 54 meta-rows g_meta_rows'
)
# After the table, on a clean sheet only: 60 health-boots-k, 61 health-boots-nok; then fatal 70 freeze-k, 71 freeze-nok, 72 render, and gate 73 placement-proof on the frozen builds.
TAIL_STAGES=('60 health-boots-k' '61 health-boots-nok' '70 freeze-stack11-s2' '71 freeze-stack11-s2-nok' '72 render-and-check-items' '73 placement-proof')

# Source anchors the gates rely on (a renamed one fails here, not on the GPU).
source_checks() {
    local t d n
    for t in "$KROOT" "$ROOT"; do
        for n in test_meta_host_mapped_weight test_meta_subgraph_reuse_two_splits test_meta_subgraph_realloc_count test_meta_subgraph_realloc_size; do
            [[ $(grep -c "$n" "$t/tests/test-alloc.cpp") -gt 0 ]] || { echo "$t: test-alloc lacks $n"; return 1; }
        done
        [[ $(grep -c 'uses_prompt' "$t/tests/test-speculative.cpp") -gt 0 && -s $t/tests/test-host-mapped.cpp ]] || { echo "$t: tests"; return 1; }
        [[ $(grep -c '_Host_Mapped' "$t/ggml/src/ggml-cuda/ggml-cuda.cu") -gt 0 ]] || { echo "$t: no _Host_Mapped"; return 1; }
        [[ $(grep -rc 'fattn_kv_max' "$t/ggml/src" | grep -vc ':0$') == 0 ]] || { echo "$t: K2 in ggml/src"; return 1; }
    done
    [[ -f $KROOT/tests/test-topk-batched.cpp && ! -e $ROOT/tests/test-topk-batched.cpp ]] || { echo 'K10 files: K tree must have, nok tree must not'; return 1; }
    [[ -f $KROOT/ggml/src/ggml-rms-norm-scale-pair.h && ! -e $ROOT/ggml/src/ggml-rms-norm-scale-pair.h ]] || { echo 'K7 header'; return 1; }
}

# 0 the Mac CPU gate passed on these sources; 2 the evidence is missing; 1 it is wrong. TREE FILE-PREFIX
mac_evidence() {
    local tree=$1 prefix=$2 commit
    [[ -s $MACEV/$prefix.tsv && -s $MACEV/$prefix-source-commit.txt ]] || return 2
    commit=$(cat "$MACEV/$prefix-source-commit.txt")
    [[ $(grep -ac 'gate=FAIL' "$MACEV/$prefix.tsv") == 0 && $(grep -ac 'gate=PASS' "$MACEV/$prefix.tsv") -ge 40 ]] || return 1
    git -C "$tree" cat-file -e "$commit^{commit}" || return 1
    git -C "$tree" diff --quiet "$commit" HEAD -- . ':!codex' || return 1
    echo "Mac evidence $prefix: $(grep -ac 'gate=PASS' "$MACEV/$prefix.tsv") stages PASS at ${commit:0:9}, sources identical to HEAD"
}

preflight() {
    local tool p script rc
    for tool in cmake git python3 nice gpuq ccache strings timeout flock taskset g++ c++ nvidia-smi gdb journalctl curl bc ss awk sed xargs comm; do
        command -v "$tool" > /dev/null || { echo "missing tool $tool"; return 1; }
    done
    [[ $(git branch --show-current) == topic/stack11-s2 && $(git -C "$KROOT" branch --show-current) == topic/stack11-s2-k ]] || { echo 'branches'; return 1; }
    [[ -z $(git status --porcelain --untracked-files=no) && -z $(git -C "$KROOT" status --porcelain --untracked-files=no) ]] || { echo 'tracked tree is dirty'; return 1; }
    for p in stack11-s2 stack11-s2-nok; do
        if [[ -e $W/frozen/$p ]]; then echo "frozen/$p exists"; [[ $DRY == 1 ]] || return 1; fi
    done
    git merge-base --is-ancestor "$BASE" HEAD && git -C "$KROOT" merge-base --is-ancestor "$BASE" HEAD || { echo "master $BASE is not an ancestor"; return 1; }
    git merge-base --is-ancestor "$BASE_SHA" HEAD && git -C "$KROOT" merge-base --is-ancestor "$BASE_SHA" HEAD || { echo 'stack11-base is not an ancestor'; return 1; }
    for p in "$CCCL" "$BUILDGATE" "$W/bin/wt" "$W/work/aa-load/ccx01.sh" /usr/local/cuda-13.2/bin/nvcc "$EVAL/harness/mkitem.py" "$SERVE" /mnt/scratch \
             "$BASE_FROZEN/SOURCE" "$BASE_FROZEN/build/bin/test-backend-ops" "$BASE_FROZEN/build/bin/round-trace" "$BASE_FROZEN/build/bin/SHA256SUMS" \
             "$BASE_FROZEN/build/bin/libggml-cuda.so"; do
        [[ -e $p ]] || { echo "missing $p"; return 1; }
    done
    [[ $(head -n 1 "$BASE_FROZEN/SOURCE") == "$BASE_SHA" ]] || { echo "frozen/stack11-base is not $BASE_SHA"; return 1; }
    # the K tree is this branch without the two drop commits: its source differs from HEAD by exactly their diff (blob ids ignored)
    p=$(git log --format=%H --grep='^WIP s2 drop K7$' -1)
    q=$(git log --format=%H --grep='^WIP s2 drop K10$' -1)
    [[ -n $p && -n $q ]] && cmp -s <(git diff "$p~1" "$q" -- . ':!codex' | grep -v '^index ') \
        <(git diff "$(git -C "$KROOT" rev-parse HEAD)" HEAD -- . ':!codex' | grep -v '^index ') \
        || { echo 'wt/stack11-s2-k source differs from HEAD by more than the K7/K10 drop commits'; return 1; }
    [[ -w $W/work ]] || { echo 'work dir not writable'; return 1; }
    source_checks || { echo 'source checks'; return 1; }
    for script in "$WIN"/*.sh; do bash -n "$script" || { echo "bash -n $script"; return 1; }; done
    python3 -c 'import ast, sys; [ast.parse(open(f).read(), f) for f in sys.argv[1:]]' "$WIN"/*.py || { echo 'python syntax'; return 1; }
    bash "$WIN/np2_smoke.sh" --preflight || { echo 'np2 smoke preflight'; return 1; }
    python3 "$WIN/engagement.py" selftest > /dev/null || { echo 'engagement selftest'; return 1; }
    python3 "$WIN/s2items.py" selftest > /dev/null || { echo 's2items selftest'; return 1; }
    python3 "$WIN/health.py" --selftest > /dev/null || { echo 'health selftest'; return 1; }
    python3 "$WIN/placement.py" selftest > /dev/null || { echo 'placement selftest'; return 1; }
    g++ -std=c++17 -Wall -Wextra -Werror -I"$KROOT/ggml/include" -fsyntax-only "$WIN/meta_rows.cpp" || { echo 'meta_rows.cpp does not compile'; return 1; }
    python3 "$WIN/health.py" --parse 18099 > /dev/null || { echo 'serve.sh launch lines'; return 1; }
    python3 "$WIN/check_pins.py" > /dev/null || { echo 'pins'; return 1; }
    for p in "$ROOT nok" "$KROOT k"; do
        rc=0
        mac_evidence "${p% *}" "$(case ${p#* } in k) echo with-k ;; nok) echo nok ;; esac)" || rc=$?
        case $rc in
            0) ;;
            2) echo "Mac evidence MISSING for ${p#* } (required in the real run)"; [[ $DRY == 1 ]] || return 1 ;;
            *) echo "Mac evidence WRONG for ${p#* }"; return 1 ;;
        esac
    done
    return 0
}

list_stages() {
    local row
    for row in "${STAGES[@]}"; do read -ra F <<< "$row"; echo "${F[1]} ${F[2]}"; done
    for row in "${TAIL_STAGES[@]}"; do echo "$row"; done
}

dry_run() {
    local d=$ROOT/codex/window/stack11-s2-dry dry=$WIN/rendered-dry
    rm -rf "$d"
    mkdir -p "$d"
    echo "bw-s2: DRY_RUN commit=$(git rev-parse HEAD) k-commit=$(git -C "$KROOT" rev-parse HEAD) components=[$COMPONENTS]"
    preflight || { echo 'bw-s2: FAIL dry preflight'; exit 8; }
    echo 'bw-s2: stages'
    list_stages | sed 's/^/  /'
    python3 "$WIN/health.py" --parse 18099 > "$d/health-parse.jsonl"
    rm -rf "$dry"
    mkdir -p "$dry"
    # stand-ins, labelled in every rendered item: frozen/stack11-base for frozen/stack11-s2 and frozen/stack11-s2-nok
    python3 "$WIN/render_items.py" --out "$dry" --dry --s2-root "$BASE_FROZEN" --nok-root "$BASE_FROZEN" --components "$COMPONENTS" \
        --evidence "$MACEV/with-k.tsv"
    python3 "$WIN/mk_window_toml.py" > "$d/bw-stack11-s2.toml"
    cmp -s "$d/bw-stack11-s2.toml" "$WIN/bw-stack11-s2.toml" || { echo 'bw-s2: FAIL bw-stack11-s2.toml is stale (rerun mk_window_toml.py)'; exit 8; }
    python3 "$WIN/check_pins.py" "$WIN/bw-stack11-s2.toml"
    gpuq submit --check "$WIN/bw-stack11-s2.toml" > "$dry/bw-stack11-s2-contract.log" 2>&1 || { cat "$dry/bw-stack11-s2-contract.log"; echo 'bw-s2: FAIL window toml contract'; exit 8; }
    cat "$dry/bw-stack11-s2-contract.log"
    echo "bw-s2: DRY_RUN PASS: preflight, $(list_stages | wc -l) stages listed, selftests, $(ls "$dry"/*.gpuq.toml | wc -l) dry items + window toml checked (NOT submitted); GPU stages and compiles not run"
}

if [[ ${1:-} == --list-stages ]]; then list_stages; exit 0; fi
if [[ ${1:-} == --selftest ]]; then
    python3 "$WIN/engagement.py" selftest && python3 "$WIN/s2items.py" selftest && python3 "$WIN/health.py" --selftest \
        && python3 "$WIN/placement.py" selftest \
        && { echo 'bw-s2: PASS selftest'; exit 0; }
    echo 'bw-s2: FAIL selftest'; exit 8
fi
if [[ ${1:-} == --preflight ]]; then
    if preflight; then echo 'bw-s2: PASS preflight'; exit 0; fi
    echo 'bw-s2: FAIL preflight'; exit 8
fi
if [[ $DRY == 1 ]]; then
    [[ $# == 0 && -z ${GPUQ_ITEM:-} ]] || { echo 'DRY_RUN=1 takes no arguments and never runs under gpuq'; exit 8; }
    dry_run
    exit 0
fi
[[ ${GPUQ_ITEM:-} == "$ITEM" && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in $OUT_GLOB) ;; *) exit 8 ;; esac
mkdir -p "$OUT"
exec > >(tee "$OUT/payload.log") 2>&1
preflight || { echo 'bw-s2: FAIL preflight'; exit 8; }
SHA=$(git rev-parse HEAD)
KSHA=$(git -C "$KROOT" rev-parse HEAD)
SHORT=$(git rev-parse --short=9 HEAD)
KSHORT=$(git -C "$KROOT" rev-parse --short=9 HEAD)
STARTED=$(date '+%Y-%m-%d %H:%M:%S')
echo "bw-s2: components=[$COMPONENTS] nok=$SHA k=$KSHA visible=$VISIBLE"
[[ $VISIBLE == GPU-*,GPU-* ]] || { echo 'bw-s2: STOP no two GPU UUIDs in CUDA_VISIBLE_DEVICES'; exit 4; }

for row in "${STAGES[@]}"; do
    read -ra F <<< "$row"
    "${F[0]}" "${F[1]}" "${F[2]}" "${F[3]}"
done
if [[ ${#FAILS[@]} == 0 ]]; then
    gate 60 health-boots-k g_health_k
    gate 61 health-boots-nok g_health_nok
else
    echo 'bw-s2: health boots skipped, an earlier gate failed'
fi
if [[ ${#FAILS[@]} != 0 ]]; then finish_fail; fi

fatal 70 freeze-stack11-s2 freeze_k
fatal 71 freeze-stack11-s2-nok freeze_nok
fatal 72 render-and-check-items items
# after the freeze and the render: a failure here leaves both frozen builds and the checked items in place and fails the window
gate 73 placement-proof g_placement
if [[ ${#FAILS[@]} != 0 ]]; then finish_fail; fi
python3 - "$OUT" "$SHA" "$KSHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
stages = [dict(code=int(c), name=n, rc=int(rc), wall_s_M=int(t)) for c, n, rc, t in (l.split('\t') for l in (p/'stages.tsv').read_text().splitlines())]
(p/'summary.json').write_text(json.dumps(dict(result='PASS', commit_nok=sys.argv[2], commit_k=sys.argv[3], GPU_execution=True, stages=stages,
                                              frozen=['frozen/stack11-s2', 'frozen/stack11-s2-nok'],
                                              items='codex/stack11-s2/window/rendered; checked, NOT submitted'), indent=2)+'\n')
PY_RESULT
echo 'bw-s2: PASS; frozen as stack11-s2 and stack11-s2-nok, items checked, NOT submitted, placement proof passed'
