#!/usr/bin/env bash
# Combined stack10b CUDA window (stack10 minus K9 plus the K4 np2 fix; parent queue only, no A/B timing): CUDA build, the np2
# joint-replay server smoke (the stack10 crash), every selected component's own CUDA gates, the meta subgraph unit test, the fafix
# SASS gate, test-backend-ops in full on both GPUs, K11a sync count, freeze as frozen/stack10, render and check the A/B items.
#   env COMPONENTS (default: p55 k7 k4 k6a k6b k2 k10 k8 k11a k11b) selects the gates and the canaries of the rendered items;
#   the base ops, the np2 smoke, the meta unit test, the fafix gates and the full suite always run. gdn4 and K9 are not part of it.
#   env STACK10_TAIL_FROM=<codex/window dir> (set by a tail item only) reuses that attempt's stages 40-65 instead of rerunning
#   them: tail_check.py must prove them still valid (rc 6 otherwise) before any GPU stage; then only 66, 63, 70, 71 run.
# rc: 0 pass | 3 Xid | 4 transient (GPU busy/missing) | 5 void negative control or too little data | 6 integrity
#     7 timeout | 8 deterministic failure. A failed gate does not stop the other gates (the drop map needs every failing component),
#     but nothing is frozen or rendered after one; the exit code is the worst of them (3 > 6 > 5 > 8 > 7 > 4), an Xid stops at once.
set -euo pipefail
VISIBLE=${CUDA_VISIBLE_DEVICES:-}
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BUILDGATE_MIN_GIB=10 CPUGATE_MIN_GIB=10
export PATH="$HOME/.local/bin:/usr/local/cuda-13.2/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
cd "$ROOT"
W=$HOME/repos/qwen38-eval-worktrees
EVAL=$HOME/repos/qwen38-eval
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
GPUS=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
BUILDGATE=$W/bin/buildgate
CPUS=0-3,12-15
STACK9=cd776e104dbd516c7b74506992d2e8efd7600155
CAND2=def4a4e20a689047e68e8edd81c97790b69a1de7
S9_LIB=$W/frozen/stack9/build/bin
CAND_LIB=$W/frozen/stack10-cand/build/bin
P55_LIB=$W/frozen/stack9-p55/build/bin
EVIDENCE=codex/stack10/mac-evidence/summary.json
ALL_COMPONENTS='p55 k7 k4 k6a k6b k2 k10 k8 k11a k11b'
COMPONENTS=$(echo "${COMPONENTS:-$ALL_COMPONENTS}" | xargs)
TAIL_FROM=${STACK10_TAIL_FROM:-}
if [[ -n $TAIL_FROM ]]; then
    [[ $TAIL_FROM =~ ^stack10b-a[0-9]+$ ]] || { echo "bad STACK10_TAIL_FROM $TAIL_FROM"; exit 8; }
    A1=$ROOT/codex/window/$TAIL_FROM
    ITEM=bw-stack10b-tail
    OUT_GLOB=$ROOT/codex/window/stack10b-tail-a[0-9]*
else
    ITEM=bw-stack10b
    OUT_GLOB=$ROOT/codex/window/stack10b-a[0-9]*
fi
# shellcheck source=gates.sh
source "$ROOT/codex/stack10/gates.sh"

# Same flags as bw-stack10-cand.sh (LLAMA_BUILD_TESTS=ON for the test binaries).
flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)

# kind code name function [components: any one selects the stage]. The single definition of what the window runs before the
# full suite; the tail's integrity precheck derives the stages it must find at rc 0 from the same table.
STAGES=(
    'fatal 40 cuda-build build'
    'fatal 41 cuda-stamps stamps'
    'fatal 43 np2-smoke np2_smoke'
    'fatal 42 stock-mmq-build stock_build k8'
    'gate 50 base-ops g_base'
    'gate 54 meta-subgraphs g_mfix'
    'gate 52 k7-pair g_k7 k7'
    'gate 53 k4-conv-rs g_k4 k4'
    'gate 55 k6a-q8-reuse g_k6a k6a'
    'gate 56 k6b-add-rms g_k6b k6b'
    'gate 57 k6-memcheck g_k6_memcheck k6a k6b'
    'gate 58 k2-shared-mask g_k2 k2'
    'gate 59 k10-topk g_k10 k10'
    'gate 61 k8-mmq-tiles g_k8 k8'
    'gate 62 k11b-input-stage g_k11b k11b'
    'gate 64 fafix-sass g_sass'
    'gate 65 fafix-identity g_fafix'
)
# Fields of a STAGES row into F; succeeds when the stage is selected (no component listed, or one of them is).
stage_fields() {
    local comp
    read -ra F <<< "$1"
    [[ ${#F[@]} == 4 ]] && return 0
    for comp in "${F[@]:4}"; do has "$comp" && return 0; done
    return 1
}
expected_stages() {
    local row F out=()
    for row in "${STAGES[@]}"; do
        if stage_fields "$row"; then out+=("${F[1]}:${F[2]}"); fi
    done
    local IFS=,
    echo "${out[*]}"
}

# Tail only: controls (every sub-check must refuse its one-input mutation) and then the real precheck of the reused attempt.
tail_proof() {
    local expected
    expected=$(expected_stages)
    python3 codex/stack10/tail_check.py controls --root "$ROOT" --a1 "$A1" --expected "$expected" || return 6
    python3 codex/stack10/tail_check.py check --root "$ROOT" --a1 "$A1" --expected "$expected" ${1:+--out "$1"} || return 6
}

preflight() {
    local tool word p script
    for word in $COMPONENTS; do
        [[ " $ALL_COMPONENTS " == *" $word "* ]] || { echo "unknown component $word"; return 1; }
    done
    [[ $(git branch --show-current) == topic/stack10b ]] || { echo 'branch is not topic/stack10b'; return 1; }
    [[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'tracked tree is dirty'; return 1; }
    [[ ! -e $W/frozen/stack10 ]] || { echo 'frozen/stack10 exists'; return 1; }
    for tool in cmake git python3 nice gpuq ccache strings timeout patchelf flock taskset g++ cuobjdump nsys compute-sanitizer nvidia-smi \
                journalctl curl bc ss awk sed xargs; do
        command -v "$tool" > /dev/null || { echo "missing tool $tool"; return 1; }
    done
    git merge-base --is-ancestor "$STACK9" HEAD || { echo 'stack9 is not an ancestor'; return 1; }
    git merge-base --is-ancestor "$CAND2" HEAD || { echo 'stack10-cand2 is not an ancestor'; return 1; }
    local log
    log=$(git log --format=%s "$CAND2..HEAD")
    [[ $log != *'process 4 GDN columns per warp'* && $log != *'time the fused GDN+CPY pair'* ]] || { echo 'gdn4 commits are still in HEAD'; return 1; }
    for p in "$CCCL" "$BUILDGATE" "$W/bin/cpugate" "$W/bin/wt" "$W/frozen/stack6/RECIPE.txt" "$W/work/aa-load/ccx01.sh" /usr/local/cuda-13.2/bin/nvcc \
             "$EVAL/harness/mkitem.py" "$W/frozen/stack9/SOURCE" "$W/frozen/stack9-p55/SOURCE" "$W/frozen/stack10-cand/SOURCE" \
             "$S9_LIB/libggml-cuda.so" "$CAND_LIB/libggml-cuda.so" "$P55_LIB/test-backend-ops" "$P55_LIB/round-trace" "$EVIDENCE" /mnt/scratch; do
        [[ -e $p ]] || { echo "missing $p"; return 1; }
    done
    [[ $(head -n 1 "$W/frozen/stack9/SOURCE") == "$STACK9" ]] || { echo 'frozen/stack9 is not cd776e104'; return 1; }
    [[ -w $W/work ]] || { echo 'work dir not writable'; return 1; }
    for script in codex/stack10/bw-stack10b.sh codex/stack10/gates.sh codex/stack10/np2_smoke.sh codex/stack10/gpu/gpu_ab.sh codex/k8/gpu/k8-gpu.sh codex/k11a/sync_count.sh; do
        bash -n "$script" || { echo "bash -n $script"; return 1; }
    done
    python3 codex/stack10/shfn.py selftest > /dev/null || { echo 'shfn selftest'; return 1; }
    bash codex/stack10/np2_smoke.sh --preflight || { echo 'np2 smoke preflight'; return 1; }
    python3 codex/stack10/full_checks.py --fixtures > /dev/null || { echo 'full_checks fixtures'; return 1; }
    python3 codex/stack10/engagement.py selftest > /dev/null || { echo 'engagement selftest'; return 1; }
    python3 codex/stack10/stock_tree.py list > /dev/null || { echo 'stock tree commits'; return 1; }
    python3 codex/stack10/check_pins.py > /dev/null || { echo 'pins'; return 1; }
    local tmp
    tmp=$(mktemp -d)
    python3 codex/stack10/k8_gpu_patch.py codex/k8/gpu/k8-gpu.sh "$tmp/k8-gpu.sh" "$tmp/k8-bin" "$tmp/stock-bin" && bash -n "$tmp/k8-gpu.sh" || { rm -rf "$tmp"; echo 'k8 derivation'; return 1; }
    rm -rf "$tmp"
    HARNESS_CODE_ROOT=$EVAL PYTHONPATH=$EVAL python3 -P codex/stack10/gpu/gpu_s10.py --fixtures > /dev/null || { echo 'item fixtures'; return 1; }
    python3 - "$EVIDENCE" <<'PY_GATE' || { echo 'Mac evidence'; return 1; }
import hashlib, json, subprocess, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
assert r['result'] == 'PASS', r['result']
files = subprocess.check_output(['git', 'ls-files', 'src', 'ggml', 'tests', 'common'], text=True).splitlines()
assert files == r['source_files'], 'source file list differs from the Mac gate'
h = hashlib.sha256()
for name in files:
    h.update(name.encode() + b'\0' + Path(name).read_bytes())
if h.hexdigest() != r['source_sha256']:
    # 10-09: test-alloc went in-process (fork deadlock); its Linux CPU run is in codex/stack10/linux-mfix, the Mac rerun runs beside
    changed = subprocess.check_output(['git', 'diff', '--name-only', r['commit'], 'HEAD', '--', 'src', 'ggml', 'tests', 'common'],
                                      text=True).split()
    assert changed == ['tests/test-alloc.cpp'], f'source differs from the Mac CPU gate: {changed}'
    print('Mac evidence: only tests/test-alloc.cpp changed since the Mac gate', file=sys.stderr)
PY_GATE
    [[ -z $TAIL_FROM || -d $A1 ]] || { echo "missing $A1"; return 1; }
    if has k11a; then
        DRY_RUN=1 ROOT="$ROOT" OUT="$ROOT/codex/window/preflight" bash codex/k11a/sync_count.sh > /dev/null || { echo 'k11a dry run'; return 1; }
    fi
    return 0
}

if [[ ${1:-} == --list-stages ]]; then
    for row in "${STAGES[@]}"; do if stage_fields "$row"; then echo "${F[1]} ${F[2]}"; fi; done
    exit 0
fi
if [[ ${1:-} == --preflight ]]; then
    if preflight && { [[ -z $TAIL_FROM ]] || tail_proof; }; then echo 'bw-stack10: PASS preflight'; exit 0; fi
    echo 'bw-stack10: FAIL preflight'; exit 8
fi
[[ ${GPUQ_ITEM:-} == "$ITEM" && ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$1
case $OUT in $OUT_GLOB) ;; *) exit 8 ;; esac
ATTEMPT=${OUT##*-a}
mkdir -p "$OUT"
exec > >(tee "$OUT/payload.log") 2>&1
preflight || { echo 'bw-stack10: FAIL preflight'; exit 8; }
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short=9 HEAD)
BUILD_SHA=$SHA
P55_SHA=$(head -n 1 "$P55_LIB/../../SOURCE")
STARTED=$(date '+%Y-%m-%d %H:%M:%S')
echo "bw-stack10: components=[$COMPONENTS] commit=$SHA visible=$VISIBLE tail_from=${TAIL_FROM:--}"
[[ $VISIBLE == GPU-*,GPU-* ]] || { echo 'bw-stack10: FAIL no two GPU UUIDs in CUDA_VISIBLE_DEVICES'; exit 4; }

if [[ -n $TAIL_FROM ]]; then
    # Reuse of an earlier attempt's build and gates: proven before any GPU stage, evidence copied next to the tail's own.
    if ! tail_proof "$OUT/tail-precheck.json"; then echo 'bw-stack10: STOP tail precheck refused rc=6'; finish_fail 6; fi
    BUILD_SHA=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["a1_commit"])' "$OUT/tail-precheck.json")
    for f in stages.tsv summary.json payload.log; do cp -p "$A1/$f" "$OUT/a1-$f"; done
else
    for row in "${STAGES[@]}"; do
        stage_fields "$row" || continue
        "${F[0]}" "${F[1]}" "${F[2]}" "${F[3]}"
        if [[ ${F[1]} == 42 ]]; then
            STOCK_DIR=$(sed -n 's/^STOCK_DIR=//p' "$OUT/stock-dir.txt")
            [[ -d $STOCK_DIR/stock-bin && -d $STOCK_DIR/k8-bin ]]
        fi
    done
fi
gate 66 full-suite g_full
# The server-launching capture goes last and only on a clean sheet: it costs the most GPU time and adds no drop-map information
# once another gate has failed.
if has k11a; then
    if [[ ${#FAILS[@]} == 0 ]]; then gate 63 k11a-sync-count g_k11a; else echo 'bw-stack10: k11a-sync-count skipped, an earlier gate failed'; fi
fi
if [[ ${#FAILS[@]} != 0 ]]; then finish_fail; fi

fatal 70 freeze-stack10 freeze
fatal 71 render-and-check-items items
python3 - "$OUT" "$SHA" "$COMPONENTS" "$TAIL_FROM" "$BUILD_SHA" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
stages = [dict(code=int(c), name=n, rc=int(rc), wall_s_M=int(t)) for c, n, rc, t in (l.split('\t') for l in (p/'stages.tsv').read_text().splitlines())]
result = dict(result='PASS', commit=sys.argv[2], components=sys.argv[3].split(), GPU_execution=True, stages=stages,
              frozen='frozen/stack10', items='codex/stack10/rendered; checked, NOT submitted')
if sys.argv[4]:
    a1 = json.loads((p/'tail-precheck.json').read_text())
    result.update(tail_from=sys.argv[4], build_commit=sys.argv[5], reused_stages=a1['stages'], precheck='tail-precheck.json',
                  note='stages is the tail only; reused_stages passed rc 0 in tail_from (a1-stages.tsv) and were re-proved by tail-precheck.json')
(p/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
PY_RESULT
echo 'bw-stack10: PASS; frozen as stack10 and items checked, NOT submitted'
