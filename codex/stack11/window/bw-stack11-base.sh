#!/usr/bin/env bash
# stack11-base CUDA window (parent queue only, no serving A/B): CUDA build, the np2 joint-replay smoke, every component's CUDA gates,
# the meta subgraph unit test, the fafix SASS and identity gates and the FA q8_0 op timing against frozen/stack10, the touched ops and the whole
# test-backend-ops on both GPUs, K11a sync count, served-flag health boots of the four swift15 profiles, freeze as frozen/stack11-base,
# the kld_dump pair, render and `gpuq submit --check` the A/B and KLD items (never submitted).
#   env COMPONENTS (default: p55 k7 k4 k6a k6b k10 k8 k11a k11b) selects the component gates and the canaries of the rendered items;
#   base ops, FA, meta subgraphs, gdn4, fafix, touched ops, full suite and health boots always run. K2 and K9 are not in stack11.
#   --preflight, --selftest (gpuq stages, CPU only), --list-stages.
#   DRY_RUN=1 (CPU only, no gpuq): preflight with the Mac evidence reported but not required, stage list, every selftest and dry run
#   the tree can do without a build, the KLD CPU stub, dry renders against labelled stand-ins into rendered-dry/ with
#   `gpuq submit --check`, and the window toml check.
# rc: 0 pass | 3 Xid | 4 transient (GPU busy/missing) | 5 void negative control or too little data | 6 integrity
#     7 timeout | 8 deterministic failure. A failed gate does not stop the other gates (the drop map needs every failing component),
#     but nothing is frozen or rendered after one; the exit code is the worst of them (3 > 6 > 5 > 8 > 7 > 4), an Xid stops at once.
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
EVAL=$HOME/repos/qwen38-eval
CCCL=$HOME/repos/qwen38-weekend/deps/cccl-v3.4.3
GPUS=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
BUILDGATE=$W/bin/buildgate
CPUS=0-3,12-15
BASE=abee0c8476b585b0d081d580b5ce21d0addc22c4
S10=$W/frozen/stack10
S10_LIB=$S10/build/bin
S10_SHA=a588f2a44382a361df3e2a456be0c80ca10dd1d6
S10_REPO=$W/wt/stack10b
CAND_LIB=$W/frozen/stack10-cand/build/bin
MASTER=$W/frozen/pkg89-base
MASTER_LIB=$MASTER/build/bin
SERVE=$HOME/.claude/skills/local-qwen-server/bin/serve.sh
EVIDENCE=codex/stack11/mac-evidence/summary.json
ALL_COMPONENTS='p55 k7 k4 k6a k6b k10 k8 k11a k11b'
COMPONENTS=$(echo "${COMPONENTS:-$ALL_COMPONENTS}" | xargs)
ITEM=bw-stack11-base
OUT_GLOB=$ROOT/codex/window/stack11-base-a[0-9]*
# shellcheck source=gates.sh
source "$WIN/gates.sh"

flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)

# kind code name function [components: any one selects the stage]: the single definition of what runs before the full suite.
STAGES=(
    'fatal 40 cuda-build build'
    'fatal 41 cuda-stamps stamps'
    'fatal 43 np2-smoke np2_smoke'
    'gate 50 base-ops g_base'
    'gate 51 fa-ops g_fa'
    'gate 52 k7-pair g_k7 k7'
    'gate 53 k4-conv-rs g_k4 k4'
    'gate 54 meta-subgraphs g_mfix'
    'gate 55 k6a-q8-reuse g_k6a k6a'
    'gate 56 k6b-add-rms g_k6b k6b'
    'gate 57 k6-memcheck g_k6_memcheck k6a k6b'
    'gate 59 k10-topk g_k10 k10'
    'gate 60 gdn4 g_gdn4'
    'gate 61 k8-mmq-tiles g_k8 k8'
    'gate 62 k11b-input-stage g_k11b k11b'
    'gate 64 fafix-sass g_sass'
    'gate 65 fafix-identity g_fafix'
    'gate 73 fa-q8-perf g_faperf'
    'gate 67 mmvq g_mmvq p55'
    'gate 68 touched-ops g_touched'
)
# After the table: 66 full-suite; on a clean sheet only 63 k11a-sync-count (if k11a) and 69 health-boots; then fatal 70 freeze,
# 71 kld-dump-build, 72 render-and-check-items.
TAIL_STAGES=('66 full-suite' '63 k11a-sync-count' '69 health-boots' '70 freeze-stack11-base' '71 kld-dump-build' '72 render-and-check-items')
stage_fields() {
    local comp
    read -ra F <<< "$1"
    [[ ${#F[@]} == 4 ]] && return 0
    for comp in "${F[@]:4}"; do has "$comp" && return 0; done
    return 1
}

# Env switches, mutants, test names and target names the gates rely on, in this tree's source (a renamed one fails here, not on the GPU).
SWITCHES='GGML_CUDA_ALPHA_GATE_MUTANT GGML_CUDA_ALPHA_ALIAS_MUTANT GGML_CUDA_RMS_NORM_SCALE_PAIR GGML_CUDA_RMS_NORM_SCALE_PAIR_MUTANT
    GGML_RMS_NORM_SCALE_PAIR_ALIAS_MUTANT GGML_SSM_CONV_RS_MUTANT GGML_SSM_CONV_RS_OFF GGML_SSM_CONV_RS_LOG GGML_CUDA_Q8_REUSE
    GGML_CUDA_Q8_REUSE_STALE_MUTANT GGML_CUDA_ADD_RMS_NORM GGML_CUDA_ADD_RMS_NORM_MUTANT GGML_CUDA_TOPK_MUTANT GGML_CUDA_TOPK_BATCHED
    GGML_CUDA_TOPK_LOG GGML_CUDA_INPUT_STAGE_MUTANT GGML_META_INPUT_STAGE_MUTANT GGML_CUDA_INPUT_STAGE_ORDER GGML_CUDA_MMQ_SK_MUTANT
    GGML_CUDA_MMQ_TILES GGML_CUDA_MMQ_TILES_LOG GGML_CUDA_GDN_JOINT_MUTANT GGML_CUDA_DEBUG_GDN_STATE GGML_META_SUBGRAPH_MUTANT
    GGML_CUDA_Q8_DECODE GGML_CUDA_FA_MMA_Q8_0 INPUT_STAGE_REQUIRE_CUDA LLAMA_SYNC_ELIDE_DEBUG TOPK_MUTANT'
TBO_OPS='GATED_DELTA_NET_INDEXED GATED_DELTA_NET_CACHE_FUSION GATED_DELTA_NET_INDEXED_CACHE_FUSION SSM_CONV_STATE_FUSION
    SSM_CONV_BIAS_SILU SSM_CONV_RS MUL_MAT_SHARED MUL_MAT_ID_SHARED ADD_RMS_NORM RMS_NORM_MUL_ADD RMS_NORM_MUL_ROPE RMS_NORM_SCALE_PAIR
    ADD_SOFTPLUS_MUL'
source_checks() {
    local s t
    for s in $SWITCHES; do
        [[ $(grep -rlw --include='*.c*' --include='*.h' --include='*.cuh' "$s" ggml src tests common | wc -l) -gt 0 ]] || { echo "switch $s not in source"; return 1; }
    done
    for t in $TBO_OPS; do
        [[ $(grep -c "\"$t\"" tests/test-backend-ops.cpp) -gt 0 ]] || { echo "test-backend-ops op $t missing"; return 1; }
    done
    for t in $(targets); do
        [[ $t == llama-server || $t == test-backend-ops || $(grep -cE "llama_build(_and_test)?\(${t}\.cpp\)" tests/CMakeLists.txt) -gt 0 ]] || { echo "target $t missing"; return 1; }
    done
    [[ $(grep -c 'test_meta_subgraph_realloc_count' tests/test-alloc.cpp) -gt 0 && $(grep -c 'test_meta_subgraph_realloc_size' tests/test-alloc.cpp) -gt 0 ]] || { echo 'test-alloc names'; return 1; }
    [[ $(grep -c 'getenv("TOPK_MUTANT")' tests/test-topk-batched.cpp) -gt 0 ]] || { echo 'topk emulator mutant'; return 1; }
    [[ $(grep -rc 'fattn_kv_max' ggml/src | grep -vc ':0$') == 0 ]] || { echo 'K2 (fattn_kv_max) is still in ggml/src'; return 1; }
    [[ $(grep -c 'units=%d parts=%d blocks=%d' ggml/src/ggml-cuda/mmq.cuh) -gt 0 ]] || { echo 'K8 tile log format'; return 1; }
}

# 0 the Mac CPU gate passed on these sources; 2 the evidence file is missing; 1 it is wrong.
mac_evidence() {
    [[ -s $EVIDENCE ]] || return 2
    python3 - "$EVIDENCE" <<'PY_GATE' || return 1
import hashlib, json, subprocess, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
assert r['result'] == 'PASS', r['result']
files = subprocess.check_output(['git', 'ls-files', 'src', 'ggml', 'tests', 'common'], text=True).splitlines()
assert files == r['source_files'], 'source file list differs from the Mac gate'
h = hashlib.sha256()
for name in files:
    h.update(name.encode() + b'\0' + Path(name).read_bytes())
assert h.hexdigest() == r['source_sha256'], 'source differs from the Mac CPU gate'
print(f"Mac evidence: PASS at {r['commit'][:9]}, {len(r.get('stages', []))} stages, sources identical")
PY_GATE
}

preflight() {
    local tool word p script rc=0
    for word in $COMPONENTS; do
        [[ " $ALL_COMPONENTS " == *" $word "* ]] || { echo "unknown component $word"; return 1; }
    done
    [[ $(git branch --show-current) == topic/stack11-base ]] || { echo 'branch is not topic/stack11-base'; return 1; }
    [[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'tracked tree is dirty'; return 1; }
    if [[ -e $W/frozen/stack11-base ]]; then
        echo 'frozen/stack11-base exists'
        [[ $DRY == 1 ]] || return 1
    fi
    for tool in cmake git python3 nice gpuq ccache strings timeout flock taskset g++ c++ cuobjdump nsys compute-sanitizer nvidia-smi gdb \
                journalctl curl bc ss awk sed xargs tar ldd; do
        command -v "$tool" > /dev/null || { echo "missing tool $tool"; return 1; }
    done
    git merge-base --is-ancestor "$BASE" HEAD || { echo "master $BASE is not an ancestor"; return 1; }
    for p in "$CCCL" "$BUILDGATE" "$W/bin/cpugate" "$W/bin/wt" "$W/work/aa-load/ccx01.sh" /usr/local/cuda-13.2/bin/nvcc \
             /usr/local/cuda-13.2/bin/compute-sanitizer "$EVAL/harness/mkitem.py" "$S10/SOURCE" "$S10_LIB/libggml-cuda.so" \
             "$S10_LIB/test-backend-ops" "$S10_LIB/round-trace" "$S10_LIB/SHA256SUMS" "$CAND_LIB/libggml-cuda.so" "$MASTER_LIB/libggml-cuda.so" \
             "$SERVE" "$K4_MODEL" /mnt/scratch; do
        [[ -e $p ]] || { echo "missing $p"; return 1; }
    done
    [[ $(head -n 1 "$S10/SOURCE") == "$S10_SHA" ]] || { echo "frozen/stack10 is not $S10_SHA"; return 1; }
    [[ $(head -n 1 "$MASTER/SOURCE") == "$BASE" ]] || { echo "frozen/pkg89-base is not master $BASE"; return 1; }
    git -C "$S10_REPO" cat-file -e "$S10_SHA^{commit}" || { echo "wt/stack10b lacks $S10_SHA (headers of frozen/stack10)"; return 1; }
    [[ -w $W/work ]] || { echo 'work dir not writable'; return 1; }
    source_checks || { echo 'source checks'; return 1; }
    for script in "$WIN"/*.sh "$WIN"/k11a/*.sh "$WIN"/kld/*.sh; do
        bash -n "$script" || { echo "bash -n $script"; return 1; }
    done
    python3 -c 'import ast, sys; [ast.parse(open(f).read(), f) for f in sys.argv[1:]]' "$WIN"/*.py "$WIN"/k11a/*.py "$WIN"/kld/*.py \
        "$WIN"/kld/tests/*.py || { echo 'python syntax'; return 1; }
    bash "$WIN/np2_smoke.sh" --preflight || { echo 'np2 smoke preflight'; return 1; }
    python3 "$WIN/full_checks.py" --fixtures > /dev/null || { echo 'full_checks fixtures'; return 1; }
    python3 "$WIN/sass_gate.py" --selftest > /dev/null || { echo 'sass_gate selftest'; return 1; }
    python3 "$WIN/fa_q8_perf.py" selftest > /dev/null || { echo 'fa_q8_perf selftest'; return 1; }
    python3 "$WIN/k10_perf.py" selftest > /dev/null || { echo 'k10_perf selftest'; return 1; }
    python3 "$WIN/engagement.py" selftest > /dev/null || { echo 'engagement selftest'; return 1; }
    python3 "$WIN/s11items.py" selftest > /dev/null || { echo 's11items selftest'; return 1; }
    python3 "$WIN/health.py" --selftest > /dev/null || { echo 'health selftest'; return 1; }
    python3 "$WIN/health.py" --parse 18099 > /dev/null || { echo 'serve.sh launch lines'; return 1; }
    python3 "$WIN/check_pins.py" > /dev/null || { echo 'pins'; return 1; }
    mac_evidence || rc=$?
    case $rc in
        0) ;;
        2) echo "Mac evidence MISSING: $EVIDENCE (required in the real run)"; [[ $DRY == 1 ]] || return 1 ;;
        *) echo "Mac evidence WRONG: $EVIDENCE"; return 1 ;;
    esac
    if has k11a; then
        DRY_RUN=1 ROOT="$ROOT" OUT="$ROOT/codex/window/preflight-k11a" bash "$WIN/k11a/sync_count.sh" > /dev/null || { echo 'k11a dry run'; return 1; }
    fi
    return 0
}

list_stages() {
    local row
    for row in "${STAGES[@]}"; do if stage_fields "$row"; then echo "${F[1]} ${F[2]}"; fi; done
    for row in "${TAIL_STAGES[@]}"; do
        [[ $row != '63 '* ]] || has k11a || continue
        echo "$row"
    done
}

# CPU-only rehearsal of everything the window can show without a GPU or a compile.
dry_run() {
    local d=$ROOT/codex/window/stack11-base-dry dry=$WIN/rendered-dry
    rm -rf "$d"
    mkdir -p "$d"
    echo "bw-stack11: DRY_RUN commit=$(git rev-parse HEAD) components=[$COMPONENTS]"
    preflight || { echo 'bw-stack11: FAIL dry preflight'; exit 8; }
    echo 'bw-stack11: stages'
    list_stages | sed 's/^/  /'
    python3 "$WIN/health.py" --parse 18099 > "$d/health-parse.jsonl"
    python3 - "$d/health-parse.jsonl" <<'PY_PARSE'
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    print('  health %s: np=%d, served exe %s replaced by build-cuda/bin/round-trace, gating %s, %d patterns'
          % (r['profile'], r['np'], r['exe'], ' '.join(r['gating']), r['patterns']))
PY_PARSE
    CUDA_VISIBLE_DEVICES='' python3 "$WIN/kld/tests/stub_e2e.py" "$d/kld-stub"
    rm -rf "$d/kld-stub"
    # Stand-ins, clearly labelled in every rendered item: frozen/stack10 for frozen/stack11-base, stack10b's Mac evidence, and the
    # stack9 kld_dump for both members of the KLD pair.
    rm -rf "$dry"
    mkdir -p "$dry"
    local m
    for m in j5q iq4; do
        python3 "$WIN/kld/mkconfig.py" --gen "$dry/kld-gen" --model "$m" --s11 "$S10" \
            --binary-s10 "$W/work/kld-s9-gdn4/bin/kld_dump" --binary-s11 "$W/work/kld-s9-gdn4/bin/kld_dump"
    done
    python3 "$WIN/render_items.py" --out "$dry" --dry --s11-root "$S10" --evidence "$S10_REPO/codex/stack10/mac-evidence/summary.json" \
        --kld-gen "$dry/kld-gen" --components "$COMPONENTS"
    python3 "$WIN/mk_window_toml.py" > "$d/bw-stack11-base.toml"
    cmp -s "$d/bw-stack11-base.toml" "$WIN/bw-stack11-base.toml" || { echo 'bw-stack11: FAIL bw-stack11-base.toml is stale (rerun mk_window_toml.py)'; exit 8; }
    python3 "$WIN/check_pins.py" "$WIN/bw-stack11-base.toml"
    gpuq submit --check "$WIN/bw-stack11-base.toml" > "$dry/bw-stack11-base-contract.log" 2>&1 || { cat "$dry/bw-stack11-base-contract.log"; echo 'bw-stack11: FAIL window toml contract'; exit 8; }
    cat "$dry/bw-stack11-base-contract.log"
    local ev=present
    mac_evidence > /dev/null 2>&1 || ev=MISSING
    echo "bw-stack11: DRY_RUN PASS: preflight, $(list_stages | wc -l) stages listed, selftests, KLD stub, $(ls "$dry"/*.gpuq.toml | wc -l) dry items + window toml checked (NOT submitted); Mac evidence $ev$([[ $ev == present ]] || echo ' (required in the real run)'); GPU stages and compiles not run"
}

if [[ ${1:-} == --list-stages ]]; then list_stages; exit 0; fi
# gpuq [selftest]: the gate judges on recorded good inputs (must pass) and their broken copies (must fail), CPU only.
if [[ ${1:-} == --selftest ]]; then
    python3 "$WIN/full_checks.py" --fixtures && python3 "$WIN/engagement.py" selftest && python3 "$WIN/s11items.py" selftest \
        && python3 "$WIN/health.py" --selftest && python3 "$WIN/sass_gate.py" --selftest && python3 "$WIN/fa_q8_perf.py" selftest \
        && python3 "$WIN/k10_perf.py" selftest && { echo 'bw-stack11: PASS selftest'; exit 0; }
    echo 'bw-stack11: FAIL selftest'; exit 8
fi
if [[ ${1:-} == --preflight ]]; then
    if preflight; then echo 'bw-stack11: PASS preflight'; exit 0; fi
    echo 'bw-stack11: FAIL preflight'; exit 8
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
preflight || { echo 'bw-stack11: FAIL preflight'; exit 8; }
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short=9 HEAD)
STARTED=$(date '+%Y-%m-%d %H:%M:%S')
echo "bw-stack11: components=[$COMPONENTS] commit=$SHA visible=$VISIBLE"
[[ $VISIBLE == GPU-*,GPU-* ]] || { echo 'bw-stack11: STOP no two GPU UUIDs in CUDA_VISIBLE_DEVICES'; exit 4; }

for row in "${STAGES[@]}"; do
    stage_fields "$row" || continue
    "${F[0]}" "${F[1]}" "${F[2]}" "${F[3]}"
done
gate 66 full-suite g_full
# The server-launching stages go last and only on a clean sheet: they cost the most GPU time and add no drop-map information once
# another gate has failed.
if [[ ${#FAILS[@]} == 0 ]]; then
    if has k11a; then gate 63 k11a-sync-count g_k11a; fi
    gate 69 health-boots g_health
else
    echo 'bw-stack11: k11a-sync-count and health-boots skipped, an earlier gate failed'
fi
if [[ ${#FAILS[@]} != 0 ]]; then finish_fail; fi

fatal 70 freeze-stack11-base freeze
fatal 71 kld-dump-build kld_build
fatal 72 render-and-check-items items
python3 - "$OUT" "$SHA" "$COMPONENTS" <<'PY_RESULT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
stages = [dict(code=int(c), name=n, rc=int(rc), wall_s_M=int(t)) for c, n, rc, t in (l.split('\t') for l in (p/'stages.tsv').read_text().splitlines())]
(p/'summary.json').write_text(json.dumps(dict(result='PASS', commit=sys.argv[2], components=sys.argv[3].split(), GPU_execution=True,
                                              stages=stages, frozen='frozen/stack11-base',
                                              items='codex/stack11/window/rendered; checked, NOT submitted'), indent=2)+'\n')
PY_RESULT
echo 'bw-stack11: PASS; frozen as stack11-base and items checked, NOT submitted'
