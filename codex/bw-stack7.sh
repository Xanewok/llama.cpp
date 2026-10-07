#!/usr/bin/env bash
# All builds and CPU executions are deferred to this gpuq filler payload.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export CPUGATE_MIN_GIB=10 BUILDGATE_MIN_GIB=10
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=8
export PATH="$HOME/.local/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2
unset LLAMA_MTP_CHAIN LLAMA_MTP_FOLD_C LLAMA_MTP_FOLD_DEBUG_MUTANT LLAMA_MTP_DRAFT_DUMP LLAMA_MTP_CHAIN_DUMP
unset LLAMA_MTP_CHAIN_DEBUG_REVERSE_U LLAMA_MTP_CHAIN_DEBUG_SHIFT_EMBD LLAMA_ROUND_TRACE GGML_ROUND_TRACE_SYNC
unset LLAMA_HOST_TRIMS_SEQRM LLAMA_HOST_TRIMS_BULKH LLAMA_HOST_TRIMS_VERIFY LLAMA_HOST_TRIMS_MUTANT LLAMA_HOST_TRIMS_PROOF LLAMA_STACK7_PROOF
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
cd "$ROOT"
W=$HOME/repos/qwen38-eval-worktrees
EVAL=$HOME/repos/qwen38-eval
BUILDGATE=$ROOT/codex/build.sh
CPUGATE=$W/bin/cpugate
CCCL=/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3
MODEL=/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf
STACK6=ac7465885dc9f13de396ccd91b4be5eaffde3730
trap 'echo "bw-stack7: FAIL setup rc=8"; exit 8' ERR
trap 'echo "bw-stack7: FAIL interrupted rc=9"; exit 9' INT TERM
preflight() {
    for path in "$MODEL" "$CCCL" "$W/bin/buildgate" "$CPUGATE" "$W/bin/wt" \
                "$W/work/build-window-1007/bw-run.sh" "$W/frozen/stack6/RECIPE.txt" \
                "$EVAL/harness/mkitem.py" "$EVAL/orchestration/gpuq/HARNESS-CONTRACT.md" \
                /usr/local/cuda-13.2/bin/nvcc \
                /home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json \
                "$EVAL/recipes/_common/kld/tokens64k.bin"; do
        [[ -e $path ]] || return 1
    done
    for tool in gpuq cmake ccache g++ git strings python3 nice timeout patchelf flock; do
        command -v "$tool" >/dev/null || return 1
    done
    [[ $(git branch --show-current) == topic/stack7 && ! -e $W/frozen/stack7 ]] || return 1
    git merge-base --is-ancestor "$STACK6" HEAD || return 1
    [[ -z $(git status --porcelain --untracked-files=no) ]] || return 1
    for script in codex/{build,bw-stack7,check-getter,check-host,check-np2-interactions,gpu_ab}.sh; do bash -n "$script" || return 1; done
    python3 - <<'PY' || return 1
from pathlib import Path
import tomllib
for name in ('chain_cpu_identity.py', 'fold_cpu_identity.py', 'bw-np2-compare.py', 'check-chain.py', 'check-interactions.py', 'check_trace_writer.py', 'test_nh1_meta_engagement.py', 'gpu_stack7.py', 'make_stack7_items.py'):
    p = Path('codex') / name
    compile(p.read_text(), str(p), 'exec')
for p in Path('codex/items').glob('*.toml.in'):
    tomllib.loads(p.read_text())
PY
}
if [[ ${1:-} == --preflight ]]; then
    if preflight; then echo 'bw-stack7: PASS preflight'; exit 0; fi
    echo 'bw-stack7: FAIL preflight rc=10'; exit 10
fi
[[ ${GPUQ_BUILD_WINDOW:-0} == 1 && $# == 1 ]] || exit 8
OUT=$(readlink -m "$1")
case $OUT in "$W"/work/build-window-1007/stack7-a[0-9]*) ;; *) echo 'wrong wrapper output path'; exit 10 ;; esac
mkdir -p "$OUT"
run_stage() {
    local name=$1 code=$2 rc
    shift 2
    trap - ERR
    set +e
    (trap - ERR; set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    trap 'echo "bw-stack7: FAIL setup rc=8"; exit 8' ERR
    printf '%s\t%s\t%s\n' "$name" "$code" "$rc" >> "$OUT/stages.tsv"
    if [[ $rc != 0 ]]; then
        echo "bw-stack7: FAIL $name rc=$code command_rc=$rc log=$OUT/$name.log"
        exit "$code"
    fi
    echo "bw-stack7: PASS $name log=$OUT/$name.log"
}
run_stage preflight 10 preflight
SHA=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)
printf '%s\n' "$SHA" > "$OUT/source-commit.txt"
mkdir -p build-cpu
exec 9> build-cpu/.stack7-model.lock
flock -n 9 || exit 10
cpu_flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON)
cpu_build() {
    "$BUILDGATE" cmake -S codex/stack7-test -B build-cpu "${cpu_flags[@]}"
    "$BUILDGATE" cmake --build build-cpu --target test-recurrent-state-rollback test-alloc bw-np2-driver chain-cpu fold-cpu outproj-cpu row0-mutant -j4
    mkdir -p build-stack6-source
    git archive "$STACK6" | tar -x -C build-stack6-source
    "$BUILDGATE" cmake -S codex/stack7-test -B build-stack6-cpu -DLLAMA_SOURCE="$ROOT/build-stack6-source" "${cpu_flags[@]}"
    "$BUILDGATE" cmake --build build-stack6-cpu --target outproj-cpu -j4
    mkdir -p build-trace
    git show "$STACK6:ggml/src/ggml-round-trace.cpp" > build-trace/old-round-trace.cpp
    python3 - <<'PY'
from pathlib import Path
s = Path('ggml/src/ggml-round-trace.cpp').read_text()
old = 'void ggml_round_trace_flush() {\n    if (ggml_round_trace_enabled()) {\n        trace_flush(0);'
new = 'void ggml_round_trace_flush() {\n    if (ggml_round_trace_enabled()) {'
assert s.count(old) == 1
Path('build-trace/mutant-round-trace.cpp').write_text(s.replace(old, new))
PY
    "$BUILDGATE" cmake -S codex/trace-test -B build-trace -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DCMAKE_BUILD_TYPE=Release
    "$BUILDGATE" cmake --build build-trace -j4
}
run_stage cpu-build 20 cpu_build
run_stage getter-gate-and-mutants 21 "$CPUGATE" bash codex/check-getter.sh "$OUT/getter"
run_stage host-np2-exactness-and-mutants 22 "$CPUGATE" bash codex/check-host.sh "$OUT/host"
chain_check() {
    "$CPUGATE" python3 codex/chain_cpu_identity.py --out "$OUT/chain"
    "$CPUGATE" python3 codex/check-chain.py "$OUT/chain"
}
run_stage chain-identity-and-controls 23 chain_check
run_stage trace-byte-identity-and-control 24 "$CPUGATE" python3 codex/check_trace_writer.py build-trace "$OUT/trace"
run_stage fold-plain-and-meta-identity 25 "$CPUGATE" python3 codex/fold_cpu_identity.py "$OUT/fold"
run_stage fold-engagement-negative-control 26 "$CPUGATE" python3 codex/test_nh1_meta_engagement.py "$OUT/fold/meta-k3-off"
run_stage meta-planner-test-alloc 27 "$CPUGATE" build-cpu/bin/test-alloc
run_stage fold-chain-interaction 28 "$CPUGATE" python3 codex/check-interactions.py "$OUT/interactions"
run_stage np2-chain-fold-gates 29 "$CPUGATE" bash codex/check-np2-interactions.sh "$OUT/np2-interactions"
outproj_check() {
    mkdir -p "$OUT/outproj"
    local rc=0
    "$CPUGATE" build-stack6-cpu/bin/outproj-cpu "$MODEL" "$OUT/outproj/stack6.bin" 1 > "$OUT/outproj/stack6.log" 2>&1 || rc=$?
    [[ $rc == 42 ]]
    grep -E 'outproj_graph: nodes=[1-9][0-9]* batched=[1-9][0-9]* n_seqs=2' "$OUT/outproj/stack6.log" >/dev/null
    "$CPUGATE" build-cpu/bin/outproj-cpu "$MODEL" "$OUT/outproj/stack7.bin" 1 > "$OUT/outproj/stack7.log" 2>&1
    grep -E 'outproj_graph: nodes=[1-9][0-9]* batched=0 n_seqs=2' "$OUT/outproj/stack7.log" >/dev/null
    cmp "$OUT/outproj/stack6.bin" "$OUT/outproj/stack7.bin"
}
run_stage outproj-np2-logits-tokens-and-rank-control 30 outproj_check
run_stage recorded-gpu-parser-controls 31 "$CPUGATE" python3 codex/gpu_stack7.py --fixtures
cuda_flags=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
    -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a
    -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL="$CCCL"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DGGML_CCACHE=ON)
cuda_build() {
    "$BUILDGATE" cmake -B build-cuda "${cuda_flags[@]}"
    "$BUILDGATE" cmake --build build-cuda --target llama-server -j4
    cp build-cuda/bin/llama-server build-cuda/bin/round-trace
}
run_stage cuda-build 40 cuda_build
stamps() {
    local n
    n=$(strings build-cuda/bin/libggml-cuda.so | grep -c DeviceTopK)
    echo "DeviceTopK_count_M=$n"
    [[ $n == 25 ]]
    strings build-cuda/bin/libggml-base.so | grep -x "$SHORT" >/dev/null
    [[ $(git rev-parse HEAD) == "$SHA" && -z $(git status --porcelain --untracked-files=no) ]]
}
run_stage cuda-stamps-and-clean-tree 41 stamps
freeze() {
    "$W/bin/wt" freeze "$ROOT" stack7 --build build-cuda --desc 'stack6 + gated getter, host trims v2, chain, async trace, NH1/Meta fix, 2D SSM projection; CPU gates passed; CUDA 13.2 120a CCCL343 NCCL OFF'
    local source
    IFS= read -r source < "$W/frozen/stack7/SOURCE"
    [[ $source == "$SHA" ]]
    printf 'commit %s\nconfigure: cmake -B build-cuda ' "$SHA" > "$OUT/RECIPE.txt"
    printf '%q ' "${cuda_flags[@]}" >> "$OUT/RECIPE.txt"
    printf '\nbuild: codex/build.sh cmake --build build-cuda --target llama-server -j4\nCUDA_VISIBLE_DEVICES=""; no CUDA binary executed\n' >> "$OUT/RECIPE.txt"
    cp "$OUT/RECIPE.txt" "$W/frozen/stack7/RECIPE.txt"
}
run_stage freeze-stack7 42 freeze
# The item pins this real wrapper attempt's evidence, never a codex/bw-results path.
python3 - "$OUT" "$SHA" <<'PY'
from pathlib import Path
import json,sys
out=Path(sys.argv[1])
stages=[dict(name=n,exit_code=int(c),command_rc=int(r)) for n,c,r in (l.split('\t') for l in (out/'stages.tsv').read_text().splitlines())]
assert all(s['command_rc']==0 for s in stages)
(out/'summary.json').write_text(json.dumps(dict(commit=sys.argv[2],stage1='PASS',stage2='PASS',stages=stages,cpu_numbers='See per-stage logs; no GPU measurements',items='pending contract validation'),indent=2)+'\n')
PY
run_stage generate-and-check-gpu-items 43 "$CPUGATE" python3 codex/make_stack7_items.py "$OUT/summary.json"
python3 - "$OUT" <<'PY'
from pathlib import Path
import json,sys
out=Path(sys.argv[1])
p=out/'summary.json'
r=json.loads(p.read_text())
r['stages']=[dict(name=n,exit_code=int(c),command_rc=int(v)) for n,c,v in (l.split('\t') for l in (out/'stages.tsv').read_text().splitlines())]
assert all(s['command_rc']==0 for s in r['stages'])
p.write_text(json.dumps(r,indent=2)+'\n')
PY
echo 'bw-stack7: PASS; stack7 frozen; GPU items checked, NOT submitted'
