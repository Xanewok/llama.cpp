#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 8
OUT=${1:?output}
mkdir -p "$OUT"
GATE=$ROOT/codex/mac-model-gate.sh
MODEL=${BW_MODEL:?}
unset LLAMA_KV_ROT_PERSIST LLAMA_KV_ROT_ALLOC_FAIL LLAMA_KV_ROT_MUTANT
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    [[ -z ${S10_ONLY:-} || " $S10_ONLY " == *" $code "* ]] || return 0
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-stack10-cand: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit "$code"
}
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
build() {
    "$BW_BUILDGATE" cmake -S codex/stack7-test -B build-cpu "${flags[@]}" "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-cpu --target test-backend-ops test-sampling test-alloc test-recurrent-state-rollback bw-np2-driver -j4
}
units() {
    "$GATE" build-cpu/bin/test-sampling
    "$GATE" build-cpu/bin/test-alloc
    "$GATE" build-cpu/bin/test-backend-ops test -b CPU -o ADD_SOFTPLUS_MUL -j 1
    "$GATE" build-cpu/bin/test-backend-ops test -b CPU -o FLASH_ATTN_EXT -p 'hsk=(64|96|128|256),.*kv=2048,nb=1,.*type_K=q8_0,type_V=q8_0' -j 1
}
rotations() {
    mkdir -p "$OUT/rot"
    for arm in off on fail; do
        case $arm in
            off) envs=(LLAMA_KV_ROT_PERSIST=0) ;;
            on) envs=(LLAMA_KV_ROT_PERSIST=1) ;;
            fail) envs=(LLAMA_KV_ROT_ALLOC_FAIL=1) ;;
        esac
        "$GATE" env "${envs[@]}" build-cpu/bin/test-recurrent-state-rollback --getter-gate --getter-logits "$OUT/rot/$arm.bin" \
            -m "$MODEL" -c 4096 -t 4 -ngl 0 -ctk q8_0 -ctv q8_0 -lv 5 > "$OUT/rot/$arm.log" 2>&1
    done
    cmp "$OUT/rot/off.bin" "$OUT/rot/on.bin"
    cmp "$OUT/rot/off.bin" "$OUT/rot/fail.bin"
    grep -q 'persistent rotations backend=CPU' "$OUT/rot/on.log"
    grep -q 'rotation allocation failed; using graph inputs' "$OUT/rot/fail.log"
    if "$GATE" env LLAMA_KV_ROT_MUTANT=1 build-cpu/bin/test-recurrent-state-rollback --getter-gate --getter-reference "$OUT/rot/off.bin" \
        -m "$MODEL" -c 4096 -t 4 -ngl 0 -ctk q8_0 -ctv q8_0 -lv 5 > "$OUT/rot/mutant.log" 2>&1; then
        echo 'FAIL rotation mutant escaped'; return 1
    fi
    grep -q 'rotation matrix bytes differ from the Hadamard reference' "$OUT/rot/mutant.log"
    for meta in 0 1; do
        for arm in off on fail; do
            case $arm in
                off) envs=(LLAMA_KV_ROT_PERSIST=0) ;;
                on) envs=(LLAMA_KV_ROT_PERSIST=1) ;;
                fail) envs=(LLAMA_KV_ROT_ALLOC_FAIL=1) ;;
            esac
            "$GATE" env "${envs[@]}" HOST_KV_Q8=1 NP2_THREADS=2 LLAMA_MTP_CHAIN=0 LLAMA_MTP_FOLD_C=1 \
                build-cpu/bin/bw-np2-driver "$MODEL" "$OUT/rot/$meta-$arm.json" "$meta" 2 1729 64 > "$OUT/rot/$meta-$arm.log" 2>&1
        done
    done
    python3 - "$OUT/rot" <<'PY_COMPARE'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
for meta in (0, 1):
    rows = [json.loads((p/f'{meta}-{arm}.json').read_text()) for arm in ('off','on','fail')]
    assert rows[0] == rows[1] == rows[2], 'rotation tokens/counters differ'
    assert 'persistent rotations backend=' in (p/f'{meta}-on.log').read_text()
print('rotation plain/Meta tokens and counters PASS')
PY_COMPARE
    echo 'rotation raw logits, fallback, negative control PASS'
}
static_checks() {
    bash -n codex/bw-stack10-cand-mac.sh codex/bw-stack10-cand.sh codex/gpu_ab.sh
    python3 -m py_compile codex/gpu_stack10.py codex/render_stack10_items.py
    PYTHONPATH="$ROOT/codex/shlint" "$BW_GATE" python3 -m gpuq.shlint $(git ls-files "codex/*.sh")
    S9_EVAL="$HOME/qwen38-bw/stack8/bdae9b6aa3f29c761b70e6a133a7a0a54bbc3c9a/run.935rrX/src/build-parser-inputs" HARNESS_CODE_ROOT="$ROOT/codex/parser-harness" "$BW_GATE" python3 -P codex/gpu_stack10.py --fixtures
}
stage 20 cpu-build build
stage 21 unit-tests units
stage 22 rotation-identity rotations
stage 31 recorded-parser-controls static_checks
python3 - "$OUT" <<'PY_SUMMARY'
import hashlib, json, subprocess, sys
from pathlib import Path
p = Path(sys.argv[1])
files = subprocess.check_output(['git','ls-files','src','ggml','tests','common'], text=True).splitlines()
h = hashlib.sha256()
for name in files:
    h.update(name.encode()+b'\0'+Path(name).read_bytes())
stages = [dict(code=int(c),name=n,rc=int(rc),wall_s_M=int(t)) for c,n,rc,t in (l.split('\t') for l in (p/'stages.tsv').read_text().splitlines())]
payload_files = sorted(set(['codex/bw-stack10-cand-mac.sh','codex/bw-stack10-cand-static-mac.sh','codex/bw-stack10-cand.sh','codex/bw-stack10-cand.toml','codex/gpu_ab.sh','codex/gpu_stack10.py','codex/gpu_stack9.py','codex/render_stack10_items.py','codex/mac-model-gate.sh'] + [str(f) for pattern in ('codex/items/s10c-*.toml.in','codex/parser-harness/harness/*.py','codex/shlint/gpuq/*.py') for f in Path('.').glob(pattern)]))
ph = hashlib.sha256()
for name in payload_files:
    ph.update(name.encode()+b'\0'+Path(name).read_bytes())
(p/'summary.json').write_text(json.dumps(dict(result='PASS',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_files=files,source_sha256=h.hexdigest(),payload_files=payload_files,payload_sha256=ph.hexdigest(),stages=stages),indent=2)+'\n')
PY_SUMMARY
