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
unset GGML_SSM_CONV_RS_OFF GGML_SSM_CONV_RS_MUTANT GGML_SSM_CONV_RS_LOG
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    [[ -z ${K4_ONLY:-} || " $K4_ONLY " == *" $code "* ]] || return 0
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-k4: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit "$code"
}
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
build() {
    "$BW_BUILDGATE" cmake -S codex/stack7-test -B build-cpu "${flags[@]}" "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-cpu --target test-backend-ops test-recurrent-state-rollback bw-np2-driver -j4
}
# The identity run must engage (cases > 0) and the off switch must hand every case back (cases = 0).
ops() {
    "$GATE" build-cpu/bin/test-backend-ops test -b CPU -o SSM_CONV_RS -j 1 | tee "$OUT/ops-on.txt"
    grep -aq 'byte identity with the gathered path and changing-id replay PASS cases=[1-9]' "$OUT/ops-on.txt"
    "$GATE" build-cpu/bin/test-backend-ops test -b CPU -o SSM_CONV -j 1 > "$OUT/ops-ssm-conv.txt"
    "$GATE" env GGML_SSM_CONV_RS_OFF=1 build-cpu/bin/test-backend-ops test -b CPU -o SSM_CONV_RS -j 1 | tee "$OUT/ops-off.txt"
    grep -aq 'PASS cases=0 ' "$OUT/ops-off.txt"
}
ops_mutants() {
    local m
    for m in 1 2; do
        if "$GATE" env GGML_SSM_CONV_RS_MUTANT=$m build-cpu/bin/test-backend-ops test -b CPU -o SSM_CONV_RS -j 1 > "$OUT/ops-mutant$m.txt" 2>&1; then
            echo "FAIL ops mutant $m escaped"; return 1
        fi
        grep -aq 'FAIL SSM_CONV_RS byte identity' "$OUT/ops-mutant$m.txt"
    done
}
# arm NAME RS_SEQ EXPECT ENV...: writes $OUT/model/NAME.bin; with REF set compares against that file.
arm() {
    local name=$1 rs_seq=$2 expect=$3 ref=$4
    shift 4
    local cmp=(--getter-logits "$OUT/model/$name.bin")
    [[ -z $ref ]] || cmp+=(--getter-reference "$OUT/model/$ref.bin")
    "$GATE" env "$@" build-cpu/bin/test-recurrent-state-rollback --getter-gate "${cmp[@]}" --conv-expect "$expect" \
        --getter-rs-seq "$rs_seq" -m "$MODEL" -c 4096 -t 4 -ngl 0 -lv 5 > "$OUT/model/$name.log" 2>&1
}
model() {
    mkdir -p "$OUT/model"
    arm gathered8 8 gathered '' GGML_SSM_CONV_RS_OFF=1
    arm rs8 8 rs gathered8 GGML_SSM_CONV_RS_LOG=0
    cmp "$OUT/model/gathered8.bin" "$OUT/model/rs8.bin"
    arm gathered0 0 gathered '' GGML_SSM_CONV_RS_OFF=1
    arm rs0 0 rs gathered0 GGML_SSM_CONV_RS_LOG=0
    cmp "$OUT/model/gathered0.bin" "$OUT/model/rs0.bin"
    local m
    for m in 1 2; do
        if arm mutant$m 8 rs gathered8 GGML_SSM_CONV_RS_MUTANT=$m; then
            echo "FAIL model mutant $m escaped"; return 1
        fi
        grep -aq 'logits differ' "$OUT/model/mutant$m.log"
    done
}
np2() {
    mkdir -p "$OUT/np2"
    local meta
    for meta in 0 1; do
        "$GATE" env GGML_SSM_CONV_RS_OFF=1 HOST_KV_Q8=1 NP2_THREADS=2 LLAMA_MTP_CHAIN=0 LLAMA_MTP_FOLD_C=1 \
            build-cpu/bin/bw-np2-driver "$MODEL" "$OUT/np2/$meta-off.json" "$meta" 2 1729 64 > "$OUT/np2/$meta-off.log" 2>&1
        "$GATE" env HOST_KV_Q8=1 NP2_THREADS=2 LLAMA_MTP_CHAIN=0 LLAMA_MTP_FOLD_C=1 \
            build-cpu/bin/bw-np2-driver "$MODEL" "$OUT/np2/$meta-on.json" "$meta" 2 1729 64 > "$OUT/np2/$meta-on.log" 2>&1
        cmp "$OUT/np2/$meta-off.json" "$OUT/np2/$meta-on.json"
    done
}
static_checks() {
    bash -n codex/bw-k4-mac.sh codex/bw-k4.sh
    PYTHONPATH="$ROOT/codex/shlint" "$BW_GATE" python3 -m gpuq.shlint codex/bw-k4-mac.sh codex/bw-k4.sh
}
stage 20 cpu-build build
stage 21 ops-identity ops
stage 22 ops-mutants ops_mutants
stage 23 model-gate model
stage 24 np2-driver np2
stage 31 static-checks static_checks
python3 - "$OUT" <<'PY_SUMMARY'
import hashlib, json, subprocess, sys
from pathlib import Path
p = Path(sys.argv[1])
files = subprocess.check_output(['git','ls-files','src','ggml','tests','common'], text=True).splitlines()
h = hashlib.sha256()
for name in files:
    h.update(name.encode()+b'\0'+Path(name).read_bytes())
stages = [dict(code=int(c),name=n,rc=int(rc),wall_s_M=int(t)) for c,n,rc,t in (l.split('\t') for l in (p/'stages.tsv').read_text().splitlines())]
(p/'summary.json').write_text(json.dumps(dict(result='PASS',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_files=files,source_sha256=h.hexdigest(),stages=stages),indent=2)+'\n')
PY_SUMMARY
