#!/usr/bin/env bash
# CPU gate for --spec-draft-swa (bin/bw-mac payload). A tree without swa_window in include/llama.h is the base: it records the
# reference outputs (copy them into codex/swa/base-ref/). Otherwise: unit test, off == base reference, target logits on == off,
# drafts on != off, a window larger than the context == off, np1/np2/Meta server runs with prefix reuse, then the mutants.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
OUT=${1:?output}
mkdir -p "$OUT"
GATE=$ROOT/codex/mac-model-gate.sh
[[ $(uname) == Darwin ]] || GATE=${BW_GATE:?}
MODEL=${BW_MODEL:?}
read -r -a extra <<< "${BW_CPU_EXTRA:-}"
BIN=build-cpu/bin
REF=$ROOT/codex/swa/base-ref
base=1
grep -q swa_window include/llama.h && base=0

stage() {
    local name=$1 rc=0 t0=$SECONDS
    shift
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\n' "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-swa: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit 1
}

targets=(test-kv-mask bw-np2-driver swa-target)
[[ $base == 1 ]] || targets+=(test-kv-window)
build() {
    "${BW_BUILDGATE:-nice}" cmake -S codex/stack7-test -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON \
        -DGGML_OPENMP_FETCH=OFF -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_TESTS=ON \
        -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF "${extra[@]}"
    "${BW_BUILDGATE:-nice}" cmake --build build-cpu --target "${targets[@]}" -j "${BW_JOBS:-4}"
}

# target logits and drafts: OUT/tgt/<tag>.{tgt,dft}.txt
tgt() {
    local tag=$1 meta=$2 w=$3 s=$4
    mkdir -p "$OUT/tgt"
    "$GATE" "$BIN/swa-target" "$MODEL" "$OUT/tgt/$tag" "$meta" "$w" "$s" > "$OUT/tgt/$tag.log" 2>&1
    grep -c 'swa-target: .* done' "$OUT/tgt/$tag.log" > /dev/null
}

# server path, two phases (cold, prefix reuse): OUT/srv/<tag>.json
srv() {
    local tag=$1 meta=$2 np=$3 w=$4 s=$5
    mkdir -p "$OUT/srv"
    "$GATE" env SWA_W="$w" SWA_S="$s" NP2_THREADS=4 NP2_N_MAX=5 "$BIN/bw-np2-driver" "$MODEL" "$OUT/srv/$tag.json" "$meta" "$np" 1729 96 \
        > "$OUT/srv/$tag.log" 2>&1
}

reference() {
    tgt plain-w0 0 0 0
    tgt meta-w0 1 0 0
    srv np1-w0 0 1 0 0
    srv np2-w0 0 2 0 0
    srv np2meta-w0 1 2 0 0
}

same() { cmp "$1" "$2" && echo "SAME $1 $2"; }
differ() { if cmp -s "$1" "$2"; then echo "FAIL identical: $1 $2"; return 1; fi; echo "DIFFER $1 $2"; }

if [[ $base == 1 ]]; then
    stage build build
    stage reference reference
    exit 0
fi

unit() {
    "$GATE" "$BIN/test-kv-window" | tee "$OUT/unit-window.txt"
    grep -c '^test-kv-window: 48 windowed cache cases OK$' "$OUT/unit-window.txt" > /dev/null
    "$GATE" "$BIN/test-kv-mask"
}

off_vs_base() {
    reference
    for f in plain-w0.tgt.txt plain-w0.dft.txt meta-w0.tgt.txt meta-w0.dft.txt; do same "$OUT/tgt/$f" "$REF/tgt/$f"; done
    for f in np1-w0 np2-w0 np2meta-w0; do same "$OUT/srv/$f.json" "$REF/srv/$f.json"; done
}

window_on() {
    tgt plain-w16s4 0 16 4
    tgt meta-w16s4 1 16 4
    tgt plain-w16s0 0 16 0
    tgt plain-w4096s0 0 4096 0
    for t in plain-w16s4 plain-w16s0 plain-w4096s0; do same "$OUT/tgt/$t.tgt.txt" "$OUT/tgt/plain-w0.tgt.txt"; done
    same "$OUT/tgt/meta-w16s4.tgt.txt" "$OUT/tgt/meta-w0.tgt.txt"
    differ "$OUT/tgt/plain-w16s4.dft.txt" "$OUT/tgt/plain-w0.dft.txt"
    differ "$OUT/tgt/plain-w16s4.dft.txt" "$OUT/tgt/plain-w16s0.dft.txt"
    # nothing leaves a window wider than the context: the windowed cache must then draft exactly like the full one
    diff <(grep -v '^draft pos_min' "$OUT/tgt/plain-w4096s0.dft.txt") <(grep -v '^draft pos_min' "$OUT/tgt/plain-w0.dft.txt")
    grep -c 'swa_window            = 16, swa_sinks = 4, kv_size = 256' "$OUT/tgt/plain-w16s4.log" > /dev/null
    grep 'draft pos_min' "$OUT/tgt/plain-w16s4.dft.txt" "$OUT/tgt/plain-w0.dft.txt"
}

server_on() {
    srv np1-w32s4 0 1 32 4
    srv np2-w32s4 0 2 32 4
    srv np2meta-w32s4 1 2 32 4
    for t in np1 np2 np2meta; do
        grep -c 'swa_window            = 32, swa_sinks = 4' "$OUT/srv/$t-w32s4.log" > /dev/null
        python3 - "$OUT/srv/$t-w32s4.json" "$OUT/srv/$t-w0.json" <<'PY'
import json, sys
on, off = (json.load(open(p)) for p in sys.argv[1:])
for ph_on, ph_off in zip(on["phases"], off["phases"]):
    for a, b in zip(ph_on["stats"], ph_off["stats"]):
        print(f"phase {ph_on['phase']} slot {a['slot']}: on accepted/draft {a['accepted']}/{a['draft']} rounds {a['rounds']} cached {a['cached']}"
              f" | off {b['accepted']}/{b['draft']} rounds {b['rounds']} cached {b['cached']}")
assert on != off, "window changed nothing"
assert any(s["cached"] > 0 for s in on["phases"][1]["stats"]), "phase 2 did not reuse the prompt"
PY
    done
}

mutants() {
    cp "$OUT/tgt/plain-w16s4.dft.txt" "$OUT/correct-w16s4.dft.txt"
    for m in codex/swa/mutants/*.patch; do
        local name
        name=$(basename "$m" .patch)
        git checkout -- src
        git apply "$m"
        "${BW_BUILDGATE:-nice}" cmake --build build-cpu --target test-kv-window swa-target -j "${BW_JOBS:-4}" > "$OUT/mutant-$name-build.log" 2>&1
        if "$GATE" "$BIN/test-kv-window" > "$OUT/mutant-$name-unit.txt" 2>&1; then
            echo "FAIL mutant $name escaped test-kv-window"; git checkout -- src; return 1
        fi
        echo "mutant $name: $(grep -m1 'FAIL' "$OUT/mutant-$name-unit.txt")"
        tgt "mutant-$name" 0 16 4
        same "$OUT/tgt/mutant-$name.tgt.txt" "$OUT/tgt/plain-w0.tgt.txt"
        differ "$OUT/tgt/mutant-$name.dft.txt" "$OUT/correct-w16s4.dft.txt"
    done
    git checkout -- src
}

stage build build
stage unit unit
stage off-vs-base off_vs_base
stage window-on window_on
stage server-on server_on
stage mutants mutants
echo "bw-swa: PASS"
