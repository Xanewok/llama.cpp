#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 8
OUT=${1:?output}
mkdir -p "$OUT"
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    [[ -z ${K2_ONLY:-} || " $K2_ONLY " == *" $code "* ]] || return 0
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-s10-k2: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit "$code"
}
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
build() {
    "$BW_BUILDGATE" cmake -S . -B build-cpu "${flags[@]}" "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-cpu --target test-backend-ops test-fattn-kv-max-cache -j4
}
units() {
    "$BW_GATE" build-cpu/bin/test-fattn-kv-max-cache
    "$BW_GATE" build-cpu/bin/test-backend-ops test -b CPU -o FLASH_ATTN_EXT -p 'shared_mask=1' -j 1 > "$OUT/tbo-shared-mask.log" 2>&1
    grep -aq 'FLASH_ATTN_EXT shared mask: same-backend byte identity, replay and overwrite PASS cases=5' "$OUT/tbo-shared-mask.log"
    grep -aEc '19/19 tests passed' "$OUT/tbo-shared-mask.log" > /dev/null
}
# Each mutant of the host bookkeeping must fail the unit test, and each mutant of the identity check must fail on the CPU.
mutants() {
    local libs=(-Lbuild-cpu/bin -lggml -lggml-base -lggml-cpu "-Wl,-rpath,$PWD/build-cpu/bin")
    mkdir -p "$OUT/mut"
    python3 - "$OUT/mut" <<'PY_MUTANTS'
import sys
from pathlib import Path
out = Path(sys.argv[1])
h = Path('ggml/src/ggml-cuda/fattn-kv-max-cache.h').read_text()
muts = {
    'invalidate-noop': ('g.valid = false;', ';'),
    'overlap-boundary': ('mask - start < size :', 'mask - start <= size :'),
    'ncols1-ignored': ('if (g.ncols1 != 0 && g.ncols1 != ncols1) {', 'if (false) {'),
    'iter-k-ignored': (' && g.iter_k == K->ne[1] / GGML_CUDA_FATTN_KV_MAX_STRIDE;', ';'),
    'stride-ignored': ('g.mask_nb1 == mask->nb[1] && g.mask_nb3 == mask->nb[3] &&', ''),
    'sequences-ignored': ('g.nq == Q->ne[1] && g.ns == Q->ne[3] &&', 'g.nq == Q->ne[1] &&'),
    'single-user-kept': ('g.uses < 2', 'g.uses < 1'),
    'single-sequence-scanned': ('Q->ne[3] > 1)', 'Q->ne[3] >= 1)'),
    'slot-unaligned': ('(n + 63) / 64 * 64', 'n'),
    'mask-end-ignored': ('start - mask < g.mask_bytes', 'start - mask < 1'),
}
for name, (old, new) in muts.items():
    assert h.count(old) == 1, name
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / 'fattn-kv-max-cache.h').write_text(h.replace(old, new))
(out / 'names.txt').write_text('\n'.join(muts) + '\n')
t = Path('tests/test-backend-ops.cpp').read_text()
tmuts = {
    # the alias reference keeps the pre-overwrite bytes, as a stale scan would
    'identity-stale-reference': ('init_kq_mask_lens(own[l], c.nb, l < half ? lens : lens_next);', 'init_kq_mask_lens(own[l], c.nb, lens);'),
    # the layers after the overwrite read the unchanged mask
    'identity-no-overwrite': ('return l < half ? mask : written;', 'return mask;'),
}
for name, (old, new) in tmuts.items():
    assert t.count(old) == 1, name
    (out / (name + '.cpp')).write_text(t.replace(old, new))
(out / 'tnames.txt').write_text('\n'.join(tmuts) + '\n')
PY_MUTANTS
    local name
    while read -r name; do
        c++ -std=c++17 -O1 -I "$OUT/mut/$name" -I ggml/include tests/test-fattn-kv-max-cache.cpp -o "$OUT/mut/$name/test" "${libs[@]}"
        if "$BW_GATE" "$OUT/mut/$name/test" > "$OUT/mut/$name/run.log" 2>&1; then
            echo "FAIL unit mutant escaped: $name"; return 1
        fi
        grep -aq 'FAIL' "$OUT/mut/$name/run.log"
        echo "unit mutant killed: $name"
    done < "$OUT/mut/names.txt"
    while read -r name; do
        cp "$OUT/mut/$name.cpp" tests/_mutant-tbo.cpp
        c++ -std=c++17 -O1 -I ggml/include -I ggml/src tests/_mutant-tbo.cpp -o "$OUT/mut/$name.bin" "${libs[@]}" -pthread
        rm -f tests/_mutant-tbo.cpp
        if "$BW_GATE" "$OUT/mut/$name.bin" test -b CPU -o FLASH_ATTN_EXT -p 'shared_mask=1' -j 1 > "$OUT/mut/$name.log" 2>&1; then
            echo "FAIL identity mutant escaped: $name"; return 1
        fi
        grep -aq 'FAIL FLASH_ATTN_EXT shared mask' "$OUT/mut/$name.log"
        echo "identity mutant killed: $name"
    done < "$OUT/mut/tnames.txt"
}
static_checks() {
    bash -n codex/bw-s10-k2-mac.sh codex/bw-s10-k2.sh
    python3 -m py_compile codex/s10_k2_perf.py
    PYTHONPATH="$ROOT/codex/shlint" "$BW_GATE" python3 -m gpuq.shlint codex/bw-s10-k2-mac.sh codex/bw-s10-k2.sh
}
stage 20 cpu-build build
stage 21 unit-tests units
stage 22 mutants mutants
stage 31 static static_checks
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
