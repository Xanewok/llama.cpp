#!/usr/bin/env bash
# Mac CPU gates for s9-gdn4 (kernel and test-harness change only; the CPU path is untouched). Run via bin/bw-mac.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 8
OUT=${1:?output}
mkdir -p "$OUT"
OLD_TESTS=codex/s9gdn4/base-test-backend-ops.cpp.orig  # stack9 cd776e104 tests/test-backend-ops.cpp; the bundle has no history
T=tests/test-backend-ops.cpp
CHECK=codex/s9gdn4/check_ops.py
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-s9-gdn4-mac: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit "$code"
}
[[ -z $(git status --porcelain --untracked-files=no) ]]
git rev-parse HEAD > "$OUT/source-commit.txt"
trap 'git checkout -q -- "$T"' EXIT
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
build() {
    "$BW_BUILDGATE" cmake -S . -B build-cpu "${flags[@]}" "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-cpu --target test-backend-ops -j4
}
# CPU against CPU: the indexed cases still check indexed-versus-gathered inside one backend, and every new graph builds and runs.
cases() {
    "$BW_GATE" build-cpu/bin/test-backend-ops test -b CPU -o 'GATED_DELTA_NET.*' -j 1 > "$OUT/cases.log" 2>&1
    python3 "$CHECK" passed "$OUT/cases.log" 70
    grep -Ec '^ +GATED_DELTA_NET_INDEXED_CACHE_FUSION\(.*head_size=128,n_seq_tokens=6,n_seqs=[1234],K=6.*OK' "$OUT/cases.log" | tee "$OUT/fusion128.count"
    [[ $(cat "$OUT/fusion128.count") -ge 8 ]]
}
# perf mode: the repeated nodes of the fusion cases are gdn + cpy, so the row must now cost far more than the old sum-only row.
perf_rows() {
    local shape='head_count=24,head_size=128,n_seq_tokens=6,n_seqs=1,'
    "$BW_GATE" build-cpu/bin/test-backend-ops perf -b CPU -o 'GATED_DELTA_NET.*' -p 'head_count=24,' > "$OUT/perf-new.log" 2>&1
    python3 "$CHECK" perf "$OUT/perf-new.log" 12
    cp "$OLD_TESTS" "$T"
    "$BW_BUILDGATE" cmake --build build-cpu --target test-backend-ops -j4
    "$BW_GATE" build-cpu/bin/test-backend-ops perf -b CPU -o 'GATED_DELTA_NET.*' -p 'head_count=24,' > "$OUT/perf-old.log" 2>&1
    python3 "$CHECK" perf "$OUT/perf-old.log" 12
    git checkout -q -- "$T"
    python3 - "$OUT" <<'PY'
import re, sys
from pathlib import Path
out = Path(sys.argv[1])
row = re.compile(r'^\s+(GATED_DELTA_NET\w*)\((.*)\):\s+(\d+) runs\s+-\s+([\d.]+) us/run')
def load(name):
    rows = {}
    for l in (out / name).read_text(errors='replace').splitlines():
        m = row.match(re.sub(r'\x1b\[[0-9;]*m', '', l))
        if m:
            rows[(m[1], m[2])] = float(m[4])
    return rows
new, old = load('perf-new.log'), load('perf-old.log')
assert set(new) == set(old) and len(new) >= 12, (len(new), len(old))
print('case\told_us\tnew_us\tnew/old')
ratios = {}
for (op, params), t in sorted(new.items()):
    r = t / old[(op, params)]
    print(f'{op}({params})\t{old[(op, params)]:.1f}\t{t:.1f}\t{r:.2f}')
    if op.endswith('CACHE_FUSION') and 'n_seq_tokens=6' in params:
        ratios[(op, params)] = r
    elif not op.endswith('CACHE_FUSION'):
        # untouched cases: same code path, the ratio is the CPU timing noise
        assert 0.33 < r < 3.0, (op, params, r)
assert ratios and all(r > 1.5 for r in ratios.values()), f'fusion rows did not grow: {ratios}'
print(f'fusion rows with 6 tokens grew {min(ratios.values()):.2f}..{max(ratios.values()):.2f}x (old rows timed ggml_sum)')
PY
}
stage 20 cpu-build build
stage 21 gdn-cases cases
stage 22 perf-repeats-the-pair perf_rows
python3 - "$OUT" <<'PY'
import hashlib, json, sys
from pathlib import Path
p = Path(sys.argv[1])
stages = [dict(code=int(c), name=n, rc=int(rc), wall_s_M=int(t)) for c, n, rc, t in (l.split('\t') for l in (p / 'stages.tsv').read_text().splitlines())]
files = ['ggml/src/ggml-cuda/gated_delta_net.cu', 'tests/test-backend-ops.cpp']
d = hashlib.sha256()
for name in files:
    d.update(name.encode() + b'\0' + Path(name).read_bytes())
cases = int(next(l for l in (p / 'cases.log').read_text().splitlines() if 'tests passed' in l).split('/')[1].split()[0])
(p / 'summary.json').write_text(json.dumps({'result': 'PASS', 'commit': (p / 'source-commit.txt').read_text().strip(), 'source_files': files,
                                           'source_sha256': d.hexdigest(), 'gdn_cases': cases, 'stages': stages}, indent=2) + '\n')
PY
echo 'bw-s9-gdn4-mac: PASS'
