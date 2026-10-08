# gdn4 (4 GDN columns per warp, CUDA-only kernel) and the fusion-pair perf rows in the harness. Codes 130-139.
# The old-harness perf control is compiled from a patched copy of tests/test-backend-ops.cpp; the tree itself stays untouched.
T=tests/test-backend-ops.cpp
CHECK=codex/s9gdn4/check_ops.py
TBO_LIBS=(-Lbuild-cpu/bin -lggml -lggml-base -lggml-cpu "-Wl,-rpath,$ROOT/build-cpu/bin")
# CPU against CPU: the indexed cases still check indexed-versus-gathered inside one backend, and every new graph builds and runs.
cases() {
    "$BW_GATE" build-cpu/bin/test-backend-ops test -b CPU -o 'GATED_DELTA_NET.*' -j 1 > "$D/cases.out" 2>&1
    python3 "$CHECK" passed "$D/cases.out" 70
    grep -Ec '^ +GATED_DELTA_NET_INDEXED_CACHE_FUSION\(.*head_size=128,n_seq_tokens=6,n_seqs=[1234],K=6.*OK' "$D/cases.out" | tee "$D/fusion128.count"
    [[ $(cat "$D/fusion128.count") -ge 8 ]]
}
# perf mode: the repeated nodes of the fusion cases are gdn + cpy, so the row must now cost far more than the old sum-only row.
perf_rows() {
    "$BW_GATE" build-cpu/bin/test-backend-ops perf -b CPU -o 'GATED_DELTA_NET.*' -p 'head_count=24,' > "$D/perf-new.out" 2>&1
    python3 "$CHECK" perf "$D/perf-new.out" 12
    # the old harness: the fusion cases repeat only their output node (ggml_sum)
    [[ $(grep -cF 'return { gdn_node, cpy_node };' "$T") == 1 ]] || { echo 'FAIL perf anchor moved in tests/test-backend-ops.cpp'; return 1; }
    perl -pe 's#return \{ gdn_node, cpy_node \};#return { out };#' "$T" > "$D/old-tbo.cpp"
    [[ $(diff "$T" "$D/old-tbo.cpp" | grep -c '^>') == 1 ]]
    c++ -std=c++17 -O1 -I ggml/include -I ggml/src -I tests "$D/old-tbo.cpp" -o "$D/old-tbo" "${TBO_LIBS[@]}" -pthread
    "$BW_GATE" "$D/old-tbo" perf -b CPU -o 'GATED_DELTA_NET.*' -p 'head_count=24,' > "$D/perf-old.out" 2>&1
    python3 "$CHECK" perf "$D/perf-old.out" 12
    python3 - "$D" <<'PY'
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
new, old = load('perf-new.out'), load('perf-old.out')
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
gdn4_cheap() {
    stage 130 gdn-cases cases
    stage 131 perf-repeats-the-pair perf_rows
}
