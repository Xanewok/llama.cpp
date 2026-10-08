# K2 shared flash-attention KV-max scan. Codes 70-79.
TBO_LIBS=(-Lbuild-cpu/bin -lggml -lggml-base -lggml-cpu "-Wl,-rpath,$ROOT/build-cpu/bin")
units() {
    "$BW_GATE" build-cpu/bin/test-fattn-kv-max-cache
}
shared_mask_identity() {
    "$BW_GATE" build-cpu/bin/test-backend-ops test -b CPU -o FLASH_ATTN_EXT -p 'shared_mask=1' -j 1 > "$D/tbo-shared-mask.out" 2>&1
    grep -aq 'FLASH_ATTN_EXT shared mask: same-backend byte identity, replay and overwrite PASS cases=5' "$D/tbo-shared-mask.out"
    grep -aEc '19/19 tests passed' "$D/tbo-shared-mask.out" > /dev/null
}
# Each mutant of the host bookkeeping must fail the unit test, and each mutant of the identity check must fail on the CPU.
gen_mutants() {
    mkdir -p "$D/mut"
    python3 - "$D/mut" <<'PY_MUTANTS'
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
    assert h.count(old) == 1, f'anchor moved: {name}'
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
    assert t.count(old) == 1, f'anchor moved: {name}'
    (out / (name + '.cpp')).write_text(t.replace(old, new))
(out / 'tnames.txt').write_text('\n'.join(tmuts) + '\n')
PY_MUTANTS
}
unit_mutants() {
    gen_mutants
    local name
    while read -r name; do
        c++ -std=c++17 -O1 -I "$D/mut/$name" -I ggml/include tests/test-fattn-kv-max-cache.cpp -o "$D/mut/$name/test" "${TBO_LIBS[@]}"
        if "$BW_GATE" "$D/mut/$name/test" > "$D/mut/$name/run.log" 2>&1; then
            echo "FAIL unit mutant escaped: $name"; return 1
        fi
        grep -aq 'FAIL' "$D/mut/$name/run.log"
        echo "unit mutant killed: $name: $(grep -a 'FAIL' "$D/mut/$name/run.log" | sed -n 1p)"
    done < "$D/mut/names.txt"
}
identity_mutants() {
    gen_mutants
    local name
    while read -r name; do
        c++ -std=c++17 -O1 -I ggml/include -I ggml/src -I tests "$D/mut/$name.cpp" -o "$D/mut/$name.bin" "${TBO_LIBS[@]}" -pthread
        if "$BW_GATE" "$D/mut/$name.bin" test -b CPU -o FLASH_ATTN_EXT -p 'shared_mask=1' -j 1 > "$D/mut/$name.out" 2>&1; then
            echo "FAIL identity mutant escaped: $name"; return 1
        fi
        grep -aq 'FAIL FLASH_ATTN_EXT shared mask' "$D/mut/$name.out"
        echo "identity mutant killed: $name"
    done < "$D/mut/tnames.txt"
}
k2_cheap() {
    stage 70 kv-max-cache-unit units
    stage 71 shared-mask-identity shared_mask_identity
    stage 72 kv-max-unit-mutants unit_mutants
    stage 73 shared-mask-identity-mutants identity_mutants
}
