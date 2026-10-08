# Base stack10-cand gates: sampling and allocator units, softplus identity, q8_0 flash-attention slice, persistent KV rotation
# identity (raw logits, fallback, negative control, np2 driver plain and Meta), recorded parser controls, payload lint. Codes 20-29.
unset LLAMA_KV_ROT_PERSIST LLAMA_KV_ROT_ALLOC_FAIL LLAMA_KV_ROT_MUTANT
units_plain() {
    "$BW_GATE" build-cpu/bin/test-sampling
    "$BW_GATE" build-cpu/bin/test-alloc
}
units_ops() {
    "$BW_GATE" build-cpu/bin/test-backend-ops test -b CPU -o ADD_SOFTPLUS_MUL -j 1 > "$D/add-softplus-mul.out" 2>&1
    grep -aq 'ADD_SOFTPLUS_MUL: same-backend byte identity and changing-input replay PASS cases=[1-9]' "$D/add-softplus-mul.out"
    "$BW_GATE" build-cpu/bin/test-backend-ops test -b CPU -o FLASH_ATTN_EXT -p 'hsk=(64|96|128|256),.*kv=2048,nb=1,.*type_K=q8_0,type_V=q8_0' -j 1 > "$D/fa-q8.out" 2>&1
    grep -aEq '^ +([1-9][0-9]*)/\1 tests passed' "$D/fa-q8.out"
}
rotations() {
    local arm meta envs
    mkdir -p "$D/rot"
    for arm in off on fail; do
        case $arm in
            off) envs=(LLAMA_KV_ROT_PERSIST=0) ;;
            on) envs=(LLAMA_KV_ROT_PERSIST=1) ;;
            fail) envs=(LLAMA_KV_ROT_ALLOC_FAIL=1) ;;
        esac
        "$MGATE" env "${envs[@]}" build-cpu/bin/test-recurrent-state-rollback --getter-gate --getter-logits "$D/rot/$arm.bin" \
            -m "$MODEL" -c 4096 -t 4 -ngl 0 -ctk q8_0 -ctv q8_0 -lv 5 > "$D/rot/$arm.log" 2>&1
    done
    cmp "$D/rot/off.bin" "$D/rot/on.bin"
    cmp "$D/rot/off.bin" "$D/rot/fail.bin"
    grep -aq 'persistent rotations backend=CPU' "$D/rot/on.log"
    grep -aq 'rotation allocation failed; using graph inputs' "$D/rot/fail.log"
    if "$MGATE" env LLAMA_KV_ROT_MUTANT=1 build-cpu/bin/test-recurrent-state-rollback --getter-gate --getter-reference "$D/rot/off.bin" \
        -m "$MODEL" -c 4096 -t 4 -ngl 0 -ctk q8_0 -ctv q8_0 -lv 5 > "$D/rot/mutant.log" 2>&1; then
        echo 'FAIL rotation mutant escaped'; return 1
    fi
    grep -aq 'rotation matrix bytes differ from the Hadamard reference' "$D/rot/mutant.log"
    echo 'rotation mutant killed: rotation matrix bytes differ from the Hadamard reference'
    for meta in 0 1; do
        for arm in off on fail; do
            case $arm in
                off) envs=(LLAMA_KV_ROT_PERSIST=0) ;;
                on) envs=(LLAMA_KV_ROT_PERSIST=1) ;;
                fail) envs=(LLAMA_KV_ROT_ALLOC_FAIL=1) ;;
            esac
            "$MGATE" env "${envs[@]}" HOST_KV_Q8=1 NP2_THREADS=2 LLAMA_MTP_CHAIN=0 LLAMA_MTP_FOLD_C=1 \
                build-cpu/bin/bw-np2-driver "$MODEL" "$D/rot/$meta-$arm.json" "$meta" 2 1729 64 > "$D/rot/$meta-$arm.log" 2>&1
        done
    done
    python3 - "$D/rot" <<'PY_COMPARE'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
for meta in (0, 1):
    rows = [json.loads((p/f'{meta}-{arm}.json').read_text()) for arm in ('off', 'on', 'fail')]
    assert rows[0] == rows[1] == rows[2], 'rotation tokens/counters differ'
    assert 'persistent rotations backend=' in (p/f'{meta}-on.log').read_text()
print('rotation plain/Meta tokens and counters PASS')
PY_COMPARE
    echo 'rotation raw logits, fallback, negative control PASS'
}
parsers() {
    S9_EVAL="$HOME/qwen38-bw/stack8/bdae9b6aa3f29c761b70e6a133a7a0a54bbc3c9a/run.935rrX/src/build-parser-inputs" HARNESS_CODE_ROOT="$ROOT/codex/parser-harness" \
        "$BW_GATE" python3 -P codex/gpu_stack10.py --fixtures
}
# shlint runs over every tracked payload script of the tree, the stack10 libs included
payload_lint() {
    local f
    for f in codex/stack10/bw-stack10-mac.sh codex/stack10/mac/*.sh; do bash -n "$f"; done
    bash -n codex/bw-stack10-cand-mac.sh codex/bw-stack10-cand.sh codex/gpu_ab.sh
    python3 -m py_compile codex/gpu_stack10.py codex/render_stack10_items.py
    PYTHONPATH="$ROOT/codex/shlint" "$BW_GATE" python3 -m gpuq.shlint $(git ls-files 'codex/*.sh')
}
base_cheap() {
    stage 20 payload-lint payload_lint
    stage 21 sampling-and-alloc-units units_plain
    stage 22 softplus-and-fa-q8-ops units_ops
    stage 23 recorded-parser-controls parsers
}
base_model() {
    stage 24 rotation-identity rotations
}
