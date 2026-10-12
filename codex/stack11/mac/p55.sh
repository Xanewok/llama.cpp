# p55 (MMVQ per-type tables, CUDA-only): table unit test and its negative control on CPU, TP sampling (test-llama-archs, no model), then the stack9 CPU stages that need a
# model: recurrent-state rollback, np2 chain identity, recorded parser controls. Codes 30-39.
mmvq_units() {
    "$BW_GATE" build-cpu/bin/test-mmvq-params
}
# Dropping IQ4_XS from the Blackwell n>=5 table must abort the test; rc 134 (SIGABRT from require()) so a compile or link failure
# cannot pass as a caught mutant.
mmvq_mutant() {
    local d=$D/mmvq-mutant rc=0
    mkdir -p "$d/tests" "$d/ggml/src/ggml-cuda"
    cp tests/test-mmvq-params.cpp "$d/tests/"
    python3 - ggml/src/ggml-cuda/mmvq-params.h "$d/ggml/src/ggml-cuda/mmvq-params.h" <<'PY'
import sys
from pathlib import Path
s = Path(sys.argv[1]).read_text()
old = "                case GGML_TYPE_Q4_K:\n                case GGML_TYPE_IQ4_XS:\n                    return 1;\n"
assert s.count(old) == 1, 'mmvq-params.h anchor moved'
Path(sys.argv[2]).write_text(s.replace(old, "                case GGML_TYPE_Q4_K:\n                    return 1;\n"))
PY
    c++ -std=c++17 -O0 -I ggml/include -o "$d/test-mmvq-params-mutant" "$d/tests/test-mmvq-params.cpp" -L build-cpu/bin -lggml-base -Wl,-rpath,"$PWD/build-cpu/bin"
    "$d/test-mmvq-params-mutant" > "$d/run.log" 2>&1 || rc=$?
    echo "mutant rc=$rc"
    [[ $rc == 134 ]]
}
rollback() {
    "$MGATE" build-cpu/bin/test-recurrent-state-rollback -m "$MODEL" -c 4096 -t 4 -ngl 0
}
tp_sampling() {
    local seed
    for seed in 42 1729; do
        "$BW_GATE" build-cpu/bin/test-llama-archs --tp-sampling -s "$seed" > "$D/tp-sampling-$seed.log" 2>&1
        grep -aq 'test_banned_candidates: .*: OK' "$D/tp-sampling-$seed.log"
        grep -aq 'unextended chain caught.*: OK' "$D/tp-sampling-$seed.log"
    done
}
np2_chain_identity() {
    bash codex/check-np2-interactions.sh "$D/np2"
    grep -aq 'MTP chain disabled: needs one sequence (n_seq=2)' "$D/np2/0-1.log"
    grep -aq 'MTP chain disabled: needs one sequence (n_seq=2)' "$D/np2/1-1.log"
}
p55_cheap() {
    stage 30 mmvq-table-test mmvq_units
    stage 31 mmvq-table-mutant mmvq_mutant
    stage 33 tp-sampling tp_sampling
}
p55_model() {
    stage 32 recurrent-state-rollback rollback
    stage 35 np2-chain-identity np2_chain_identity
}
