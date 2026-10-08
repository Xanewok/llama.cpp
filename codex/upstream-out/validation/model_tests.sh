#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
export PATH="$HOME/qwen38-tools/bin:/opt/homebrew/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
root=$(cd "${1:?run root}" && pwd)
pkg=${2:?slug}
run=$root/$pkg
[[ $pkg != 68-* ]] || exit 2
src=$run/src
build=$run/build-cpu
gate=$root/validation/gate.sh
model=$HOME/qwen38-stack7/models/Qwen3.5-4B-UD-Q4_K_XL.gguf
mkdir -p "$run/model-tests"
out=$run/model-tests
cd "$src"

compile_probe() {
    local name=$1 chain=${2:-0}
    if [[ $name != outproj-cpu && $name != upstream-model-probe ]]; then
        bash "$gate" cmake --build "$build" -j4 --target server-context > "$out/server-build.log" 2>&1
    fi
    local extra=(-lllama)
    if [[ $name != outproj-cpu && $name != upstream-model-probe ]]; then
        extra+=("$build/tools/server/libserver-context.a" -lllama-common -lmtmd "$build/common/libllama-common-base.a")
    fi
    bash "$gate" c++ -std=c++17 -O2 -pthread -DPACK_CHAIN="$chain" \
        -Isrc -Iinclude -Iggml/include -Icommon -Itools/server -Itools/mtmd -Ivendor -I. \
        "$root/validation/$name.cpp" -L"$build/bin" -Wl,-rpath,"$build/bin" \
        "${extra[@]}" -lllama -lggml -lggml-base -lggml-cpu -o "$build/bin/$name" \
        > "$out/$name-build.log" 2>&1
}

rebuild() {
    bash "$gate" cmake --build "$build" -j4 --target llama-common > "$out/$1-build.log" 2>&1
}

case "$pkg" in
    69-*)
        compile_probe outproj-cpu
        git apply -R "$root/patches/upstream/llama.cpp/$pkg/"0001-*.patch
        rebuild baseline
        set +e
        bash "$gate" "$build/bin/outproj-cpu" "$model" "$out/base.bin" 1 > "$out/base.log" 2>&1
        rc=$?
        set -e
        printf '%s\n' "$rc" > "$out/base.rc"
        [[ $rc == 42 ]] || exit 1
        git apply "$root/patches/upstream/llama.cpp/$pkg/"0001-*.patch
        rebuild positive
        bash "$gate" "$build/bin/outproj-cpu" "$model" "$out/patch.bin" 1 > "$out/patch.log" 2>&1
        cmp "$out/base.bin" "$out/patch.bin"
        ;;
    71-*)
        compile_probe bw-np2-driver
        bash "$gate" "$build/bin/bw-np2-driver" "$model" "$out/patch.json" 0 2 1729 16 > "$out/patch.log" 2>&1
        git apply -R "$root/patches/upstream/llama.cpp/$pkg/"0001-*.patch
        rebuild baseline
        bash "$gate" "$build/bin/bw-np2-driver" "$model" "$out/base.json" 0 2 1729 16 > "$out/base.log" 2>&1
        cmp "$out/base.json" "$out/patch.json"
        git apply "$root/patches/upstream/llama.cpp/$pkg/"0001-*.patch
        rebuild restored
        python3 - <<'PY'
from pathlib import Path
p = Path('common/speculative.cpp')
t = p.read_text()
old = 'h_all + (size_t) i_batch_beg[seq_id] * n_embd'
assert t.count(old) == 1
p.write_text(t.replace(old, 'h_all'))
PY
        rebuild mutant
        bash "$gate" "$build/bin/bw-np2-driver" "$model" "$out/mutant.json" 0 2 1729 16 > "$out/mutant.log" 2>&1
        if cmp -s "$out/patch.json" "$out/mutant.json"; then echo 'bulk-row mutant escaped'; exit 1; fi
        git restore common/speculative.cpp
        rebuild restored-final
        ;;
    21-*)
        if [[ ${PACK_LOCAL_APPLIED:-0} != 1 ]]; then
            git am "$root/patches/upstream/llama.cpp/$pkg/local/"*.patch > "$out/local-am.log" 2>&1
        fi
        git apply "$root/validation/shape-instrumentation.patch"
        rebuild controls
        compile_probe bw-np2-driver 1
        for np in 1 2; do
            for chain in 0 1; do
                LLAMA_MTP_CHAIN=$chain LLAMA_MTP_CHAIN_DUMP="$out/np$np-$chain.dump" \
                    bash "$gate" "$build/bin/bw-np2-driver" "$model" "$out/np$np-$chain.json" 0 "$np" 1729 16 \
                    > "$out/np$np-$chain.log" 2>&1
            done
        done
        python3 "$root/validation/check_model_results.py" chain "$out"
        for control in reverse-u shift-embd; do
            if [[ $control == reverse-u ]]; then
                control_env=LLAMA_MTP_CHAIN_DEBUG_REVERSE_U=1
            else
                control_env=LLAMA_MTP_CHAIN_DEBUG_SHIFT_EMBD=1
            fi
            env "$control_env" LLAMA_MTP_CHAIN=1 LLAMA_MTP_CHAIN_DUMP="$out/$control.dump" \
                bash "$gate" "$build/bin/bw-np2-driver" "$model" "$out/$control.json" 0 1 1729 16 > "$out/$control.log" 2>&1
            grep -q 'MTP chain: k=3' "$out/$control.log"
            if cmp -s "$out/np1-1.json" "$out/$control.json" && cmp -s "$out/np1-1.dump" "$out/$control.dump"; then
                echo "chain control escaped: $control"; exit 1
            fi
        done
        cp common/speculative.cpp "$out/fixed-speculative.cpp"
        python3 - <<'PY'
from pathlib import Path
p = Path('common/speculative.cpp')
t = p.read_text()
assert t.count('params.mtp_chain && n_seq == 1 &&') == 1
assert t.count('params_spec.mtp_chain && params.n_parallel == 1 &&') == 1
t = t.replace('params.mtp_chain && n_seq == 1 &&', 'params.mtp_chain &&')
t = t.replace('params_spec.mtp_chain && params.n_parallel == 1 &&', 'params_spec.mtp_chain &&')
p.write_text(t)
PY
        rebuild np2-mutant
        compile_probe bw-np2-driver 1
        set +e
        LLAMA_MTP_CHAIN=1 LLAMA_MTP_CHAIN_DUMP="$out/np2-mutant.dump" \
            bash "$gate" "$build/bin/bw-np2-driver" "$model" "$out/np2-mutant.json" 0 2 1729 16 > "$out/np2-mutant.log" 2>&1
        rc=$?
        set -e
        printf '%s\n' "$rc" > "$out/np2-mutant.rc"
        cp "$out/fixed-speculative.cpp" common/speculative.cpp
        [[ $rc != 75 && $rc != 124 ]] || exit "$rc"
        if [[ $rc == 0 ]]; then
            python3 - "$root/validation/check_model_results.py" "$out" <<'PY'
import runpy
import sys
from pathlib import Path
checker, directory = sys.argv[1:]
sys.argv = [checker, 'chain', directory]
check = runpy.run_path(checker)
p = Path(directory)
same_output = (p / 'np2-0.json').read_bytes() == (p / 'np2-mutant.json').read_bytes() and (p / 'np2-0.dump').read_bytes() == (p / 'np2-mutant.dump').read_bytes()
same_shapes = check['decode_shapes']((p / 'np2-0.log').read_text()) == check['decode_shapes']((p / 'np2-mutant.log').read_text())
assert not (same_output and same_shapes), 'np2 mutant escaped'
PY
        fi
        git restore src/llama-context.cpp
        rebuild restored-final
        compile_probe bw-np2-driver 1
        ;;
    72-*)
        if [[ ${PACK_LOCAL_APPLIED:-0} != 1 ]]; then
            git am "$root/patches/upstream/llama.cpp/$pkg/local/"*.patch > "$out/local-am.log" 2>&1
        fi
        rebuild controls
        compile_probe upstream-model-probe
        bash "$gate" "$build/bin/upstream-model-probe" "$model" > "$out/eligibility.log" 2>&1
        compile_probe fold-cpu
        cp common/speculative.cpp "$out/positive-speculative.cpp"
        cp src/llama-context.cpp "$out/context-original.cpp"
        python3 - <<'PY'
from pathlib import Path
p = Path('src/llama-context.cpp')
t = p.read_text()
old = 'mtp_fold_n_tok = (int32_t) n_tokens_all;'
assert t.count(old) == 1
p.write_text(t.replace(old, old + '\n        std::fprintf(stderr, "PACK_FOLD rows=%zu\\n", size_t(n_tokens_all));'))
PY
        rebuild engagement
        args=(-m "$model" -c 4096 -np 1 -ngl 0 -fa on -b 256 -ub 256 -t 4 -tb 4
              --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0
              --spec-draft-sampling probabilistic --spec-draft-backend-sampling --backend-sampling)
        FOLD_CASES="$root/validation/cases.json" FOLD_OUT="$out/patch.json" FOLD_META=0 LLAMA_MTP_DRAFT_DUMP="$out/patch.dump" \
            bash "$gate" "$build/bin/fold-cpu" "${args[@]}" > "$out/patch.log" 2>&1
        grep -E 'PACK_FOLD rows=[1-9][0-9]*' "$out/patch.log" > "$out/engagement.log"
        python3 - <<'PY'
from pathlib import Path
p = Path('common/speculative.cpp')
t = p.read_text()
old = 'fold = llama_mtp_fold_init(ctx_tgt, ctx_dft);'
assert t.count(old) == 1
p.write_text(t.replace(old, 'fold = false;'))
PY
        rebuild baseline
        FOLD_CASES="$root/validation/cases.json" FOLD_OUT="$out/base.json" FOLD_META=0 LLAMA_MTP_DRAFT_DUMP="$out/base.dump" \
            bash "$gate" "$build/bin/fold-cpu" "${args[@]}" > "$out/base.log" 2>&1
        cmp "$out/base.json" "$out/patch.json"
        cmp "$out/base.dump" "$out/patch.dump"
        cp "$out/positive-speculative.cpp" common/speculative.cpp
        rebuild mutant
        FOLD_CASES="$root/validation/cases.json" FOLD_OUT="$out/mutant.json" FOLD_META=0 \
            LLAMA_MTP_FOLD_DEBUG_MUTANT=1 LLAMA_MTP_DRAFT_DUMP="$out/mutant.dump" \
            bash "$gate" "$build/bin/fold-cpu" "${args[@]}" > "$out/mutant.log" 2>&1
        if cmp -s "$out/patch.json" "$out/mutant.json" && cmp -s "$out/patch.dump" "$out/mutant.dump"; then
            echo 'fold-row mutant escaped'; exit 1
        fi
        cp "$out/context-original.cpp" src/llama-context.cpp
        rebuild restored-final
        ;;
    *) exit 2 ;;
esac
git diff --exit-code
printf 'PASS\n' > "$out/result"
