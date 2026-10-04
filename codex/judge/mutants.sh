#!/usr/bin/env bash
# Mutation test of the prefix certificate and the reuse/setter guards against the shipped CPU checks.
# Usage: [ONLY=regex] judge/mutants.sh
# Needs codex/check-mask-upload.sh then judge/run-judge.sh (reference outputs).
set -uo pipefail
if [[ ${JUDGE_NICED:-} != 1 ]]; then export JUDGE_NICED=1; exec nice -n 19 bash "$0" "$@"; fi
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 TMPDIR="$PWD/build-tmp"
OUT=build-mask-cpu/judge; mkdir -p "$OUT"
link() { # $1 lib, $2 out (test source: $SRC)
  c++ -O3 -UNDEBUG -std=c++17 -Iinclude -Isrc -Icommon -Iggml/include -Iggml/src -Ivendor -Ibuild-mask-cpu/common "$SRC" -o "$2" \
    -Wl,--start-group "$1" build-mask-cpu/common/libllama-common.a build-mask-cpu/ggml/src/libggml.a build-mask-cpu/ggml/src/libggml-cpu.a \
    build-mask-cpu/ggml/src/libggml-base.a build-mask-cpu/common/libllama-common-base.a build-mask-cpu/vendor/cpp-httplib/libcpp-httplib.a \
    -Wl,--end-group -pthread -ldl -lm -lssl -lcrypto
}
# name@@file@@needle@@replacement  (python-literal strings)
MUTANTS=$(cat <<'M'
K1_no_hole_check@@llama-kv-cache@@"if (cells.used_max_p1() != n_used || positions.size() != n_used) {\n            return 0;"@@"if (positions.size() != n_used) {\n            return 0;"
K2_no_owner_check@@llama-kv-cache@@"if (cells.used_max_p1() != n_used || positions.size() != n_used) {\n            return 0;"@@"if (cells.used_max_p1() != n_used) {\n            return 0;"
K3_no_mrope_tie@@llama-kv-cache@@"                    prefix = std::min(prefix, it->second);\n                }\n            }\n        }"@@"                }\n            }\n        }"
K4_no_future@@llama-kv-cache@@"        for (auto it = end; it != positions.end(); ++it) {\n            prefix = std::min(prefix, it->second);\n        }"@@""
K5_no_n_used_cap@@llama-kv-cache@@"        prefix = std::min(prefix, n_used);\n"@@""
K6_no_suffix_reserve@@llama-kv-cache@@"return std::min(prefix, n_kv - 1)/256*256;"@@"return prefix/256*256;"
K7_no_setter_assert@@llama-kv-cache@@"GGML_ASSERT(prefix == 0 || get_kq_mask_prefix(*ubatch, n_kv, causal_attn) >= prefix);"@@""
G1_no_reuse_recheck@@llama-graph@@"        res &= mctx->get_kq_mask_prefix(ubatch, cparams.causal_attn) >= kq_mask->src[0]->ne[0];\n"@@""
G4_fill_negzero@@llama-graph@@"zeros = ggml_fill(ctx, zeros, 0.0f);"@@"zeros = ggml_fill(ctx, zeros, -0.0f);"
G1K7_no_reuse_guards@@llama-graph@@"        res &= mctx->get_kq_mask_prefix(ubatch, cparams.causal_attn) >= kq_mask->src[0]->ne[0];\n"@@""
G1K7_no_reuse_guards@@llama-kv-cache@@"GGML_ASSERT(prefix == 0 || get_kq_mask_prefix(*ubatch, n_kv, causal_attn) >= prefix);"@@""
M
)
# suites: "tag src ref_cpu_logits ref_meta_logits"
SUITES=("shipped codex/host-depth.cpp build-mask-cpu/patch-logits.bin build-mask-cpu/full-meta-logits.bin"
        "judge codex/judge/host-depth-judge.cpp build-mask-cpu/judge/good-logits.bin build-mask-cpu/judge/full-mlogits.bin")
run_one() { # name lib
  local name=$1 lib=$2 line=""
  for suite in "${SUITES[@]}"; do
    read -r TAG SRC REF_LOGITS REF_MLOGITS <<< "$suite"
    local bin=$OUT/hd-$TAG-$name verdict=""
    link "$lib" "$bin" || { line+=",$TAG=BUILD_FAIL"; continue; }
    if ! "$bin" "$OUT/$TAG-$name-masks.bin" "$OUT/$TAG-$name-logits.bin" > "$OUT/$TAG-$name.csv" 2> "$OUT/$TAG-$name.log"; then
      verdict="killed(cpu: $(tail -n 1 "$OUT/$TAG-$name.log"))"
    elif ! MASK_UPLOAD_META=1 "$bin" "$OUT/$TAG-$name-mmasks.bin" "$OUT/$TAG-$name-mlogits.bin" > "$OUT/$TAG-$name-meta.csv" 2> "$OUT/$TAG-$name-meta.log"; then
      verdict="killed(meta: $(tail -n 1 "$OUT/$TAG-$name-meta.log"))"
    elif ! cmp -s "$OUT/$TAG-$name-mlogits.bin" "$REF_MLOGITS"; then
      verdict="killed(meta logits != forced-full)"
    elif ! cmp -s "$OUT/$TAG-$name-logits.bin" "$REF_LOGITS"; then
      verdict="killed(cpu logits differ)"
    else
      verdict="SURVIVED"
    fi
    line+=",$TAG=$verdict"
  done
  echo "$name$line"
}
if [[ -n ${ONLY:-} ]]; then MUTANTS=$(grep -E "^(${ONLY})@@" <<< "$MUTANTS"); fi
names=$(awk -F'@@' 'NF && !seen[$1]++ {print $1}' <<< "$MUTANTS")
for name in $names; do
  dir=$OUT/$name; rm -rf "$dir"; mkdir -p "$dir"
  grep -E "^${name}@@" <<< "$MUTANTS" | python3 -c '
import ast, sys, pathlib
d = sys.argv[1]
for line in sys.stdin:
    name, f, needle, repl = line.rstrip("\n").split("@@", 3)
    needle, repl = ast.literal_eval(needle), ast.literal_eval(repl)
    dst = pathlib.Path(f"{d}/{f}.cpp")
    s = dst.read_text() if dst.exists() else pathlib.Path(f"src/{f}.cpp").read_text()
    assert s.count(needle) == 1, (name, f, needle)
    dst.write_text(s.replace(needle, repl))
' "$dir" || { echo "$name,NEEDLE_FAIL"; continue; }
  cp build-mask-cpu/src/libllama.a "$dir/libllama.a"
  ok=1
  for src in "$dir"/*.cpp; do
    f=$(basename "$src")
    c++ -O3 -DNDEBUG -DGGML_USE_CPU -std=c++17 -Isrc -Ibuild-mask-cpu/src -Iinclude -Iggml/include -c "$src" -o "$dir/$f.o" || ok=0
    (cd "$dir" && ar r libllama.a "$f.o")
  done
  [[ $ok == 1 ]] || { echo "$name,COMPILE_FAIL"; continue; }
  run_one "$name" "$dir/libllama.a"
done
