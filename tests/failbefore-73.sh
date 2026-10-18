#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
out=$1
mkdir -p "$out"
cat > "$out/production.diff" <<'PRODUCTION'
diff --git a/ggml/src/ggml-backend-meta.cpp b/ggml/src/ggml-backend-meta.cpp
index 679d5216d..197533fa3 100644
--- a/ggml/src/ggml-backend-meta.cpp
+++ b/ggml/src/ggml-backend-meta.cpp
@@ -2179 +2179,2 @@ static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend,
-                    if (next->view_src != nullptr && next->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(next->view_src->buffer)) {
+                    if (next->view_src != nullptr && next->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(next->view_src->buffer) &&
+                            !ggml_backend_buffer_is_meta(next->view_src->buffer)) {
@@ -2218 +2219,2 @@ static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend,
-                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
+                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer) &&
+                        !ggml_backend_buffer_is_meta(node->view_src->buffer)) {
PRODUCTION
if git apply --unidiff-zero --reverse --check "$out/production.diff" 2>/dev/null; then
    git apply --unidiff-zero --reverse "$out/production.diff"
fi
build=$PWD/build-failbefore-73
read -r -a extra <<< "${BW_CPU_EXTRA:-}"
"${BW_BUILDGATE:?}" cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON     -DGGML_CUDA=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF     -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF "${extra[@]}" > "$out/configure.log" 2>&1
"$BW_BUILDGATE" cmake --build "$build" -j"${BW_JOBS:-4}" --target test-alloc > "$out/build-before.log" 2>&1
export GGML_NO_BACKTRACE=1
printf 'case\tbefore_rc\tcontrol_rc\tafter_rc\n' > "$out/summary.tsv"
cases=(view-mirrored view-split cpy-mirrored cpy-split set-rows-mirrored set-rows-split)
for case_name in "${cases[@]}"; do
    rc=0
    "$BW_GATE" "$build/bin/test-alloc" --meta-host-view "$case_name" > "$out/$case_name-before.log" 2>&1 || rc=$?
    [[ $rc != 0 ]]
    grep -Fq 'GGML_ASSERT(i_start == cgraph->n_nodes) failed' "$out/$case_name-before.log"
    "$BW_GATE" "$build/bin/test-alloc" --meta-host-view "$case_name" tail > "$out/$case_name-control.log" 2>&1
    grep -Fq "73 PASS $case_name tail=1" "$out/$case_name-control.log"
    printf '%s\t%s\t0\n' "$case_name" "$rc" >> "$out/before.tsv"
done
git apply --unidiff-zero "$out/production.diff"
"$BW_BUILDGATE" cmake --build "$build" -j"${BW_JOBS:-4}" --target test-alloc > "$out/build-after.log" 2>&1
for case_name in "${cases[@]}"; do
    "$BW_GATE" "$build/bin/test-alloc" --meta-host-view "$case_name" > "$out/$case_name-after.log" 2>&1
    grep -Fq "73 PASS $case_name tail=0" "$out/$case_name-after.log"
done
while IFS= read -r row; do printf '%s\t0\n' "$row" >> "$out/summary.tsv"; done < "$out/before.tsv"
"$BW_GATE" "$build/bin/test-alloc" > "$out/full-after.log" 2>&1
printf 'PASS: six independent planner aborts, six tail controls, six fixed fixtures, full test-alloc\n' > "$out/verdict.txt"
