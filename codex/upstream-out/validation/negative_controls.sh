#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
run=${1:?package run directory}
pkg=${2:?slug}
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$run/src"
gate=$root/validation/gate.sh
case "$pkg" in
    70-*)
        file=src/llama-kv-cache.cpp
        target=test-llama-archs
        controls=(bound head)
        ;;
    73-*)
        file=ggml/src/ggml-backend-meta.cpp
        target=test-alloc
        controls=(planner)
        ;;
    *) exit 2 ;;
esac
run_test() {
    if [[ $target == test-alloc ]]; then
        bash "$gate" "$run/build-cpu/bin/$target"
    else
        bash "$gate" "$run/build-cpu/bin/$target" --seq-rm
    fi
}
for control in "${controls[@]}"; do
    git show "HEAD:$file" > "$run/original-$control.cpp"
    if [[ $control == planner ]]; then
        git apply -R --include="$file" "$root/patches/upstream/llama.cpp/$pkg/"0001-*.patch
    else
        python3 - "$file" "$control" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
t = p.read_text()
old, new = ('it->first < p1', 'it->first <= p1') if sys.argv[2] == 'bound' else (
    'new_head = std::min(new_head, i);', 'if (new_head == cells.size()) { new_head = i; }')
assert t.count(old) == 1
p.write_text(t.replace(old, new))
PY
    fi
    bash "$gate" cmake --build "$run/build-cpu" -j4 --target "$target" > "$run/logs/$pkg-$control-build.log" 2>&1
    set +e
    run_test > "$run/logs/$pkg-$control.log" 2>&1
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$run/logs/$pkg-$control.rc"
    cp "$run/original-$control.cpp" "$file"
    [[ $rc -ne 0 ]] || { echo "negative control escaped: $control"; exit 1; }
done
bash "$gate" cmake --build "$run/build-cpu" -j4 --target "$target" > "$run/logs/$pkg-restored-build.log" 2>&1
run_test > "$run/logs/$pkg-restored.log" 2>&1
git diff --exit-code
printf 'PASS\n' > "$run/logs/$pkg-negative-result"
