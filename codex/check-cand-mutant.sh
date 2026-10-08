#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${1:?output}
mkdir -p "$OUT"
BUILDGATE=${BW_BUILDGATE:?}
CPUGATE=${S7_GATE:?}
die() { local rc=$1; shift; echo "$*"; exit "$rc"; }
mutant_stage() {
    local m=$OUT/mutant
    if [[ ${BW_CPU_ONLY:-0} == 1 ]]; then
        mkdir -p "$m" || return 1
    else
        rm -rf "$m" && mkdir -p "$m" || return 1
    fi
    sed 's/llama_sampler_init_top_k(candidate_k + 1 + n_ban)/llama_sampler_init_top_k(candidate_k + 1)/' common/sampling.cpp > "$m/sampling.cpp"
    ! cmp -s common/sampling.cpp "$m/sampling.cpp" || { echo "mutant sed changed nothing"; return 1; }
    # the library's own compile command for sampling.cpp, pointed at the mutant source and object
    python3 - "$ROOT" "$m" > "$m/compile.sh" <<'EOF' || return 1
import json, shlex, sys
root, m = sys.argv[1:]
for e in json.load(open(f"{root}/build-cpu/compile_commands.json")):
    if e["file"].endswith("/common/sampling.cpp"):
        args = e.get("arguments") or shlex.split(e["command"])
        i = args.index("-o")
        args[i + 1] = f"{m}/sampling.o"
        args = [f"{m}/sampling.cpp" if a == e["file"] or a.endswith("/common/sampling.cpp") else a for a in args]
        print("cd " + shlex.quote(e["directory"]) + " && " + shlex.join(args))
        break
else:
    sys.exit("common/sampling.cpp not in compile_commands.json")
EOF
    "$BUILDGATE" bash "$m/compile.sh" || return 1
    # the test's own link line, plus the mutant object ahead of libllama-common (executable definitions win)
    local link
    if [[ -f build-cpu/build.ninja ]]; then
        ninja -C build-cpu -t commands test-llama-archs | tail -1 > "$m/link.txt" || return 1
        link=$(python3 - "$m" <<'EOF'
import shlex, sys
from pathlib import Path
m = Path(sys.argv[1])
args = shlex.split((m / 'link.txt').read_text())
if args[:2] == [':', '&&']:
    args = args[2:]
if args[-2:] == ['&&', ':']:
    args = args[:-2]
i = args.index('-o')
args[i + 1] = str(m / 'test-llama-archs-mutant')
args.insert(i + 2, str(m / 'sampling.o'))
print(shlex.join(args))
EOF
        ) || return 1
    else
        link=$(sed "s| -o [^ ]*test-llama-archs | -o $m/test-llama-archs-mutant $m/sampling.o |" build-cpu/tests/CMakeFiles/test-llama-archs.dir/link.txt)
    fi
    [[ $link == *"$m/sampling.o"* ]] || { echo "could not rewrite the link line"; return 1; }
    if [[ -f build-cpu/build.ninja ]]; then
        (cd build-cpu && "$BUILDGATE" bash -c "$link") || return 1
    else
        (cd build-cpu/tests && "$BUILDGATE" bash -c "$link") || return 1
    fi
    nm -C "$m/test-llama-archs-mutant" | grep " T common_sampler_init" >/dev/null || { echo "mutant common_sampler_init not in the executable"; return 1; }
}
mutant_stage > "$OUT/mutant-build.log" 2>&1 || die 8 "mutant build (see $OUT/mutant-build.log)"
mrc=0
"$CPUGATE" "$OUT/mutant/test-llama-archs-mutant" --tp-sampling -s 42 > "$OUT/mutant-run.log" 2>&1 || mrc=$?
[[ $mrc != 0 ]] || die 3 "mutant escaped: test-llama-archs passed without the extension (see $OUT/mutant-run.log)"
grep -q "banned: read the full row although the top k is unique" "$OUT/mutant-run.log" ||
    die 3 "mutant failed (rc=$mrc) but not on a full-row read (see $OUT/mutant-run.log)"
echo "mutant (no extension): rc=$mrc, caught by: $(grep -m1 'banned: read the full row' "$OUT/mutant-run.log")"
echo "mutant caught"
