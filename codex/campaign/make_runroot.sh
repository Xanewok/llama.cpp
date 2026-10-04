#!/usr/bin/env bash
# The shared runner wants <engine>/build/bin/llama-server, <engine>/gguf-py and a commit. Present the frozen stack3-mask bytes that way
# without touching them: symlinks to the frozen bin/, gguf-py from the SOURCE commit, sums for the copied python.
set -euo pipefail
cd "$(dirname "$0")/../.."
src=$(head -n1 codex/frozen/stack3-mask/SOURCE)
run=codex/frozen/stack3-mask-runroot
[[ ! -e $run ]] || { echo "runroot exists" >&2; exit 1; }
mkdir -p "$run/build/bin"
for f in codex/frozen/stack3-mask/bin/*.so*; do ln -s "../../../stack3-mask/bin/$(basename "$f")" "$run/build/bin/$(basename "$f")"; done
ln -s ../../../stack3-mask/bin/round-trace "$run/build/bin/llama-server"
git archive "$src" gguf-py | tar -x -C "$run"
(cd "$run" && find gguf-py -type f | sort | xargs sha256sum > RUNROOT.SHA256SUMS)
echo "$src" > "$run/SOURCE"
