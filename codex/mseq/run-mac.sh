#!/usr/bin/env bash
# Exact committed snapshot; foreground transfer, detached bounded payload, evidence copied only into this worktree.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
ATTEMPT=${1:?attempt name}
[[ $ATTEMPT =~ ^a[0-9]+$ ]] || exit 125
# The owner's prompt edits are not build inputs. Any other tracked edit must be committed first.
while IFS= read -r path; do
    [[ $path == codex/PROMPT.md || $path == codex/PURPOSE ]] || { echo "uncommitted input: $path"; exit 125; }
done < <(git diff HEAD --name-only)
SHA=$(git rev-parse HEAD)
mkdir -p build-mac-transfer "codex/mseq/mac-out/$ATTEMPT"
nice -n 19 git archive --format=tar HEAD -o "build-mac-transfer/$ATTEMPT.tar"
scp "build-mac-transfer/$ATTEMPT.tar" "mini-builder:qwen38-fold-mseq-$ATTEMPT.tar"
ssh mini-builder bash -s -- "$ATTEMPT" "$SHA" <<'REMOTE'
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
export PATH="$HOME/qwen38-tools/bin:/opt/homebrew/bin:$PATH"
run=$HOME/qwen38-fold-mseq/$1
[[ ! -e $run ]] || exit 125
mkdir -p "$run/src" "$run/out"
tar -xf "$HOME/qwen38-fold-mseq-$1.tar" -C "$run/src"
printf '%s\n' "$2" > "$run/out/COMMIT"
cd "$run/src"
nohup bash -c 'export CUDA_VISIBLE_DEVICES=""; python3 codex/mseq/bounded.py 3600 bash codex/mseq/mac.sh "$1/out" > "$1/out/payload.log" 2>&1; echo $? > "$1/rc"' _ "$run" > "$run/runner.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$run/runner.pid"
REMOTE
start=$SECONDS
while :; do
    state=$(ssh mini-builder "export CUDA_VISIBLE_DEVICES=''; if test -f \"\$HOME/qwen38-fold-mseq/$ATTEMPT/rc\"; then cat \"\$HOME/qwen38-fold-mseq/$ATTEMPT/rc\"; else echo RUN; fi")
    [[ $state == RUN ]] || break
    [[ $((SECONDS - start)) -lt 3660 ]] || exit 125
    sleep 15
done
scp -r "mini-builder:qwen38-fold-mseq/$ATTEMPT/out/." "codex/mseq/mac-out/$ATTEMPT/"
printf 'Mac payload rc=%s, evidence codex/mseq/mac-out/%s\n' "$state" "$ATTEMPT"
exit "$state"
