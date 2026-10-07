#!/usr/bin/env bash
# Usage: bw-mac.sh TREE PAYLOAD [OUT_NAME]. PAYLOAD is a committed, tree-relative Bash script taking OUT as its only argument.
# Payloads honor BW_CPU_ONLY=1, BW_MODEL, BW_GATE/BW_BUILDGATE (nice -n 10), BW_JOBS=5, BW_CCACHE=OFF and CMAKE_GENERATOR=Ninja.
# BW_CPU_EXTRA is a whitespace-separated cmake argument list: native/i8mm, Metal and BLAS off on the Mac; PATH includes ~/qwen38-tools/bin.
# Linux payload defaults stay unchanged. Model runs use <=4 threads, explicit small contexts, and wait for >=10 GiB and no other model.
# Bundles contain the exact HEAD snapshot, with history cut at HEAD; payloads cannot assume older revisions are present.
# Copies OUT to work/build-window-1007/<name>-mac-a<N>; payload rc is preserved. Runner/setup/transport failures return 125.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1
trap 'exit 125' ERR

if [[ ${1:-} == --remote ]]; then
    shift
    name=$1 sha=$2 payload=$3 attempt=$4
    export PATH="$HOME/qwen38-tools/bin:/opt/homebrew/bin:$PATH"
    base=$HOME/qwen38-bw
    run=$base/$name/$sha/$attempt
    cache=$base/objects.git
    mkdir "$base/objects.lock" || { echo 'bw-mac: object cache busy; retry later' >&2; exit 125; }
    trap 'rmdir "$base/objects.lock"' EXIT
    [[ -d $cache ]] || git init --bare "$cache"
    git -C "$cache" config gc.auto 0
    git -C "$cache" config maintenance.auto false
    printf '%s\n' "$sha" >> "$cache/shallow"
    git -C "$cache" bundle verify "$run/source.bundle"
    git -C "$cache" fetch "$run/source.bundle" +refs/heads/source:refs/heads/bundle
    rmdir "$base/objects.lock"
    trap - EXIT
    git clone --shared --no-checkout "$cache" "$run/src"
    git -C "$run/src" checkout --detach "$sha"
    [[ $(git -C "$run/src" rev-parse HEAD) == "$sha" ]]
    mkdir "$run/out"
    printf '#!/usr/bin/env bash\nexec nice -n 10 "$@"\n' > "$run/gate.sh"
    chmod +x "$run/gate.sh"
    export BW_CPU_ONLY=1 BW_MODEL="$HOME/qwen38-stack7/models/Qwen3.5-4B-UD-Q4_K_XL.gguf"
    export BW_GATE="$run/gate.sh" BW_BUILDGATE="$run/gate.sh" BW_JOBS=5 BW_CCACHE=OFF
    # i8mm changes 2-row accumulation order and breaks np2 byte identity.
    export BW_CPU_EXTRA='-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod -DGGML_METAL=OFF -DGGML_BLAS=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON'
    export CMAKE_GENERATOR=Ninja OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
    export GPUQ_BUILD_WINDOW=1 PYTHONDONTWRITEBYTECODE=1
    {
        printf 'HEAD=%s\nHOST=%s\nPAYLOAD=%s\nSTART=%s\n' "$sha" "$(hostname)" "$payload" "$(date -Iseconds)"
        env | LC_ALL=C sort | grep -E '^(BW_|CMAKE_GENERATOR=|CUDA_VISIBLE_DEVICES=|GIT_NO_LAZY_FETCH=|GPUQ_BUILD_WINDOW=|PATH=|OMP_NUM_THREADS=|OPENBLAS_NUM_THREADS=|MKL_NUM_THREADS=|VECLIB_MAXIMUM_THREADS=|PYTHONDONTWRITEBYTECODE=)'
    } > "$run/out/mac-run.txt"
    cd "$run/src"
    if bash "$payload" "$run/out" > "$run/out/payload.log" 2>&1; then rc=0; else rc=$?; fi
    printf 'END=%s\nEXIT_CODE=%s\n' "$(date -Iseconds)" "$rc" >> "$run/out/mac-run.txt"
    printf '%s\n' "$rc" > "$run/exit-code"
    exit 0
fi

[[ $# == 2 || $# == 3 ]] || { echo 'usage: bw-mac.sh TREE PAYLOAD [OUT_NAME]' >&2; exit 125; }
tree=$(cd "$1" && pwd -P)
[[ $(git -C "$tree" rev-parse --show-toplevel) == "$tree" ]] || { echo 'bw-mac: TREE must be a Git root' >&2; exit 125; }
payload=${2#"$tree"/}
name=${3:-$(basename "$tree")}
[[ $name =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || { echo 'bw-mac: invalid OUT_NAME' >&2; exit 125; }
[[ $payload != /* && $payload != -* && /$payload/ != *'/../'* ]] || { echo 'bw-mac: PAYLOAD must be tree-relative' >&2; exit 125; }
[[ -z $(git -C "$tree" status --porcelain --untracked-files=no) ]] || { echo 'bw-mac: dirty tracked tree' >&2; exit 125; }
sha=$(git -C "$tree" rev-parse HEAD)
git -C "$tree" cat-file -e "$sha:$payload" || { echo 'bw-mac: PAYLOAD is not committed' >&2; exit 125; }
scratch=$(mktemp -d "$tree/build-bw-mac.XXXXXX")
ssh_bash() {
    local command='bash' arg
    for arg in "$@"; do printf -v command '%s %q' "$command" "$arg"; done
    ssh mini-builder "$command"
}
remote_base=$(ssh_bash -c 'export CUDA_VISIBLE_DEVICES=""; printf "%s/qwen38-bw" "$HOME"')
[[ $remote_base =~ ^/[a-zA-Z0-9/._-]+$ ]] || exit 125
previous=$(ssh_bash -c 'export CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1; git -C "$HOME/qwen38-bw/objects.git" rev-parse --verify refs/heads/bundle 2>/dev/null || true')
git init --bare "$scratch/export.git" >/dev/null
common=$(git -C "$tree" rev-parse --path-format=absolute --git-common-dir)
printf '%s/objects\n' "$common" > "$scratch/export.git/objects/info/alternates"
printf '%s\n' "$sha" > "$scratch/export.git/shallow"
git -C "$scratch/export.git" update-ref refs/heads/source "$sha"
exclude=()
if [[ $previous =~ ^[0-9a-f]{40}$ && $previous != "$sha" ]] && git -C "$tree" cat-file -e "$previous^{commit}" 2>/dev/null; then
    printf '%s\n' "$previous" >> "$scratch/export.git/shallow"
    exclude+=("^$previous")
fi
nice -n 19 git -C "$scratch/export.git" -c pack.threads=1 bundle create "$scratch/source.bundle" refs/heads/source "${exclude[@]}"
[[ $(git -C "$tree" rev-parse HEAD) == "$sha" && -z $(git -C "$tree" status --porcelain --untracked-files=no) ]] || exit 125
output_base=$HOME/repos/qwen38-eval-worktrees/work/build-window-1007
mkdir -p "$output_base"
n=1
while ! mkdir "$output_base/$name-mac-a$n" 2>/dev/null; do
    [[ -e $output_base/$name-mac-a$n ]] || exit 125
    n=$((n + 1))
done
out=$output_base/$name-mac-a$n
attempt=$(ssh_bash -c 'export CUDA_VISIBLE_DEVICES=""; mkdir -p "$1"; mktemp -d "$1/run.XXXXXX"' _ "$remote_base/$name/$sha")
remote=$attempt
attempt=${attempt##*/}
printf 'bw-mac: HEAD=%s remote=%s out=%s\n' "$sha" "$remote" "$out"
printf 'HEAD=%s\nREMOTE=%s\n' "$sha" "$remote" > "$out/transport.txt"
scp "$scratch/source.bundle" "$0" "mini-builder:$remote/" || exit 125
runner=${0##*/}
ssh_bash -c 'export CUDA_VISIBLE_DEVICES=""; nohup bash "$1" --remote "$2" "$3" "$4" "$5" > "$6/setup.log" 2>&1 < /dev/null & printf "%s\n" "$!" > "$6/runner.pid"' _ "$remote/$runner" "$name" "$sha" "$payload" "$attempt" "$remote" || exit 125
while :; do
    state=$(ssh_bash -c 'export CUDA_VISIBLE_DEVICES=""; if [[ -f $1/exit-code ]]; then cat "$1/exit-code"; elif kill -0 "$(cat "$1/runner.pid")" 2>/dev/null; then echo RUN; else echo FAILED; fi' _ "$remote") || exit 125
    [[ $state == RUN ]] || break
    sleep 15
done
scp "mini-builder:$remote/setup.log" "$out/" || exit 125
if [[ $state == FAILED ]]; then echo "bw-mac: setup failed; see $out/setup.log" >&2; exit 125; fi
[[ $state =~ ^[0-9]+$ && $state -le 255 ]] || exit 125
scp -r "mini-builder:$remote/out/." "$out/" || exit 125
printf 'bw-mac: payload rc=%s out=%s\n' "$state" "$out"
exit "$state"
