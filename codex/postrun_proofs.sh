#!/usr/bin/env bash
# Post-run proofs on every measured boot of an ar-readfirst A/B item. Usage: postrun_proofs.sh RUN_DIR N_BOOTS | --selftest
# Measured boots run at default verbosity, which does not print ggml_cuda_ar_pipeline_init's INFO lines, so the arm proof is: each
# boot ran its frozen root's server (command.json), and the same attempt's canary (verbosity 4) logged the rf kernel configuration
# line on rf and only the master AllReduce init line on base. (Adapted from work/zc-pr-window/postrun_proofs.sh.)
set -uo pipefail
W=/home/xanewok/repos/qwen38-eval-worktrees
AR_INIT='ggml_cuda_ar_pipeline_init: initialized AllReduce pipeline: 2 GPUs'
RF_CFG='ggml_cuda_ar_pipeline_init: chunked kernel round_vecs=1 store_first=0 mutant=0 stall_cycles=0'
exe_is() {  # BOOT_DIR FROZEN_NAME
    grep -aqF "\"$W/frozen/$2/build/bin/round-trace\"" "$1/command.json"
}
canary_ok() {  # CANARY_DIR
    local d=$1 ok=1 f
    for f in "$d"/[0-9]*-rf/server.log; do
        [[ -f $f ]] || { ok=0; continue; }
        grep -aqF "$RF_CFG" "$f" && grep -aqF "$AR_INIT" "$f" || ok=0
    done
    for f in "$d"/[0-9]*-base/server.log; do
        [[ -f $f ]] || { ok=0; continue; }
        grep -aqF "$AR_INIT" "$f" || ok=0
        if grep -aqF 'chunked kernel round_vecs=' "$f"; then ok=0; fi
    done
    [[ $ok == 1 ]]
}
check() {  # RUN_DIR N
    local run=$1 want=$2 n=0 bad=0 dir canary
    for dir in "$run"/[0-9]*-base "$run"/[0-9]*-rf; do
        [[ -f $dir/server.log && -f $dir/command.json ]] || continue
        n=$((n + 1))
        case $dir in
            *-rf) exe_is "$dir" ar-readfirst || { echo "postrun_proofs: FAIL $dir: not frozen/ar-readfirst"; bad=1; } ;;
            *) exe_is "$dir" pkg89-base || { echo "postrun_proofs: FAIL $dir: not frozen/pkg89-base"; bad=1; } ;;
        esac
    done
    [[ $n == "$want" ]] || { echo "postrun_proofs: FAIL $run: $n boot logs, expected $want"; return 5; }
    canary=${run%/full-a*}/canary-a${run##*/full-a}
    canary_ok "$canary" || { echo "postrun_proofs: FAIL $canary: AllReduce init lines"; bad=1; }
    [[ $bad == 0 ]] || return 6
    echo "postrun_proofs: PASS $n boots ($run), canary $canary"
}
if [[ ${1:-} == --selftest ]]; then
    d=$(mktemp -d)
    trap 'rm -rf "$d"' EXIT
    mk() {  # NAME RF_EXE BASE_EXE RF_LINE BASE_LINE
        local r=$d/$1/full-a1 c=$d/$1/canary-a1 i
        for i in 0 3; do mkdir -p "$r/$i-base"; printf '["%s/frozen/%s/build/bin/round-trace"]\n' "$W" "$3" > "$r/$i-base/command.json"; : > "$r/$i-base/server.log"; done
        for i in 1 2; do mkdir -p "$r/$i-rf"; printf '["%s/frozen/%s/build/bin/round-trace"]\n' "$W" "$2" > "$r/$i-rf/command.json"; : > "$r/$i-rf/server.log"; done
        mkdir -p "$c/0-base" "$c/1-rf"
        printf '0.50.8 I %s, 1024 KB chunked kernel staging\n' "$AR_INIT" > "$c/0-base/server.log"
        printf '0.50.8 I %s, 1024 KB chunked kernel staging\n' "$AR_INIT" > "$c/1-rf/server.log"
        [[ $4 == cfg ]] && printf '0.50.8 I %s\n' "$RF_CFG" >> "$c/1-rf/server.log"
        [[ $5 == cfg ]] && printf '0.50.8 I %s\n' "$RF_CFG" >> "$c/0-base/server.log"
        echo "$r"
    }
    good=$(mk good ar-readfirst pkg89-base cfg none)
    check "$good" 4 > /dev/null || { echo 'postrun_proofs: FAIL selftest (good)'; exit 8; }
    if check "$good" 8 > /dev/null; then echo 'postrun_proofs: FAIL selftest (boot count)'; exit 8; fi
    for bad in "swap pkg89-base ar-readfirst cfg none" "noline ar-readfirst pkg89-base none none" \
               "both ar-readfirst pkg89-base cfg cfg"; do
        # shellcheck disable=SC2086
        r=$(mk $bad)
        if check "$r" 4 > /dev/null; then echo "postrun_proofs: FAIL selftest (accepts ${bad%% *})"; exit 8; fi
    done
    echo 'postrun_proofs: PASS selftest'; exit 0
fi
check "${1:?run dir}" "${2:?boots}"
