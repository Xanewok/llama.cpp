#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "usage: $0 /path/to/test-ar-p2p out-dir" >&2
    exit 2
fi
binary=$1
out=$2
mkdir -p -- "$out"
failed=0
# Thresholds are read at pipeline init: each arm/configuration is a fresh process.
for copy in default 0 1; do
    for bf16 in 0 1; do
        for chunk in auto 262144; do
            [[ $chunk == auto || $copy == 1 ]] || continue
            name="copy-$copy-bf16-$bf16-chunk-$chunk"
            ok=1
            for arm in host p2p; do
                if (
                    unset GGML_CUDA_AR_P2P GGML_CUDA_AR_COPY_THRESHOLD GGML_CUDA_AR_COPY_CHUNK_BYTES
                    export GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_BF16_THRESHOLD=$bf16
                    [[ $copy == default ]] || export GGML_CUDA_AR_COPY_THRESHOLD=$copy
                    [[ $chunk == auto ]] || export GGML_CUDA_AR_COPY_CHUNK_BYTES=$chunk
                    [[ $arm == host ]] || export GGML_CUDA_AR_P2P=1
                    exec "$binary"
                ) > "$out/$name.$arm.stdout" 2> "$out/$name.$arm.stderr"; then
                    grep -v '^time ' "$out/$name.$arm.stdout" > "$out/$name.$arm.result" || ok=0
                    canary='allreduce: host staging'
                    [[ $arm == host ]] || canary='allreduce: P2P staging (peer device buffers)'
                    if ! grep -F -c "$canary" "$out/$name.$arm.stderr" > /dev/null ||
                       ! grep -c '^case ' "$out/$name.$arm.result" > /dev/null ||
                       [[ $(grep -c '^stress ' "$out/$name.$arm.result") != 3 ]]; then
                        ok=0
                    fi
                else
                    status=$?
                    echo "FAIL $name $arm exit=$status" >&2
                    ok=0
                fi
            done
            if [[ $ok == 1 ]] && diff -u "$out/$name.host.result" "$out/$name.p2p.result" > "$out/$name.diff"; then
                echo "PASS $name"
            else
                echo "FAIL $name (see $out/$name.*)" >&2
                failed=1
            fi
        done
    done
done
exit "$failed"
