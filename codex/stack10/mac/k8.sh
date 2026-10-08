# K8 MMQ stream-k small-I tiles (+ L1 prerequisite): CUDA-only kernels, so CPU gates check the partition model, case plumbing and the
# mmq-hash plumbing. Codes 100-109.
T=build-cpu/bin/test-backend-ops
all_passed() { grep -aEq '^ +([1-9][0-9]*)/\1 tests passed' "$1"; }
sim() {
    "$BW_BUILDGATE" c++ -std=c++17 -O2 -ffp-contract=off -o "$D/mmq_sk_sim" codex/k8/mmq_sk_sim.cpp
    "$BW_GATE" "$D/mmq_sk_sim" | tee "$D/sim.txt"
    grep -aq '^PASS$' "$D/sim.txt"
}
cases() {
    "$BW_GATE" $T test -b CPU -o MUL_MAT -p 'type_a=(iq4_xs|q4_K|q5_K|q6_K|q8_0),type_b=f32,m=(3072|5120|6144|8704|9216),n=(9|12|16|17|24|25),k=1024,' > "$D/mm.out" 2>&1
    all_passed "$D/mm.out"
    "$BW_GATE" $T test -b CPU -o MUL_MAT_ID -p 'type_a=(iq4_xs|q4_K|q5_K|q6_K|q8_0),type_b=f32,n_mats=4,n_used=2,b=0,m=(3072|5120),n=(24|48),k=1024,' > "$D/id.out" 2>&1
    all_passed "$D/id.out"
}
hash_plumbing() {
    "$BW_GATE" $T mmq-hash CPU 3072 > "$D/hash-a.out"
    "$BW_GATE" $T mmq-hash CPU 3072 > "$D/hash-b.out"
    cmp "$D/hash-a.out" "$D/hash-b.out"
    [[ $(grep -ac '^id_' "$D/hash-a.out") == 8 ]]
    [[ $(wc -l < "$D/hash-a.out") -gt 100 ]]
}
k8_cheap() {
    stage 100 stream-k-partition-model sim
    stage 101 mul-mat-cases cases
    stage 102 mmq-hash-plumbing hash_plumbing
}
