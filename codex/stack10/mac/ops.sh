# Union test-backend-ops CPU runs for touched ops the component libs do not cover. CPU against CPU: shape and case plumbing plus the
# same-backend identity checks; no CUDA code runs. Stage codes 140-149.
T=build-cpu/bin/test-backend-ops
# every tests-passed line must be N/N with N >= 1: an empty -p match must fail loudly
all_passed() { grep -aEq '^ +([1-9][0-9]*)/\1 tests passed' "$1"; }
mmvq_small_n() {
    "$BW_GATE" $T test -b CPU -o MUL_MAT -p 'type_a=(q4_0|q4_K|q5_K|q6_K|q8_0|iq4_xs),type_b=f32,m=16,n=[1-8],k=256,' -j 1 > "$D/mmvq-small-n.out" 2>&1
    all_passed "$D/mmvq-small-n.out"
    grep -ac 'tests passed' "$D/mmvq-small-n.out"
}
mul_mat_id() {
    "$BW_GATE" $T test -b CPU -o MUL_MAT_ID -p 'type_a=(q4_K|q6_K|q8_0|iq4_xs),type_b=f32,n_mats=4,n_used=2,b=0,m=512,n=[1-8],k=256,' -j 1 > "$D/mul-mat-id.out" 2>&1
    all_passed "$D/mul-mat-id.out"
}
id_shared_ops() {
    "$BW_GATE" $T test -b CPU -o MUL_MAT_ID_SHARED -j 1 > "$D/id-shared.out" 2>&1
    all_passed "$D/id-shared.out"
}
ops_cheap() {
    stage 140 mmvq-mul-mat-small-n mmvq_small_n
    stage 141 mul-mat-id-small-n mul_mat_id
    stage 142 mul-mat-id-shared-ops id_shared_ops
}
