# K8 MMQ stream-k occupancy (topic/l1-k8, base ace9a6d01)

Status: design + CPU model + Mac gates done; CUDA kernels uncompiled (no CUDA on this host). First compile and all GPU claims are window gates.

## Commits (ace9a6d01..HEAD)
- 8d7ca5a74 cuda : MMQ stream-k partitions in default-tile units for narrow tiles (general; `mmq.cuh`)
- e410fd6dc cuda : take SMALL_I MMQ tiles on stream-k launches too (selector no longer needs plain tiling)
- d859734f2 tests : cover MMQ small-I tiles on stream-k shapes (test_mul_mat m 3072/5120/6144/8704/9216; MUL_MAT_ID m 3072/5120 x n 24/48 at k=1024)
- 09c80fd7f local : `GGML_CUDA_MMQ_TILES=auto-l1` (L1 policy), `GGML_CUDA_MMQ_SK_MUTANT`, `mmq-hash` shapes for stream-k and MUL_MAT_ID
- fef62ffd3 local : MUL_MAT_ID hash k 1536 -> 1024
- 68c919c22, ae0cfb16a, 045e49d31, 54b6009f3, 4e6eb863d codex : model/sim, window, k8-gpu checks+perf, perf item template, Mac payload, Mac evidence

Source changes (all): `ggml/src/ggml-cuda/mmq.cuh` (device helper `ggml_cuda_mmq_get_sk_group`; `mul_mat_q` and `mul_mat_q_stream_k_fixup` use sk_group/part/sub; host `mmq_stream_k_plan`; `launch_mul_mat_q` uses it; `ggml_cuda_mmq_use_small_i`), `tests/test-backend-ops.cpp`. No other compile fixes.

## Design and exactness
Exact vs stock IS possible. fp32 accumulation order is fixed by the k split points of each output tile and the fixup add order. A narrow (I=64) tile that partitions the k range in default-tile (I=128) units, with `sk_group = I_default/I_tile = 2` consecutive blocks sharing one partition (same kbc range; `sub = blockIdx.x % group` picks the row half), reproduces the default tile's k ranges, tile ownership and fixup walk exactly, for every shape. The fixup kernel reads slot `bidx*group+sub`. Grid is `nparts*group` blocks (72 at nsm=36 = same warps/SM as the default's 36x256 threads): "occupancy-aware", `group <= config.occupancy` asserted. Default tiles have group 1: launch unchanged (only the 8 L1 small-I configs have occupancy > 1).
Selector and launcher share one plan, so the L1 gate "plain tiling only" is dropped: small-I now is taken on stream-k shapes too (3072/5120/6144-row at 12/24 cols), bitwise equal to default. Requires `nrows_x % 128 == 0`, else default tile.
Side effect (deviation from L1): L1 `auto` differed from default at m=11520 (20/362 hash cases; L1's own 64-row grid was plain, default stream-k). K8 makes that exact too. 11520 is not in the Swift model, unmeasured.
Not proven for CUDA: the model transcribes the kernel arithmetic; real bitwise equality is the window hash gate (auto == default, zero diffs).

## Mac results (M)
`bw-mac . codex/k8/bw-k8-mac.sh l1-k8` -> out `work/build-window-1007/l1-k8-mac-a1`, HEAD 54b6009f3, rc 0, total 2m41s wall: 1-sim 3 s, 2-build 84 s (CPU test-backend-ops), 3-cases 20 s (MUL_MAT 60/60, MUL_MAT_ID 20/20 pass), 4-hash 30 s (`mmq-hash CPU 3072` run twice, identical, 128 lines, 8 id_ lines). Evidence: `codex/k8/mac-evidence/{summary.json,sim.txt}` (source sha checked by window preflight).
Sim (5256 cases, k split + float add order, nsm 36): K8 vs default 0 differ; integer-term completeness failures 0.

## Negative controls
- Sim: L1 own-tile grid differs from default in 2348/5256 cases, forced grid=nsm 4043/5256, misaligned 4863/5256; fixup sub-block mix-up caught 3/3 (all > 0 required, PASS printed).
- Window (not yet run): `auto-drop` (small-I launches dropped) must fail MUL_MAT on stream-k shapes and not q8_0 (`must_fail`); hash arm auto-drop must differ on stream-k keys and on gated plain keys; `SK_MUTANT` (nparts = nsm+1, misaligned partition) must differ on stream-k keys, not on plain keys, and be a subset of auto-drop diffs; default == stock (frozen/stack8) hash; tile-log engagement strings (e.g. `m=3072 cols=12 ... small_i=1 units=24 parts=36 blocks=72`) with small_i=0 strings for auto-drop.

## Window
`/home/xanewok/repos/qwen38-eval-worktrees/wt/l1-k8/codex/k8/bw-l1-k8.toml` (filler, priority 18, preemptible=false, host_sensitive=false, est 3000/budget 9000 s, same GPU UUIDs/recipe as stack10-cand: 120a, CUDA 13.2, NCCL OFF, CCCL 3.4.3, DeviceTopK count 25). Stages: cuda-build, stamps, bin-assemble (`k8-bin`, no freeze), gates+controls on both GPUs (`codex/k8/gpu/k8-gpu.sh checks`), render + check perf item. Run `--preflight` passes.
- `gpuq submit --check codex/k8/bw-l1-k8.toml` -> `bw-l1-k8: valid`
- perf item `k8-mmq-perf` (priority 29, measure, ccx01) rendered by the window to the drafts dir; dry run with stand-in k8-bin: `k8-mmq-perf: valid` (`[fix] protected` accepted). Submit it only after the checks window passes.
- shlint + `bash -n` clean on bw-l1-k8.sh, k8-gpu.sh, bw-k8-mac.sh.
Perf arms (`k8-gpu.sh perf`, `perf_summary.py`): default / auto-l1 / auto / auto-drop, hot n=12|24, cold n=12, shapes 8704/5120, 5120/8704, 3072/5120, 6144/5120, 5120/3072, 5120/5120, 9216/5120, 17408/5120, 124160/5120, both GPUs.

## Expected gain (G)
np2 stream-k class (3072/5120/6144-row at 12/24 cols) is ~10.85 ms/round; if the narrow tile gives L1's ~5 % there: -0.1..-0.55 ms/round, 0 if parity; depth-independent (same at 8k and 90k). Proposal-2 hypothesis: old small-I stream-k loss (+3..+30 us) was the nsm-block grid (4 warps/SM); 72 blocks restores default warp count.
Falsifier: `auto - auto-l1` and `auto - default` in perf.tsv on 3072/5120/6144 x 12/24 not below A/A spread (negative = win); if >= 0 K8 is correctness-only (still removes the 11520 inexactness), then default-off the stream-k selection is a one-line change (use plain-only).

## Blockers / uncertainty
- CUDA never compiled; first compile errors would show in window stage 40.
- Gain unmeasured; MUL_MAT_ID small-I engagement at the id path is only a note in the checks (expert-split columns).
- `nrows_x % 128 != 0` falls back to default tiles (no small-I) by design.
