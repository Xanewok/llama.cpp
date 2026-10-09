# Restrict probe: does #24030 (no `__restrict__` on Hopper+ with PDL) cost kernel time outside MMQ?

Status 2026-10-09: window built, offline-validated, `gpuq submit --check` valid; not submitted (parent submits). No GPU data yet.
Labels: M measured, D derived, G guess.

## Tree and window
- Tree `~/repos/qwen38-eval-worktrees/wt/restrict-probe`, branch `topic/restrict-probe`, base `45d90d3e4` (= `frozen/stack9-p55` SOURCE).
- `e7ddefd3d` local: `-DGGML_CUDA_FORCE_RESTRICT` makes `GGML_CUDA_RESTRICT` expand to `__restrict__` while PDL stays on (common.cuh).
  Perf-only arm: it reintroduces the #24030 hazard; never served, outputs not checked for identity.
- `ac887c950` local: perf cases at the served decode shapes in `tests/test-backend-ops.cpp` (second commit, beyond the brief's one: the stock
  perf list has none of these shapes, and the window cannot run `-o OP -p REGEX` without them).
- Window `codex/bw-restrict-probe.{toml,sh}` + `restrict_sass.py` + `restrict_summary.py`; item `bw-restrict-probe`, filler, priority 17,
  `host_sensitive = false`, run wrapped in `work/aa-load/ccx01.sh` (CPUs 0-5,12-17), 120a, CUDA 13.2, NCCL OFF, CCCL 3.4.3.
  Both builds from this tree, build dirs `/mnt/scratch/restrict-probe/{stock,forced}/build`; only `-DCMAKE_CUDA_FLAGS=-DGGML_CUDA_FORCE_RESTRICT` differs
  (stamp stage checks flags.make equality apart from the switch, DeviceTopK == 25 in both libs).
- est_s 7500 (G), budget_s 12000. Basis (D): 2 cold CUDA builds ~700 s each (bw-k9 a1), SASS census 5-10 min, CUDA0 5 arm-runs x (242 MUL_MAT at ~2 s/case,
  l1-mmq-perf hot: 72 cases in 133 s, plus ~30 other cases), CUDA1 5 arm-runs x 120 cases.
- `gpuq submit --check codex/bw-restrict-probe.toml` -> `codex/bw-restrict-probe.toml: bw-restrict-probe: valid` (rc 0). Preflight under `env -i`: PASS.
  `python3 -m gpuq.shlint` on the .sh: rc 0. Output under `codex/window/restrict-a{attempt}/` (`summary.json`, `perf.tsv`, `served.tsv`,
  `sass-counts.tsv`, `perf-verdict.txt`).

## Stages (rc 5 control void, 6 integrity, 8 deterministic)
1. builds, 2. stamps, 3. SASS census, 4. perf arms (stock, forced, stock, forced, stock per GPU), 5. summary.
- SASS (`restrict_sass.py`): per object of the #24030 files (fattn*, getrows, mmvf, mmvq, quantize, sumrows, mean, set-rows, ssm-conv, ssm-scan)
  count `LDG*.CONSTANT`, plain `LDG*`, generic `LD`. Must-differ control: `mmvq.cu.o` forced has more `.CONSTANT` and fewer plain LDG, else rc 5.
  Census code checked against grep on a real `mmvq.cu.o` (11370 / 4145, identical); void path checked with identical dirs (rc 5).
- Perf: `test-backend-ops perf -b CUDAn -o OP -p REGEX --output sql`, one process per (arm, op set). Each filter's row count must equal the case
  count (rc 8 otherwise; stricter than "matches >= 1").

## Shapes and sources
Model metadata (M, read from the GGUF header of `Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf`): arch qwen35, 64 blocks + 1 MTP layer, n_embd 5120, ffn 17408,
24 Q / 4 KV heads, key/value length 256, `full_attention_interval` 4 (16 attention + 48 GDN), ssm conv 4, state 128, group 16, time_step_rank 48,
inner 6144; vocab shard 124160 per GPU. Per GPU of the TP2 split (D): 12 Q / 2 KV heads (GQA 6), GDN 8 K groups x 3 V heads, 5120 conv channels
(matches `test_ssm_conv_state_fusion::check_gallocr` c = 5120). MUL_MAT (m x k) list = the L1 perf set (`wt/l1-mmq-smallj` test, results
`2026-10-08-l1-mmq-perf`): 8704x5120 ffn gate/up, 5120x8704 ffn down, 6144x5120 attn q+gate, 3072x5120 GDN z, 5120x5120 GDN qkv, 5120x3072 out
projections, head 124160x5120 q6_K (n = 1, 6 only).

| set | op | cases (expected rows) |
|---|---|---|
| mm | MUL_MAT iq4_xs, q4_K, q5_K, q6_K, q8_0 x 6 shapes x n=1..8, + head q6_K n=1,6 | 242 (CUDA0) |
| mm1 | same types, shapes 8704x5120, 5120x8704, 5120x5120, n=1..8 | 120 (CUDA1) |
| fa | FLASH_ATTN_EXT hsk=hsv=256, nh=2, nr23=[6,1], f16 KV laid out like the cache, KV 8192/32768/90112 x nb 1/6/12 | 9 |
| conv | SSM_CONV ne_a {3+n_tok, 5120, n_seqs}: (1,1), (6,1), (6,2) | 3 |
| gdn | GATED_DELTA_NET head_count 8, head_size 128, v_repeat 3, K = n_tok: same three | 3 |
| gr | GET_ROWS f32 393216 cols (GDN state row) m=8 r=1,2; 5120 cols m=6 r=6 | 3 |
| sr | SET_ROWS f32 -> f16, i64 idx, ne {512, 90112}, r = 1, 6, 12 | 3 |
| scan | SSM_SCAN generate cases already in the perf list (Swift has no SSM_SCAN; upstream coverage only) | 2 |

Filter validation (M, offline): a CPU-only build of this tree (`/mnt/scratch/restrict-probe/cpu-check`) ran `perf -b CPU --output sql` with the exact
window filters: fa 9, conv 3, gdn 3, gr 3, sr 3, scan 2 rows, all supported and passed; the first 16 mm rows matched in order. mm = 242 and mm1 = 120 are D
(5 x 6 x 8 + 2; no other perf-list case has these m/k pairs); the window checks the count and fails rc 8 at once if wrong. The summary and the
perf stage plumbing ran end to end against a fake binary (synthetic +6 % on iq4_xs MMVQ rows is flagged `cost`, MMQ-routed rows stay null).

## Verdict rules (fixed before the run)
- Row: stock median over 3 runs, forced median over 2; A/A spread = max - min of the stock runs; `cost` when both forced repeats exceed
  stock median + spread. Family verdict: `cost` when >= 50 % of rows are flagged and the median delta >= +0.5 %; `none` when <= 15 % flagged; else `mixed`.
- Null control inside the data: MUL_MAT rows that cc 12.0 routes to MMQ (q4_K n > 5, q5_K n > 6, q6_K n > 7) run kernels #24030 did not touch
  (mmq.cuh keeps `__restrict__`); > 25 % of them flagged voids the verdict (rc 5, outputs kept).
- Row G closes if the SASS control passes (forced has more `.CONSTANT` loads) and every touched family is `none`: #24030 costs nothing measurable outside MMQ.
  Expected (G): MMVQ +2...+8 %, FA decode 0...+5 %.

## Served impact (D), computed by `restrict_summary.py` into `served.tsv`
Per verify round and GPU, call count x (forced - stock) median delta on CUDA0, per pure type (type mix is G, so MUL_MAT is totalled per type; both
"sum of all measured deltas" and "cost-flagged rows only" are printed).
- np1 k5: n=6 target GEMMs 128 gate/up, 64 down, 16 attn q, 48 GDN z, 48 GDN qkv, 64 out proj (16 attn + 48 GDN), 1 head; MTP draft 5 steps at n=1
  (10 gate/up, 5 down, 5 attn q, 5 out); FA nb=6 x 16 and nb=1 x 5 at each KV; SSM_CONV 48; GDN 48; GET_ROWS 48 (rollback gather) + 1; SET_ROWS r=6 x 32, r=1 x 10.
- np2 k5: the 12-column target GEMMs are MMQ (untouched, 0); GDN out projection at n=6 read twice (3D input, 10-06 note; 96 calls, G); draft steps per slot
  at n=1 (20 gate/up, 10 down, 10 q, 10 out); FA nb=6 x 32, nb=1 x 10; SSM_CONV/GDN at (6,2) x 48; GET_ROWS r=2 x 48; SET_ROWS r=12 x 32, r=1 x 20.
- Not covered: draft vocab head (25879 rows), MUL_MAT_ID (dense model), reduce_rows consumers (SUM_ROWS/MEAN, rms norm paths), the fused GDN/conv-state
  variants (same kernel families; fusion tests are whole-graph and do not run under `perf`).

## Limits of the probe (decision-relevant)
- Weights of shapes below the 32 MiB L2 stay hot across the repeated nodes (this tree's `test-backend-ops` has no `GGML_TEST_PERF_COLD_MB`; it exists
  at `wt/l1-mmq-smallj` 275ad4038). Hot rows bound the delta from above for latency-bound kernels; 8704x5120 q6_K/q8_0 and the head exceed L2. If rows
  are `none` hot and the question stays open, port the cold-weight option as one more local commit.
- Microbench = repeated nodes of one op in a stream: PDL overlap and launch gaps are as in a decode graph, but there is no neighbouring-kernel interaction.
- The forced build is only a timing arm. A positive result still needs the K9-style fix form (`wt/k9-fix` 117c87e14, offset by the `griddepcontrol.wait` value) before anything is served.
