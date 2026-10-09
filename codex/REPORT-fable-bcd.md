# Fable B/C/D existence probes: design, controls, verdict rules (written before the run, 2026-10-09)

Status: NOT RUN. Nothing was compiled with nvcc and nothing ran on a GPU (brief limit). Syntax was checked host-only with g++ for
`tests/test-fable-bcd.cpp` and `fable-probe.cu`; the device code, the PTX strings (notably `cp.async.bulk.prefetch.L2.global`) and the
K9-fix MMQ substrate are unvalidated until the window's build stage. A build failure ends the window with rc 8 and costs no GPU verdict.

Tree: `~/repos/qwen38-eval-worktrees/wt/fable-bcd`, branch `topic/fable-bcd`, base `45d90d3e4` (= frozen/stack9-p55 SOURCE).
Commits: `4e571e2ca` substrate (MMQ PDL in the K9-fix form, 3 files from wt/k9-fix `117c87e14`), `2e803a2af` probes, `8f9b04841` window.
All of it is local probe code, never served.

## Window
- Item `bw-fable-bcd`: `codex/bw-fable-bcd.toml`, `codex/bw-fable-bcd.sh`; output `codex/window/fable-bcd-a{attempt}/`
  (`verdict.txt`, `summary.tsv`, `served.tsv`, `summary.json`, per-run `run-*.tsv`, `stages.tsv`, `clocks.csv`).
- filler, priority 16, `host_sensitive=false`, retry never, run wrapped in `work/aa-load/ccx01.sh` (CPUs 0-5,12-17).
- Build: recipe 120a, CUDA 13.2, `-DGGML_CUDA_NCCL=OFF`, CCCL v3.4.3, build dir `/mnt/scratch/fable-bcd/build`, via `bin/buildgate -j4`, target `test-fable-bcd` only.
- Stages: cuda-build, stamps (probe API in libggml-cuda.so, 120a flags, clean tree at HEAD), sass-info (informational CCTL count per object),
  runs (CUDA1 all levers -> CUDA1 PDL-off lever C stock -> CUDA0 all levers), summary. Exit: 0 pass, 5 control void, 6 integrity, 8 deterministic.
- est_s = 2700 (G), budget_s = 5400. Basis: one cold build of ggml-cuda + the test target ~700-1500 s (bw-k9 a1; ccache may cut it) plus
  three driver runs at ~5-10 min each. Within the brief's 3600 s est limit. Budget = 2x est.
- `gpuq submit --check` output (M): `/home/xanewok/repos/qwen38-eval-worktrees/wt/fable-bcd/codex/bw-fable-bcd.toml: bw-fable-bcd: valid`.
  Also: `bw-fable-bcd.sh --preflight` under `env -i HOME=$HOME PATH=/usr/local/bin:/usr/bin:/bin` -> `PASS preflight`; `python3 -m gpuq.shlint` rc 0.

## Common method
Probe program `tests/test-fable-bcd.cpp` (dlsym of `fable_probe_api` in `ggml/src/ggml-cuda/fable-probe.cu`, so no new libggml API). Each
config builds one ggml graph on one CUDA device, runs 4 warmup computes, then reps of: flush kernel (>= 64 MB sweep, > 32 MB L2, G) ->
optional prefetch/spin kernel -> event -> graph -> event. Metric `trim_us` = 10 %-trimmed mean of the graph interval; `host_us` per rep is reported
to flag a host-bound graph path. Arms are interleaved round-robin over R = 7 rounds. CUDA1 (headless) is the verdict device, CUDA0 the replicate.
Every lever carries an identical control pair (A, A2) measured each round. **S = max - min over all 2R control samples. base_r = mean(A_r, A2_r).
eff_r = base_r - X_r (positive = X faster). An arm "fires" iff min_r eff_r > S** (every repeat beats the spread); else "none" (inside the A/A).
Numbers are M only from the run; everything below marked G/D is expectation or derivation.

## B: L2 prefetch of the next GEMM's weights during the AllReduce wait
- Design: one MMVQ GEMM (the served op, `GGML_OP_MUL_MAT`, graph of one node) with the weight at a cold L2 (flush before every rep). Preceded
  by a kernel standing in for the AR wait: 8 blocks x 256 threads (as `ggml_cuda_ar_kernel`) that issue `prefetch.global.L2` over the first
  5/10/19 MB of the weight in address order, and thread 0 spins on `%globaltimer` for 23 us (n=6) / 42 us (n=12) (the measured AR waits, Fable §3.B M).
- Configs (8): {iq4_xs, q6_K} x {8704x5120 (ffn), 5120x5120} x n {6, 12}. Weight pool of 2 matrices (target and decoy).
- Arms: `cold_a`, `cold_b` (A/A), `warm` (no flush: must-fire), `pf5`, `pf10`, `pf19` (line prefetch), `wrong10` (decoy matrix prefetch: must-not-fire),
  `bulk10` (`cp.async.bulk.prefetch.L2`), `ldcg10` (`ld.global.cg` into a sink).
- Verdict: **exists** iff `pf10` fires on >= 50 % of the 8 configs (>= 4); report saved us per MB = median eff / 10 MB. Fable expectation: ~22 of ~45 us for 10 MB (D).
  **none** otherwise (B closes with its number). **VOID (rc 5)** if `warm` fires on < 75 % of configs (baseline not cold) or `wrong10` fires on > 25 % (not L2 retention of the target).
- Caveat: the spin kernel serializes before the GEMM, so a prefetch that is still in flight when the GEMM starts is counted as helping; the real AR
  kernel overlaps its wait the same way (the GEMM starts after the AR returns), so this matches the served ordering. Bandwidth contention with the AR's PCIe traffic is not modelled.

## C: weight prefetch above `ggml_cuda_pdl_sync()` in MMVQ and MMQ
- Design: variant arms set the global `g_fable_probe_pf` (read at launch; in a captured graph it is baked into kernel params, so each arm has its own graph).
  MMVQ (`mmvq.cu`, `mul_mat_vec_q`): when `fusion.probe_pf != 0 && ids == nullptr`, before `ggml_cuda_pdl_sync()` each thread issues `prefetch.global.L2`
  for the first `probe_pf` k-iterations' weight blocks (both ends of each block) of its rows. Inputs written by the producer stay after the wait (brief; `.nc` visibility under PDL is unproven).
  MMQ (`mmq.cuh`, `mul_mat_q`, stream-K and non-stream-K): between `ggml_cuda_pdl_lc()` and `ggml_cuda_pdl_sync_offset()` replay the first-tile selection and prefetch
  128 B lines of the first `probe_pf * pf_bpi` k-blocks of the tile rows. Base `45d90d3e4` MMQ has no PDL, so MMQ-C exists only on the K9-fix substrate (`4e571e2ca`); its
  result therefore answers "would the lever pay if K9 is served", not "stock stack9".
- Chains (48 links each; per link add -> rms_norm -> mul -> mul_mat; weights rotated so reuse distance >= 128 MB > L2): `updown` (5120->8704->5120), `sq5120`, `small` (5120->3072->5120).
- Configs (12): {iq4_xs n=6, q6_K n=6, iq4_xs n=1, iq4_xs n=12} x 3 chains. Routing on cc 12.0: n<=8 iq4_xs and n<=7 q6_K -> MMVQ, n=12 -> MMQ; asserted by `ROUTE` rows (mismatch -> rc 8).
- Arms: `stock_a`, `stock_b` (A/A; stock_a vs stock_b is also the null control: variant code compiled in, `probe_pf=0`), `pf1`, `pf2`, `pf4`, `pf16` (k-iterations / depth prefetched).
  Diagnostic (not a gate): stock with `GGML_CUDA_PDL=0` (`run-nopdl-CUDA1.tsv`) vs PDL on; if PDL gives nothing in this chain, C cannot gain.
- Verdict per class (MMVQ n=6, MMVQ n=1 draft, MMQ n=12): **exists** iff `pf1` fires on >= 50 % of the class's configs; **exists (deeper only)** if only `pf2/pf4/pf16` fire; else **none**.
  us per call-link = median eff / 48 links. **VOID** if S > 2 % of base on > 50 % of C configs. Fable: -1…-1.5 ms/round total (G), i.e. ~2-4 us per weight GEMM pair.
- Caveat: a 48-link isolated chain has the producer kernel immediately before every GEMM, like the served graph, but the producer here is a cheap elementwise op, so the
  PDL-shadow is shorter than behind a heavy kernel; the result is a floor for the shadow benefit.

## D: GDN verify writes one state instead of K = 6 snapshots
- Design: 48 `GGML_OP_GATED_DELTA_NET` ops (head 128, 8 K-heads, 24 V-heads, each layer its own state and output) built with `ggml_gated_delta_net(..., K)` where K = 6
  writes 6 snapshots (`keep_rs_t`) and K = 1 writes the final state only (perf-only; outputs ignored). K = 3 is the dose point.
- Modes: `iso` (scans only, upper bound) and `mixed` (each scan followed by 2 IQ4_XS GEMMs, 8704x5120 and 5120x8704 from 6-matrix pools, so DRAM is busy like the served layer; primary).
- Shapes: (n_tokens, n_seqs) = (6,1) np1 k5 verify and (6,2) np2 k5 (the same shapes `restrict_summary.py` times at the op level). 4 configs.
- Arms: `K6_a`, `K6_b` (A/A), `K3`, `K1`. Per-sequence saving = eff / n_seqs per round (the graph is the 48 GDN layers).
- Verdict: **exposed** iff K1 fires and >= 0.6 ms/seq; **partial** iff fires and 0.2..0.6; else **none** (Fable thresholds). Dose check eff(K1) > eff(K3) > 0 reported.
  **VOID** if S > 2 % of base on > 2 of 4 D configs. Fable expectation: up to -0.84 ms/round/seq if exposed (D), realized -0.4…-0.8 (G); replay cost +0.37 ms/seq (D) is NOT in the probe.

## Shapes and sources
- Weight shapes 8704x5120 / 5120x5120 / 5120x3072: Swift (Qwen3.8-27B) FFN, attention/projection and small projections as in `restrict-probe/codex/restrict_summary.py` MM_NP1 set (G for the mapping to layer roles).
- AR wait 23 / 42 us at 6 / 12 rows: Fable REPORT §3.B (M, prior probe). L2 32 MB: assumed (G); the program reports `cudaDevAttrL2CacheSize` in the META row.
- GDN geometry and K = n_rs_seq + 1 = 6: `delta-net-base.cpp:595`, Fable §3.D; state 128*128*24*4 B = 1.57 MB/slot/layer, 75.5 MB over 48 layers (D).

## Served impact (D = derived at verdict time; call counts G; written by `served.tsv`)
Per verify round and GPU, ms = calls x measured us (the window fills the us). Assumptions: np1 k5 = 1 sequence, 6-column target verify + 5 draft n=1 steps; np2 k5 = 2 sequences, 12-column target verify.
- B: 128 AR calls/round (Fable §3.B, G). np1: 128 x `pf10` saved us (n=6 waits); np2: 128 x `pf19` saved us (n=12). Fable bound: -1…-2 (np1), -1.5…-2.5 (np2) ms/round (G).
- C: np1: 192 gate/up/down + 48 qkv + 112 z/out-proj + 17 other = 369 target GEMMs at MMVQ n=6 (counts from `restrict_summary.py`, G) x per-link `pf1` us, + 25 draft n=1 GEMMs x n=1 per-link us.
  np2: the same 369 at MMQ n=12 (needs the K9-fix PDL to be served: K9 was cancelled, so this is conditional) + 50 draft GEMMs at n=1. Fable bound -1…-1.5 ms/round (G).
- D: 48 GDN layers, K6 -> K1 saving per sequence from the measured (6,1)/(6,2) `mixed` rows, minus the replay cost 0.37 ms/seq (D, Fable) to get the net; `served.tsv` prints the gross (replay not subtracted).
- If a lever's verdict is none, its served impact is 0 and the number printed is the inside-spread value, not a gain.

## Self-test of the verdict logic
`codex/test_fable_bcd_summary.py`: synthetic data, 12 checks, all pass (M, CPU only): null data (all arms = control) gives rc 0 and "none" for B, C (every class) and D (negative control); injected effects give B exists, C exists, D partial at 0.3 ms/seq and exposed at 1.0 ms/seq; silent `warm`, firing `wrong10` decoy and a noisy A/A each give void rc 5.

## Known risks (decision-relevant)
- Not compiled: first window attempt may die in the build (rc 8). The summary and window scripts are validated; the CUDA code is not. If it fails, the log is `stages`/`cuda-build.log`; fixes are code-only, no verdict lost.
- MMQ-C needs the K9-fix PDL substrate, never built in this exact form (bw-k9-fix was cancelled); an MMQ build error or a PDL hazard there affects only the `MMQ n=12` class.
- `ggml_backend_graph_compute_async` of small graphs may be host-bound; `host_us` / `trim_us` ratio is reported per row and the summary flags rows where host time dominates.
- The B model serializes the stand-in spin kernel before the GEMM; the real AR kernel's HBM/PCIe contention is not modelled, so B "exists" is an upper bound for the served gain.
