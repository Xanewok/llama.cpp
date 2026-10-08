# K6: residual ADD+RMSNorm fusion and shared activation quantization, split

Base `0cfc0723e`, branch `topic/s10-k6`. Source from `wt/q8-dedupe` (`849f16315`, `7967e192c`), not edited there. Labels: M measured, D derived, G guess.

## Commits
- `8bf9c7492` cuda : reuse shared matmul activations of up to 16 rows within a graph (b)
- `aac6d37fb` cuda : fuse residual add with RMS normalization of up to 16 rows (a)
- local: shared-activation off switch, engagement log, stale-cache mutant; residual fusion off switch, engagement log, numerical mutant
- codex: Mac gates, host contracts, window, evidence

(a) and (b) are independent (each applies to the base alone). No env var, switch or log in the general commits.

## Design and exactness
- (a) `add_rms_norm_f32` (`norm.cu`) writes the ADD result and the MUL result and never materializes the RMS_NORM result. Same block-size rule, `block_reduce`, `rsqrtf(mean+eps)` and `(scale*x)*w` order as `rms_norm_f32<..., true>`. Matcher `add-rms-norm.h: ggml_cuda_can_fuse_add_rms_norm` is host code (Mac-testable). Row cap 8 -> 16 (`GGML_CUDA_ADD_RMS_NORM_MAX_ROWS`) is a measurement bound, not an exactness bound. Added guard vs q8-dedupe: no operand may share the buffer the unfused RMS_NORM would have written. Any refusal falls back to the unfused path.
- (b) `q8-cache.h` + hooks in `mmq.cu`, `mmvq.cu`, `ggml-cuda.cu`. Entries keyed by (src1 tensor identity, quantizer variant); a write overlapping an input's byte span (every node, every node of a fused group) invalidates it; each capture starts invalid, so replays re-quantize current data. Row bound 8 -> 16 (`ggml_cuda_q8_reuse_input_ok`) replaces the per-dimension and MMVQ_MAX_BATCH_SIZE guards; it is a retention bound. Invariant: ops write only their dst span or a fused group's members.
- D engagement: Swift J5q census regrouped at 12 rows (all MMQ): 128 shared groups / 144 removable quantizations (same as 6 rows; 8 rows gives 80); 128 residual pairs.

## Expected effect and falsifier
D ceiling -0.3 ms/round at np2 (128 x 1.5 us + 144 x 0.94 us), depth-flat; G realized 0.3 to 1.0 of it (-0.09 to -0.3). np1 null may have been PDL-hidden (PDL saves 0.27 ms/round at np1); np2 MMQ launches are not PDL-overlapped. Falsifier: with positive engagement counters (`add_rms_norm:` rows=12, `q8_reuse:` skipped>0; per verify `quantize_mmq_q8_1` ~394 -> ~250, `k_bin_bcast<add>` ~177 -> ~49), a np2 8k/90k round change under 0.1 ms beyond between-boot spread falsifies usefulness.

## Tests and negative controls
- (b) `MUL_MAT_SHARED` NMSE cases + `run_shared_activation_identity` (160 cases: shared vs private-copy consumers, byte-compared, in-graph overwrite, 4 replays, built-in comparison control).
- (a) `RMS_NORM_MUL_ADD` rows 12/16/17 + `run_add_rms_norm_identity` (324 cases: chain vs output-flagged unfused reference; plain, in-place add, in-place chain, partial overlap; cols 1..5120, rows 1..33).
- Host contracts `codex/k6/q8-cache-contract.cpp`, `codex/k6/add-rms-norm-matcher.cpp` with sed-mutated header copies that must exit 1 (stale invalidation, bound 17, cap 17, norm-alias guard removed, residual-overlap guard removed).
- CUDA must-fail (window): `GGML_CUDA_Q8_REUSE_STALE_MUTANT`, `GGML_CUDA_ADD_RMS_NORM_MUTANT`; off switches `GGML_CUDA_Q8_REUSE=0`, `GGML_CUDA_ADD_RMS_NORM=0` give zero markers; rows 17/33 give no marker. Removing a matcher overlap guard is not detectable by a CUDA mutant (kernel is per-row/per-thread safe); only the Mac matcher table covers it (G).

## Mac results
MAC_RESULTS

## Window
`codex/k6/bw-s10-k6.{toml,sh}`: filler, priority 18, non-preemptible, host_sensitive=false, stack10-cand recipe (arch 120a, CUDA 13.2, NCCL OFF, CCCL 3.4.3, GCC 13, DeviceTopK 25). Builds test-backend-ops; per GPU: both identity tests, engagement greps, off switches, both mutants must fail; one memcheck on CUDA0. No freeze, no timing item. `gpuq submit --check`: CHECK_LINE

## Open
CUDA code uncompiled/unexecuted (first compile in window stage 40); CUDA byte identity is a window gate; capture/replay engagement is G; real-graph engagement (adjacency after graph_optimize, Meta splits) unmeasured; MUL_MAT_ID reuse untested; a gate/up pair fused in MMVQ reserves a slice that is never hit (wasted scratch).
