# Flat-cost dispatch sweep

Private bench-only experiment on served fork tip `726f57ffe`. No production threshold changes.

`GGML_CUDA_MMVQ_MAX_COLS=0..8` is read once in `ggml_cuda_should_use_mmvq`. It overrides dense Blackwell IQ4_XS, Q5_K and Q6_K only. Zero selects MMQ at every positive width. Eight exposes Q5_K n=7/8 and Q6_K n=8 controls. Unset preserves dispatch exactly; invalid values abort. MMVQ above eight columns is unavailable.

Existing thresholds: `ggml/src/ggml-cuda/mmvq.cu` Blackwell switch, originally lines 310-321: Q5_K <=6, Q6_K <=7, IQ4_XS <=8 (`MMVQ_MAX_BATCH_SIZE`, `mmvq-params.h:8`). Current switch starts at line 326.

## Artifacts

- `bench.py --prepare --out DIR` reads GGUF metadata and tensor tables only. Each source header is bounded to 200 MiB; the observed reads are 10,995,377 and 10,995,988 bytes. No tensor data is read or mapped.
- `evidence/shapes.json` records source paths, file sizes, header SHA256, actual target tensor names and counts, and equal-TP2 shapes. `evidence/shapes.md` is its table. Split policy follows `llama_meta_device_get_split_state`; FFN gate and up remain separate, output uses 124160 rows. The 65-block metadata includes one MTP block, excluded from target counts. Embedding lookup and types outside IQ4_XS/Q5_K/Q6_K are outside the weighted proxy.
- `bench.py --run --binary PATH --out DIR` requires `GPUQ_ACTIVE=1` and explicit visible GPU IDs. Each child sees one GPU as CUDA0; logs must identify RTX 5060 Ti / CC 12.0. Synthetic input data are deterministic across route processes. Identical type/shape/width/batch cells share measurements across model inventories; each inventory retains its own tensor counts.
- `item/flatcost-sweep.{toml,sh}` are unsubmitted drafts. The shell entry remains in this worktree when the parent copies the manifest. It pins the committed source tip and builds through buildgate during GPU-hidden preflight, using CUDA 13.2, local CCCL, no network fetch, NCCL off, four build threads, nice 19 and `/mnt/scratch/tmp-build`.

## Measurement

Widths are 1, 5, 6, 7, 8 and 12. Each legal cell measures served dispatch, forced MMQ and forced MMVQ; n=12 measures served dispatch and MMQ. GDN also measures n=6 with two activation batches separately.

The perf-only selector is `GGML_TEST_FLATCOST_CASE=type_id,m,n,k,batches`. It creates one exact `test_mul_mat` case. The perf path adapts `5934ca651`: distinct initialized weight copies, `copies=ceil(64 MiB/weight_bytes)+1`, a full warm rotation before timing, allocation and initialization outside timing, no reuse before completing the ring. Distinct activation tensors prevent Q8 producer reuse from hiding quantization cost. CUDA graphs are disabled. This is a cold-weight method, not measured proof of DRAM traffic; that requires profiling.

Six process-level paired blocks alternate route order. Default/default bookends in each block enforce maximum absolute drift <=0.5%. Each cell sample has a one-second timed floor; 93 unique cells per GPU give a 5400-second full-run timing floor. The JSON reports raw samples, bootstrap percentile 95% CIs, paired speedup, count-weighted model totals and separate GDN controls. The Markdown tables report latency and original weight bytes / latency. Global cutoff controls 4..8 provide exploratory weighted n6 ratios, sum of positive tensor-count-weighted n7->n8 increments removal and T8/T1; per-type/shape decisions use raw cells and require confirmation.

All times cover the complete operation, including quantization, MMQ fixup and amortized synchronous graph submission. Pure multiply kernel times, DRAM bandwidth saturation, correctness and stall attribution are unmeasured. No performance conclusion follows from syntax checks.

Expected (hypothesis): some 36-SM thresholds outperform RTX-5090-tuned dispatch. Design gate: IQ4 n7/8 geometric-mean speedup >=1.10x, neither large shape loses, n6 regression <=1%; full count-weighted gate removes >=50% of the positive increment. T8/T1 <=1.10 is an objective, not a kill rule. A/A failure invalidates comparisons. No eligible threshold win falsifies dispatch as the cheap lever; no inference about the best new kernel follows without the design's profile.

Planning estimate: 18000 seconds of queue occupancy, including CPU initialization and GPU timing, plus preflight build variability. The manifest allows 14400 seconds for GPU-hidden preflight, 900 for canary and 28500 for the full run within 45000 total. This estimate is uncalibrated; no benchmark was run.
