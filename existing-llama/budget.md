# Attribution budget: pre-fill from EXISTING llama.cpp traces (no GPU); k=3 builds, not the k=5 stack2 the item will measure

Labels: M = read from the trace; D = derived (arithmetic on trace values, model sizes or published constants); G = guess. Axis times are M (priority partition of GPU0's timeline); floors are D (bytes / (tp*448 GB/s), AR table 10 KiB 9 us .. 40 KiB 14 us, best demonstrated, extrapolated above); the prefill compute floor uses a G peak.

- model: Swift 1.5 IQ4_XS on llama.cpp stack2 (draft-vocab subset head), -sm tensor, 2 GPUs
- engine: llama

### item47-np1-8k (2026-09-29, host draft sampling, full 248k draft head, k=3)

engine llama, depth 8192, k=3, 70 steady rounds, round 38.36 ms (GPU0 window, sd 0.51; host launch-to-launch 38.36; GPU1 differs by <= 0.00 ms), draft forwards per round 4, 75.0 tok/s at 2.88 tok/round

| axis | ms/round | % round | target ms | draft ms | kernel-sum ms | floor ms | headroom ms | headroom % round |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FWD | 25.94 | 67.6 | 20.89 | 5.05 | 25.94 | 20.80 | 5.14 | 13.4 |
| KV | 2.62 | 6.8 | 2.41 | 0.21 | 2.98 | 0.82 | 1.81 | 4.7 |
| COMM | 2.35 | 6.1 | 2.18 | 0.17 | 2.35 | 1.87 | 0.49 | 1.3 |
| ELEM | 0.71 | 1.9 | 0.65 | 0.06 | 2.94 | 1.02 | 0.00 | 0.0 |
| STEP | 6.73 | 17.5 | 1.83 | 4.90 | 0.00 | 0.52 | 6.21 | 16.2 |
| **sum** | 38.36 | 100.0 | 27.96 | 10.40 | | 25.02 | 13.64 | |

All floors sum to 25.02 ms: ideal speedup 1.53x if every axis hit its floor (floors use the theoretical 448 GB/s per GPU, the GDN state read+write charge (guess) and, for ELEM and STEP, values demonstrated by HQ at k=3 (not physical limits); FWD and KV are additive memory traffic).

Per GPU (ms/round): GPU0: FWD 25.94, COMM 2.35, KV 2.62, round 38.36; GPU1: FWD 25.91, COMM 2.22, KV 2.61, round 38.36

Sub-rows (ms/round): FWD.lm_head target 1.26 / draft 3.70; KV_ATTN target 1.01 / draft 0.21; KV_GDN target 1.40 / draft 0.00; STEP.SAMP target 0.00 / draft 0.00; STEP.COPY target 0.20 / draft 0.18; STEP.IDLE target 1.63 / draft 4.72
Host gap split (llama, summed over the round's inter-phase gaps, ms): readback 0.52, cpu_pre 4.11, inputs 1.36, launch_lat 0.13
AllReduce: V: 128 calls x 14.0 us floor at 40 KiB; C: 2 calls x 10.7 us floor at 20 KiB; D1: 2 calls x 9.0 us floor at 10 KiB; D2: 2 calls x 9.0 us floor at 10 KiB; D3: 2 calls x 9.0 us floor at 10 KiB

Top kernels GPU0 (ms/round, calls/round): mmvq23x4 17.32 (328); mmvq14x1 3.70 (3); mmvq14x4 2.52 (31); ggml_cuda_ar_kernel 2.35 (136); quantize_q8_1 1.07 (433); mmvq2x1 1.01 (21); rms_norm_f32 0.95 (333); flash_attn_ext_f16 0.57 (20)

### round-trace-0.7k (2026-10-01, backend draft sampling + subset draft head, k=3, prompts 600-790 tokens)

engine llama, depth 700, k=3, 87 steady rounds, round 31.02 ms (GPU0 window, sd 0.21; host launch-to-launch 31.02; GPU1 differs by <= 0.00 ms), draft forwards per round 4

| axis | ms/round | % round | target ms | draft ms | kernel-sum ms | floor ms | headroom ms | headroom % round |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FWD | 22.51 | 72.6 | 20.76 | 1.76 | 22.51 | 17.67 | 4.84 | 15.6 |
| KV | 1.94 | 6.2 | 1.87 | 0.07 | 2.29 | 0.39 | 1.55 | 5.0 |
| COMM | 1.98 | 6.4 | 1.84 | 0.14 | 1.98 | 1.87 | 0.11 | 0.4 |
| ELEM | 0.79 | 2.6 | 0.70 | 0.10 | 3.11 | 1.02 | 0.02 | 0.1 |
| STEP | 3.80 | 12.2 | 1.46 | 2.34 | 0.16 | 0.52 | 3.28 | 10.6 |
| **sum** | 31.02 | 100.0 | 26.61 | 4.41 | | 21.47 | 9.79 | |

All floors sum to 21.47 ms: ideal speedup 1.44x if every axis hit its floor (floors use the theoretical 448 GB/s per GPU, the GDN state read+write charge (guess) and, for ELEM and STEP, values demonstrated by HQ at k=3 (not physical limits); FWD and KV are additive memory traffic).

Per GPU (ms/round): GPU0: FWD 22.51, COMM 1.98, KV 1.94, round 31.02; GPU1: FWD 22.27, COMM 2.25, KV 1.85, round 31.02

Sub-rows (ms/round): FWD.lm_head target 1.25 / draft 0.40; KV_ATTN target 0.40 / draft 0.07; KV_GDN target 1.46 / draft 0.00; STEP.SAMP target 0.08 / draft 0.08; STEP.COPY target 0.03 / draft 0.05; STEP.IDLE target 1.34 / draft 2.20
Host gap split (llama, summed over the round's inter-phase gaps, ms): readback 0.17, cpu_pre 1.79, inputs 0.80, launch_lat 0.09
AllReduce: V: 128 calls x 14.0 us floor at 40 KiB; C: 2 calls x 10.7 us floor at 20 KiB; D1: 2 calls x 9.0 us floor at 10 KiB; D2: 2 calls x 9.0 us floor at 10 KiB; D3: 2 calls x 9.0 us floor at 10 KiB

Top kernels GPU0 (ms/round, calls/round): mmvq23x4 17.20 (328); mmvq14x4 2.48 (31); ggml_cuda_ar_kernel 1.98 (136); mmvq2x1 1.17 (26); quantize_q8_1 1.10 (433); rms_norm_f32 0.99 (333); concat_non_cont 0.55 (48); gdn 0.45 (48)

### round-trace short prefill (505-token warm-up prompt; NOT a 32k prefill)

engine llama, 505 tokens in 0.5 s of GPU0 window = 935 tok/s (1070 ms per 1k tokens), batch 128 tokens

| class | ms | share % | kernel-sum share % | Q1 % | Q2 % | Q3 % | Q4 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| GEMM | 206 | 38.1 | 63.9 | 38.8 | 36.1 | 36.7 | 41.5 |
| GDN | 44 | 8.2 | 14.1 | 8.8 | 8.3 | 8.3 | 7.1 |
| ATTN | 6 | 1.0 | 1.7 | 0.9 | 0.9 | 1.0 | 1.4 |
| COMM | 36 | 6.6 | 11.2 | 0.7 | 0.6 | 1.6 | 28.7 |
| ELEM | 26 | 4.9 | 9.1 | 5.3 | 4.7 | 4.8 | 4.8 |
| SAMP | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| COPY | 76 | 14.0 | 0.0 | 17.8 | 16.7 | 16.4 | 2.4 |
| IDLE | 147 | 27.2 | 0.0 | 27.7 | 32.8 | 31.1 | 14.1 |

Quarter midpoints (tokens of context): 63, 189, 315, 441. GEMM = weight GEMV/GEMM (incl. MMQ); GDN = gated-delta-net/conv/chunk kernels; ATTN = flash attention + KV writes.

Floors: weight re-stream 64 ms (4 batches of 128 tokens, each re-reads target weights / (tp*BW)); compute 68 ms (2*params*T + causal attention FLOPs over tp * 190 TOPS (int8 tensor-core peak per GPU used by the MMQ path (guess, vendor-class number; fp16 peak is about half))). Headroom vs the larger floor: 472 ms (87% of the window).

Top kernels GPU0 (ms, calls): mmq23 172 (1312); gdn 37 (192); ggml_cuda_ar_kernel 33 (129); mmq14 13 (120); rms_norm_f32 11 (1246); mul_mat_q_stream_k_fixup 8 (1111); k_bin_bcast 6 (903); unary_gated_op_kernel 5 (520)

