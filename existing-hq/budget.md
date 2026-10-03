# Attribution budget: pre-fill from the EXISTING HyperQwen 8k trace (idle desktop, k=3, NCCL AllReduce, default top-k); the item will measure codexh k=5 with host-staged AR and fast top-k

Labels: M = read from the trace; D = derived (arithmetic on trace values, model sizes or published constants); G = guess. Axis times are M (priority partition of GPU0's timeline); floors are D (bytes / (tp*448 GB/s), AR table 10 KiB 9 us .. 40 KiB 14 us, best demonstrated, extrapolated above); the prefill compute floor uses a G peak.

- model: Swift 1.5 W4A16 g128 syv-fast on HyperQwen (existing 8k traces), TP2
- engine: hq

### hq-idle-8k (2026-09-29, g128 syv-fast, k=3)

engine hq, depth 8192, k=3, 197 steady rounds, round 29.81 ms (GPU0 window, sd 0.62; host launch-to-launch 29.77; GPU1 differs by <= 0.00 ms), draft forwards per round 3

| axis | ms/round | % round | target ms | draft ms | kernel-sum ms | floor ms | headroom ms | headroom % round |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FWD | 21.06 | 70.7 | 18.97 | 2.09 | 21.06 | 18.42 | 2.64 | 8.9 |
| KV | 1.33 | 4.5 | 1.21 | 0.12 | 1.33 | 0.71 | 0.63 | 2.1 |
| COMM | 4.17 | 14.0 | 3.71 | 0.45 | 4.17 | 1.96 | 2.21 | 7.4 |
| ELEM | 1.11 | 3.7 | 0.98 | 0.13 | 1.11 | 1.00 | 0.11 | 0.4 |
| STEP | 2.14 | 7.2 | 1.99 | 0.15 | 1.56 | 0.52 | 1.79 | 6.0 |
| **sum** | 29.81 | 100.0 | 26.86 | 2.94 | | 22.60 | 7.37 | |

All floors sum to 22.60 ms: ideal speedup 1.32x if every axis hit its floor (floors use the theoretical 448 GB/s per GPU, the GDN state read+write charge (guess) and, for ELEM and STEP, values demonstrated by HQ at k=3 (not physical limits); FWD and KV are additive memory traffic).

Per GPU (ms/round): GPU0: FWD 21.06, COMM 4.17, KV 1.33, round 29.81; GPU1: FWD 21.08, COMM 4.27, KV 1.34, round 29.81

Sub-rows (ms/round): FWD.lm_head target 1.53 / draft 0.50; KV_ATTN target 0.70 / draft 0.12; KV_GDN target 0.51 / draft 0.00; STEP.SAMP target 1.54 / draft 0.02; STEP.COPY target 0.03 / draft 0.02; STEP.IDLE target 0.42 / draft 0.11
AllReduce: V: 130 calls x 14.0 us floor at 40 KiB; D1: 5 calls x 9.0 us floor at 10 KiB; D2: 5 calls x 9.0 us floor at 10 KiB; D3: 5 calls x 9.0 us floor at 10 KiB

Top kernels GPU0 (ms/round, calls/round): void marlin::Marlin<(long)1125899906909960, (long)1125899907 16.07 (256); ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsSt 3.81 (138); void marlin::Marlin<(long)1125899906909960, (long)1125899923 3.62 (19); _topk_topp_kernel 1.39 (1); void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616g 1.37 (48); kernel_mha(unsigned int, unsigned int, unsigned int, const u 0.66 (17); fused_sigmoid_gating_delta_rule_update_kernel 0.38 (48); ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(un 0.36 (7)

