# Grouped one-row q8_0 attention: evidence

Private engineering evidence, not PR text. Root BASE: a7b94df2c; root source: 47cf5e79b (tests: 9b574c486). Local BASE: 6ef137376; local source: e7f0447e9. Cumulative patches include CUDA attention, its instances and native operation cases. Private CPU dispatch/harness files stay outside the package. Full analysis: [REPORT-q8-general-fix.md](../../../REPORT-q8-general-fix.md).

## Measured regression and cause

M probe6: `codex/runs/q8-probe6/full-a1/{results,correctness}.csv`, both GPUs, ABBA, M 250 ms windows. All q8 cases pass correctness; f16 failures agree across arms; M f16 median time ratio is 1.000. M patch/up medians at KV8192 / 32768 / 90112:

| D | GQA | M patch/up |
|---|---|---|
| 256 | 4 | 0.70 / 0.67 / 0.64 (M long-KV 398 -> 255 us) |
| 256 | 2 | 1.22 / 1.13 / approximately 1.00 |
| 256 | 1,6,8 | within approximately 4% |
| 128 | 2 | 5.1 / 6.8 / 12.1 (M long-KV 142 -> 1728 us) |
| 128 | 4 | 2.9 / 3.7 / 7.3 |
| 128 | 6 | 2.2 / 2.5 / 5.1 |
| 128 | 8 | 1.8 / 1.9 / 3.9 |
| 128 | 1 | approximately 1.00 |
| 64 | 2 | 2.6 / 3.6 / 3.8 |
| 64 | 4 | 1.8 / 2.0 / 1.9 |
| 64 | 6 | 1.35 / 1.45 / 1.35 |
| 64 | 8 | approximately 1.0-1.1 |
| 96 | all tested | approximately 1.04 / 0.86 / 1.8 (M long-KV 732 -> 1330 us) |

M D128 patch times are approximately 87 / 308 / 1720 us across GQA. D identical head tile and stream-K geometry explain the fixed work across GQA: ncols1=1, ncols2=8, four warps; each KV head is split across the same blocks regardless of unused query-head slots.

Source: `fattn-mma-f16.cuh:501-537` uses 16-byte cp.async only when raw q8 row bytes divide by 16. D64/96/128 rows are 68/102/136 bytes and use synchronous 2-byte global reads/shared writes; D256's 272-byte row uses asynchronous copies. D raw-copy instruction amplification is eight. D smaller heads use KV tiles of 128, D256 uses 64. M nvcc -Xptxas=-v and cuobjdump show all four q8 kernels consume 255 registers; D128 spills M 56 bytes stores and 64 bytes loads (M stack 48 bytes), while others have no spills (M stack 16 bytes). D dynamic shared sizes are 41744 / 66576 / 83216 / 83088 bytes, plus M static 1024 bytes. D standard sm_120 resource limits admit 2 / 1 / 1 / 1 resident blocks per SM. This is resource-derived occupancy, not a GPU query.

D stream-K at 36 SMs and two KV heads launches 72 blocks for D64, 36 for D96/128/256, splitting each destination 36 or 18 ways, followed by uniform fixup. Source: `fattn-common.cuh:1181-1228,1307-1324`. No split-one or unfilled-SM explanation survives. Direct q8 launch disables f16 scratch conversion; allocation checks prove that it was absent. Static SASS has M LDG counts 417/624/825/18 and LDGSTS counts 6/6/6/114 for D64/96/128/256, including all compiled loop variants. Source/SASS establish the loading penalty and register pressure. Runtime attribution among these and cache/bank effects remains unmeasured.

## Eligibility design

The pipelined grouped path now estimates copy work, rather than counting only unique global bytes. D cost per grouped KV pass: 16*(raw_K/chunk_K + raw_V/chunk_V) + expanded_f16_KV, where chunk is the actual loader width (16 or 2). Multiply by ceil(GQA/ncols2). Compare with D VEC cost GQA*(raw_K+raw_V), or D conversion cost raw_K+raw_V+(1+groups)*expanded_f16_KV when VEC has no instance. Strictly lower estimated cost is required.

This is a G scheduling proxy, not a proven cycle model. Single-stage partial converters retain their prior eligibility. No model name, GPU name, context threshold or shape blacklist is added. Existing supported head-pair and layout gates remain. D aligned staging costs approximately 2.882 unique q8 row equivalents per group; narrow staging costs 9.882. A tile holds at most eight query heads, so the latter cannot amortize against VEC under this proxy. D96 conversion is also cheaper. The expanded shared-store term rejects D256/GQA2, retaining the observed D256/GQA4 win.

D routes: D64/D128 -> VEC; D96 -> existing f16 conversion/MMA; D256/GQA1-2 -> VEC; D256/GQA>=3 -> direct grouped q8 where the inequality and original layout gates pass. D192/128 -> conversion/MMA. D320/256,512/512,576/512 retain single-stage eligibility. Multi-query dispatch and sampling are unchanged. GQA3 is CPU-covered but awaits GPU evidence.

## Validation and queue

M 210 actual selector/allocation shapes pass on both branches; M six negative controls fail, including bypassed cost gating. M 16 loader/swizzle cases preserve half values; M eight CPU attention cases pass, max split NMSE 5.61224804e-11. D summed measured case time: 6.188 s, one thread. CUDA compilation used nice 19, at most eight build workers, hidden CUDA devices, CUDA 13.2, sm_120a, NCCL OFF and offline CCCL. No CUDA executable ran.

M frozen q8-general-fix2 SOURCE and libggml-base stamp match e7f0447e9, the enclosing HEAD at configure/freeze. Probe7 keeps q8-general-base2 as its base, the full case matrix, both-GPU ABBA, exact-PID cleanup, word-bounded hazard checks, strict q8 NMSE and matched f16 failures. M 13 unit tests include full recorded probe6 SQL and the existing real GPU/device/memory/hazard fixtures. CPU preflight and gpuq submit --check pass; no submission occurred. D run budget 1500 s, canary timeout 300 s, manifest budget 1800 s; G estimate 1000 s. Canary is one paired kernel A/B cycle, no server boots.

G small-head regression cases should return to approximately 1.00 patch/up because they restore base routes; D256/GQA4 should retain approximately 0.70/0.67/0.64. These are predictions, not new GPU results. Portability of the staging heuristic, GQA3 speed and wide-head GPU correctness remain open. The former byte-only G prediction of 250-410 GB/s for grouped D128 is withdrawn. D unique-byte/time throughput for the reported two-head D128 long-KV case is 28.4 GB/s; the brief's 57 GB/s implies double the unique bytes. Neither is measured DRAM bandwidth. Network policy prevented upstream issue/duplicate checks.
