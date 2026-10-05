# Grouped one-row q8_0 flash attention

Implemented; GPU adoption remains blocked on the queued correctness/performance canary. No GPU or network work ran. Master code is `topic/q8-1row-general-master` at `e6d897de6`, based on `a7b94df2c`; stack code is `3eab968dd`, based on `6ef137376`. Package: `codex/upstream/llama.cpp/54-grouped-q8-one-row-fa/`.

## Cause and traffic

Observed source at `6ef137376`: `fattn.cu:157` restricts direct q8 MMA to equal head widths of 256 and GQA > 4; `fattn.cu:721` repeats that grouping gate. `fattn-mma-f16.cuh:2328` and the three q8 instance files cover only `(256,256,{1,4,8},8)`. Thus D=128 uses VEC for every tested GQA, and D=256/GQA=4 uses VEC. GQA=1 already reads each KV head once and reached M 402/411 GB/s for D=128/256 in probe3; retaining VEC there is intentional.

D unique q8 bytes, with two KV heads and separate K/V:

| D head | KV 8192 | KV 32768 | KV 90112 |
|---|---:|---:|---:|
| 128 | 4,456,448 | 17,825,792 | 49,020,928 |
| 256 | 8,912,896 | 35,651,584 | 98,041,856 |

D VEC logical reads are GQA times these bytes: D=128/GQA=1,4,6,8 reads 1,4,6,8 times the floor; D=256/GQA=4 reads four times it. Logical loads include cache hits. D=96 previously converts the whole view: D logical K/V read/write traffic is q8 bytes plus twice the f16 bytes. `q8-general-traffic.csv` covers each valid requested shape, its traffic, ideal time at the prior f16 bandwidth of M 410 GB/s, and G bandwidth ranges.

## General design

`fattn.cu:211,431,817,929` share eligibility between selection, direct dispatch and scratch sizing. Grouping uses eight head slots, including partial groups; GQA=1 keeps VEC where available. Eligibility uses existing layout, capability and KV-padding contracts, with no model, GPU-name, or arbitrary context threshold.

All block-valid f16 MMA head pairs are covered: 64/64, 96/96, 128/128, 192/128, 256/256, 320/256, 512/512, 576/512, with its admissible GQA ratios. Special wide tiles follow the existing f16 configuration coverage. D=80/112 cannot be q8 tensor widths: `ggml.c:1339` requires complete 32-value blocks. Their f16 cases remain; the probe explicitly records D 30 unsupported q8 combinations.

`fattn-mma-f16.cuh:501,614` handles half-aligned raw rows, odd block counts and dimension-batched large heads. Byte strides stay in bytes. Dequantization rounds q*d once to f16, as the existing conversion does. `:1622` bounds q8 sink reads in padded head slots; `:2235` allocates raw staging only when used. Sampling stays unchanged. GPU accumulation tolerance remains an open gate, not a CPU-proven result.

## Evidence

M CUDA compile: GGML_CUDA=OFF/ON builds used CUDA hidden, nice 19, no more than eight build workers, CUDA 13.2, architecture 120, NCCL OFF, offline CCCL. Logs: `codex/logs/general-{master,fix,base}-*.log`; master snapshot identities are explicit.

D q8 host instances: root adds 10; stack grows from 3 to 12, adding 9. M sm_120 CUDA library bytes: root 71,018,120 -> 72,411,504 (D +1,393,384, +1.96%); stack 71,724,192 -> 72,937,992 (D +1,213,800, +1.69%). Other targets are unbuilt.

M CPU: `CUDA_VISIBLE_DEVICES="" nice -n 19 bash codex/run_q8_general_cpu.sh`; 210 dispatch shapes, 16 head/swizzle loader cases, and five failing negative controls (disabled dispatch, unwanted f16 scratch, lost offset, unbounded sinks, integer-cast half scales). M eight CPU attention cases have maximum split NMSE 5.6123e-11; padding is bit-identical and every dropped-split control fails. D summed CPU case time is 6.535 s, one thread. M 12 probe tests pass on real recorded GPU SQL/logs, including identity, metric, budget and exact-PID cleanup gates. M both package patches apply to their bases and reproduce the committed files.

G bandwidth predictions: cache-resident shapes 250-1400 GB/s; nonresident D=96 150-350; D=128/256 grouped 250-410; large/asymmetric heads 100-350. No new GPU measurements. Below the requested KV lengths, launch overhead and padded compute may dominate. Large/asymmetric GPU correctness is open. Inherited f16 padded sink indexing is unchanged; existing f16 ERR failures require identical ERR across arms, while every q8 row must pass its original tolerance.

## Queue

`wt freeze` used `QWEN38_WT_ROOT=$PWD/codex` to obey the worktree-only write rule: `codex/frozen/q8-general-{base,fix}` have SOURCE, hashes and ORIGIN RUNPATH. M frozen preflight and `gpuq submit --check` pass. Master freeze was omitted; its CUDA library was built.

D full probe: 150 valid cases plus 16 controls, both GPUs, ABBA. Perf windows are D 250 ms per case; total budget is D 1800 s, including the D 300 s canary and D 1500 s run cap. G estimate is 1000 s. Canary is one paired cycle with ten cases and the controls; two arm suites are necessary to exercise the A/B integrity gate, with no server boots. Nothing was queued.

Parent submit command:
`gpuq submit /home/xanewok/repos/qwen38-eval-worktrees/wt/q8-1row-general/codex/q8-probe4/item.toml`
