# g32-step: extra bytes explain most of the Swift HQ step penalty

## Conclusion

The best supported explanation is memory traffic, not a demonstrated Marlin tile regression. The supplied checkpoint A/B is 30.9 ms g128 versus 33.2 ms g32i8/codexh (M, parent measurements). The header-based incremental traffic model gives 2.093 ms/step (D): 1.185 ms for extra GDN/MLP scales and 0.908 ms for attention. This accounts for 91.0% of the 2.3 ms difference (D). The remaining 0.207 ms is a model residual (D), **not a measured kernel-efficiency penalty**.

A kernel-only recovery of the entire 2.3 ms while preserving these checkpoint values is unsupported. Start with the delivered cold/warm per-shape microbenchmark, then test g64 on the same calibration recipe if a recipe trade is acceptable. Do not enable the existing sm86 tuning table on sm120 without rebuilding and retuning it. No production kernel or sampling code changed; no GPU work ran.

Labels throughout: M = measured, including supplied measurements and existing trace samples; D = calculated from headers or a stated model; G = unmeasured estimate. Structural counts, shapes, source line numbers and version identifiers describe artifacts rather than performance measurements. All byte/time tables below explicitly carry their label. Bandwidth 448 GB/s/card is the supplied premise, not newly measured here.

## Dispatch and checkpoint schemes

Paths abbreviated below: `V` = `/home/xanewok/repos/HyperQwen/venv/lib/python3.12/site-packages/vllm`; `C` = `/home/xanewok/repos/vllm-patches/csrc/libtorch_stable/quantization/marlin`; `SP` is the scratchpad in the brief. Read the four requested background reports before design. Their llama.cpp host-gap budget concerns a different engine and does not explain this HQ checkpoint delta.

Observed checkpoint configuration and source dispatch:

| Runtime linear | TP2 local K,N | g128 | g32i8 and codexh |
|---|---|---|---|
| `mlp.gate_up_proj` | 5120,17408 | W4A16 g128 | W4A16 g32 |
| `mlp.down_proj` | 8704,5120 | W4A16 g128 | W4A16 g32 |
| `linear_attn.in_proj_qkvz` | 5120,8192 | W4A16 g128 | W4A16 g32 |
| `linear_attn.out_proj` | 3072,5120 | W4A16 g128 | W4A16 g32 |
| `self_attn.qkv_proj` | 5120,7168 | W4A16 g128 | W8A16 per channel |
| `self_attn.o_proj` | 3072,5120 | W4A16 g128 | W8A16 per channel |
| `linear_attn.in_proj_ba` | 5120,48 | BF16 unquantized | BF16 unquantized |
| `lm_head`, MTP linears | unchanged | W8A16 g128 | W8A16 g128 |

All quantized linears above select `MarlinLinearKernel` on sm120 under the measured default activation settings (source-derived dispatch, high confidence). The 8-bit attention scheme is symmetric `pack-quantized`, `strategy=channel`, `group_size=null`, with no activation quantization. `CompressedTensorsWNA16` converts this to `group_size=-1` and `scalar_types.uint8b128`. It is **not W8A8**, not grouped int8, not a CUTLASS int8 activation GEMM. BF16 activation and loaded scales are the model default. Serialized body/attention scales are BF16; some unchanged MTP scales are F16 and are loaded through the configured parameter dtype.

Source chain: `V/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:809` selects `CompressedTensorsWNA16`; `schemes/compressed_tensors_wNa16.py:36`, `:74`, `:128`, `:142` maps bit type, channel groups, parameters and kernel choice. The priority list at `V/model_executor/kernels/linear/__init__.py:499` tries CUTLASS, Machete and AllSpark before Marlin. They reject sm120: `mixed_precision/cutlass.py:35`, `machete.py:37`, `V/model_executor/layers/quantization/utils/allspark_utils.py:26`. `mixed_precision/marlin.py:95` accepts these groups/types; `:189`, `:227`, `:304` repacks once, permutes scales once and calls `apply_gptq_marlin_linear`. `V/model_executor/layers/quantization/utils/marlin_utils.py:695` calls `ops.marlin_gemm`.

Fusion matters: `V/model_executor/models/qwen3_5.py:220` and `:303` maps q/k/v, gate/up, GDN qkv/z and b/a into fused linears. `q_proj` includes the output gate, so its global N is 12288, not 6144. `V/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:431`, `:442`, `:496`, `:573` establishes GDN fusion and TP sharding. Embeddings use their gather implementation, not a full linear GEMM. The BF16 b/a projection uses the unquantized linear path; backend/compiler-specific GEMM names cannot be established from config alone.

The measured overlays change AllReduce and sampling. No supplied evidence changes this linear dispatch. `VLLM_MARLIN_INPUT_DTYPE=int8/fp8` would change the body arithmetic and is outside the measured W4A16 comparison; W8A8 is explicitly rejected by the installed Marlin wrapper.

## Header inventory and bandwidth model

`codex/analyze.py` reads only the safetensors length/header, not weight payloads. `evidence/modules.csv` records every packed tensor's bytes, scale bytes/dtype/shape; `evidence/inventory.json` records config/header hashes, totals and trace kernel statistics. All three checkpoints have identical byte layout outside the body/attention changes. This establishes layout identity, not equality of weight values or acceptance.

The following are global model totals in decimal MB, D from tensor header offsets. TP2 divides the affected tensors equally. There are 64 MLP layers, 48 GDN layers and 16 full-attention layers; the six fused shapes account exactly for the affected packed weights.

| Header family | Weight MB g128 (D) | Weight MB g32i8 (D) | Scale MB g128 (D) | Scale MB g32i8 (D) |
|---|---:|---:|---:|---:|
| `mlp.gate_proj` | 2852.127 | 2852.127 | 89.129 | 356.516 |
| `mlp.up_proj` | 2852.127 | 2852.127 | 89.129 | 356.516 |
| `mlp.down_proj` | 2852.127 | 2852.127 | 89.129 | 356.516 |
| GDN `in_proj_qkv` | 1258.291 | 1258.291 | 39.322 | 157.286 |
| GDN `in_proj_z` | 754.975 | 754.975 | 23.593 | 94.372 |
| GDN `out_proj` | 754.975 | 754.975 | 23.593 | 94.372 |
| attention `q_proj` | 503.316 | 1006.633 | 15.729 | 0.393 |
| attention `k_proj` | 41.943 | 83.886 | 1.311 | 0.033 |
| attention `v_proj` | 41.943 | 83.886 | 1.311 | 0.033 |
| attention `o_proj` | 251.658 | 503.316 | 7.864 | 0.164 |

Model: `extra_ms = extra_global_bytes / 2 / 448e9 * 1000`. One target forward reads each affected weight and scale once, independent of its row count. MTP does not execute the full target body again for each draft; unchanged draft/head traffic cancels. Tiny norms, b/a, KV and AllReduce sizes also cancel in this comparison. Ignore cache residency, redundant reads, overlap, thermal state and launch gaps. This is an incremental streaming estimate, **not a formal lower bound on the difference between two complete engines**.

| Contribution | Extra bytes/card (D) | Extra ms/step (D) |
|---|---:|---:|
| GDN/MLP g128 -> g32 scales | 530841600 | 1.185 |
| attention W4 g128 -> W8 channel, including scale savings | 406634496 | 0.908 |
| total | 937476096 | 2.093 |

Plain g32 everywhere adds 570163200 scale bytes/card, predicting 1.273 ms/step (D). Earlier measured g32 overhead was 1.35-1.39 ms (M, `SP/vast-hq/ab_summary_g32.md:3`, `:6`), consistent with that estimate. Upgrading attention from W4 g32 to W8 channel predicts a further 0.820 ms (D); adding it to the earlier 1.37 ms midpoint gives 2.19 ms (D). Those campaigns differ in engine/context/MTP settings and cannot be subtracted as a paired attribution. The fidelity figures supplied in the brief are g128 KLD 0.0254, g32i8 0.0159 and codexh 0.0134 (M, REF6).

At batch 1-6 rows the arithmetic intensity is `2*m*K*N / ((bits/8 + 2/group)*K*N)`: W4 g128 3.88*m FLOP/B, W4 g32 3.56*m FLOP/B (D). A W4 g32 tensor has 12.5% scale/weight traffic, versus 3.125% for g128 (D); the total weight-plus-scale increase is 9.09% (D). This makes memory traffic a credible primary limit for these small-row GEMMs. It does not prove achieved bandwidth.

## What the existing kernels and traces establish

`C/marlin.cu:81` ranks small-batch tiles `(K=128,N=128,threads=256)`, `(64,128,128)`, `(128,64,128)`. `:107` allocates scales proportional to `ceil(tile_K/group_size)`; `:204` selects the first valid kernel rather than benchmarking candidates. `:294` uses four stages outside Turing, `:319` uses the small-row path and `:409` requests the full available dynamic shared-memory budget. The normal W4A16/W8A16 path does not have an sm120-specific tile policy; `C/marlin_mma.h:71` uses BF16 `mma.sync`, not a native low-bit weight MMA. Architecture support exists; a missing sm120 fast path is not established.

At a K128/N128 tile, four-stage scale storage is 1024 B g128 versus 4096 B g32 (D). Scale fetch/register paths differ at `C/marlin_template.h:534`, `:779`, `:847`; W8 per-channel scale application is in the epilogue at `:1736`. More group scale loads/shared-memory instructions may add cost, but their runtime contribution is unknown. W8 also doubles the packed-weight tile storage. Compiler register counts, stalls and actual occupancy for g32/W8 channel need a matched trace or profiler run.

Existing trace evidence (M): `SP/vast-hq-d/ab/prof/nsys-n3-1.sqlite`, corroborated by `serve-n3-1.log:11`, uses the g128 checkpoint. It contains only Marlin group-blocks=8 (g128), no group-blocks=2 (g32) or -1 (channel) Marlin kernel. GPU0 W4 kernels use 256 threads, K128/N128 tiles, four stages, 102 registers/thread, 101376 B dynamic shared memory and no local memory. Their whole-capture mean is 62.975 us (M). W8 g128 kernels use 125 or 128 registers/thread (M). Whole-capture means mix shapes/rows and cannot attribute individual linears.

The supplied steady summary `SP/vast-hq-d/ab/prof/nsys_per_gpu.txt:4` reports GPU0 Marlin 19.791 ms/step and mean 71.82 us (M); this is an older g128 MTP run, not the current patched k=5 checkpoint A/B. `SP/engine-plan/step0/out/profile/summary-nsys.json` reports GPU0 Marlin 21.389 ms/step over 199 steps (M), also g128. Its raw trace confirms the same W4 tile/register configuration. No real g32 or g32i8 trace exists in the requested three trace directories. `step0/mock/` fixtures were excluded.

Provenance limit: the installed Python is HQ-patched vLLM; the available C++ is the adjacent `vllm-patches` source checkout. It is not proven to be the build source of the installed CUDA library. Existing template arguments corroborate the tile policy on g128 only. `evidence/source-sha256.txt` pins the inspected sources; a future GPU run must also pin its built library.

## Candidate fixes, ranked

Expected savings below refer to the current g32i8/codexh step. G ranges are prioritization guesses, not GPU results. Byte savings are D streaming estimates; actual step gains can differ.

| Rank / candidate | Expected ms/step | Fidelity / exactness | Implementation and scope |
|---|---|---|---|
| 1. Test group-aware sm120 Marlin tiles/staging before a rebuild | 0-0.3 ms (G); the byte model leaves only 0.207 ms unexplained (D), though common-path gains could exceed that | Same checkpoint values; require bit-identical outputs. Tile/reduction reorder can change logits and is not automatically distribution-exact | Kernel selector patch, upstream-shaped if architecture/shape evidence supports it. Existing HQ tuning hook is local, off by default |
| 2. g64 GDN/MLP, retain W8 channel attention | 0.790 ms (D) traffic saving; 0.6-0.9 ms (G) step saving | KLD unknown; does not preserve current checkpoint or sampling distribution | Recipe change, fresh GPTQ/calibration build and REF6 KLD needed; supported Marlin group, no kernel patch |
| 3. g128 GDN/MLP, retain W8 channel attention | 1.185 ms (D) traffic saving; 1.0-1.3 ms (G) step saving | Gives up some g32 fidelity; own-recipe control is still missing | Recipe change and new GPTQ/KLD. Stronger speed experiment than speculative scale packing |
| 4. Keep W8 q/k/v, quantize only o to W4 g32 | 0.246 ms (D); 0.2-0.3 ms (G) | KLD unknown; q/k/v fusion remains intact | Recipe change, GPTQ/KLD; supported local recipe |
| 5. Keep W8 o, quantize q/k/v to W4 g32 | 0.574 ms (D); 0.5-0.7 ms (G) | KLD unknown, likely riskier than o-only (G). q/k/v must share a scheme | Recipe change; GPTQ/KLD. Quantizing just q or k/v differently needs an unfused model/kernel path first |
| 6. Lossless scale packing/load scheduling | 0-0.2 ms (G); no general lossless byte reduction is established | BF16 -> F16 still uses two bytes and may change range/rounding. FP8 scale compression changes values/distribution | Kernel/storage patch only if exact decode is proven. Generic exponent compression is speculative/local; ordinary 8-bit scales are a recipe change |
| 7. Faster W8A16 kernel on sm120 | 0-0.2 ms (G); cannot eliminate the 0.908 ms (D) additional traffic on its own | Retain per-channel W8 values and BF16 activation; enforce exact outputs | New/ported kernel, potentially upstream-shaped. Existing AllSpark excludes sm120; Machete/CUTLASS candidates exclude it too |
| 8. Selective g32 tensors | 0-1.185 ms (D) traffic range, no defensible expected KLD | Previous sensitive set gave KLD 0.0242 vs 0.0250 (M) and failed to recover global g32 benefit | Recipe change. `PLAN.md:482` supports rejection of that set, not a universal impossibility. No new sensitivity evidence here |

`VLLM_MARLIN_TUNE=1` exists at `V/_custom_ops.py:1215` and loads a standalone vLLM 0.27.1 sm86 table. The referenced `~/marlin-tune` installation is absent. Its table key documented at `:1220` has shape/bits/m but no group size; a group-aware sm120 sweep is needed. Failures silently fall back to stock at `:1260`, so a speed experiment must verify the chosen provider. HQ's prior W4A8/prefill tuning gain washed out end-to-end (`/home/xanewok/repos/HyperQwen/docs/optimizations.md:192`); this is not a small-row W4A16 result.

`VLLM_MARLIN_USE_ATOMIC_ADD=1` is not a useful primary knob here: `marlin_utils.py:625` only enables it for N<2048 and K>=2048; all six fused affected shapes have larger N. Splitting q/k/v merely to unlock it sacrifices fusion and alters reduction numerics. `VLLM_MARLIN_INPUT_DTYPE=int8/fp8` quantizes activations, changes the distribution and does not supply W8A8 support. None of these flags is an exact shortcut to g128 speed.

## Delivered parent GPU experiment

`codex/gpu_ab.sh` defaults to a CPU-only dry plan. `marlin_bench.py --execute` refuses unless `PARENT_GPU=1` and `CUDA_VISIBLE_DEVICES` is nonempty. It imports GPU libraries only after that guard. It tests all six Swift TP2 fused shapes at rows 1 through 6, W4 g128/g32/g64 and W8 channel. W8 on GDN/MLP shapes is an exploratory kernel comparison; the current checkpoint uses W8 only on full attention. Both ranks are measured separately, not concurrently: this isolates per-card kernels and excludes TP/AllReduce/power interaction.

Use two **already frozen, independently installed environment prefixes** (up/patch), including Python, vLLM .py/.so, torch libraries and any candidate extension under SHA256SUMS. Do not point at a live checkout. Module path guards reject vLLM/torch resolving outside the prefix. The parent must include all runtime files in the manifest; the script validates hashes but does not prove manifest completeness. Run from the GPU queue:

    PARENT_GPU=1 CUDA_VISIBLE_DEVICES=<two-GPU-UUIDs> UP_ROOT=<frozen-up> PATCH_ROOT=<frozen-patch> bash codex/gpu_ab.sh --execute

Optional `UP_MARLIN_TUNE` / `PATCH_MARLIN_TUNE` select a validated candidate hook; default is stock. The six fresh boot order is up, patch, patch, up, up, patch. There is a hard overall budget of 1200 s and each rank process gets at most 180 s. The script refuses existing Xids, busy GPUs or an existing `llama-server`, and stops only its own exact timeout PID. No server is started; these are fresh microbenchmark boots. Outputs stay under this worktree. Interrupted/incomplete matrices fail summary rather than silently produce partial A/B results.

Benchmark design: deterministic synthetic quantized weights, BF16 inputs/scales, FP32 reduction enabled and atomic reduction disabled as in the serving wrapper. CUDA graphs measure warm cache and a separate cold mode that clears 128 MiB before each timed GEMM. External timing events exclude the flush. Cold mode approximates rotating target weights; repeated warm microbenchmarks alone could incorrectly hide scale/weight traffic. No repack, initialization or reference GEMM is timed. Median/range are recorded; no minimum-only speed claim.

Correctness checks: each variant is checked against a dequantized BF16 matmul with numerical tolerance; corrupted output must fail that check. Identical deterministic output hashes must match across frozen up/patch boots; graph replay must also be bit-identical. The tolerance check alone does not establish exact sampling. A candidate failing the hash gate is not accepted as an exact optimization. The real checkpoint still needs matched-logit/seed and REF6 checks before deployment; synthetic equivalence is necessary, not sufficient.

To estimate target-step delta from cold results, sum `count * (g32-g128)` for the four GDN/MLP shapes plus `count * (W8channel-W4g128)` for the two full-attention shapes, divide microseconds by 1000 (D). Use each rank separately and the slower rank for a TP critical-path hypothesis. Do not sum draft passes into that delta: their quantization is unchanged. Confirm with the existing parent end-to-end checkpoint harness at fixed k/context/flags; this script does not establish end-to-end throughput or acceptance.

## CPU verification and scope

All executed commands used `CUDA_VISIBLE_DEVICES=''` and `nice -n 19`; no GPU executable, server, network request, model inference, CUDA build or llama.cpp build ran. This is a source/header/trace study, so the Qwen3.5 CPU inference test does not validate its subject and was not run.

Executed verification: `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 codex/check_cpu.py`. It runs `python3 codex/analyze.py` against three checkpoints and two SQLite traces read-only; fused-shape packed-byte and fourfold body-scale checks pass. `bash codex/gpu_ab.sh` takes the no-import CPU dry path. Syntax and synthetic summary tests, including a mismatched-output negative control, are recorded in `evidence/cpu-checks.txt`. CPU analyzer wall time is recorded there (M); it is not a decoding performance number. GPU graph event compatibility, frozen-prefix preparation and numerical tolerance on real hardware remain unverified.

The supplied `.git` is a read-only placeholder. An isolated local topic checkout was initialized with `.topic-git` inside this workspace, fetched the requested base from the local `llama.cpp-stack-resync-fmq` checkout without network and checked out `topic/g32-step`. Use `git --git-dir=.topic-git` here. The repository source is unchanged; only `codex/` study artifacts are committed. No other checkout/branch was modified.

Open items: matched g32/g32i8 trace; cold microbenchmark; group-aware sm120 selector sweep if justified; own-recipe g64 or g128+W8 control with REF6 KLD. Evidence currently supports a bandwidth explanation and bounded experiments, not a proven exact 7% decode win.
