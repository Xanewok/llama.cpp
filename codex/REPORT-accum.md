# Swift MMA value accumulation

## Verdict

Fix implemented for NVIDIA Turing and newer when `GGML_PREC_F32` is requested. The default half path is retained. GPU correctness and runtime cost of the fix remain unmeasured; queue `codex/gpu_accum_ab.sh` before adopting it.

The accumulator precision defect is real. The parent overflow result is explained by the unnormalized half numerator, not q8 staging or unused query columns. Swift's measured activations do not establish overflow at served 100k, but neither the short prompt nor the weights exclude it there. CPU models using real V values show a credible >1e-3 relative output error under diffuse attention. Precision is therefore a correctness risk worth fixing independently of whether Swift produces infinity in normal traffic.

Labels: M = measured/observed, D = derived, G = assumed scenario. Source line numbers, identifiers, flags, shapes and suite parameters are references/settings. No GPU measurement was executed here. Fix commit `20d8cfee8`. Branch `topic/fa-mma-f32-vkq` starts at `0d18ebdb3`, retaining the earlier fixture over fork base `68d99c67d`.

## Source and overflow mechanism

`src/llama-graph.cpp:2661` requests `GGML_PREC_F32` for flash attention. The original MMA launcher never reads that precision. Its wide `T_C_VKQ` is `tile<16,8,half2>` and its thin variant is `tile<16,4,half2>` (`fattn-mma-f16.cuh`, original lines 1202/1211). `mma.cuh` emits `mma.sync...f16.f16.f16.f16` for these tiles. QK and softmax row sums use float, but that does not promote VKQ.

`fattn-common.cuh:19` sets `c=3*0.6931`, approximately log(8). Each tile updates a running maximum M and uses weights `exp(logit-M-c)` represented as half before the V multiply. Before adding the next tile, `fattn-mma-f16.cuh:1009` rescales the previous numerator and denominator by `exp(M_old-M_new)`. This happens once per `nbatch_fa` keys, but the scale is one when the maximum is unchanged. There is no periodic division by the number of accumulated keys. Division by the row sum happens only at writeback; warp/block combination cannot repair an already infinite half value. The old shared-memory combine also stores unnormalized half values.

For one value coordinate and a warp/block partition S:

`A = sum(i in S, V_i * exp(logit_i-M-c))`.

Define `N_a=|S|`, `R=sum(exp(logit_i-M))`, and flatness `F=R/N_a`, with `0<F<=1`. For coherent constant-signed values of magnitude v, `|A|=v*N_a*F*exp(-c)`. The half finite-range limit is D 65504; under round-to-nearest the infinity boundary is D 65520. Thus the practical overflow condition is `v*N_a*F/8 >= 65520`. For mixed signed values use the weighted signed mean in that expression; `max|V|*N_a*F/8` is a sufficient upper bound on magnitude, not a prediction of overflow. Changing maxima can erase old mass; a flat local tail retains it. Each partition has its own maximum, so a global attention peak does not guarantee that every partial numerator is small.

For `Q=K=0`, represented weights round to D 0.125. `V=1024` contributes D 128 per key. D 512 equally weighted keys produce D 65536, beyond half range, although the correct normalized output is finite. Even a float numerator is finally divided by the float exponential row sum rather than its rounded half weights; D about 1.4e-4 relative weight-rounding bias remains for this flat fixture. The fix targets accumulation, not operand precision.

## Keys per accumulator

M: an existing, read-only Nsight SQLite capture (`SP/queue/out/np2-nsys/np1-k3-d8192.sqlite`, `TARGET_INFO_GPU`) records 36 SMs, 65536 registers/SM and 102400 shared bytes/SM on both RTX 5060 Ti cards. No GPU discovery command was run. Original compile resources plus the launch's shared-memory sizing give D two resident blocks/SM for the aligned verify widths, hence D 72 stream-K blocks.

`launch_fattn` in `fattn-common.cuh:1184-1224` uses runtime occupancy, not a fixed per-head block count. MMA calls it with `stream_k=true`. Ada and newer always choose stream-K: `B=min(SM_count*active_blocks_per_SM, KV_tiles*output_tiles)`, optionally rounded down to a multiple of output tiles if the loss is at most D 5%. The non-stream-K `parallel_blocks` path is not used by this MMA launch. It instead cuts the flattened tile/key iteration space at `floor(block_index*total_KV_tiles/B)` (`fattn-mma-f16.cuh`, `kbc`/`kbc_stop`). Fixup combines the resulting float partials.

For the aligned single-stream local Swift fixture, two KV heads and GQA 6 produce D two output tiles. Query tile width 4 has `ncols=32`, D four warps/block, D 16 columns/warp, and `np=4*16/32=2` parallel warps per column. Width 8 has `ncols=64` and D `np=1`. Both use D 32-key softmax tiles. A warp owns alternating 16-key pieces, not a separate full sequence. For an output tile split uniformly over P blocks, the maximum is `N_a <= (nbatch_fa/np)*ceil(KV_tiles/P)`; use the actual flattened block interval for an exact count.

D bounds for the original aligned fixture, with P=36:

| KV setting | Maximum keys/warp, width 4 | Maximum keys/warp, width 8 | Largest V=1024 numerator, width 4 / 8 |
|---|---:|---:|---:|
| 4096 | D 64 | D 128 | D 8192 / 16384 |
| 32768 | D 464 | D 928 | D 59392 / 118784 |
| 65536 | D 912 | D 1824 | D 116736 / 233472 |
| 102400 | D 1424 | D 2848 | D 182272 / 364544 |

This explains why every short `large` case passes and why both aligned long widths overflow. The query width boundary doubles per-warp exposure; it does not mix unused columns into live ones.

Raw KV 102399 disables GQA optimization (`fattn.cu:263`). It uses query width 8, `ncols2=1`, D 12 output tiles, D 64-key tiles and D four parallel warps/column. The original sm_120 generic kernel uses M 210 registers/thread and D 33920 dynamic shared bytes, giving D two blocks/SM and P=6. The largest partial has D 267 tiles and D 4272 keys/warp, so its large constant numerator can reach D 546816. Served padded KV keeps the aligned GQA geometry and should not be conflated with this raw-length diagnostic.

Coverage correction: M the parent's 30 failures are exactly `nb={3,4,5,6,8}`, both types, at the three long settings. The original fixture has no `large` cases for nb 1, 2 or 7. Their passing ordinary patterns are not large-value controls. Aligned q8 nb 1/2 dispatches VEC; f16 can dispatch MMA at both, with `np=4` and fewer stream-K blocks because its larger shared tile permits only D one block/SM. Source predicts long f16 large cases can overflow there too. Added 24 optional cases for these missing widths under `LLAMA_TEST_SWIFT_FA_ACCUM=1`; parent must verify them.

## Swift activations and weight bounds

M: ran the real `~/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf` on CPU, with 149 prompt tokens, eight threads, nice 19, GPU layers zero, flash attention disabled and no sampling/generation. SHA256 `e3b59faa312a6202d7ac0a2400123956104bc720e77f6ff21c754aec5c3740df`. Eval callback dumps F32 `Vcur` and softmax probabilities for all full-attention layers. Each V dump contains M 152576 values (149 x 4 x 256). Wall time around `llama_decode` including callbacks was M 12363.910 ms. Built-in perf counters report incorrect token/time totals in this fork; they were not used.

`codex/swift_v_probe.cpp` is the reproducible probe; raw dumps and stderr are in `build-accum/probe/` and `build-accum/probe.log`. `codex/probe.csv` preserves the measured summary. No proxy architecture was used. The GGUF has M 65 blocks including the MTP block, full-attention interval 4, 24 Q heads, four KV heads, 256 head dimensions and 5120 embedding dimensions. The main forward visits 16 full-attention layers.

`codex/swift_weight_bounds.py` dequantizes the real V projection and attention RMS gain on CPU. For RMS-normalized input x with `||x||_2<=sqrt(5120)`, the exact-dot bound is `|V_j|<=sqrt(5120)*||Wv_j*RMS_gain||_2`. It assumes the loaded model without LoRA/bias or non-finite input and ignores dot-product rounding. This is an analytical bound, not an activation estimate or a claim that the extremum is reachable by a prompt.

| Layer | M max abs V | M RMS V | D max row bound from weights |
|---|---:|---:|---:|
| 3 | M 4.821 | M 0.532 | D 143.775 |
| 7 | M 7.325 | M 0.599 | D 120.265 |
| 11 | M 9.166 | M 0.683 | D 106.874 |
| 15 | M 5.787 | M 0.711 | D 109.485 |
| 19 | M 6.399 | M 0.766 | D 108.788 |
| 23 | M 13.280 | M 0.884 | D 123.641 |
| 27 | M 26.686 | M 1.011 | D 124.913 |
| 31 | M 41.466 | M 1.108 | D 133.149 |
| 35 | M 37.159 | M 1.138 | D 127.573 |
| 39 | M 31.295 | M 1.060 | D 177.853 |
| 43 | M 48.594 | M 1.271 | D 138.086 |
| 47 | M 48.314 | M 1.402 | D 172.027 |
| 51 | M 45.433 | M 1.779 | D 174.909 |
| 55 | M 32.885 | M 2.060 | D 193.729 |
| 59 | M 95.058 | M 3.639 | D 402.309 |
| 63 | M 136.743 | M 5.535 | D 319.909 |

M short-prompt mean entropy ranges from 1.746 to 3.115 nats across layers. Mean `1/p_max` is M 2.775-7.609; the largest observed row has M `1/p_max=42.015`. These quantities describe the short causal attention distribution, not its long-context extrapolation. A global peak may leave locally flat block partitions.

D at served 32768, the conservative aligned width-8 overflow threshold for coherent flat V is about 565. Every exact-dot weight bound is below that, with substantial margin, so the weights support a finite-range safety argument there for the stated model/configuration. D at 102400, thresholds are about 368 for width 4 and 184 for width 8. The measured max 136.743 is below both, but late-layer weight bounds cross them. The raw 102399 geometry has a threshold about 123. The measured maximum is not a bound over unseen prompts, tokens, quantizations or LoRA.

G normal served 100k overflow looks less likely than significant rounding error: the measured persistent channel means top out at M 59.658 after f16 conversion, far below the aligned flat-sum thresholds. Overflow still requires a coherent weighted mean, not just one outlier. Neither an actual 100k distribution nor a long CPU decode was measured. The synthetic V=1024 test is a range control, above these model weight bounds; it must not be presented as a measured Swift activation.

## Rounding growth and CPU models

D half unit roundoff is `u=2^-11`, about 4.883e-4. In a model where each 16-key MMA dot is computed exactly and its running result is rounded to half, r=N_a/16 accumulation rounds give the usual bound `gamma_r=r*u/(1-r*u)` times the cancellation condition number `sum(abs(V_i*w_i))/abs(sum(V_i*w_i))`. Turing emits two 8-key instructions instead; its r is larger. This model is not a promise about hardware's internal dot order/rounding. A per-key rounding model gives a more pessimistic bound. Changing maxima adds rescaling rounds; multiplication by one is exact for flat attention. Cancellation can make coordinate-relative errors arbitrarily large, so reported realistic errors use relative L2 norm.

D example gamma_r values for 464/928/1424/2848 assigned keys are about 0.0144/0.0291/0.0454/0.0952 under the 16-key model. Under independent unbiased roundoff with coherent approximately uniform additions, an illustrative RMS estimate is `u*sqrt(r)/3`, about D 0.00088/0.00124/0.00154/0.00217. This is an error-growth illustration, not a hardware error prediction. Float accumulation's corresponding unit roundoff is D 2^-24; half operand and probability rounding remain separate.

`codex/accum_analysis.py` models the rounding of each 16-key update on CPU, using measured values rounded to f16. Its negative control requires half infinity and float 65536 after the 512-key large constant; that fails if the modeled accumulator is broken. `MODEL` rows isolate individual partial sums, including G diffuse-tail weights. `OUTPUT` rows combine all D 36 blocks/output and np 2/1 warp partitions, then compare their sum with a double oracle over represented inputs. They use G flat logits and either repeated measured V sequences or a persistent V vector made from each layer's measured channel means. Persistent means represent a correlated-value stress, not real long-context inference. No GPU bit-level equivalence is claimed for this software MMA model.

M results of the CPU model under those G inputs (relative L2 error after block/warp combination, ranges over the 16 layers):

| Context setting | np | Repeated measured V | Persistent channel mean |
|---|---:|---:|---:|
| 32768 | 2 | M 0.000118-0.000195 | M 0.001594-0.001930 |
| 32768 | 1 | M 0.000252-0.000351 | M 0.002974-0.003752 |
| 102400 | 2 | M 0.000394-0.000637 | M 0.004370-0.006027 |
| 102400 | 1 | M 0.000877-0.002369 | M 0.009406-0.011262 |

D the model shows that >1e-3 output error is possible at realistic V magnitudes without overflow. It does not establish how often such attention/value correlations occur in Swift. At 100k, eight layers cross that threshold in the np=1 repeated-value case. The per-partial error can be larger than the final output error; the output table includes combination rather than treating every partial error as final. `codex/model-errors.csv` preserves both half and float model results.

## Fix

Only `ggml/src/ggml-cuda/fattn-mma-f16.cuh` changes production behavior. `ggml_cuda_flash_attn_ext_mma_f16_case` reads precision and selects a compile-time `accum_f32` specialization for NVIDIA Turing+. The process tile promotes half C tiles to the corresponding float C tile, retaining the same logical matrix shape. Existing `mma.cuh` float-accumulator instructions are reused; no new assembly or subsystem is introduced.

VKQ scales stay float in the f32 specialization, including attention sinks. Shared-memory values remain float through warp combination and final normalization/fixup. Metadata keeps the original half-tile column mapping: scalar float C coordinates have a different element packing and cannot be substituted for those KQ metadata indices. The combine chunk is halved where it remains a whole MMA tile, reusing the same shared byte budget for Swift instead of doubling the full combine buffer. The default specialization retains its original half multiplication, scaling, scratch layout and launch geometry.

Evidence locations: precision dispatch `fattn-mma-f16.cuh:2309`; accumulator type `:1356`; per-tile float rescale `:1023`; sink rescale `:1597`; combine stride `:1665`; metadata mapping `:1700`; float stores `:1811`; float combine loads `:1925`; host scratch sizing `:2218`. Q/K scaling, exponential conversion to half, cache quantization, FTZ policy and sampling are unchanged. F32 logits can differ from the defective half path; token identity across this numerical fix is not promised. No sampling law or RNG code changes.

M CUDA configuration and full compile succeeded with CUDA 13.2, NCCL off and requested architecture 120 (this fork's CMake maps it to 120a). No resulting binary was run. CPU fixture tests do not execute this kernel.

### Compile resource cost

M `nvcc -Xptxas=-v` on old/new focused instances for sm_75/80/86/89/120 succeeded. `codex/ptxas-accum.csv` records registers, stack, spill stores and spill loads; full logs are `build-accum/ptxas/{old,new}-sm*.log`. Registers and all reported spill/stack quantities for each default specialization are M identical to the old instance.

D a wide CUDA warp holds 256 x 16 output scalars: VKQ storage rises from 64 packed registers/thread to 128 scalar registers/thread. There is no second persistent accumulator array. Compiler totals are frequently capped at M 255 registers and expose the extra pressure as spills. This is the main runtime risk; correctness has been prioritized over adding a shape-specific occupancy/config retune.

M table: ncols1 is the template query width; spill counts are bytes reported by ptxas per compiled entry, not measured runtime traffic. Each row is the non-softcap, non-sparse, separate-V variant.

| sm | ncols1 | KV | M registers, old -> f32 | M spill stores/loads bytes, old -> f32 |
|---|---:|---|---|---|
| 75 | 4 | f16 | 255 -> 255 | 240/284 -> 2392/1908 |
| 80 | 4 | f16 | 227 -> 255 | 0/0 -> 340/204 |
| 80 | 4 | q8_0 | 255 -> 255 | 0/0 -> 980/856 |
| 80 | 8 | f16 | 255 -> 255 | 0/0 -> 1064/700 |
| 80 | 8 | q8_0 | 255 -> 255 | 88/192 -> 1672/1540 |
| 86 | 4 | f16 | 231 -> 255 | 0/0 -> 340/204 |
| 86 | 4 | q8_0 | 255 -> 255 | 0/0 -> 948/840 |
| 86 | 8 | f16 | 255 -> 255 | 0/0 -> 1136/740 |
| 86 | 8 | q8_0 | 255 -> 255 | 84/184 -> 1652/1532 |
| 89 | 4 | f16 | 231 -> 255 | 0/0 -> 340/204 |
| 89 | 4 | q8_0 | 255 -> 255 | 0/0 -> 948/840 |
| 89 | 8 | f16 | 255 -> 255 | 0/0 -> 1136/740 |
| 89 | 8 | q8_0 | 255 -> 255 | 84/184 -> 1652/1532 |
| 120 | 4 | f16 | 252 -> 255 | 0/0 -> 132/120 |
| 120 | 4 | q8_0 | 255 -> 255 | 44/32 -> 200/188 |
| 120 | 8 | f16 | 254 -> 255 | 40/40 -> 264/264 |
| 120 | 8 | q8_0 | 255 -> 255 | 296/296 -> 1088/1332 |

sm_75 q8 direct staging and ncols1=8,ncols2=8 are unavailable by the existing guards; their compiled stubs are not meaningful resource measurements. Turing processes nb=8 via its supported width-4 tile instead. Volta/HIP/MUSA behavior is unchanged; this patch does not repair their precision handling.

D dynamic shared memory for Swift is unchanged by the float combine: on sm_80/86/89/120 the width-4/8 f16 launches use 33792/33792 bytes, and q8 direct launches use 41792/42112 bytes. sm_75's supported width-4 f16 launch uses 33344 bytes because cp.async is absent. The compiled MMA entries report M no static shared allocation. These dynamic sizes are host-formula derivations, not a new runtime observation. Parent traces record actual launch registers, shared bytes and grid for both arms.

## CPU checks, queue and reproduction

Executed with `CUDA_VISIBLE_DEVICES=""`, nice 19 and at most eight worker threads:

```
cmake -B build-cpu -DGGML_CUDA=OFF
cmake --build build-cpu --target llama test-backend-ops -j8
c++ -O2 -std=c++17 codex/swift_v_probe.cpp -Iinclude -Iggml/include -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lllama -lggml -lggml-base -o build-accum/swift-v-probe
build-accum/swift-v-probe ~/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf build-accum/probe
python3 codex/swift_weight_bounds.py ~/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf
OPENBLAS_NUM_THREADS=1 python3 codex/accum_analysis.py build-accum/probe
cmake -B build-cuda -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda --target test-backend-ops -j8
bash -n codex/gpu_accum_ab.sh
bash codex/gpu_accum_ab.sh
```

The commands above omit the repeated nice/environment prefix for readability. The probe directory must exist. The executable is CPU-only, named `swift-v-probe`; the CPU fixture binary is copied to `build-cpu/bin/swift-fa-cpu`. No llama-server, llama-cli, test-backend-ops runtime, nvidia-smi, sanitizer or CUDA runtime executable was invoked.

M existing `large` CPU subset at KV 4096/65536/102400/102399, nb 4/8: 8/16 pass. All eight q8 cases pass; all eight f16 cases fail with non-finite output from the pre-existing CPU half attention implementation. This remains an independent CPU defect, not failure evidence for the changed CUDA kernel. Logs: `build-accum/cpu-large.log`. M added nb 1/2/7 q8 short cases pass 3/3 (`cpu-extra.log`). M corrupted q8 large case exits 1 and reports failure (`cpu-negative.log`). No CPU failure was relabeled as a passing CUDA test.

M the parent result parser reproduces 332/362 and exactly 30 large infinities from its archived old log. Kernel-trace extraction was checked against an existing read-only SQLite capture. Shell syntax and PLAN=1 pass. GPU A/B has not run.

Parent-only queue: `RUN_PARENT_GPU=1 PLAN=0 nice -n 19 bash codex/gpu_accum_ab.sh`. Default old bytes are the parent's `build-gpu-ab-20261001-110357/frozen`; new bytes are this worktree's compiled CUDA test/library files, checked against `codex/cuda-accum.sha256`. The script freezes both independently, rechecks hashes, interleaves old/new/new/old on each GPU, requires the old 30 known failures and new 386/386 passes, and checks corrupted new output. It captures individual MMA/fixup microseconds with Nsight at nb 4/8 and KV 8192/32768/98304 for f16 and q8 direct, separately per setting. It trims the first tenth of kernel calls and records median/mean plus launch resources. These isolated kernel timings do not establish served MTP-round speed or TP behavior. Budget setting is 7200 seconds; idle/no-compute-process and readable clean Xid guards apply before/after each owned process. Cleanup signals only its exact owned timeout PID, never a process-name match. It starts no server.

## Upstream and remaining uncertainty

M read-only upstream clone HEAD is `ed7ac35e1`. Its `fattn-mma-f16.cuh:1090/1099` has the same half VKQ accumulator and no precision selection in its MMA case; `fattn-common.cuh:19` has the same fixed log(8) offset. Local history contains `e95d0bc8f`, "CUDA: fix FA VKQ accumulator overflow (#17746)", and the offset comment references issue #18606. Those are relevant prior local references; their current GitHub status/discussion was not checked.

One history `git show` in that partial clone attempted an implicit promisor fetch and failed DNS; no download completed. All subsequent Git reads set `GIT_NO_LAZY_FETCH=1`. No deliberate network request, posting, push or external file mutation was performed.

Open: run the parent's new suite and kernel A/B; assess the spill cost before daily adoption; test remaining architectures/backends separately if expanding the precision contract; obtain long-context Swift activation/attention samples before claiming practical overflow frequency. The source change is ready for GPU verification, not certified for serving. Raw dumps/build logs remain worktree-local; compact numerical summaries and runnable probes are committed.
