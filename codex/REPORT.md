# Reduced-vocabulary Qwen3.5 MTP draft head

Implementation and frozen stack ready for the parent GPU A/B. CPU correctness checks and CUDA compile/link checks pass. GPU speed and Swift acceptance are unknown. No GPU tool, CUDA binary, server, CLI, or GPU benchmark was executed.

Labels: M = measured here, D = derived from source/metadata/measurements, G = guess. There is no new GPU performance estimate.

## Commits

Topic implementation: `42a4b559f` (reduced head) and `82282bb94` (small-subset guards). The artifact commit contains this report and the reproducible CPU/GPU harnesses.

`stack/draft-vocab-host` is `b8668a92e5188c8294e4e133f4426a3506fac50c`, based on `68d99c67d958c28b6bff5fdc2db096f81c324ef9` plus these clean commits:

- `e7071647c`: cherry-pick of `f46f8bbdb`, host input uploads.
- `70036f47d`: cherry-pick of `60c8cbfb2`, backend verify sampling.
- `faee46be9`: reduced head with the deferred-readback conflict resolved.
- `b8668a92e`: small-subset guards.

No DO-NOT-MERGE commit was added above the requested base. The nested stack worktree is `build-stack-src`, and its tracked tree is clean.

## Design and evidence

- `common/arg.cpp:4210`: `--spec-draft-vocab FILE` loads a nonempty JSON array of unique nonnegative int32 IDs. `src/llama-model.cpp:1517` checks vocabulary bounds and requires the dense Qwen3.5 architecture, one loaded MTP layer, and GGUF weight data. Swift/Qwen3.8 uses this architecture. Other architectures are rejected.
- `include/llama.h:349` and `common/common.cpp:1625`: optional model-load parameters carry the IDs. The model owns a copy. A separate draft model receives its own subset through `common/speculative.cpp:2608`; its target does not allocate a reduced head.
- `src/llama-model.cpp:1950`: allocate the compact head once on the output device, preserving the original head type. Read selected rows directly from GGUF before backend repacking, then upload them together. This avoids slicing CPU-repacked weights or dequantizing/requantizing rows. The original target head remains intact. The allocation participates in model memory fitting, including the no-allocation pass.
- `src/models/qwen35.cpp:645`: only the MTP graph uses `draft_output`; head normalization and scalar weight scaling retain the original path. LoRA with a reduced head is unsupported and explicitly asserts rather than silently ignoring adapter rows.
- `src/llama-model.cpp:589` and `src/llama-context.cpp:1389`: the compact head is split by vocabulary, and its F32 token-ID tensor follows the same contiguous partition. `src/llama-graph.cpp:3880` supplies real IDs to a single-device backend chain; tensor parallelism uses the existing per-shard top-k gather. IDs are exactly representable as F32 for this vocabulary.
- `src/llama-context.cpp:2031`: host sampling reads compact logits into the existing pinned output arena, then scatters them into full-vocabulary rows with `-inf` outside the subset. The public logits layout remains unchanged. Scatter occurs before output reordering. A per-row flag prevents a second scatter over an already expanded row.
- The stack conflict was in `output_reorder`. Both deferred readback and expansion are retained, in that order. Deferred MTP readback also uses compact dimensions before scatter; otherwise the target-verify patch would copy a full-vocabulary byte count from a compact tensor. The real-model harness explicitly reads deferred raw logits after backend candidates, including multiple microbatches.
- A subset smaller than leading backend top-k falls back to host sampling before backend initialization. Tensor-parallel sampling also retains the existing minimum-shard-size gate. The loader rejects a tensor split that gives any device an empty compact-head shard; this prevents a meta allocation assertion even when sampling would fall back to the host.

Default-off behavior leaves the original head, logits, sampler chain, and random streams in use. M: target logits are bit-identical with a compact head present, on CPU and Meta(CPU,CPU). The GPU script requires identical completions between the original stack and the new stack without a subset.

## Exactness

Let the actual filtered draft distribution be `q`, normalized on subset `S`, with `q(x)=0` outside `S`. Let `p` be the existing target sampling law at that history. Acceptance contributes `min(p(x),q(x))`. Rejection probability is `Z = 1 - sum_x min(p(x),q(x))`; drawing its residual from normalized `max(p-q,0)` contributes `max(p(x)-q(x),0)`. Their sum is `p(x)`. Conditioning on each emitted history gives the same target sequence law. Restriction changes proposals and acceptance, so it need not preserve the full-vocabulary draft's per-seed completion.

The code records the post-filter probabilities used for the actual draft draw, with real token IDs; verification arithmetic is unchanged by the topic implementation. The stack retains the requested backend-verify implementation.

M: `tests/test-sampling-prefilter.cpp:331` applies the normal top-k/top-p/min-p/temperature chain to a small subset. Its recorded probability sum is `1.000000015` (float rounding). A target with mass outside the subset passes the independent target-law chi-square check, `p=0.489897`, over M: 20,000 draws. Recording only 0.1 times the actual subset probabilities gives M: `p=0` and fails the same law oracle. Topic and stack verify outputs match for M: 300 recorded seed streams. `codex/test-stack-verify.cpp:180` additionally verifies M: 512 restricted-q rounds with identical host/backend candidate seed streams. Its real-model check compares M: four rows x 64 seeds and their candidate probabilities exactly.

M: the real reduced-head harness also runs the sampling chain from host rows and backend candidates, checks that recorded q sums to one within `1e-6`, and checks that every positive-probability candidate and every draw stays in the subset. It rejects a deliberately displaced token-ID mapping. It covers unsorted IDs, high token IDs, excluded-token masking, repeated reads, multiple output rows and microbatches, and global backend top-k.

## Memory

D, from the local Swift IQ4_XS GGUF: `output.weight` is Q6_K, shape `[5120,248320]`, with 4,200 bytes per row. The supplied JSON contains 25,879 distinct IDs, all in range. This is 10.422% of the full vocabulary.

| Added allocation | First GPU | Second GPU | Label |
|---|---:|---:|---|
| Compact Q6_K head payload | 54,343,800 B | 54,348,000 B | D |
| F32 subset-to-vocabulary map payload | 51,756 B | 51,760 B | D |
| Combined payload | 54,395,556 B | 54,399,760 B | D |
| Combined allocations with CUDA 128-byte padding | 54,395,648 B | 54,399,872 B | D |

This is approximately D: 51.876 / 51.880 MiB per GPU. It excludes driver allocation granularity and changed compute scratch. The full target head is retained. Host state adds the ID vector and a pinned compact readback arena of `sizeof(float)*subset_size*n_outputs_max` per MTP context; host full-vocabulary buffers remain.

## CPU measurements

Model: `/home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf`. The timing fixture takes every tenth ID from this model's own vocabulary, in descending order: D: 24,832 IDs, Q6_K, 52,147,200 head bytes. It never applies the Swift list to the 4B model. This fixture measures correctness and draft-step work; it does not estimate real acceptance or token-frequency coverage.

M: one pass per mode, 16 steady single-token steps after a four-token warm-up, fixed synthetic hidden state and fixed token inputs. Timings include host work, compute, synchronization, a top-k/top-p/min-p/temperature/draw chain and output inspection. The fixture uses the public sampler chain directly; it does not reproduce the common sampler's host prefilter optimization, and its validation overhead is included. Deferred raw readback is forced during backend correctness checks.

| Stack CPU mode | Full head, host | Compact head, host | Compact head, backend | Label |
|---|---:|---:|---:|---|
| Single CPU, eight threads | 23.867 ms | 7.617 ms | 5.977 ms | M |
| Meta(CPU,CPU), four threads per backend | 44.052 ms | 8.583 ms | 6.572 ms | M |

These are noisy wall times under competing host load, not a controlled speed estimate. Earlier runs varied substantially. No CPU timing is extrapolated to GPU performance. M: all retained subset logits match their source-row logits with maximum error zero; target logits match bit-for-bit in each mode.

## Commands and artifacts

All shell commands set `CUDA_VISIBLE_DEVICES=""`. CPU work used `nice -n 19` and at most eight configured worker threads. Builds use worktree-local scratch after the initial setup.

CPU configuration:

```sh
export CUDA_VISIBLE_DEVICES="" TMPDIR="$PWD/build-tmp" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nice -n 19 cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
nice -n 19 cmake --build build-cpu -j8 --target llama-server test-sampling test-sampling-prefilter test-llama-archs
```

The stack uses the same command from `build-stack-src`. `codex/run_cpu.sh` rebuilds the standalone harnesses and runs the stack checks. It uses existing `tests/*` infrastructure; no new file was added to `tests/*`.

M: final `codex/run_cpu.sh` exit status zero (`codex/stack-final-tests.log`). The real-head chain check was extended afterward and passes again (`codex/stack-final-head-{cpu,tp}.log`). Successful checks:

- `test-sampling`, on topic and stack.
- Stack `test-llama-archs --input-upload -s 42`: ordering, fallback and dropped-transfer negative control pass.
- Stack `test-llama-archs --tp-sampling -s 42`: candidate IDs/logits, four-row chains, distribution and tiny-shard fallback pass. M: total variation 0.038 over 4,000 draws.
- `test-sampling-prefilter MODEL --draft-vocab-only --write codex/subset-verify-seeds.txt` on topic; `--check` on stack. The restricted-q law and broken-recording control pass.
- One-token single-CPU and two-token Meta(CPU,CPU) subsets use host top-k and retain their support; an empty TP shard fails model loading with the expected error.
- `test-draft-vocab MODEL single 17 deferred` and `test-draft-vocab MODEL tp 17 deferred`, linked against stack CPU libraries. Raw compact readback, scatter, mapping and target identity pass.

Logs: `codex/{cpu-build,stack-cpu-build,stack-input-upload,stack-tp-sampling,subset-verify,stack-subset-verify,stack-draft-vocab-cpu,stack-draft-vocab-tp-cpu}.log`. They are retained locally and ignored by Git. `codex/subset-verify-seeds.txt` is the reproducible seed reference.

CUDA configuration uses `/usr/local/cuda-13.2`, `GGML_CUDA_NCCL=OFF`, architecture `120`, and the original frozen baseline's FA types `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`. M: final CPU and CUDA build exit status zero (`codex/stack-{cpu,cuda}-final-build.log`). CUDA builds are compile/link checks only. The final CUDA configure log reports commit `b8668a92e`, architecture `120` and NCCL off; the frozen binary was not executed.

The original GPU binary was reused from the read-only queue freeze at `$SP/queue/build/stack-gpu/resync-fmq/bin` after verifying its manifest. Its source descriptor identifies `68d99c67d958c28b6bff5fdc2db096f81c324ef9`; the copied executable bytes are unchanged and renamed `vocab-ab`. The stack is built here. `codex/freeze.py` copies each server and its libraries into `codex/frozen/{up,stack}/bin`, writes `SHA256SUMS`, and puts the full commit SHA alone on the first line of `SOURCE`. The runner selects these frozen libraries with `LD_LIBRARY_PATH`.

The first CPU build attempted CMake's default UI fetch; DNS blocked it. This was unintended. UI downloads and ccache were then disabled. No remote data was obtained. Git commits and nested worktree creation required sandbox escalation for linked-worktree metadata.

## Parent GPU experiment

`codex/gpu_ab.sh` defaults to a dry run. The parent can queue:

```sh
CUDA_VISIBLE_DEVICES="" DRY_RUN=1 nice -n 19 bash codex/gpu_ab.sh
RUN_GPU_AB=1 DRY_RUN=0 nice -n 19 bash codex/gpu_ab.sh
```

Arms: original k=3; stack k=3 without a subset; stack with the Swift subset at k=3, k=4, k=5. D: two boots per arm, forward then reverse order. Draft backend sampling is explicitly on in every arm. Stack arms also use target backend sampling. Flags fix tensor split 50/50, NCCL off in the binary, q8_0 target KV, f16 draft KV and 32k context; memory fitting is disabled to preserve arm configuration.

Each boot runs warm-up, a tool request, a thinking request and a tokenizer-sized approximately 8k-token prompt. The driver preserves the same seed and messages across arms. It reports M: emitted-token timings and measured verification-round counters; the summary derives ms/round, acceptance, tokens/round and tok/s. It checks baseline versus stack-without-subset completion identity for every request and both boots. Subset completions can differ while preserving the target law.

Guards: frozen manifests and source SHAs, input hashes, model size, selected GPU memory, occupied port, existing server names, kernel Xids, exact child-PID cleanup and a maximum D: 2,400-second wall budget. The GPU runner was not executed here. M: dry-run exit status zero (`codex/gpu-ab-dry-run.log`), all frozen manifests verify, and shell/Python syntax checks pass. The summary's synthetic oracle recovers D: 33.333 ms/round, 0.66667 acceptance, 3 tokens/round and 100 tok/s from known inputs, and rejects a changed completion hash. These synthetic values are not GPU measurements.

## Risks and open items

- GPU speed, acceptance and the best k remain unmeasured. Run the delivered experiment before adopting the subset. The HyperQwen list's coverage of this workload is not established here.
- CUDA asynchronous behavior and device allocation were only compile-checked; CPU meta tests cannot prove them. Compute scratch and CUDA driver allocation rounding remain unmeasured.
- The public model-parameter struct changed; callers must rebuild with the matching header and libraries.
- LoRA, other architectures, user-supplied synthetic model weights and empty tensor-parallel vocabulary shards are unsupported with a reduced head. Separate-MTP GGUF loading is implemented but not exercised with a sidecar fixture.
- The local upstream checkout contains no checked-out draft-vocabulary implementation, and local history search did not locate #29143. No network lookup was permitted. HyperQwen's local patch and preparation notes were read; this implementation uses their restricted-proposal idea without adding their full-vocabulary device scatter.
- Patch A was read, including its sampler-ownership and meta-plan review findings, but it is not included in this stack.
