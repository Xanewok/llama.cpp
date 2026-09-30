# Subset MTP draft chain

CPU exactness and CUDA compile/link checks pass. Final frozen artifacts use a CUDA configuration that skips CMake native-device probing. GPU speed, GPU seed identity and total GPU memory growth remain unmeasured.

Labels: M = measured, D = derived, G = guess. CPU timings do not predict GPU speed.

## Commits and scope

Base: `b8668a92e` (`stack/draft-vocab-host`), descended from `68d99c67d` through batched input uploads, backend verify candidates and the draft subset.

- `034e51d38`: subset chain graph, compact device embeddings and RNG peek API.
- `eb7d0b606`: opt-in driver with exact host validation and CPU harness.
- `872b20070`: one active slot in a parallel context; stronger q bit comparison and parallel fallback coverage.
- `0e2960142`: concise staging-API exactness contract.

The artifact commit contains this report and the reproducible runners. No other branch was changed. Patch A, fused catch-up, target verification, AllReduce and its pool are unchanged. The port reuses the block/chain design of `0bef2df11` and `ebfc900d5`; it does not copy the full token table.

## Design and exactness

`--spec-mtp-chain` is off by default. It needs probabilistic drafting, draft backend sampling, a positive request temperature, a Qwen3.5 single MTP head, the subset, flash attention and one active drafting sequence. Multiple configured slots work; concurrent active drafts use the existing loop. Unsupported configurations report `MTP chain unavailable` and use that loop.

`src/llama-context.cpp:1278` validates the chain. `:1305` copies selected quantized embedding rows once to the output backend without dequantizing/requantizing them. The table is mirrored under tensor split. Its companion F32 compact-index tensor uses the existing draft-vocabulary partition. The initial token still uses the full host embedding table because it can lie outside the subset.

`src/models/qwen35.cpp:647` runs the MTP block repeatedly in one graph. Each later step reads its embedding through device `get_rows` at the previous compact index and takes the previous hidden row directly. Per-step positions, KV writes and causal masks retain the normal semantics. `src/llama-graph.cpp:2524` gathers shard-local top-k logits, original IDs and compact indices, then computes the next pick. The returned row contains the token, candidate IDs, raw logits and device probabilities. `src/llama-context.cpp:2261` copies all rows together; no hidden row is read back between steps.

`src/llama-sampler.cpp:1446` peeks at the existing dist RNG using a local copy and the same double uniform draws. `common/speculative.cpp:1661` snapshots the host sampler, runs the graph, then applies the existing host sampler to each returned candidate row. Only matching device picks are published. The recorded q is the host sampler's exact q, including its normalization and temperature arithmetic; device softmax rounding is not substituted for it.

If a device pick crosses a rounded CDF boundary, `common/speculative.cpp:1719` discards the provisional result, restores the sampler state, removes draft KV at and after `pos0`, and replays the normal loop. Early stops consume only the host draws the normal path consumes. Request-level maximum truncation retains the normal driver's RNG behavior.

The law follows from this validation: every published token is the existing host draw from the q recorded for that history. A mismatch commits neither a token nor RNG state and is recomputed by the original path. The target verifier and its independent RNG are untouched. Same-seed GPU identity still requires the delivered A/B: device embedding/attention arithmetic can differ from the separate-step graph even when the rejection law remains valid.

## CPU evidence

Model: `/home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf`. The test subset is deliberately unsorted and includes high token IDs. Run `CUDA_VISIBLE_DEVICES="" nice -n 19 bash codex/run_cpu.sh`.

The final runner passed:

- M: 2,700 single-CPU comparison rounds over 100 seeds and 1,080 Meta(CPU,CPU) rounds over 40 seeds. Configurations cover k=3/4/5, temperatures 0.4/1/1.7, p_min=0/0.4/1, RNG continuation and request-level draft truncation. Tokens, IDs, logits and q probabilities match; float q fields are compared by bytes. Synthetic target rows exercise the unchanged rejection verifier. Every intended chain round has an observed multi-token chain decode, so these are not fallback-only comparisons.
- M: 54 single-CPU and 54 meta rounds with slot 1 active in a two-slot context match. M: 54 meta rounds with both slots active match through the normal fallback, with no observed chain picks.
- M: 54 rounds per backend with deliberately wrong device picks (both reported token and next-step index) recover with identical tokens/q/verification outputs. Zeroing a chained embedding makes the comparison fail with `FAIL: draft tokens/q differ`; changing a recorded q probability also fails the comparator.
- M: 12 paired CPU server completion checks pass across one/two configured slots, k=3/4/5 and seeds 1/7. Completion tokens, content, drafted counts and accepted counts match. The CPU executable is renamed `chain-cpu`; the harness uses only a Unix socket inside this worktree.
- `test-sampling`, `test-llama-archs --tp-sampling -s 42` and `--input-upload -s 42` pass, including their existing negative controls.

M: mean elapsed CPU milliseconds per draft, p_min=0, callback instrumentation disabled; initial round excluded. Single CPU uses 8 threads; meta uses 4 per simple CPU backend.

| k | CPU plain | CPU chain | Meta plain | Meta chain |
|---|---:|---:|---:|---:|
| 3 | 14.171 | 13.562 | 16.679 | 16.092 |
| 4 | 18.787 | 17.999 | 22.198 | 21.516 |
| 5 | 23.519 | 22.571 | 27.849 | 26.844 |

Logs: `codex/cpu-{single,tp,two-slots,tp-two-slots,tp-parallel,wrong-pick,tp-wrong-pick,wrong-embedding}.log`, `cpu-server{,-two-slots}.log` and `test-{sampling,tp-sampling,input-upload}.log`. Logs and generated binaries are ignored; runner sources are committed.

## Memory

M: the actual Swift GGUF embedding is `token_embd.weight`, IQ4_XS, shape `[5120,248320]`; its subset file has 25,879 entries. `codex/memory.py MODEL VOCAB_JSON` reproduces the metadata read without a backend.

D: each row is 2,720 bytes, so the additional table is 70,390,880 bytes = 67.129974 MiB per GPU. The compact indices add 103,516 bytes total, vocabulary-sharded, approximately 0.049 MiB per GPU for the equal split. Existing global-ID storage is already present in the baseline. D: chain output readback is 124 bytes per step.

Allocation alignment, the larger graph's compute scratch, CUDA capture and driver allocations are unmeasured. The A/B records per-device resident memory after each request set and reports the paired difference. The table is context-owned and is not included by the current `llama_context::memory_breakdown` (`src/llama-context.cpp:3699`); memory fitting therefore needs explicit headroom. Meta allocation can abort on OOM. The high-context two-slot deployment still needs a parent GPU fit check.

## Remaining host versus GPU cost

The item-96b measurements precede both the subset and batched input uploads. M: internal gaps were 0.24 and 0.23 ms; D: their sum is 0.47 ms for k=3. This chain removes those two host boundaries, not the final round readback. Extrapolating the old average gap gives G: 0.71 ms for k=4 and 0.94 ms for k=5 in the current stack; these are stale-budget hypotheses, not predicted savings.

M: the old draft GPU busy sum was approximately 5.00 ms per k=3 round, including 3.69 ms in the full head. D: non-head GPU work was approximately 0.44 ms per step. The subset changes head work, and the larger chain graph changes rebuild/scheduling cost. Neither the post-subset host gap nor post-subset GPU busy time is measured by the available trace.

The parent's `SP/codex/dv-ab2/summary.txt` gives D: k=3->4 / k=4->5 round-time differences of 1.365/6.176 ms (tool), 1.506/0.699 ms (think), and 2.691/2.960 ms (long). They are not isolated draft-step times: verify grows, acceptance and completions change across k, and the tool k=5 requests contain only M: 5 rounds per boot. Subtracting an old 0.24 ms gap and calling the remainder GPU work would be invalid.

Post-subset host/GPU attribution remains unknown. A same-k A/B with identical completions measures the realized chain benefit; an updated nsys trace is needed to separate host gaps from GPU work. Larger C->chain graph/meta rebuilds can offset the eliminated boundaries. No GPU speedup is claimed.

## Builds and queued GPU A/B

CPU configuration: `cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF`; build with `nice -n 19 cmake --build build-cpu --target llama-server test-sampling test-llama-archs -j8`.

CUDA configuration: `cmake -B build-cuda-no-probe -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER_FORCED=ON -DGGML_CCACHE=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF '-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16'`; build `llama-server` with `nice -n 19 ... -j8`. Compile/link exits successfully (`codex/build-cuda-no-probe.log`); no model-serving CUDA binary was executed. See the configure-probe deviation below.

`codex/frozen/chain` freezes implementation commit `0e29601422f35572757734544cfdad673df3f621`, executable `chain-ab`, all project libraries, `SOURCE` and `SHA256SUMS`. The baseline remains the parent's read-only `~/repos/llama.cpp-mtp-draft-vocab/codex/frozen/stack` at `b8668a92e`.

Parent queue command: `RUN_GPU_AB=1 DRY_RUN=0 nice -n 19 bash codex/gpu_ab.sh`. The script defaults to a safe dry run, which passes manifest/source/input checks without executing binaries or querying GPUs. It uses the draft-vocab A/B's exact warmup/tool/think/long request driver, frozen libraries, interleaved forward/reverse boots, MTP k=3/4/5 and equal tensor split. D: two boots per arm, twelve boots total, a 2,400-second wall budget. Xid, idle-memory, port, process-name, exact-child-PID cleanup and immutable-input guards are retained. Its enablement guard reads the common logger's `MTP subset chain enabled` line, not a library INFO line. The summary rejects changed completions or acceptance counters at the same k. Synthetic summary arithmetic and changed-completion negative control pass.

## Limits and deviations

- GPU seed identity, CUDA execution/capture behavior, net speed and peak high-context memory remain open. Do not enable this in the serving stack until the parent A/B passes.
- Concurrent active drafts, greedy/zero-temperature drafting, missing subsets, tiny shards, other architectures, multi-head MTP and non-flash attention use the original path. LoRA with a subset remains unsupported. Positive p_min can waste GPU work because the full chain runs before the host truncates it.
- The first CPU server build used an obsolete UI-disabling option and CMake attempted its default UI download. DNS failed; no remote data was received. The attempt violated the no-network constraint. Subsequent builds disable the current UI options. No push or posting occurred.
- The first CUDA configure ran CMake 3.28's `CMakeDetermineCompilerABI_CUDA.bin` to discover native architectures, despite the explicit architecture. GPUs were hidden and it reported `No CUDA devices found`, but this violated the CUDA-binary restriction. `codex/build_cuda.sh` uses `CMAKE_CUDA_COMPILER_FORCED=ON` to skip the probe, while the full target build checks compilation/linking. A fresh configure and full build confirm that CUDA ABI/native probing is skipped; `build-cuda-no-probe/CMakeFiles/3.28.3` contains no CUDA ABI executable.
- The sandbox blocks socket binds. CPU completion tests received automatic approval for a worktree-local Unix socket with a renamed CPU-only server. Topic commits received approval for linked-worktree Git metadata. No process named `llama-server`, GPU tool or GPU workload was started.
