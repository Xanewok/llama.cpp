# Input upload batching

Prototype ready for GPU A/B. CPU and Meta(CPU,CPU) logits are bit-identical to the frozen parent. CUDA compiles; no CUDA binary or GPU tool was executed. GPU benefit is unknown.

## Commits

- `f46f8bbdb`: clean implementation, `ggml : batch scheduler host input uploads`. Five source/test files, +224/-1 lines (D from Git).
- `3d98588fd`: DO NOT MERGE comparison toggle and counters. Remove this commit from an upstream candidate. `GGML_INPUT_BATCH_DISABLE` selects the original per-input copy path; `GGML_INPUT_UPLOAD_PROFILE` measures input filling and scheduler uploads; `GGML_INPUT_UPLOAD_TRACE` reports meta layout dispatch.
- The subsequent `codex` artifact commit contains this report, the CPU harnesses, and the parent-only GPU queue script. No public submission was made.

## Design and exactness

The main uploads occur in `ggml_backend_sched_compute_splits`, after `set_inputs`: many input setters write directly into host tensors. Batching just `ggml_backend_tensor_set` calls inside `set_inputs` would miss masks, KV indices, and output IDs.

`ggml/src/ggml-backend.cpp:1723` collects nonempty host user inputs for each scheduler split. It waits for the destination copy slot once, uploads the batch synchronously, and excludes those inputs from the old copy loop. Non-user inputs, device sources, MoE weight copies, and cross-backend dependencies retain their original paths. `ggml_backend_tensor_set_batch` groups consecutive transfers with the same backing buffer and falls back to the existing setters when the optional callback is absent.

`ggml/src/ggml-backend-meta.cpp:1728` batches MIRRORED tensors by simple device. Other layouts use the existing splitter, including its segmented and PARTIAL behavior. M: the real Qwen3.5 MTP harness dispatched mirrored batches for every decode, with five or six tensors per batch and two simple devices; it used no layout fallback.

`ggml/src/ggml-cuda/ggml-cuda.cu:802` retains pinned staging and descriptor storage per CUDA buffer. It snapshots the host bytes, submits one `cudaMemcpyBatchAsync` call for disjoint destinations on CUDA 13 or newer, and synchronizes the upload stream once. Overlapping destinations and older CUDA/HIP/MUSA builds use ordered async copies followed by one upload-stream synchronization. Existing host-buffer allocation handles pinned-memory fallback; allocation failure uses the original setters.

This is one batch-copy API call per device buffer, not one contiguous DMA transfer. There is a reusable staging allocation per destination CUDA buffer, rather than one allocation shared across devices. The scheduler's preceding dependency wait is also retained; this does not claim one total synchronization across the entire decode. A literal contiguous transfer would require a separate input-allocation change or unsafe copying across gaps in the compute arena.

Exactness argument: transfer payloads are copied as bytes without conversion; each transfer retains its offset and size. CUDA's unordered batch path is used only when destination ranges do not overlap. Ordered fallback preserves last-writer behavior. The post-upload wait prevents staging reuse and host-source reuse before completion. No sampler logic, random draws, or distribution arithmetic changes.

The optional buffer callback changes the backend ABI, so `GGML_BACKEND_API_VERSION` is bumped in `ggml/src/ggml-backend-impl.h:11`. Dynamic backend plugins must be rebuilt. Other backend aggregate initializers retain a null trailing callback; those backends were not built here.

## Reuse and pipeline interactions

- `src/llama-context.cpp:1468` retains the existing pipeline-parallel wait before reused graph inputs are overwritten. Graph rebuild/allocation ordering is unchanged. The brief's local reports were the source for upstream #28058; no network lookup was performed.
- Transfers are reconstructed from the current split and current copy slot on every compute. No tensor address, graph result, meta slice, or sampler binding is cached by this patch. CUDA staging contains only host bytes and reusable descriptor capacity.
- `can_reuse` and meta graph-plan reuse are unchanged. Patch A was read but not incorporated; its reported cache and sampler ownership defects are outside this patch.
- M: `codex/check-pipeline.cpp` passed with four pipeline copy slots and twelve computes of a reused Meta(CPU,CPU) graph, both with batching and with the old copy path. It overwrites the original host inputs immediately after compute returns. CPU backends are synchronous; this does not prove CUDA event or stream behavior.

## Verification

All commands used `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`, and at most eight worker threads. All artifacts are in this worktree. Git's linked-worktree metadata required sandbox escalation for the requested topic-branch commits.

CPU configuration and build:

```sh
export CUDA_VISIBLE_DEVICES=""
nice -n 19 cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DGGML_CCACHE=OFF -DCMAKE_CXX_COMPILER_LAUNCHER= -DCMAKE_C_COMPILER_LAUNCHER=
nice -n 19 cmake --build build-cpu -j8 --target test-llama-archs
nice -n 19 build-cpu/bin/test-llama-archs --input-upload -s 42
nice -n 19 build-cpu/bin/test-llama-archs --tp-sampling -s 42
nice -n 19 bash codex/run_cpu.sh
```

The initial build failed because ccache attempted to write outside the worktree. Disabling ccache resolved it. CPU `llama-server` was built but never started; the real-model checks use `codex/check-inputs` directly.

`tests/test-llama-archs.cpp:1213` checks default setter fallback, optional batch dispatch, overlapping writes, and zero-byte input handling. Its injected dropped-transfer implementation fails the same comparison oracle. The existing TP sampler tests pass candidate IDs/logits, greedy, top-k/top-p/min-p/temp/dist, and the distribution check; M: total variation 0.038 over 4000 draws.

`codex/check-inputs.cpp` loads `/home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf` with its MTP head. It runs target verify batches, output-free MTP catch-up, and consecutive single-token draft inputs. M: each dump is 27,811,840 bytes; all 6,952,960 floats are finite and nonzero. D: this covers 28 complete vocabulary rows across four rounds.

The clean patch dumps match the frozen parent byte-for-byte independently on CPU and Meta(CPU,CPU). CPU and Meta outputs are not compared against each other. The instrumented build also matches with batching on/off and graph reuse disabled/enabled in each mode. Every interleaved timing run has the same dump. Both real-model negative controls change the token at position 4 and fail equality; M: the first differing output byte is 6,952,960 in both modes. The initial catch-up test exposed a zero-byte meta input abort; the clean patch skips those transfers and the regression passes.

Pipeline check:

```sh
export CUDA_VISIBLE_DEVICES=""
nice -n 19 g++ -std=c++17 -O2 -Iggml/include codex/check-pipeline.cpp -Lbuild-cpu/bin -Wl,-rpath,'$ORIGIN/../build-cpu/bin' -lggml -lggml-base -lggml-cpu -o codex/check-pipeline
nice -n 19 codex/check-pipeline
GGML_INPUT_BATCH_DISABLE=1 nice -n 19 codex/check-pipeline
```

CUDA compile check, passed for the clean implementation and the counter build:

```sh
export CUDA_VISIBLE_DEVICES=""
nice -n 19 cmake -B build-cuda -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CCACHE=OFF -DCMAKE_CUDA_COMPILER_LAUNCHER= -DCMAKE_C_COMPILER_LAUNCHER= -DCMAKE_CXX_COMPILER_LAUNCHER= -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF
nice -n 19 cmake --build build-cuda -j8 --target ggml-cuda
```

`bash -n` passed for the shell artifacts; Python compilation passed for the helpers. The GPU A/B script was not run, including its dry-run mode.

## CPU input cost

M: medians of three interleaved run means per arm, eighteen decodes per run after excluding the initial verify/catch-up pair. `input_fill` measures `set_inputs`; `input_upload` measures user-input upload preparation, dependency waiting, and copies. It excludes graph build, allocation, and forward compute. CPU uses eight threads; each Meta CPU backend uses four. The microsecond timer and instrumentation limit precision for these short operations.

| Backend | Arm | Fill + upload, us/decode (M) | Scheduler upload, us/decode (M) |
|---|---|---:|---:|
| CPU | old path | 6.111 | 0.000 |
| CPU | batch | 6.500 | 0.000 |
| Meta(CPU,CPU) | old path | 16.667 | 9.389 |
| Meta(CPU,CPU) | batch | 19.278 | 12.333 |

D: Meta fill/upload overhead is +2.611 us/decode in this run. CPU has no CUDA synchronization penalty to remove; these numbers do not predict GPU speed. The brief's approximately 0.9 ms/round opportunity is D from the earlier GPU attribution, not a measured saving for this patch.

## Parent-only GPU A/B

`codex/gpu_ab.sh` compares frozen parent and clean-patch servers in interleaved boots. It checks commit IDs, binary/library SHA256 manifests, model size, selected GPU memory, port ownership, another `llama-server`, kernel Xids before/after boots, and a wall-time budget. It launches `input-upload-srv`, never a process named `llama-server`, and stops only its own exact PID. It uses probabilistic draft sampling with backend draft top-k, MTP k=3, tensor split, and the same tool/think requests as the reference queue pattern.

Prepare these frozen directories before queuing it:

| Arm | Directory default | First line of `../SOURCE` |
|---|---|---|
| up | `codex/frozen/up/bin` | `68d99c67d958c28b6bff5fdc2db096f81c324ef9` |
| patch | `codex/frozen/patch/bin` | `f46f8bbdb8a6b73eb6ddeabad49f24534e70e0cd` |

Each `bin` directory needs the frozen server copied as `input-upload-srv`, its shared libraries, and a `SHA256SUMS` manifest covering every executable/library. Build those revisions in separate source directories under this worktree; do not point the queue at the mutable counter-build binaries. `UP_DIR`, `PATCH_DIR`, and `OUT` can override paths. The default budget is 1200 seconds (configuration, not a measured duration).

`codex/gpu_request.py` reuses the reference request driver and adds a completion hash without unstable tool-call IDs. The summary refuses timing comparisons if completion content, predicted count, draft count, or accepted count differs across boots/arms. M: request durations and counts come from the server; D: ms/round uses `draft_n/3`, an estimated round count. A follow-up trace is needed to verify actual copy/sync counts and isolate the upload saving.

## Risks and open items

- CUDA runtime correctness, batch-copy support on the installed driver, CUDA graphs, and real asynchronous pipeline behavior remain untested. Compilation and CPU comparisons cannot close these items.
- Non-mirrored meta batches use per-tensor upload fallback. Multiple destination buffers or scheduler splits can produce multiple batches per device per decode. Only the local Qwen3.5 CPU model's mirrored path was observed.
- Pinned staging is retained at the high-water mark per CUDA buffer. Very large prefill inputs can increase pinned host memory. Device allocation sizes and graph topology are unchanged (D from source); pinned RSS was not measured.
- Reordering independent user-input transfers ahead of other split inputs preserves the documented user-input contract and passed the tested graphs. Custom backends with hidden ordering dependencies require review.
- Other backend builds and older CUDA/HIP/MUSA compilation are open. No network or upstream duplication check was performed under the brief's no-network rule.
- CPU-only overhead is small but measured. GPU gain must be established by the queued A/B; no GPU speed claim is made.
