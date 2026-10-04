# Host bundle

Status: three independent product fixes are committed. Input-upload sharing is not implemented; the original four-target task is therefore incomplete. Default inference is byte-identical in the CPU server checks. Host buffer selection is opt-in and does not remove CPU embedding dequantization by itself. No GPU result is claimed.

Labels: M = measured or observed in this run, D = derived, G = estimated. Parent trace measurements are identified separately from this run's CPU measurements.

## Product commits and design

- `c35e649a2` - `kv-cache : use the position index for sequence removal`. `src/llama-kv-cells.h:268` uses the existing `(position, cell)` index for a specific sequence. Increment the iterator before erasing its entry; compute the minimum freed cell index, not the first position encountered. The wildcard path scans only the immutable used extent. `src/llama-kv-cache.cpp:403` and `:415` retain the original head update and range normalization. No extra index, allocation, model/backend condition, or environment variable.
- `ef1a40f39` - `server : borrow text prompts during speculative drafting`. `tools/server/server-common.cpp:648` returns the existing text vector when multimodal input is disabled, and the original filtered copy in caller-owned storage otherwise. `tools/server/server-context.cpp:3278` passes that reference to the drafter. Drafting finishes before `handle_last_sampled_token` changes the prompt. While the queue yields, `process_single_task` accepts only metrics and slot reads. The reference remains a vector reference through reallocations; callers must not mutate its contents during drafting. Valid text-only prompts contain no media placeholders, as required by the existing `get_tokens()` path.
- `85b13faf2` - `common : accept host buffers in tensor overrides`. `common/arg.cpp:262` also enumerates each device's host buffer type. `emplace` preserves the meaning of existing buffer names when a host type has a colliding name. This extends the existing `--override-tensor` option rather than adding a model-specific flag or changing input-layer defaults. The CPU test supplies a synthetic host type, then checks both discovery and a name collision.

M: each final commit passed the independent cherry-pick and mutation checks recorded in `host-results/master.json`. The scratch branch `topic/host-bundle-master` stays at master `050439614`; checks use `cherry-pick --no-commit`, then restore that scratch worktree. No scratch commits or other branch histories are rewritten.

The initial test locations depended on stack-only tests. They were moved to standalone tests and common CMake context before final validation.

## CPU evidence

Model: `/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf`. Baseline source is an unmodified `git archive 98c695111` in `build-baseline-src`; its CPU libraries are built in `build-baseline`. Product libraries are built in `build-cpu` with `-DGGML_CUDA=OFF`.

`host_bundle_cpu.py` builds a fixture named `host-server-cpu` against each arm's real server context. It runs the real server loop without HTTP, GPU backends, or a process named `llama-server`. It reuses `reuse-server-cpu.cpp`, including its reduced draft vocabulary; both arms use the same vocabulary. The MTP depth is five, drafts are probabilistic, and backend sampling is enabled. Each case sends two requests so cached-prompt reuse is exercised.

M: six cases passed: CPU and Meta(CPU,CPU), each at temperature zero and temperature one with seeds 1729 and 2718. M: twelve arm executions and twenty-four requests. The binary records contain token IDs and every byte of the final target logit row; draft and accepted counts are compared separately. Every request drafted. A perturbed token byte and a perturbed draft counter fail the same comparator. Evidence: `host-results/identity/results.json`, `logs/cpu-identity.log` and the per-case binary/stats/log files.

M: all seven focused CTest tests passed: `test-alloc`, `test-kv-cells`, `test-server-tokens`, `test-server-tokens-view`, `test-sampling`, `test-tp-backend-sampling`, and `test-batch-alloc`. M: the stack's `--input-upload` ordering/fallback/dropped-transfer control passed. M: hybrid rollback and split replay each passed on the two generated Qwen models and skipped the two non-recurrent llama models. M: state save/load passed on all four generated models. M: `test-sampling-prefilter` passed with zero failures, including grammar/budget/rejection law and seed checks. Logs: `logs/cpu-ctest.log`, `logs/stack-input-upload.log`, `logs/stack-rollback.log`, `logs/stack-state.log`, `logs/stack-prefilter.log`.

`test-backend-ops` was built but not executed: hard rule 1 forbids executing it, whereas the later proof list asks for CPU execution. That contradiction remains an open proof item. No CUDA executable, GPU tool, or named server process was executed.

### Discriminating tests

`tests/test-kv-cells.cpp` compares removals with the old capacity scan across randomized sparse/shared cells, duplicate and non-monotonic positions, shifts, empty/reversed ranges, repeated removals, and wildcard removals. It compares the first freed index, positions, shifts, spatial/token metadata, sequence bits and position indexes. The mutant replaces `min(first, i)` with `i`; it fails because position order is not cell order.

`tests/test-server-tokens-view.cpp` checks reference identity, untouched scratch storage, visible appended tokens, media filtering, truncation and empty media input. The mutant forces the copying branch; the reference-identity assertion fails.

`tests/test-arg-parser.cpp --tensor-buffers-only` discovers a CPU-backed synthetic host buffer through the production CLI parser and preserves the existing CPU buffer when the synthetic name collides. The mutant removes host discovery; parsing fails. The default parser test includes remote download tests, so only the offline selector is run.

### CPU microbenchmarks

The optional `--bench` modes have no timing thresholds. The KV benchmark measures empty-range removal, isolating the unnecessary capacity scan; it does not include erasing cells or model decoding. The prompt benchmark compares the original filtered-copy method with the new borrowing method. These are CPU call costs under concurrent host load, not GPU round measurements.

| CPU call | Used tokens | Original cost, microseconds M | Patched cost, microseconds M |
|---|---:|---:|---:|
| KV empty-range removal | 8192 | 50.792 | 0.047 |
| KV empty-range removal | 90112 | 50.973 | 0.067 |
| Text prompt access | 8192 | 6.235 | 0.002 |
| Text prompt access | 90112 | 173.745 | 0.002 |

These values come from the final independent master run, `logs/master-kv-test.log` and `logs/master-prompt-test.log`. Timing varies under host load. No CPU dequantization cost drop is claimed for host buffer discovery; its default cost is unchanged.

## Predicted host savings and defaults

Parent trace M: host time per MTP round is 2.98 ms at 8k and 2.83 ms at 90k, on Swift 1.5 IQ4_XS with tensor split and draft depth five. The following are bounds or estimates from that trace, not measurements of this patch.

| Target | Parent cost at 8k / 90k, ms per round M | Predicted saving, ms per round | Default behavior |
|---|---|---|---|
| Input uploads | 0.52 / 0.42 | Not implemented. D upper bound 0.11 at 8k for sharing the 0.22 ms copy cost across two ranks if all copied bytes are mirrored; direct producers could remove up to 0.22. The 0.30 ms API/wait component remains. G roughly 0.08-0.11 at 90k for shared copies, pending attribution. | Existing stack staging is unchanged. |
| KV removal | 0.13 / 0.05-0.09 | D upper bounds 0.13 / 0.05-0.09. G practical saving 0.10-0.13 / 0.03-0.08 for short removals; indexed erases remain. | On. |
| Prompt copying | 0.01 / 0.03-0.05 | D upper bounds 0.01 / 0.03-0.05. G close to those bounds for text-only drafting. | Borrow text; filter multimodal input as before. |
| CPU embedding dequantization | 0.05 / 0.08-0.12 | D zero removed CPU dequantization with pinned host storage alone. G device lookup could remove the attributed host cost, before accounting for kernel/copy costs. Device numerical identity is unverified. | Placement stays off unless explicitly overridden. |

G: the two default compute-path changes could save roughly 0.11-0.14 ms per round at 8k and 0.06-0.13 at 90k. The GPU A/B must determine the net effect; there is no measured GPU speed claim. The predicted gain may be below boot-to-boot variation with two boots per arm; repeat pairs if inconclusive.

Pinned-host example: `--override-tensor 'token_embd\.weight=CUDA_Host' --load-mode none`. Memory mapping must be disabled to allocate the requested host storage. This uses CPU lookup and therefore does not eliminate `dequantize_row_iq4_xs`. Existing device buffer overrides also remain available, but can choose a different lookup implementation or introduce transfers. Do not enable device placement for an exactness experiment until it has its own identity gate. D: replicated placement would consume approximately the embedding tensor's size on each rank (about 0.68 GB per rank in the task brief); a single-device override has a different placement and transfer pattern.

## Why input sharing is still open

Master `050439614` has neither `set_tensor_batch_owned` nor the stack's owning batch-transfer API. Changing only that CUDA method would violate the standalone-master requirement. Existing graph producers already write into backend host buffers where those are supplied, but owning uploads copy those sources again to make their lifetime independent of the caller and reused graph allocation.

Sharing a source by pointer alone is unsafe: `set_inputs` and CPU compute can overwrite it before an earlier asynchronous upload retires. Sharing across backend contexts additionally needs completion from every consumer before reuse. A general solution needs a portable staging/source ownership contract, producer access before filling, and retirement before graph allocation or inputs are reused. It must preserve ordered overlapping writes, source mutation after return, allocation failure fallback, and destruction with pending work. This is an interface/design change rather than a small amendment to master. A scope question is pending; no such interface has been introduced here. The repository instruction to pause before a new pattern applies to that ownership interface, rather than to the three small fixes (`AGENTS.md`, Code and Commit Standards).

## CUDA build and frozen artifact

`freeze_host_bundle.sh` configures CUDA 13.2, NCCL off, arch `120a` as in the requested freeze reference, CCCL v3.4.3 from `/home/xanewok/repos/qwen38-weekend/deps/cccl-v3.4.3`, and disables FetchContent updates/downloads. It forces compiler detection and supplies the architecture so CMake does not run a CUDA device probe. CPU work is niced and builds use at most eight jobs.

M: CUDA configure and compile succeeded; only nvcc compilation was performed. All frozen checksums and RUNPATH checks pass (`logs/cuda-configure.log`, `logs/cuda-build.log`, `logs/frozen-sha-check.log`, `host-results/runpaths.json`).

The freeze writes `frozen/host-bundle/{SOURCE,SHA256SUMS,PRE-PATCHELF-SHA256SUMS,bin/}` and patches RUNPATH to `$ORIGIN` (plus the CUDA runtime directory for `libggml-cuda`). The executable is `bin/round-trace`. Product SOURCE is `85b13faf2`; later evidence-only commits do not change the product. No frozen executable has been run.

## Parent GPU experiment, not submitted or run

`host-bundle.experiment.toml` specifies the required upstream frozen package, the new frozen package, IQ4_XS, staging in both arms, depths 8192/32768/90112, and fresh ABBA boots. It gates native tokens and draft/accepted counters across arms and uses steady full-depth intervals, minimum output length, uncached prefill and engaged staging. The existing harness handles frozen hashes, listener reuse, GPU-idle waits, exact owned process groups, recording each request before gates, and journal attribution. gpuq owns the GPU hazard/idle/RAM gates and runtime budget.

`gpu_ab.sh --canary` runs one patch boot against the recorded CCCL343 upstream requests in `stack3-rebase-cccl-ab/canary-a1/patch/0-patch`. `host_bundle_canary.py` verifies every recorded upstream code-library digest against the specified frozen upstream package, plus model/vocabulary digests, the command and exact seeded request bodies, before seeding the normal cross-arm identity gate. This lets the one-boot canary exercise the actual cross-arm gate. Those reference artifacts and the adapter are pinned. Full execution compares fresh up and patch boots again.

M: `gpu_ab.sh --parsers` passes on the supplied real SSE/GPU-memory/fault fixtures and recorded native Swift streams at all requested depths. It tests missing identities, short data, null/partial/repeated counters, disabled staging, token corruption and accepted-counter corruption. Evidence: `host-results/parsers/results.json`, `logs/parsers.log`, `logs/recorded-reference.log`.

G budget: 350 seconds per boot, one canary boot inside 600 seconds, four full boots with a 1.5 margin and cleanup. The derived item budget is 3000 seconds D; estimated workload is 1800 seconds G, below its 70 percent envelope D. gpuq enforces expiry; direct GPU execution is rejected without `GPUQ_ACTIVE=1`.

Use `CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/write_host_bundle_item.py > codex/host-bundle.gpuq.toml` to render the draft item. It uses `harness.mkitem` and changes only the canary entry to the recorded-reference adapter declared in the experiment. Calling `harness.mkitem` directly would omit that adapter and would not test cross-arm identity in a single boot. No queue submission is performed.

## Open items

- Resolve the input-ownership scope and implement a separate portable input fix if requested.
- Resolve the contradictory `test-backend-ops` instructions before declaring every requested proof complete.
- Run the parent GPU canary and full A/B. GPU identity, net savings and optional device embedding lookup remain unverified.
- Host-name discovery is checked on CPU mocks and compiled against CUDA; actual CUDA pinned allocation is not executed.
