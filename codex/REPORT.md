# Backend verify sampling: prompt reservation regression

Labels: M = measured/observed; D = derived, including source inference; G = guess. Configuration values are literal settings.

Read-only `SP`: `/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad`.

## Conclusion

Confirmed unnecessary scheduler reconstruction at each backend-sampled request. Slot reset detaches the sampler and request launch attaches a fresh chain; both previously forced `sched_reserve`. This reconstructs the scheduler, frees its compute buffers, and reserves prompt/generation/prompt graphs inside the next prompt timer.

The fix retains scheduler capacity, invalidates sampler graph bindings on attachment/removal/refusal, and requests a full reservation only when the sampler node budget grows. Sampling and rejection algorithms are unchanged. The first attachment can still incur reservation; warm-up absorbs that startup path in the A/B.

D, high confidence: this lifecycle is the strongest source-supported explanation for the supplied GPU prompt regression. M: CPU controls expose the reservation and fail when it is restored. GPU attribution and gain remain unmeasured. CPU forward timings are too variable to reproduce the GPU slowdown or quantify it.

## Source evidence and design

- `tools/server/server-context.cpp:389` detaches at slot reset; `:1913` attaches the next request chain; `:3274` starts the prompt clock before the first decode.
- `src/llama-context.cpp:1879` calls `sched_reserve` before processing microbatches. `:643` reconstructs graph arenas and the scheduler; `:679`, `:701`, `:726` reserve prompt/generation/prompt graphs. D: CUDA buffer release/allocation adds a backend-specific cost beyond CPU reservation time.
- `src/llama-graph.cpp:3793` already returns before adding sampler inputs/tensors for empty TP logits. Zero-output prompt microbatches therefore build no TP top-K/gather work. This is not the regression cause in this tensor-split configuration.
- `src/llama-context.cpp:2013` and `:2025` gate full-logit extraction/deferred retention on positive output count. No zero-output microbatch copy or retention occurs. `logits_deferred_fetch` does nothing without pending outputs.
- `src/llama-graph.h:850` checks output count before sampler/output-position comparisons. Candidate top-K has no changing RNG inputs. M, supplied: GPU boot logs have identical graph reuse counts between arms.
- The final output-producing microbatch still runs top-K/gather; its GPU cost is unmeasured. This fix neither bypasses its sampling nor moves reservation into decode timing.

`src/llama-context.cpp:1347` invalidates `gf_res_prev_active` even if an allocator recycles the sampler pointer. The next `process_ubatch` rebuilds the selected graph and sampler input bindings. `:1349` compares `graph_max_nodes` against retained arena capacity, preserving other pending reservation reasons.

Changed tensor shapes remain supported by `ggml/src/ggml-backend.cpp:1627`: allocation/backend changes trigger synchronization and buffer reservation before compute. This introduces no multi-graph cache, scheduler family or meta plan cache. Candidate certification, probabilities, RNG draws, grammar transitions, rejection verification and deferred-logit lifetimes are unchanged; the exactness argument at `60c8cbfb2` still applies.

The supplied GPU inputs were read-only: `SP/codex/bv-ab/*.server.log` and `SP/codex/bv-chain.log`. M, supplied: tool prompts contain 602 tokens, think prompts contain 638 tokens, and output digests match. GPU timing observations come from the parent, not execution here.

## CPU evidence

Model: `~/models/Qwen3.5-4B-UD-Q4_K_XL.gguf`, MTP weights loaded, target context only, unmasked NextN embeddings, context/batch 2048, microbatch 128, F16 KV, eight CPU threads. One model/context is reused; memory clears between identical 602-token prompts, with one final output. Arms: off/on/on/off. The final harness detaches the sampler before each request, matching server reset. Reservation and forward time are reported separately and summed.

`codex/profile.patch` is opt-in instrumentation, absent from production. `host_us` measures memory apply + graph reuse/build/allocation + input setup; `compute_us` measures `graph_compute` + synchronization, including scheduler/meta work and CPU operators. Graph counts include non-compute nodes. Single-CPU graphs keep inactive top-K metadata, uncomputed by `ggml_build_forward_select` (`ggml/src/ggml.c:7323`); TP returns before constructing it.

M: complete real-model prefill runs return bit-identical full logits. M: TP with this MTP GGUF is refused because it loads a separate MTP head (`src/llama-model.cpp:2008`); the run stopped rather than silently measuring host sampling. Synthetic Meta(CPU,CPU) tests cover TP candidates.

Before-fix measurements:

| Run/arm | Reserve ms (M) | Forward ms (M) |
|---|---:|---:|
| off, first | 0.001 | 42428.862 |
| on, first | 23.296 | 37599.725 |
| on, repeated | 7.581 | 39095.812 |
| off, last | 65.905 | 36812.102 |

After-fix measurements, including server-style detach/reattach:

| Run/arm | Reserve ms (M) | Forward ms (M) |
|---|---:|---:|
| off, first | 0.000 | 9897.482 |
| on, first attachment | 7.094 | 8505.663 |
| on, repeated | 0.001 | 10809.695 |
| off, last | 0.000 | 9604.080 |

Before-fix per-microbatch means across two runs per arm (D, from M timings):

| Microbatch | Host off us (D) | Host on us (D) | Compute off ms (D) | Compute on ms (D) |
|---|---:|---:|---:|---:|
| first 128 tokens | 5483.5 | 10400.5 | 4396.458 | 5874.533 |
| second 128 tokens | 2040.5 | 1402.0 | 4867.280 | 5448.403 |
| third 128 tokens | 1824.5 | 2435.5 | 10922.805 | 9089.427 |
| fourth 128 tokens | 143.0 | 118.5 | 12521.494 | 10642.495 |
| final 90 tokens, one output | 14013.0 | 6116.5 | 6874.068 | 7245.639 |

M: single-CPU graphs have 1879 nodes off and 1885 on, including one inactive top-K node. Raw before/after measurements: `codex/prefill-measurements.tsv`; local logs: `codex/prefill-{before,after}-single.log`.

System load was uncontrolled. CPU compute varies substantially and does not reproduce the supplied GPU slowdown. The discriminating result is scheduler reconstruction, not those forward-time comparisons.

## Verification

Implementation commits: `ffa868b88` (attachment capacity), `846f3b4ff` (detach/refusal lifecycle).

M: CPU build and CUDA server build pass. `codex/tp-final.log` passes sampler detach/host decode/reattach with changed top-K, seed identity, deferred full rows spanning microbatches, greedy/full chains, distribution and undersized-shard fallback. `test-sampling` passes.

M: `codex/verify-final.log` passes candidate/host seed and probability identity, chi-square with corrupt-logit control, rejection verification, grammar/lazy trigger, grammar-first, FORCING, tie fallback, prefix eligibility and clone/copy. `codex/model-verify-final.log` passes real Qwen3.5 CPU forward outputs and full logits, four output rows with 64 matching seeds each. `VERIFY_NEGATIVE_CONTROL=1` exits 1 with `FAIL: seed or distribution mismatch`.

M: new regression test fails on the original implementation (`codex/prefill-negative-control.log`, exit 255), and on attachment-only retention with the old detach path (`codex/detach-negative-control.log`, exit 255). The final implementation passes both the no-reservation assertion and changed-K candidate/seed checks. Those controls compiled the new test against the old CPU library before rebuilding it.

M: the inherited grammar prefilter suite passes with zero failures and the unchanged stream hash `e97ea9873285a3f9` (`codex/prefilter-final.log`). Its truncated-resample control is rejected by the host-law check.

Commands, all with `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=8`, `TMPDIR=$PWD/build-tmp` and `nice -n 19`:

```sh
cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-cpu --target test-llama-archs test-sampling test-sampling-prefilter -j8
build-cpu/bin/test-llama-archs --tp-sampling -s 1
build-cpu/bin/test-sampling
bash codex/run_verify.sh
bash codex/run_verify.sh --decode
build-cpu/bin/test-sampling-prefilter ~/models/Qwen3.5-4B-UD-Q4_K_XL.gguf --write codex/prefilter-seeds.txt 4000
VERIFY_NEGATIVE_CONTROL=1 build-cpu/bin/verify-sampling ~/models/Qwen3.5-4B-UD-Q4_K_XL.gguf
cmake -B build-cuda -DGGML_CUDA=ON -DGGML_CCACHE=OFF -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_BUILD_TESTS=OFF '-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16'
cmake --build build-cuda --target llama-server -j8
```

CPU profiling: apply `codex/profile.patch`, rebuild CPU target `llama` with `-j8`, run `VERIFY_PREFILL_PROFILE=1 bash codex/run_prefill.sh`, then reverse the patch and rebuild. Before-fix data used the original implementation; after-fix data used the final detach/reattach lifecycle. All timing rows include opt-in instrumentation. The profiling patch must never enter the frozen CUDA image.

No GPU tool, CUDA executable, GPU query, network request or process named `llama-server` ran. Source/build/artifact writes stay in this worktree; commits update only `topic/verify-prefill`, using its linked shared Git metadata directory.

## GPU A/B and remaining uncertainty

`codex/gpu_ab.sh` defaults to a GPU-free dry run. Up: existing frozen `60c8cbfb2` at `~/repos/llama.cpp-backend-verify-sampling/codex/frozen/patch/bin`. Patch: this worktree's frozen fix. Both enable target backend sampling and retain probabilistic MTP and backend draft sampling.

Boots: up/patch/patch/up. Per boot: warm-up, tool (`follow`, seed 1, 384 max tokens, thinking off), think (`follow`, seed 1, 1024 max tokens, thinking on), matching the parent requests. Output includes measured prompt timings, derived prompt deltas, and `predicted_ms / (draft_n/3)` decode-round proxies. It checks manifests, input hashes, up revision, GPU/port activity, Xid, wall budget and exact child PID cleanup; it rejects mismatched outputs/prompt lengths, cache use and zero drafts.

M: `codex/freeze.py` froze the clean implementation at `846f3b4ff` into `codex/frozen/patch/bin` with renamed `verify-ab`, local libraries, `$ORIGIN` RUNPATH, `SOURCE` and verified `SHA256SUMS`. The clean CUDA rebuild passes (`codex/build-clean-cuda.log`); neither frozen executable was run.

M: shell syntax/default dry run pass; summary self-test recovers fixture deltas and rejects four negative controls (`codex/gpu-ab-selftest.log`). No wet run occurred. Parent-only command: `nice -n 19 env RUN_GPU_AB=1 DRY_RUN=0 bash codex/gpu_ab.sh`.

GPU prompt recovery and decode-round neutrality remain unverified. Initial attachment/budget growth still reserve. Retained capacity tracks the largest sampler budget until another scheduler reservation; removal does not shrink it. Shape changes may still reallocate buffers. GPU asynchronous behavior and pipeline parallelism remain untested; this stack uses tensor split.
