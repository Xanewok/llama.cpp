# Backend candidates for rejection verify

Labels: M = measured or observed; D = derived; G = estimate. Build and sampling configuration values are literal settings, not measurements.

## Conclusion

Implemented on `topic/backend-verify-sampling`, base `68d99c67d`, implementation commit `60c8cbfb2`. Enable target `--backend-sampling`; eligible requests now use a separate backend top-K chain even with grammar or reasoning budget. The host keeps the original probability chain, RNG, grammar transitions and rejection verification. Defaults remain opt-in.

M: CPU seed and probability identity, real Qwen3.5-4B forward outputs, Meta(CPU,CPU) TP candidates, deferred logits and CUDA compilation pass. GPU execution and GPU speed are unmeasured. Queue `codex/gpu_ab.sh` for that decision.

## Design and exactness

- `common/sampling.cpp:561`: reuse the existing prefilter eligibility check. The candidate path requires positive top-K below the TP limit, an inactive prefix and no logit biases or model suppression. Other grammar/budget configurations retain the host path.
- `common/sampling.cpp:592`: attach a separate top-K-only chain. D: configured target K=20 requests K+1=21 candidates per output row. Backend sampling consumes no RNG draws. `common/common.cpp:1295` and `tools/server/server-context.cpp:1913` use this chain; the original chain remains on the host.
- `common/sampling.cpp:154`: require finite, strictly decreasing logits through K+1. This certifies unique top-K membership and order, including the cutoff. A tie or failed certificate fetches the full row before the original host selection. Without ties, the host chain sees exactly its original top-K, so probabilities and RNG consumption are bit-identical.
- `common/sampling.cpp:811` and `:866`: budget FORCING, grammar-first sampling and a rejected first draw fetch the full row. The grammar-first retry is never confined to unmasked candidates. A valid forced or grammar token can lie outside the backend top-K.
- `common/sampling.cpp:936`: rejection verification retains its original acceptance and residual algorithm. Grammar checks occur after each accepted token, so an accepted draft can activate a lazy grammar within the verify batch. Its target law remains `c(x)*1[x in G] + (1-c(G))*c'(x)`, where `c'` is the full-row grammar-first chain. Masking the backend top-K before the first draw would change that law.
- `src/llama-context.cpp:913` and `:2019`: retain the final microbatch's full logits on the device. A fallback reads that microbatch once; a subsequent microbatch reads older pending outputs before buffer reuse. Reordering and graph/memory invalidation materialize pending rows first. A new decode discards the old outputs.
- `src/llama-context.cpp:4047`: `llama_get_logits_ith` now follows its documented full-vocabulary contract. Previously it returned packed candidate logits with a backend sampler. The TP full-row test caught this during implementation. Packed values remain available through `llama_get_sampled_logits_ith`.

D: full logits already have output lifetime and host allocation, so this adds no full-logit device buffer. A fallback copies the entire pending microbatch rather than one row. D: a four-row Swift fallback transfers 3.973 MB; the normal candidate output transfers 672 bytes of IDs and logits, excluding existing MTP hidden-state transfers.

## Evidence

All execution used `CUDA_VISIBLE_DEVICES=''`, `nice -n 19`, and CPU thread limits at or below the requested limit. No server process, GPU binary, GPU query or external network request ran. Linked-worktree commits required approved Git metadata access outside the worktree; all source, build and artifact files are inside this worktree.

M: `tests/test-llama-archs.cpp:1039`, `--tp-sampling -s 1`, exit 0:

- Four output rows on Meta(CPU,CPU), each with 64 matching seeds; backend returns K+1 logits and no token draw.
- Deferred full logits match the same TP forward pass bit-for-bit, including outputs spanning two microbatches.
- Existing greedy, full backend chain and distribution checks pass. Existing distribution control: 4000 draws, TV=0.022. Undersized vocab-shard fallback passes.

M: `codex/verify.cpp`, via `bash codex/run_verify.sh`, exit 0 (`codex/verify-final.log`):

- 256 seed identities with exact IDs, logits and probabilities. Changed candidate logits alter 243 seed outcomes.
- Candidate chi-square versus host probabilities: 10000 draws, statistic 22.359, df=18, approximate p=0.216006. Changed-logit control: statistic 143144.438, approximate p=0, rejected by the same threshold.
- 128 all-accepted four-row rejection rounds and 128 off-support residual rounds are seed-identical.
- 32 grammar-rejection cases fetch a valid token outside top-K; 32 grammar-active verify rounds match. Lazy trigger, grammar-first, FORCING, tie fallback, ineligible prefixes, clone and copy pass.
- `VERIFY_NEGATIVE_CONTROL=1 .../verify-sampling MODEL` exits 1 with `FAIL: seed or distribution mismatch` (`codex/negative-control.log`). This corrupts the actual candidate input before the identity check.

M: `bash codex/run_verify.sh --decode`, exit 0 (`codex/model-verify.log`): Qwen3.5-4B `Qwen3.5-4B-UD-Q4_K_XL.gguf`, CPU forward pass, four output rows x 64 seeds with exact probabilities and bit-identical full logits.

M: existing `test-sampling` passes (`codex/test-sampling.log`). `test-sampling-prefilter MODEL --write codex/prefilter-seeds.txt 4000` passes with zero failures (`codex/prefilter.log`): real Codex tool grammar in thinking, prose, function-name and parameter-value states, grammar-first, forcing, trigger trajectories and rejection verify. Its wrong truncated-resample control fails the function-name law check (approximate p=3.38e-89); the per-seed stream hash is `e97ea9873285a3f9`. This suite checks the host law independently; the candidate/host equivalence tests above check the new fast path.

M: CPU build and CUDA server build exit 0 (`codex/build-final-{cpu,cuda}.log`). CUDA configuration matches the base build's architecture, NCCL, FA quant set, graphs, native CPU and BLAS settings.

Commands:

```sh
CUDA_VISIBLE_DEVICES='' TMPDIR=$PWD/build-tmp nice -n 19 cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=ON
CUDA_VISIBLE_DEVICES='' TMPDIR=$PWD/build-tmp nice -n 19 cmake --build build-cpu --target test-llama-archs test-sampling test-sampling-prefilter llama-server -j8
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 nice -n 19 build-cpu/bin/test-llama-archs --tp-sampling -s 1
CUDA_VISIBLE_DEVICES='' nice -n 19 bash codex/run_verify.sh --decode
CUDA_VISIBLE_DEVICES='' TMPDIR=$PWD/build-tmp nice -n 19 cmake -B build-cuda -DGGML_CUDA=ON -DGGML_CCACHE=OFF -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_BUILD_TESTS=OFF
CUDA_VISIBLE_DEVICES='' TMPDIR=$PWD/build-tmp nice -n 19 cmake --build build-cuda --target llama-server -j8
```

## CPU host time

M: synthetic logits with the real Qwen vocabulary, vocab=248320, nice 19, sequential timing phases, 2000 rounds per arm. These timings exclude forward decode, device top-K, transfer/synchronization and server work. They compare the base host prefilter with the new candidate path, not a full-vocab sort baseline.

| Host work, us/round | Full row (M) | Candidates (M) | Difference (D) |
|---|---:|---:|---:|
| Four row chains, final run | 278.469 | 2.500 | -275.969 |
| All-accepted rejection verify, final run | 264.679 | 2.746 | -261.933 |
| All-accepted rejection verify, preceding run | 273.219 | 2.808 | -270.411 |

M: earlier chain runs ranged from 271.882 to 527.878 us/round for full rows and from 2.500 to 2.589 for candidates. Host load was uncontrolled; do not translate these CPU timings into GPU gain. The supplied Patch A report attributes about 0.46 ms/round exposed target sampling (M/D); whether the backend top-K and gather recover that time is unknown.

## Codex coverage

D: `SP/grammar-committee/r3-workload-payoff.txt:36` derives 19.5% grammar-active rounds and 80.5% inactive rounds from token mix and acceptance. With the default eligible chain, finite distinct top logits and no budget forcing, the inactive share uses candidates even though grammar/budget objects exist. Active grammar also uses candidates when its first draw is valid; its rejection-verify mask operates on the original target support.

M, supplied report: zero first-draw grammar rejects in 2043 sampled positions. G: coverage can approach all eligible rounds on that traffic, but a numerical overall fast-round fraction is unmeasured. The inactive share is a conditional coverage estimate, not an observed hit rate for this patch. Ties, forcing, grammar-first, unsupported prefixes and backend refusal reduce coverage.

## GPU A/B and open items

M: `bash -n codex/gpu_ab.sh` and its default dry run pass. Its summary recovers D: -1000 us/round from synthetic fixtures and fails mismatched-output and zero-draft controls (`codex/gpu-ab-selftest.log`). `codex/freeze.py` created `codex/frozen/{up,patch}/bin`, each with `verify-ab`, local libraries, `$ORIGIN` RUNPATH, `SHA256SUMS` and a parent `SOURCE`. The up image comes from the checked frozen base; the patch image contains the implementation commit. Neither image was executed.

Parent command, GPU execution explicitly enabled:

```sh
nice -n 19 env RUN_GPU_AB=1 DRY_RUN=0 bash codex/gpu_ab.sh
```

The script boots up/patch/patch/up. Up uses host target sampling; patch adds `--backend-sampling`; both retain probabilistic MTP and backend draft sampling. Each boot runs warm-up, tool and think requests with the same seed. It verifies manifests and input hashes, refuses busy GPUs/port or another server, checks Xid, enforces a wall budget and stops only its exact child PID. Both executable names are `verify-ab`. It fails on output digest mismatch. Reported time per round uses `draft_n/3`, a D proxy that can include partial final rounds.

Open: GPU speed and seed identity; actual fallback/fast-round rate; GPU asynchronous lifetime behavior, pipeline parallelism and reordered multi-sequence outputs. CPU and Meta CPU tests cover synchronous lifetime and split microbatches. Integration with Patch A's separate scheduler/cache prototype is untested; its review identified sampler ownership defects. The full-row accessor behavior change may affect consumers relying on its former undocumented packed prefix. No issue/PR submission or remote action was performed.
