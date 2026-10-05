# Reduced MTP draft vocabulary

## Conclusion

Stack4 already implements and serves the HyperQwen subset. This task adds GGUF-carried IDs, validates the inherited implementation against a separately built stack4 CPU baseline, adds a wrong-ID rejection-law control, and prepares a full-vocabulary versus subset GPU replay. It introduces no new compute optimization relative to the current served subset. Whether k=6 or k=7 beats k=5 remains unmeasured here; the recorded k-sweep already used the subset and favored k=5.

Labels: M = measured or observed in a recorded artifact, D = calculated, G = estimate. Commit hashes, token IDs, seeds, k, file names, flags, and configured bounds are identifiers or configuration, not measurements. No GPU program/query, network request, queue submission, push, or process named llama-server ran. Localhost CPU stub rehearsals ran with GPUs hidden.

## Provenance and design

Base: `590912a1de2097523db6c8efe185387bb5c7d4fc` on `topic/draft-vocab`. Loader/tests: `21746ba07`; baseline/GGUF payload checks and frozen source: `5e24a96c5a052d77f863191070adf3def2fb5455`. The inherited compact-head commits are `faee46be9` and `b8668a92e` (`git show 590912a1d:codex/REPORT.md`). The read-only served profile includes `--spec-draft-vocab` (`~/.claude/skills/local-qwen-server/bin/serve.sh:103`). Historical reduced-head measurements and the later subset-enabled k-sweep are in `~/repos/qwen38-eval/docs/ledger.md:65` and `:83`; they are not measurements from this task.

Use the HQ mapping, not a newly estimated frequency list: `~/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json`, SHA256 `e30af79520ade5b7ddb851d75cf271f07d407dc768bf646e4517607d562a65a6`. It contains 25,879 unique IDs (M), from token ID 0 through 248076 (M), for a target vocabulary of 248,320 tokens (M). Its ordered values match the I64 storage of the adjacent `mtp_draft_vocab_ids.pt` exactly (M, ZIP/storage inspection using Python stdlib, without importing torch). The Swift 1.0 directory is absent from the named model root; historical reuse from Swift 1.0 is supplied background, not independently verified here.

`general.draft_vocab_ids` is a proposed optional GGUF INT32 array of absolute target token IDs, in compact-head row order. It is an ordered injective mapping, not offsets to add to draft IDs. `src/llama-model.cpp:1523` reads it only for an MTP load when explicit model parameters supply no mapping. Type and count are checked before vector allocation; existing range/uniqueness checks apply (`:1527`, `:1545`). Explicit IDs override the GGUF array. Target-only loading ignores it. Models without either source keep the original path.

The existing loader gathers quantized rows once, directly from GGUF before backend repacking, into `[n_embd, |S|]` (`src/llama-model.cpp:1988`, `:2007`). It preserves the head type and blocks. Qwen3.5 MTP projects through that matrix at `src/models/qwen35.cpp:659`; the full path projects through `[n_embd, n_vocab]`. The target head and token embeddings remain full size. This adds a second, small weight matrix; it does not shrink target-model storage.

Host logits scatter to target IDs with -infinity outside S (`src/llama-context.cpp:1316`). Backend sampling attaches a compact tensor of real target IDs (`:1423`) and carries it through top-k/gather (`src/llama-graph.cpp:3916`, `:3950`); backend dist maps its selected index through those IDs (`src/llama-sampler.cpp:1337`). Tensor-parallel sampling takes the union of per-shard top-k before the global chain. Too-small subsets fall back to host sampling; empty TP shards fail loading.

`codex/gguf-draft-vocab.cpp` embeds the array by rewriting metadata and copying the original tensor payload verbatim. It refuses an existing output. A real CPU-model copy with the HQ array was loaded and decoded, with target logits unchanged. Production GPU items use the existing JSON side file for the first measurement, avoiding duplicate production GGUFs; CPU checks exercise the new GGUF loader and payload copier.

## Exactness argument

For normalized target p and the actual proposal q, extended by zero outside S, proposal x is accepted with probability min(1, p(x)/q(x)). Accepted mass at token y is min(p(y), q(y)). Rejection probability is R = sum_y max(0, p(y)-q(y)); rejection draws y from max(0, p(y)-q(y))/R. Therefore total mass at y is min(p(y),q(y)) + max(0,p(y)-q(y)) = p(y), including y outside S. The identity holds conditionally at every prefix, so sequential verification and the final target-only token preserve the target law. If R=0, no rejection occurs in the ideal law.

The code uses the distribution that actually generated the proposal:

- `common/speculative.cpp:64`: production probabilistic MTP uses top-k=10, request temperature, and dist. It does not copy the target's top-p/min-p settings. Equality of draft and target chains is unnecessary; equality of recorded q and the draft's own draw is necessary.
- `common/speculative.cpp:1734`: `common_sampler_sample` draws the draft token; `:1746` selects that token when rejection is enabled; `:1764` copies the resulting candidate IDs and probabilities into `result_q` before the next decode can replace them.
- `common/sampling.cpp:147`: backend probability rows are copied directly with their real IDs. `:834` returns the backend's sampled token without applying another CPU chain. Host sampling also leaves its final probabilities in this candidate array.
- `src/llama-sampler.cpp:1298`: backend dist softmaxes the final compact candidate logits and returns those same probabilities. Host dist similarly normalizes the final chain. Thus any top-k/top-p/min-p/temperature filters preceding dist are included in q; the verifier never reconstructs q from full-head logits or raw compact logits.
- `common/sampling.cpp:919`: missing IDs have probability zero. `:987` looks up q at the drafted target ID, `:997` accepts using p/q, and `:1007` forms the positive residual using that same sparse q. Target grammar masking is renormalized at `:980`. Rejection draws use the target sampler's separate RNG stream (`:586`, `:946`), not the draft RNG.

This is a proof of the normalized rejection-sampling law, subject to the existing floating-point and PRNG implementation. It is not a claim of bit-identical reduced-vocabulary output streams or universal numerical equivalence across GPU verify widths. The measured q sum in the law test is 1.000000015 (M), consistent with float rounding.

Greedy drafting is separate: `common/speculative.cpp:1670` clears `result_q`, and `:1746` takes the leading candidate. Verification uses target-sample-and-match instead of treating a greedy draft as the probabilistic q above. Restricting the draft argmax can change acceptance; it must not change the target sampler's law.

## CPU evidence

Model: `/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf`, the relocated model named by the patchA CPU script. CPU libraries were configured with `-DGGML_CUDA=OFF`. All commands set `CUDA_VISIBLE_DEVICES=""`, `GIT_NO_LAZY_FETCH=1`, nice 19, and at most eight CPU threads. Timing samples are small correctness runs, not speed benchmarks; concurrent low-priority compilation can affect them.

| Check | Result |
| --- | --- |
| Full-vocabulary identity map versus default path | 6 MTP rows and 600 seeded token draws bit-identical (M); 100 seeds per row (configured). |
| Separate `590912a1d` CPU library versus branch | Byte-identical logits/token dumps (M), SHA256 `0f532fa0418db830b788dad3b772b11151f20d8314b1b230defd4ea5d4c27c18`; `codex/draft-vocab-base-identity-sha256.txt`. |
| Reduced proposal law, real common_sampler verifier | 20,000 draws (M); chi-square p=0.489897 (M). Wrong full-vocabulary mass: p=0 (M). Wrong target-ID mapping: p=0 (M). `tests/test-sampling-prefilter.cpp:331`, `codex/draft-vocab-law.log`. |
| GGUF loader validation | Valid mapping; empty/oversized arrays, duplicate/negative/out-of-range IDs, UINT8 array, scalar rejected; target-only ignore and explicit override passed (M). `codex/draft-vocab-metadata.log`. |
| Reduced CPU head and two-CPU meta backend | Target logits bit-identical, compact logits equal their source rows, max error 0 (M); outside logits -infinity; global backend top-k, multi-ubatch/deferred reads, small-subset fallback, empty-shard rejection passed (M). `codex/draft-vocab-subset.log`, `codex/draft-vocab-tp.log`. |
| Actual GGUF with HQ IDs | Target logits bit-identical and all sampled compact-head rows correct, max error 0 (M); host and backend/deferred paths passed (M). `codex/draft-vocab-embedded.log`. |
| Existing sampling suite | Passed (M), `codex/draft-vocab-sampling.log`. |

| CPU setup | Full MTP ms/step (M) | Compact MTP host ms/step (M) | Compact MTP backend ms/step (M) | Steady steps (M) |
| --- | ---: | ---: | ---: | ---: |
| CPU, 8 threads, every tenth row in reverse order | 22.179232 | 5.785502 | 5.520378 | 8 |
| Two CPU devices, 4 threads/device, same subset | 40.121595 | 6.774984 | 8.017681 | 4 |
| CPU, 4 threads, HQ IDs loaded from GGUF | 24.982810 | 6.512196 | 6.407690 | 4 |

The HQ CPU compact matrix uses 54,345,900 bytes (M), preserving Q6_K. CPU measurements do not predict CUDA speed.

Reproduce with `bash codex/check-draft-vocab.sh`, then `bash codex/check-draft-vocab-base.sh`. The second script builds only a CPU library from a local archive of stack4; it neither creates a worktree nor fetches objects. `gguf-draft-vocab INPUT OUTPUT IDS.json embed` writes the carrier; `test` writes sparse fixtures for loader checks with `no_alloc` and `load_mode=NONE`.

## GPU estimate and prepared items

All numbers in this table are G. Illustrative model: legacy full-head k=3 round 34.5 ms (M, supplied trace), plus 4 ms per added draft step (M, supplied background), holding verify/catch-up time fixed. Row fraction 0.104218 is D; assuming linear scaling of the legacy 1.2 ms full draft head gives a 1.075 ms saving per draft step (G). Stack4 host/kernel changes, verify-width changes, memory traffic, and acceptance invalidate a literal extrapolation.

| k | Full round ms (G) | Subset round ms (G) | Saving ms/round (G) |
| --- | ---: | ---: | ---: |
| 5 | 42.50 | 37.13 | 5.37 |
| 6 | 46.50 | 40.05 | 6.45 |
| 7 | 50.50 | 42.98 | 7.52 |

These are estimates of the already implemented subset lever, not a new gain over serving. Larger k must deliver enough additional accepted tokens per round to offset its remaining cost. No new GPU acceptance, speed, or winning-k result exists.

Prepared, checked, not submitted:

- `codex/draft-vocab-j5q.gpuq.toml`
- `codex/draft-vocab-iq4.gpuq.toml`

Entry: `codex/gpu_ab.sh`; runner: `codex/replay/sweep.py`; generator: `codex/replay/gen_items.py`. Base is frozen stack4 without `--spec-draft-vocab`. Subset is `codex/frozen/draft-vocab/build/bin/round-trace` plus the HQ JSON mapping. Both use the same model weights, templates, probabilistic chain, backend flags, q8_0 target KV, f16 draft KV, staging, single slot and CPU thread counts; k and the subset/binary are the experimental variables. Both quants use stack4 as requested, including IQ4_XS; this does not compare with the later per-type IQ4_XS serving alternative.

Boots are `(base,5),(subset,5),(subset,6),(base,6),(base,7),(subset,7)` followed by their reverse (`sweep.py:21`). Each k therefore has interleaved ABBA or BAAB repeats. Under the time budget, each boot replays p09, p06, p03, p02 from the existing frozen corpus: one prefix per source session, rendered lengths 5857, 5857, 6365, 76839 tokens (M, recorded CPU calibration). Requests cap output at 256 tokens, below the original replay cap of 512; this keeps the expanded arm grid within budget. Other existing corpus prefixes are retained as provenance but not executed. This covers the deepest source prefix and the short contexts, not the middle-depth ladder. Prefix token hashes must match within and across arms; generated content identity is recorded but not gated because probabilistic MTP seeds did not reproduce content in the parent runs.

Summary pairs rates by prefix, reports per-prefix token count, decode tok/s, drafted/accepted counts and acceptance, and cluster-bootstraps by source session. Every arm/k repeat control requires its entire 95% CI inside [0.98,1.02], not just covering 1.0 (`sweep.py:135`). Comparisons include subset/base at each k and subset k=6/7 versus subset k=5. No round-time value is fabricated from draft counters. EOS remains enabled; replies shorter than the configured minimum of 16 tokens invalidate the run. Short replies limit speed precision, so failed equivalence yields INVALID rather than an adoption claim.

The one-boot subset k=7 canary repeats each prefix adjacently, permitting prefix-cache reuse while exercising tokenization, SSE, settings, counts, length, hashes, same-config equivalence and integrity gates. Its configured cap is 475 s; item timeout is 480 s. Each full run is capped at 3000 s, CPU preflight at 100 s, and each model item at 3600 s; the stage caps sum to 3580 s (D). Budget basis: about 94,918 rendered prefix tokens per boot (D), 650 prefill tok/s (G, conservative relative to recorded J5q deepest incremental rates near 660-669 tok/s M), 75 decode tok/s (G), and 15 s startup (G): about 2160 s full, 250 s canary and 55 s CPU preflight (G), including roughly 60 s per GPU stage for frozen-input hashing (G, based on the 44.003370 s two-model hash check M): total 2465 s, 68.5% of the item budget (D from G). The estimate assumes adjacent-repeat cache reuse; this is not established on GPUs here. The real canary must fit its cap before the full run starts. Actual expiry reports code 7/budget and writes INVALID, never a partial headline.

The harness saves requests/responses/errors before boot gates, skips incomplete timing snapshots, never sums cumulative counters, checks frozen/model/input hashes, uses word-bounded kernel-hazard detection since the queue cutoff, waits for idle GPU memory, binds ports with SO_REUSEADDR, creates outputs, and stops only its owned PID with matching process start time. Every consumed fixture and runtime input is pinned; retry is fresh. GPU scripts will only run when the parent queues them.

Parser evidence: `codex/replay/cpu-validation.json` names the real llama-SSE artifacts, recorded stack4 `/completion` response, and actual k-sweep response/token artifact used in parser and request-gate tests. The recorded replay passes all request gates; changing its EOS policy fails. Two full and two canary CPU rehearsals passed (M), with biased-repeat, wide-CI, missing-prefix, short-output, sampling drift, frozen-file mutation, impossible/null/repeated-counter and prefix-bound/type controls. This rehearsal took 21.984578 s (M); it is not model inference. Frozen inputs/argv are separately verified in `codex/replay/frozen-validation.json`.

Both commands returned `valid`, exit 0 (M):

`CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 nice -n 19 gpuq submit --check "$PWD/codex/draft-vocab-j5q.gpuq.toml"`

`CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 nice -n 19 gpuq submit --check "$PWD/codex/draft-vocab-iq4.gpuq.toml"`

## Frozen build

`bash codex/freeze-draft-vocab.sh` configured and built HEAD `5e24a96c5`, with CUDA 13.2, local CCCL v3.4.3, NCCL OFF and requested `CMAKE_CUDA_ARCHITECTURES=120`; existing CMake selects 120a. Compiler probes that could run CUDA are disabled; UI downloads and FetchContent networking are disabled. Each build used at most four jobs. Inference used at most eight CPU threads; heavy overlapping build/test jobs used four each. `codex/draft-vocab-cmake-cuda.log` reports ggml commit `5e24a96c5`.

The sanctioned freezer was run with `QWEN38_WT_ROOT="$PWD/codex"`; SOURCE is `codex/frozen/draft-vocab/SOURCE`, and RUNPATH is `$ORIGIN`. The renamed `round-trace` and `llama-server` have SHA256 `0a227d70c4b695ed51114e5f76f2b7add9f8b2f59ce95f3f2564bf4f6109bf1f` (M). Frozen checksums and library resolution passed without executing the binary. Later report/harness commits do not change the compiled runtime source. The freeze binaries remain local, untracked artifacts.

## Risks and upstream form

The GGUF representation is model-agnostic, optional, and carries the mapping with the model. The inherited runtime is still limited to a Qwen3.5 architecture with one loaded MTP layer (`src/llama-model.cpp:1539`) and does not support reduced-head LoRA (`src/models/qwen35.cpp:658`). This is upstreamable storage and sampling design, but not yet a complete general implementation for every MTP architecture. The GGUF key is proposed, not asserted to be an accepted upstream standard; #29143 was considered only from offline local code/notes. No current issue/PR status was checked.

For a production converter: accept a validated absolute mapping from the source checkpoint, emit it as INT32 in the same order as gathered rows, preserve target tokenizer/head/embeddings, and reject duplicate/out-of-range IDs. Offset-style d2t formats need conversion to absolute IDs. Emitting a pre-gathered compact weight tensor could avoid load-time gathering but needs a separate tensor name and loader contract; this change intentionally retains original weights plus one load-time compact matrix. The offline embedding helper already implements the metadata carrier and payload preservation for measurement.

Per-row external head scales, other MTP architectures/multiple heads, and adapters need explicit support before calling this universal. Compact-head allocation increases memory while the target head remains resident. Full/reduced proposals can consume RNG differently and alter acceptance/output lengths. GPU numerical exactness across different verify widths and CUDA compact-head paths is not established by CPU checks. The parent must run the real canaries and full items before drawing a performance conclusion; do not change serving based on the G table.
