# Runaway history and continuation experiment

## Conclusion

Observed: M: 6 sustained repetition failures survive deduplication: M: 4 weekend Swift1.5 IQ4_XS failures, M: 1 earlier Swift1.0 IQ3_XXS failure, and M: 1 Swift1.5 HQ/vLLM W4A16 failure. Repetition is not unique to IQ4_XS, stack2, probabilistic rejection, or the llama.cpp CUDA FA implementation. This does not exclude any of those from increasing its probability. Confidence: high for the observed failures; unresolved for cause or a quant ranking.

The history cannot establish that J5q or Q6_K prevents repetition. Same-build stack2 S5 has M: 1/43 IQ4_XS versus M: 0/16 J5q; the latter's exact nominal upper bound is D: 20.59%. Task mix, rollout duration, seeds, build changes, different model checkpoints, KV precision, and selectively repeated difficult tasks confound pooled comparisons. No unquantized same-checkpoint control was run. Current J4 has no campaign exposure here.

Recommended next action: use bounded continuations to reproduce a high conditional failure rate, then cross the same prefixes with the serving arms and quants. Add the historical S4 arm to the requested matrix. A decisive production-law verdict requires original tokenized prompts and request/sampler/grammar state; the saved archive lacks them. The delivered reconstructed study is executable with `--exploratory` and labels that limitation in every result.

M = measured/observed; D = derived; G = unmeasured assumed scenario. Parameter values, hashes, source line numbers and commit IDs are identifiers, not performance estimates. No GPU performance was measured or claimed. No model, GPU binary, GPU query, network call or serving process was executed by this worker.

## Census and detector

Local shared clone: `/home/xanewok/repos/ra-rates/stack`, topic `codex/runaway-rates`, parent `68d99c67d958c28b6bff5fdc2db096f81c324ef9`. Production source is unchanged. The serving audit additionally reads the immutable campaign source `0dba27b90` from the supplied stack2 worktree. `source-pins.json` records hashes of the inspected local sources.

Snapshot: `2026-10-01T07:02:24.917724+00:00` through `2026-10-01T07:02:45.872176+00:00` (M), M: 1975 archived attempt directories, M: 228 aliases removed, M: 1747 unique execution identities, M: 1578 recorded output windows. M: 1004 are agentic windows; short bounded profiles have a separate denominator. M: 32 output windows are incomplete/censored. M: 0 duplicate identity groups have conflicting recorded configurations. The campaign is still being extended by its owner; this report is a snapshot, not a declaration that the archive is finished.

Discovery walks both requested session2 GPU-pair roots and every weekend result directory, including aborted campaigns, nested model subcampaigns and HQ/vLLM campaigns. It prunes snapshots, task workspaces, graders and the scratchpad archive. It does not count copied repositories or planned jobs as executions. Coverage is retained at attempt, canonical execution, campaign and full configuration levels in `census/attempts.jsonl`, `unique.jsonl`, `campaign-rates.csv` and `rates.csv`.

`runaways.py:33` detects a sustained letter/punctuation span, low-diversity contiguous token n-gram blocks, or repeated line blocks. Thresholds are declared in source: D: at least 4096 characters; generic token spans also require D: at least 1024 lexical tokens, with at most D: 63 distinct four-grams per D: 256-token block. The lexical tokens are not model tokens. It inspects only generated assistant text/reasoning and tool-call arguments. User prompts and tool results are excluded. Where full rollouts are absent, it falls back to Codex completed items or joined SSE output deltas. It records field, file, JSONL line, decoded-character offsets and a short excerpt for each hit.

Independent warning signals are context truncation (`truncated = 1`), task completion length (D: threshold 50000), single-response length (D: threshold 32768), and Codex parse errors. Prompt-prefill token counts are excluded from response lengths. M: 6 truncation flags, M: 11 long-task flags, M: 14 long-response flags, and M: 1 parse-error attempts were recorded. Missing final completion metrics remain missing; the failed S5 tool-call has M: 93641 server response tokens even though its task completion counter is unavailable.

The detector's six hits have visibly pathological repetition. M: 9 other long/truncated attempts have no dense repetition hit; `census/long-review.json` retains beginning/middle/tail excerpts of their largest fields. Those excerpts contain varied deliberation, repeated solution reconsideration, and ordinary code. They are not certified healthy: semantic repetition can evade this strict detector. In particular the weekend S3 book-store (M: 64665 task tokens) and S4 doubly-linked-list (M: 58075) remain long-deliberation warnings, rather than being silently counted as confirmed alphabet/zero-row loops.

Deduplication uses recorded rollout session IDs, or profile request start identities, rather than token-content equality or directory names (`runaways.py:294`). Different seeded executions can legitimately produce identical content. All archive aliases are retained. M: 223 aliases came from reused rollout sessions and M: 5 from reused request identities in this snapshot.

## Rates and quant evidence

These are detector-hit rates within recorded windows, including the displayed censored windows. They are lower bounds on eventual complete-task incidence where recording stopped early. CIs use the exact Clopper-Pearson formula with D: 95% nominal coverage. The binomial calculation assumes independent trials with one fixed probability; heterogeneous tasks, shared histories and repeated selected prompts violate that simplification. Thus the CIs are exact binomial arithmetic, not a guarantee of coverage for the observational campaign population. Profiles are deliberately not pooled into the agentic rates.

| Engine | Checkpoint | Quant | Windows (M) | Censored (M) | Loops (M) | Rate, % (D) | Exact nominal 95% CI, % (D) |
|---|---|---|---:|---:|---:|---:|---:|
| llama.cpp | BaseQwen3.8 | IQ3_XXS | 93 | 7 | 0 | 0.00 | 0.000-3.89 |
| llama.cpp | BaseQwen3.8 | IQ4_XS | 108 | 6 | 0 | 0.00 | 0.000-3.36 |
| llama.cpp | BaseQwen3.8 | Q4_K_M | 89 | 3 | 0 | 0.00 | 0.000-4.06 |
| llama.cpp | Swift1.0 | IQ3_XXS | 86 | 0 | 1 | 1.16 | 0.029-6.31 |
| llama.cpp | Swift1.0 | IQ4_XS | 110 | 0 | 0 | 0.00 | 0.000-3.30 |
| llama.cpp | Swift1.0 | Q4_K_M | 88 | 2 | 0 | 0.00 | 0.000-4.11 |
| llama.cpp | Swift1.5 | IQ4_XS | 266 | 4 | 4 | 1.50 | 0.411-3.81 |
| llama.cpp | Swift1.5 | J1 | 16 | 0 | 0 | 0.00 | 0.000-20.59 |
| llama.cpp | Swift1.5 | J5q | 16 | 0 | 0 | 0.00 | 0.000-20.59 |
| llama.cpp | Swift1.5 | Q5_K_S | 8 | 0 | 0 | 0.00 | 0.000-36.94 |
| llama.cpp | Swift1.5 | Q6_K | 17 | 1 | 0 | 0.00 | 0.000-19.51 |
| vllm | Swift1.0 | W4A16 | 43 | 1 | 0 | 0.00 | 0.000-8.22 |
| vllm | Swift1.5 | W4A16 | 64 | 1 | 1 | 1.56 | 0.040-8.40 |

The full table `census/rates.csv` crosses engine, checkpoint, quant, speculative maximum, subset, draft policy, target/draft backend sampling, min_p, target KV and recorded build. The build key combines source commit and executable hash where available. HQ records only `vllm_version`; local patch identity is unpinned. Its historical subset setting remains `engine-default/unknown`: the current local MTP implementation automatically loads `mtp_draft_vocab_ids.pt` (`qwen3_5_mtp.py:100`), which cannot prove that file's historical presence. J1 is identified by the archived `swift15-iq4-xs-j1` spec and `IQ4_XS-ssmout6.gguf`, not merged into plain IQ4_XS.

When launch arguments omit defaults, they are inferred only for the inspected local commits: target backend sampling off, draft backend sampling on, draft greedy/match, min_p=0.05. `inferred_source_defaults` records each such inference. The original upstream MTP code selects `cur_p->data[0].id` at `03fa73cb27:common/speculative.cpp:1677`; its intermediate call to a sampling chain does not make the actual proposal probabilistic. Backend flags describe requested settings; device capability fallback is not inferred to have run. Unknown settings stay unknown. There is no usable randomized min_p contrast in these histories.

More comparable serving strata:

| Engine/build | Quant | Draft maximum/subset | Windows (M) | Loops (M) | Exact nominal 95% CI, % (D) |
|---|---|---|---:|---:|---:|
| stack2/0dba27b90 | IQ4_XS | 3/off | 28 | 0 | 0.000-12.34 |
| stack2/0dba27b90 | IQ4_XS | 3/on | 24 | 0 | 0.000-14.25 |
| stack2/0dba27b90 | IQ4_XS | 4/on | 25 | 3 | 2.547-31.22 |
| stack2/0dba27b90 | IQ4_XS | 5/on | 43 | 1 | 0.059-12.29 |
| stack2/0dba27b90 | J5q | 5/on | 16 | 0 | 0.000-20.59 |
| vLLM/version only | W4A16 | 3/engine-default/unknown | 28 | 0 | 0.000-12.34 |
| vLLM/version only | W4A16 | 4/engine-default/unknown | 36 | 1 | 0.070-14.53 |

The older Swift1.0 IQ3_XXS loop uses greedy drafting/match verification, before stack2. The HQ loop uses a separate serving engine and W4A16 weight representation. Both are counterexamples to a single-implementation explanation, but both still use speculative decoding; neither proves that no-spec is safe or unsafe. On the same task, the three S4 alphametics repetitions reproduce across different boots, which justifies a targeted mechanism experiment even though the global S4 sample is small.

| Engine/quant | Task | Draft maximum | Repeated span characters (M) | Evidence |
|---|---|---|---:|---|
| llama.cpp/IQ3_XXS | rust/alphametics | 3 | 124070 | `/home/xanewok/repos/qwen38-eval/results/session2/pair-5060ti-2060s/campaign-swift-fullsuite-20260915/swift-iq3-xxs/attempts/c166943d9bbf48fc-01/codex-state/sessions/2026/09/15/rollout-2026-09-15T06-35-26-01a0a3c7-0c62-7e22-8fa1-caa9f4899df3.jsonl:24` (`content.0.text`) |
| vllm/W4A16 | rust/alphametics | 4 | 46221 | `/home/xanewok/repos/qwen38-eval/results/session2/pair-5060ti-x2/campaign-hyperqwen15-u16-minp-k4-hsfast-longtail-20260929-rep2/attempts/90e42cc604ecf441-01/codex-state/sessions/2026/09/29/rollout-2026-09-29T17-02-31-01a0ee1e-32c4-7fc0-86bc-c2bf15430b2e.jsonl:24` (`content.0.text`) |
| llama.cpp/IQ4_XS | rust/alphametics | 4 | 129905 | `/home/xanewok/repos/qwen38-weekend/results/weekend-20260930-main/campaign-swift15-weekend-20260930-main-phase1-S4-A-IQ4_XS-e3b59faa312a-pair1/attempts/0957923a4a4b4e1b-01/codex-state/sessions/2026/09/30/rollout-2026-09-30T23-22-37-01a0f4a0-8b1d-73a1-8a1b-f316dd232125.jsonl:26` (`content.0.text`) |
| llama.cpp/IQ4_XS | rust/alphametics | 4 | 102172 | `/home/xanewok/repos/qwen38-weekend/results/weekend-20260930-main/campaign-swift15-weekend-20260930-main-phase1-S4-A-IQ4_XS-e3b59faa312a-pair2/attempts/0957923a4a4b4e1b-01/codex-state/sessions/2026/10/01/rollout-2026-10-01T00-51-09-01a0f4f1-9b77-7fa1-9b00-bd89e1b6d1dc.jsonl:28` (`content.0.text`) |
| llama.cpp/IQ4_XS | rust/alphametics | 4 | 131066 | `/home/xanewok/repos/qwen38-weekend/results/weekend-20260930-main/campaign-swift15-weekend-20260930-main-phase1-S4-A-IQ4_XS-e3b59faa312a-pair3/attempts/0957923a4a4b4e1b-01/codex-state/sessions/2026/10/01/rollout-2026-10-01T03-20-10-01a0f57a-06b9-7352-a17b-cc35bd5134ec.jsonl:31` (`content.0.text`) |
| llama.cpp/IQ4_XS | rust/nucleotide-codons | 5 | 149160 | `/home/xanewok/repos/qwen38-weekend/results/weekend-20260930-main/campaign-swift15-weekend-20260930-main-phase2-S5-A-IQ4_XS-e3b59faa312a-pair2/attempts/ad1513a00c7e62e8-01/codex-state/sessions/2026/10/01/rollout-2026-10-01T05-35-35-01a0f5f6-012e-76e0-a12f-1500776e6ec1.jsonl:34` (`arguments`) |


## Local loop bounds and exactness audit

The audit concerns stack2 `0dba27b90`; paths below are relative to that source unless stated otherwise.

- Target penalties/DRY are constructed in `common/sampling.cpp:492` and `:526`, in the declared target sampler order (`common/common.h:263`). Rejection verification calls `common_sampler_sample_row` at `sampling.cpp:952`, obtains the resulting target probabilities at `:954`, accepts the actual retained token at `:995` or `:1024`, and therefore advances target history before the next row. Ordinary target decoding uses the same sampler and accept functions (`server-context.cpp:3990`, `:3995`). The draft proposal does not have to use the same penalty chain: actual normalized q is recorded and the rejection law corrects it to configured target p. Confidence: high for this source path; no new CPU or CUDA engine-law test was run here.
- Stateful transformations cannot be preceded by an unmodified-logit top-k reduction. `sampling.cpp:279` and `:285` disable the candidate prefilter for active penalties and DRY before top-k. The candidate backend shortcut is consequently disabled (`:561`, `:562`). Grammar/budget requests then disable incompatible backend sampling (`:564`, `:570`). Penalties refuse device initialization for more than one output row (`src/llama-sampler.cpp:3028`); DRY has no device implementation (`:3641`). The backend chain stops its capable prefix when a sampler refuses (`:754`, `:762`). This costs the optimized path, rather than bypassing the configured penalty.
- Penalties and DRY intentionally change the target law. Speculative exactness means equivalence to ordinary sampling with those same settings, not equivalence to the previous unpenalized model. They cannot guarantee that no loop occurs. DRY defaults include newline as a sequence breaker (`common/common.h:261`), which limits cross-line zero-row suppression. Any new setting needs a separate controlled study; it should not be mixed into the quant/mechanism arms.
- A per-response length bound terminates generation after the requested number of retained tokens and does not alter the preceding target conditional laws. The local Codex `ResponsesApiRequest` has no `max_output_tokens` field (`/home/xanewok/repos/codex/codex-rs/codex-api/src/common.rs:170`); the campaign config has context/compaction limits but no output cap. These control context growth and between-response compaction, not a response already looping. Tool `max_output_tokens` controls tool-output display, not model generation.
- The server Responses converter maps `max_output_tokens` to `max_tokens` (`tools/server/server-chat.cpp:295`), and the schema maps it onward to `n_predict` via supported aliases (`server-schema.cpp:44`). Global `--n-predict` is only a default: explicit request limits take priority (`server-context.cpp:1932`). Recommended immediate bound: have the existing local bridge/proxy inject and clamp an explicit response limit, or add the missing outgoing request field in the client in a separate change. G: 8192 is a reasonable starting response budget for this workload, not a measured quality-preserving setting. A global server default can provide a fallback for the present client, but is not a hard cap against explicit overrides.

The prior report's checkpoint acceptance-RNG replay bug remains a separate known exactness risk. This task neither fixes it nor treats it as an explanation for the ordinary RS campaign path. Sampling production code is unchanged, so this task adds no new output-law transformation.

## Experiment design

Use the same frozen stack2 executable and libraries for all primary arms. Cross IQ4_XS and J5q with no-spec, S0 full-head k3, S3 subset k3, S5 subset k5, S5/f16 target KV, and S5/q8_0 target KV with `GGML_CUDA_FA_MMA_Q8_0=0`. Draft KV remains f16. All other server/request settings match the supplied stack: tensor mode, 50/50 split, FA on, NCCL-off frozen build, probabilistic proposals/rejection, target/draft backend sampling requested, temperature 1, top_k 20, top_p .95, min_p .05, fixed context/batch sizes. Add S4 explicitly as the historical onset control; it is optional in the harness so the requested matrix remains the default.

The convert-path arm changes dequantization placement. It does not remove MMA padding: k4/k5 still select a query tile width of eight. A clean convert-path result cannot clear the known mask-row over-read. f16 target KV also keeps the multi-query MMA family, while changing stored KV precision. A S5-to-no-spec difference within one quant implicates speculation/integration conditional on that prefix; S5-to-S0/S3 separates depth/subset, S5-to-f16 separates stored KV effects, and S5-to-convert isolates the dequantization switch. Neither numerical agreement on a short prefix nor equal observed loop rates proves correctness at long context. Weight quant is a different axis from KV quant.

`prefixes.py:41` reconstructs all six detected cases, retaining observed response items and source locations. Two primary fixtures are the first weekend S4 alphametics onset and the S5 nucleotide-codons tool onset. Cuts are D: 512 decoded characters before the detected span; they are not original token-boundary positions. For the tool case, parser JSON is decoded back to the template's XML cmd parameter syntax. The parent renders historical chat messages through `/apply-template`, appends the unfinished assistant prefix, and tokenizes it with the serving model. No sampled tools are executed.

These reconstructions explicitly lack original request schemas, exact input filtering, token IDs, lazy grammar and budget state. `tools=[]` is an intentional diagnostic simplification, not a claim that the production tool prompt is reproduced. The raw `/completion` continuation does not reproduce the production tool grammar. Results from this mode answer whether the reconstructed prompt can enter a loop, and can motivate a full-fidelity rerun; a negative does not clear the original bug.

For decisive attribution: first rerun the original failing task with bounded output and local request/returned-token capture. Save exact prompt token IDs, tools, all sampling fields, active grammar/budget configuration, retained prefix IDs, and per-round target logits before onset. Reconstruct the state from the start of that response with an audited teacher-forcing/replay helper; verify next-token target distributions against uninterrupted execution before sampling branches. Start prefixes before entry into the abnormal repetitive span, not inside it. Repeat at a later context as well as near the first onset. Existing raw server endpoints cannot import arbitrary active grammar/sampler state. The harness rejects a supplied nonempty `sampler_history_ids` rather than silently ignoring it; implementing that replay helper is an explicit open item. Ready fixtures must provide `token_ids`, `exactness_evidence`, `state_fidelity_verified`, and any audited request overrides. Those fields attest external validation; the harness cannot independently prove their truth.

Pre-register D: N=64 independent continuations per primary prefix and arm and D: L=8192 retained output tokens per continuation. The outcome is a new detector hit in the generated suffix; the prefilled history is not scanned as output. Save EOS, limit hits, generated IDs, timings and raw final response. Missing final SSE, OOM, timeout or process failure is censored and remains pending on resume, never a negative. Seeds are derived separately by prefix, quant, arm, build and trial; no cross-arm bit-identical-token assumption is made for probabilistic speculation.

Primary comparisons: S5 versus no-spec within each quant and prefix, plus IQ4_XS versus J5q in S5 for each prefix. This gives D: 6 two-sided Fisher tests, each at D: alpha=.05/6. Analyze each fixed prefix separately; do not treat the three related alphametics archives as independent sampled tasks. Secondary S0/S3/f16/convert/S4 comparisons localize a primary effect and require confirmatory follow-up or a separately adjusted family. These are conditional onset probabilities, not global task base rates.

Exact unconditional binomial power (`power.py:12`, `power.json`), with G: the displayed true rates and D: all powers:

| N per arm (D) | 1% vs 20% (G), power (D) | 1% vs 10% (G), power (D) | 1% vs 5% (G), power (D) | Zero-event upper nominal 95% bound (D) |
|---:|---:|---:|---:|---:|
| 64 | 82.79% | 18.05% | 0.76% | 5.60% |
| 128 | 99.83% | 68.09% | 10.89% | 2.84% |
| 256 | >99.99% | 98.35% | 43.98% | 1.43% |
| 512 | >99.99% | >99.99% | 87.31% | 0.72% |

Thus the default identifies large conditional differences. It cannot rule out rare runaways. For a confirmatory study aimed at G: 10% versus G: 1%, pre-register D: N=256 instead; the matrix budget grows proportionally. Length-limited negatives say nothing about entry after L. Final-horizon comparisons require equal completed N; stopping because one arm looks good/bad invalidates the stated fixed-sample power.

## Parent-only harness and CPU evidence

`gpu_ab.sh` forces CUDA hidden in its launcher. Only `RUN_PARENT_GPU=1 PLAN=0` lets `experiment.py` set devices for its own renamed child. Default plan mode makes no GPU query, subprocess or HTTP call; a test replaces those entry points with failing sentinels. Every executed command here used CUDA hidden and nice 19; numerical libraries used a single thread. No external files were modified.

Parent command for an exploratory transport smoke, not a power test:

    RUN_PARENT_GPU=1 PLAN=0 bash codex/gpu_ab.sh --exploratory --n 2 --length 512 --budget 1800 --out build-runaway-smoke

Parent command for the predeclared reconstructed study, repeat unchanged to resume:

    RUN_PARENT_GPU=1 PLAN=0 bash codex/gpu_ab.sh --exploratory --budget 7200

To include the historical depth control, use a fresh output directory and add `--arms no-spec S0 S3 S4 S5 S5-f16 S5-convert`. To cross the additional weight quants, use a fresh directory and add `--quants IQ4_XS J5q Q5_K_S Q6_K`. A full-fidelity run supplies validated ready fixtures with `--prefix-dir ...` and omits `--exploratory`. The source-only base `68d99c67d` has no subset feature; optional `--up-dir <frozen-base-bin>` adds only an S0 full-head base versus stack2 control. No new engine patch is being A/B-tested.

The harness validates upstream SHA256SUMS, copies the server under `ra-continuation-srv` and its libraries into its own frozen directory, writes a new SHA256SUMS, hashes models/template/subset/driver/detector/fixtures, and rejects changed resume settings. It rotates and reverses cell boot order, with D: 8 continuations per prefix per boot. It refuses busy selected GPUs, pre-existing Xid/fallen-off logs, a busy port or an existing `llama-server`; checks Xid again after each boot; enforces a wall watchdog with D: 20 seconds reserved for cleanup; stops only the exact owned Popen child, with bounded TERM/KILL waits. Memory fitting is off, and actual slot context is checked. If f16 KV or Q6_K does not fit, do not silently shrink its context: record censored failure and design an equal-context comparison separately.

Default plan dry run: D: 96 boots, D: 1536 trials, D: maximum 12582912 generated tokens; D: 7200-second budget per invocation. The parent must queue sufficient resumptions; the wall budget is not a promise of completing the entire study. Trials are atomic JSON files. `primary-tests.json` emits the predeclared two-sided Fisher comparisons only when the whole primary family has its planned completed sample sizes; its alpha is adjusted to the actual registered family size. Incomplete requests remain pending, and completed outcomes are summarized per build/quant/arm/prefix with nominal exact CIs.

CPU evidence is in `CPU-EVIDENCE.json`: detector positives, ordinary-output/user/tool-result negative controls, canonical deduplication, exact CI endpoint, schedule coverage, seed uniqueness, SSE completion/censor handling, controlled request fields, Fisher-family censoring, renamed frozen copies and corrupted-library rejection pass. `test_runaways.py --broken-detector` fails with `detector missed positive fixture`, as required. Shell syntax and Python compilation pass. Census runtime is M: 20.955 seconds; this is single-process CPU parsing of the local archive, not a model benchmark. A renamed model server was not started; the dry run uses in-process transport fixtures and plan generation. No CMake/CUDA build was needed because no engine source changed.

Local model inventory (observed regular GGUF files): IQ4_XS, J1 (`IQ4_XS-ssmout6`), J4, J5q, Q5_K_S, Q5_K_M, Q6_K, fdown6mid-bf16 and BF16. `experiment.py` contains exact paths for its supported matrix quants; Q5_K_S and Q6_K both exist on disk. Presence is not proof that a large model/KV/context combination fits.

Reproduce the analysis locally, without GPU execution:

    CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 19 python3 codex/runaways.py
    CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 19 python3 codex/power.py
    CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/prefixes.py
    CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 19 python3 codex/test_runaways.py
    CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/test_experiment.py
    CUDA_VISIBLE_DEVICES='' nice -n 19 bash codex/gpu_ab.sh

## Risks and open items

- Observational rates do not identify a quant cause. Current clean quant samples are underpowered, and semantic loops outside the strict detector remain possible.
- The archive contains interrupted and live attempts; preserve the census timestamp and censor counts when comparing future rescans.
- Full-fidelity onset token/state capture and a validated replay helper are required before treating continuation results as decisive for production exactness, especially the tool-argument case.
- GPU boots, rendering/tokenization against the frozen CUDA binary, model capacity and device-specific fallback have not been executed here. The parent harness has CPU contract tests, not GPU validation.
- FA convert-path and f16 KV controls do not by themselves eliminate the padded-mask bug. A direct guarded-mask/kernel control belongs to the parallel kernel investigation.
- Per-response output limits contain cost and malformed tool strings; they do not prove the underlying decoder correct. Penalty/DRY settings should be tested as a distinct policy change, not used to mask a speculative-law defect.
