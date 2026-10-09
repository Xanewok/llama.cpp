# stack10 np1 long-depth identity

The recorded np1 failure changes both native tokens and draft counters. The first visible event is an acceptance difference, before the first different token. No source defect or legitimate reassociation is established by these artifacts. My first ablation is K6a; K11b and K11a follow. The prepared item is valid, not submitted, and capped at 2690 GPU seconds.

Labels: **M** = recorded measurement/artifact; **D** = deterministic comparison or deduction from source; **G** = hypothesis or estimate. Source locations refer to `a588f2a44382a361df3e2a456be0c80ca10dd1d6`; reference source is `45d90d3e4ec1b713bef3804bac5bfc93cbff94c1`. No GPU execution, server, or test binary was used for this analysis.

## 1. Exact differences

**M/D:** Raw root is `/home/xanewok/repos/qwen38-eval/results/2026-10-09-s10-depth-iq4/full-a3/`. Both arms have byte-equivalent JSON request values at every depth: seed `1729`, `n_predict=512`, `temperature=1`, `top_k=20`, `top_p=0.95`, `min_p=0.05`, `ignore_eos=true`, `cache_prompt=false`. Every final response reports `prompt_n=depth`, `cache_n=0`, and `predicted_n=512`. This was seeded probabilistic sampling, not greedy: `0-s9p/request.json:1`, both `command.json:1` (`--spec-draft-sampling probabilistic`), and `depth-32768.json:1` (`/request` and final `/generation_settings`). The report and item preserve the recorded request; switching to greedy would test a different trajectory.

Positions below are zero-based in the full prompt-plus-output sequence. Output token ordinals are one-based. Each SSE partial chunk contains one native token; the final chunk contains none.

| Depth / request in each boot | First draft-counter difference | First native-token difference | Final `(draft_n, draft_n_accepted)`, s9p vs s10 |
|---|---|---|---|
| 8192 / first request, slot 0 task 0 | None, including every partial chunk | None; all 512 tokens equal | `(589,392)` vs `(589,392)` |
| 32768 / second request, slot 0 task 126 | Chunk 52, output token 53, sequence position 32820: `draft_n=95` in both; accepted `34` vs `35` | Chunk 53, output token 54, position 32821: `513` (`" are"`) vs `25` (`":"`) | `(862,338)` vs `(802,350)` |
| 90112 / third request, slot 0 task 318 vs task 306 | Chunk 1, output token 2, position 90113: `draft_n=5` in both; accepted `1` vs `0` | Chunk 2, output token 3, position 90114: `369` (`" is"`) vs `7701` (`" appears"`) | `(513,408)` vs `(470,417)` |

**D:** The 32k event is speculative verification round 19 (one-based): there are 19 positive `draft_n` transitions and 19 distinct positive `predicted_ms` boundaries through chunk 52, with five drafted tokens per transition. Round 18 ended at chunk 51 with `(draft_n,accepted)=(90,33)` in both arms. Round 19 accepted one vs two drafts, while its first emitted token `86633` (`" testers"`) still matched. The 90k event is round 1; the first two native tokens `760,2468` match. These are observable round boundaries, not evidence of the first different hidden activation, candidate probability, or draft token: no per-round draft/logit dump was saved.

**M:** Counter confirmations are in each `server.log:27` (8k), `:52` (32k), `:105` (90k). At 90k the task IDs already differ because the preceding 32k trajectory required different numbers of server loop tasks. The harness stopped after boot `1-s10`; this failed run contains no repeated np1 stock boot and therefore establishes neither repeatability nor a hardware/kernel nondeterminism rate. `console.log:2` is a combined identity error, not a counters-only diagnosis.

**D:** `stack10/identity/observed.json` contains the first differences and SHA-256 of all six input records. Reproduce a comparison with `CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 python3 codex/stack10/identity/compare.py REFERENCE.json CANDIDATE.json`. The existing harness identity is only concatenated native tokens plus final counters (`/home/xanewok/repos/qwen38-eval/harness/lib.py:459`); the diagnostic comparator also checks every partial draft-counter pair.

**M/D:** np2 is stronger than equal counters: both streams' 512 native tokens match across all four boots at all three depths. `stack10/identity/np2-observed.json` records the paths and counters from `2026-10-09-s10-np2-real-iq4/full-a2/`. This is a different workload: two prefills and a joint cached decode, seed `1729+depth` (`harness/depth.py:260`), and MTP chain fallback. It is not an np1 control with only chain disabled.

## 2. Source deductions and candidate ranking

**D:** None of the listed K-component enable conditions directly says "KV >= 32768". The observations locate a difference at sampled depths 32k and 90k; they do not locate a threshold or prove all intervening lengths match or fail. Longer attention reductions, larger scratch/arena plans, and near-boundary sampling can expose differences in components without a KV-length gate.

**D:** np1 builds five serial MTP steps inside one graph (`src/models/qwen35.cpp:786`, loop `:809`, per-step attention mask view `:838`). np2 rejects that graph because `n_seq != 1` (`common/speculative.cpp:1863`), and uses the step loop (`:1939`). Draft KV is f16 in both recorded arms; target KV is q8_0. Thus the chain's one-row long-KV attention is the existing f16 path, not the new grouped q8 decode path.

Ranks are **G**, not identified causes. The source constraints in each row are **D**.

| Rank | Candidate and source | Why it remains plausible, or is demoted |
|---|---|---|
| 1 | K6a: `ggml/src/ggml-cuda/q8-cache.h:12`, `:25`, `:72`, `:91`; `ggml-cuda.cu:4949`, `:5141`, `:5232`, `:5271`; `mmvq.cu:1354` | Reuses quantization for <=16-row tensors within one graph, keyed by tensor identity and layout variant, invalidated by overlapping writes. The unrolled chain adds many dependent one-row tensors and allocator aliases in one evaluation; stale quantization or capture scratch interaction would change draft probabilities before tokens. There is no observed invalidation failure and no KV-length gate; long-KV dependence would be indirect through allocation/capture/timing. |
| 2 | K11b: `ggml/src/ggml-backend-meta.cpp:2177`, `:2243`; `ggml-cuda.cu:2923`, `:2965`, `:3014` | Shared pinned inputs use two rotating slots, upload-completion events before host-slot reuse, compute-to-upload dependencies, and upload-to-compute waits. A missing dependency or wrong lifetime would be a bug, potentially exposed by long attention and the serial chain. The visible ordering is defensible; no missing wait is demonstrated. |
| 3 | K11a: `src/llama-context.cpp:841`, `:1004`, `:1935`, `:3056` | Getter sync is skipped only when `backend_pending` and queued tokens say the context is drained and the scheduler has one copy. The chain reads sampled candidates and commits sampler state between dependent steps (`common/speculative.cpp:1918`); an untracked enqueue would violate that contract. The 8k query-debug canary logged no missed sync, but its real sync at each debug check cannot certify the uninstrumented long run. |
| 4 | K10: `ggml/src/ggml-cuda/top-k.cu:242`; `src/models/qwen35.cpp:846`; `common/speculative.cpp:1837` | Changes the candidate-selection implementation, including small one-row lists and serial chain sampling. Incorrect tie ordering or candidate IDs could change draft probabilities without a large tensor error. It has no KV-length gate; exact component gates make a simple standalone error less likely, not impossible in composition. |
| 5 | K4: `src/models/delta-net-base.cpp:546`; `ggml/src/ggml-cuda/ssm-conv.cu:283`, `:300`, `:330`; Meta realloc fix `ggml-backend-meta.cpp:2653` | Changes recurrent-state reads and rollback snapshots; a wrong snapshot can poison later target features. The kernel loads all input histories before snapshot writes and keeps the old sum order; the Meta fix recreates all retained main graphs after arena reset. No wrong index or freed pointer is demonstrated; disabling the op leaves the realloc fix installed. |
| 6 | K7 and base alpha fusion / persistent rotations: `ggml-cuda.cu:4891`, `:4929`; `norm.cu:179`; `src/llama-kv-cache.cpp:365` | All change target graph execution or residency and can affect features passed to the chain. Their numerical operations or matrix bytes are intended to stay identical, and none has a long-KV enable threshold. Keep them in the ablation sweep because the actual delta includes base optimizations as well as K components. |
| 7 | Base grouped one-query q8 FA: `ggml/src/ggml-cuda/fattn.cu:183`, `:445`, `:836`; `fattn-mma-f16.cuh:2300` | This is an additional delta (`a1fff47fc`, then verify-path fix `2efa6f8dc`), omitted from the supplied K list. It can legitimately change vector-vs-MMA arithmetic and occupancy-dependent reduction partitions, but needs q8 KV and one query row; in the 8k canary its first use is the final one-token catch-up, after all drafting. It is not a supported explanation of round 19 or round 1 without evidence that it ran earlier at long depth. |
| 8 | K8 small-I selection: `ggml/src/ggml-cuda/mmq.cuh:1575`, `:1589`, `:1649` | The selector receives matmul columns `args.ncols_max`, not attention KV length. Small-I configurations exist for J=16/24; np1 verify has six columns and chain steps have one, while np2 joint verify has twelve. A shared default-MMQ code/compiler difference remains possible, but `GGML_CUDA_MMQ_TILES=default` disables tile selection, not all changed K8 source. |
| 9 | K6b / K2: `ggml/src/ggml-cuda/add-rms-norm.h:16`; `fattn-kv-max-cache.h:37` | K6b needs an adjacent eligible ADD/RMS_NORM/MUL chain with private buffers. K2 needs `Q->ne[1]>=1024 || Q->ne[3]>1`; these np1 prefill ubatches are at most 512 rows, and decode has one sequence. Neither engaged in the recorded np1 canary; K2 is excluded by these shapes, and K6b is a negative control unless a long-depth allocator layout makes its alias guard eligible. |

**M:** Recorded np1 canary `/home/xanewok/repos/qwen38-eval/results/2026-10-09-s10-depth-iq4/canary-a3/0-s10/server.log` has 1590 `q8_reuse` lines, 1152 paired RMS lines, 576 SSM_CONV_RS lines, 76 batched top-k lines, and zero K2/K6b lines. K8 has no np1 engagement evidence; its logging env was not enabled in that original canary. The new item's canary enables that log.

**M/D:** The grouped q8 FA marker first occurs at canary `server.log:55548` (`dk=256 gqa=6 seqs=1`), after `:55464` accepted the last `3/4` draft tokens at `n_tokens=8702`. All 32 such markers belong to subsequent one-token processing. The np2 canary also contains 32 markers (`canary-a2/0-s10/server.log:65866` first), so the path is not intrinsically np1-only: its enable check has no `Q->ne[3]==1` requirement. Instrumented short-depth engagement must not be promoted into long-depth causation.

**D:** Long attention partitions depend on KV tiles and actual kernel occupancy (`ggml/src/ggml-cuda/fattn-common.cuh:1193`, `:1223`), so reassociation is a concrete mechanism to inspect, not proof. Multi-query q8 verify code was explicitly restored to stack9 machine code by the SASS gate (`codex/stack10/gates.sh:228`). Its presence in the build gates reduces that candidate; it does not cover long f16 chain attention, every graph composition, or ordering races. No build/compiler difference was measured here.

## 3. Bug or valid floating-point change?

**D:** An identity-contract failure is established; a wrong mathematical result is not. The existing per-component byte gates are finite-shape tests, not a proof that all combined graph/capture/alias states are identical. Conversely, equal generated prefixes are not evidence that internal logits, candidate probabilities, or RNG draw counts matched.

**D:** This probabilistic verifier accepts using `p_x/q_x` and an independent uniform draw, and computes a residual distribution on rejection (`common/sampling.cpp:1008`, `:1019`, `:1026`). Tiny probability changes can cross an acceptance boundary; crossing `p_x>=q_x` can also change whether a uniform is consumed. Changing draft distributions can change acceptance and seeded output while preserving the target distribution if proposal probabilities and acceptance arithmetic remain consistent. Thus "different tokens" alone cannot establish a numerical bug, and "equally valid" cannot be inferred from the counters.

**G:** A restoring single-switch arm identifies a component dependency, not whether its defect is arithmetic or timing. More than one restoring arm suggests interaction. To decide wrong result versus reassociation, capture both arms' target logits/features, draft tokens and top-k IDs/probabilities, verifier uniforms and accept decisions at the same forced prefix, starting before the first visible acceptance difference. Compare the first different operator on identical inputs with a trusted higher-precision reference: small bounded error explained by a changed sum partition supports reassociation; stale inputs, wrong indices, non-finite outputs, missing candidate probability mass, or an ordering-dependent repeat supports a bug. A greedy control and repeated stock boots are useful diagnostics, but neither alone proves numerical correctness.

## 4. One prepared GPU item

Files: `codex/stack10/identity/s10-identity-32k.toml` (experiment), `s10-identity-32k.gpuq.toml` (one item), `gpu_ab.sh`, `runner.py`, `compare.py`, `render_item.py`, and the item-pinned harness snapshot. The snapshot is an unchanged copy of the qwen38-eval harness; `render_support/sitecustomize.py` redirects only mkitem's snapshot destination into this worktree. Rendering invokes `python3 -m harness.mkitem` with cwd `/home/xanewok/repos/qwen38-eval`, then uses the existing stage/CCX wrappers. It does not edit qwen38-eval.

**D:** All execution arms load only `frozen/stack10/build/bin/round-trace` with frozen source `a588f2a44`. The stack9-p55 reference is the archived run, pinned together with its original command, env, and frozen-binary digest record. No stack9 server or new build is planned. The original server flags, model, prompt corpus, target/draft KV types, seed, sampler parameters, `k=5`, `LLAMA_MTP_CHAIN=1`, and `LLAMA_MTP_FOLD_C=1` are retained.

| Arm in execution order | Only stock env override |
|---|---|
| `stock` | None |
| `k6a_off` | `GGML_CUDA_Q8_REUSE=0` |
| `k11b_off` | `GGML_META_INPUT_STAGE=0` |
| `k11a_off` | `LLAMA_SYNC_ELIDE=0` |
| `k10_off` | `GGML_CUDA_TOPK_BATCHED=0` |
| `k4_off` | `GGML_SSM_CONV_RS_OFF=1` (presence disables it) |
| `k7_off` | `GGML_CUDA_RMS_NORM_SCALE_PAIR=0` |
| `alpha_off` | `GGML_CUDA_ALPHA_GATE=0` |
| `rot_off` | `LLAMA_KV_ROT_PERSIST=0` |
| `q8decode_off` | `GGML_CUDA_Q8_DECODE=0` |
| `k8_off` | `GGML_CUDA_MMQ_TILES=default` |
| `k6b_off` | `GGML_CUDA_ADD_RMS_NORM=0` |
| `k2_off` | `GGML_CUDA_FATTN_KV_MAX_SHARE=0` |
| `stock` | None, repeat control |

**D:** There is one target depth, 32768. Each fresh boot first executes the original 8192 request as conditioning, preserving the failed A/B's preceding request and graph/capture history. Samplers reset at request begin (`common/speculative.cpp:1606`), but skipping that request would still change allocation/capture history in a lifetime investigation. The 90112 request is removed. CPU verification established that the live harness request builder exactly reproduces both archived JSON requests.

**D:** `identity="within-arm"` lets the sweep collect intentionally different component arms; it still requires the two stock boots to match. The runner then writes `identity-diagnostic.json`, comparing every arm's native tokens, final counters, and partial draft-counter pairs with the archived 32k reference and archived stack10 record. It reports `restoring_arms`; no restoration is a valid diagnostic outcome. Success requires a complete sweep, repeated stock identity, and exact reproduction of the archived stack10 failure; otherwise it returns insufficient-data rc 5 (or preserves an existing harness failure). It never converts the original mismatch into a successful performance/identity gate.

**G/D:** Budget uses 14 boots x 130 s x 1.3 + 60 s = 2426 s, rounded to 2450 s by mkitem, plus 180 s canary and 60 s cleanup = **2690 s (44 min 50 s)**. Estimated GPU wall is 1200 s; the recorded 8k and 32k request times total about 49 s per boot before load/cleanup. The inherited runtime guard permits 70 s of server lifetime per measured boot (`wall_s-60`); slower boots fail instead of consuming an unbounded run. CPU preflight has a separate 600 s timeout; the 45-minute claim is the GPU item budget, not an estimate including CPU preflight or queue wait.

**D:** The canary runs stock at 8k, proves chain/fold and active base/K7/K4/K6a/K10/K11b paths, and enables MMQ tile logging. It deliberately does not enable `LLAMA_SYNC_ELIDE_DEBUG`, which would perform the synchronization being investigated. Main boots carry no engagement or round-trace logging env. All inherited component switches/mutants are scrubbed using the original knobs plus `GGML_META_SUBGRAPH_MUTANT`; each off arm changes exactly one env value.

Limits: a one-switch sweep does not guarantee identification if two independent differences coexist. K4-off retains the Meta realloc fix; K8-default retains common MMQ changes; Q8-decode-off retains shared FA source changes. If no arm restores reference identity, inspect per-operator state and the unswitched shared code rather than concluding the change is harmless. This item isolates dependency; it does not perform the higher-precision numerical adjudication in section 3.

## Validation and scope

**M:** `CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 python3 codex/stack10/identity/render_item.py` rendered via mkitem and ran only `gpuq submit --check FILE`: `s10-identity-32k: valid` (`stack10/identity/contract.log`). No item was submitted, canceled, retried, or held. Comparator checks against real records and token-only/counter-only mutations passed; all pins existed; every non-stock arm had exactly one env override; Python parsed and `bash -n` passed (`verification.log`). No GPU or test binary ran.

Only report/item/evidence files under `codex/` were added. Existing `codex/PROMPT.md` and `codex/PURPOSE` edits were left out of the deliverable change. No production code, result directory, other worktree, or qwen38-eval file was edited.
