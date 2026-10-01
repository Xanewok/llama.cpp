# codexh MTP acceptance audit

## Conclusion

Observed: the large campaign difference is substantially inflated by a degenerate g32i8 response, not a different MTP package. g32i8 r1 emitted a malformed tool call with M: 32,768 output tokens and M: 3,530 repetitions of `.map(|t| t)`. Complete server intervals inside that response accepted M: 25,549 / 25,555 drafted tokens, D: 99.9765%. This increased the token-weighted campaign acceptance while producing unusable output.

Observed: every MTP tensor, the target and reduced draft heads, embeddings, final norm, configuration, tokenizer, template, and draft ID file are byte-identical between g32i8 and codexh. There is no packaging fix to deliver. Only body packed weights and body FP8 KV scales differ.

Inference: trajectory composition explains much of the apparent advantage in g32i8 r1. A smaller checkpoint-associated difference remains plausible, but these freely generated campaigns do not establish its cause or magnitude at equal prefixes. Hidden-state shift, changed target probabilities, and changed FP8 KV scales remain distinct candidates. A lower target KLD against BF16 does not imply a smaller target-versus-draft distance.

Labels: M = observed measurement/count; D = arithmetic derived from observations; G = untested estimate. Identifiers, dates, tensor shapes, configuration values and command arguments are literal values, not performance estimates. No GPU work or network access occurred. No production sampling or checkpoint was changed.

## Checkpoint evidence

`codex/checkpoints.py:26` reads safetensors headers and hashes raw tensor bytes through `safe_open(..., device='cpu')`, one tensor at a time. It never loads a shard or a model into a dictionary. Full local results are `codex/checkpoints.json`; the committed digest summary is `codex/checkpoints.summary.json`.

| Check | Result |
|---|---|
| Tensors in each checkpoint | M: 2,054 |
| Different dtype, shape, shard assignment, or tensor size | M: 0 |
| Different tensor payloads | M: 426 |
| Different body `weight_packed` | M: 400, across M: 64 body layers |
| Different body `k_scale` / `v_scale` | M: 12 / 14 |
| Identical `mtp.*` tensors | M: 34, including the draft head |
| Different audited auxiliary files | M: 0 |

Identical quantization declarations: body GDN projections and MLPs are symmetric int4, group size `32`; body attention q/k/v/o is symmetric int8 per channel; both heads and embeddings are int8, group size `128`; MTP fc, attention and MLP projections are int8, group size `128`. MTP norms and body norms are BF16. Body regexes exclude `mtp.` and the MTP group has its own anchored target. Ignore lists and KV schemes are identical. This is already the intended prepfix packaging.

The following hashes match in both arms:

| File or tensor | SHA256 |
|---|---|
| `config.json` | `6667dfd2feaa2beb6a7687f8b7398b405864f1591491e3c0ec28496a040ed733` |
| `mtp.fc.weight_packed` | `36f4e1c71b918bcff22d3ee7cebac4ff563164e3abcfbd76454dac33829d6753` |
| `mtp.draft_lm_head.weight_packed` | `c08d516dda7526c8a97a45a71a576f20b0e9d740014d8e9f81130a99fd2a9edf` |
| `lm_head.weight_packed` | `8e075093396845823dcade7d0879cf09bbd2c1f4a3b73d852f4dfdcb280ad1c8` |
| `model.language_model.embed_tokens.weight_packed` | `44ff7a883dfa4eab50a54e711a62cc436c46dad115ce9a08aae13e8418ac65ac` |

Both `mtp_draft_vocab_ids.pt` files also equal the literal list in `recipes/_common/hq/swift_draft_vocab_ids.json`. Neither prepared checkpoint contains a `swift_draft_vocab_ids.json`; the exported PT file is what vLLM consumes. Norms, weight scales and shape tensors also match, not just packed weights.

Additional observed confounder: changed body KV scale ratios range from D: 0.8248 to 1.2095, codexh/g32i8 (`codex/kv-scales.json`). The installed `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:1206` transfers these checkpoint scales to the attention layer; `:1217` sets the scalar variants used by the backend. The campaigns use `--kv-cache-dtype fp8`. This can change hidden states and target predictions independently of the packed weight differences; it is not an MTP quantization mismatch.

The old g32i8 `SHA256SUMS.txt` is stale after prep, including its config hash. The parent probe therefore freezes hashes from the current files instead of trusting that manifest.

## Campaign evidence

These campaigns are HyperQwen/vLLM, not llama.cpp tensor-mode campaigns. Every attempt's `launch.json` has identical command arguments, environment, template hash and vLLM version after replacing the model path (`codex/campaigns.py:40`). Logs load M: 25,879 draft vocabulary tokens and attention block size `848` for every arm. Example: g32i8 r1 `serve.log:64` and `:78`; codexh r1 `serve.log:65` and `:80`. The warning sets match except for g32i8 r1's malformed tool-call warning.

Task counters in `job-*.json` reproduce the supplied aggregate results. Campaign acceptance is total accepted drafts divided by total drafted tokens, not the unweighted task mean.

| Arm | Accepted / drafted (M) | Acceptance (D) | Existing decode rate, tok/s (D) | Reasoning share of output (D) |
|---|---:|---:|---:|---:|
| g32i8 r1 | 136,982 / 208,560 | 65.680% | 128.365 | 59.757% |
| g32i8 r2 | 110,801 / 183,365 | 60.427% | 121.848 | 79.260% |
| codexh r1 | 91,637 / 156,430 | 58.580% | 119.098 | 76.968% |
| codexh r2 | 129,488 / 221,900 | 58.354% | 117.399 | 76.013% |
| g128 r1 | 83,961 / 135,085 | 62.154% | 133.108 | 73.787% |
| g128 r2 | 116,082 / 194,595 | 59.653% | 128.702 | 81.946% |

Rates above come from existing logs, not a new speed measurement. The task-level acceptance percentages are:

| Task | g32i8 r1 (D) | g32i8 r2 (D) | codexh r1 (D) | codexh r2 (D) |
|---|---:|---:|---:|---:|
| javascript/complex-numbers | 54.78 | 52.32 | 55.41 | 50.75 |
| rust/alphametics | 59.40 | 54.35 | 56.76 | 54.81 |
| rust/book-store | 59.09 | 59.10 | 58.91 | 68.61 |
| rust/doubly-linked-list | 65.22 | 61.60 | 56.45 | 58.24 |
| rust/forth | 57.08 | 61.75 | 60.72 | 56.33 |
| rust/nucleotide-codons | 92.47 | 66.74 | 58.53 | 66.06 |
| rust/react | 60.31 | 64.17 | 63.88 | 61.93 |
| rust/xorcism | 60.37 | 64.25 | 59.62 | 59.04 |

This is not a uniform task penalty. The largest exception is `rust/nucleotide-codons`; codexh also wins on some tasks. The within-g32i8 campaign spread is D: 5.253 percentage points. The paired aggregate g32i8-minus-codexh differences are D: 7.100 and 2.072 points: one exceeds that spread and one is smaller. Independent stochastic trajectories and `seed: null` in the jobs prevent a paired-seed interpretation.

### Degenerate response

Source campaign: `~/repos/qwen38-eval/results/2026-09-28-dual-5060ti/campaign-hq15-ckptab-g32i8-r1-20261001/`.

- `attempts/02115757a053dbc3-01/model-calls.jsonl:5`: M: 32,768 output tokens, M: 17 reasoning tokens, HTTP `200` and `response.completed`.
- `serve.log:818`: `finish_reason=length`, `generated_tokens=32768`.
- `serve.log:819`: invalid JSON tool arguments, M: 77,191 characters; coerced to an empty object.
- `attempts/02115757a053dbc3-01/codex-state/sessions/2026/10/01/rollout-2026-10-01T15-48-46-01a0f827-665b-7483-9000-02fef4f7b876.jsonl:43`: the raw unfinished `exec_command` arguments contain M: 3,530 copies of `.map(|t| t)`.

`codex/cause-evidence.json` records the source path, counts and time interval. M: 17 complete nominal log intervals inside this call contain the almost perfectly accepted repetition. The task eventually passed, so job success does not identify this bad response.

Sensitivity analysis: exclude `rust/nucleotide-codons` from every arm, keeping the other tasks. g32i8 becomes D: 60.492% / 60.195%; codexh becomes D: 58.584% / 58.193%. The differences shrink to D: 1.907 / 2.002 points. This is a diagnostic exclusion, not an unbiased estimate of an intrinsic checkpoint penalty.

### Regions and rounds

`model-calls.jsonl` records total/reasoning tokens for each request but no draft counters or token-region timestamps. Attempt metrics are task totals. `serve.log` provides interval counts and rounded cumulative per-position survival, not individual rounds. Exact reasoning/prose/tool-argument acceptance and per-round outcomes cannot be recovered from these files.

The conservative proxy in `codex/campaigns.py:51` assigns only whole nominal ten-second log intervals contained within a single request. A reasoning-heavy request has reasoning share at least `0.8`. Its intervals have D: 56.40% / 56.90% acceptance in g32i8 versus D: 55.07% / 55.31% in codexh. The corresponding first-position survival is approximately D: 84.1% / 84.7% versus D: 83.2% / 83.4%. These are rounded, trajectory-confounded call proxies; they cannot identify a tool-only or late-draft-specific failure. Intervals without a unique containing call remain unassigned.

## Fixed-prefix test design

For target probabilities `p` and draft probabilities `q`, expected acceptance of one probabilistic proposal is exactly `sum_i min(p_i, q_i) = 1 - TV(p, q)`. This follows by averaging `min(1, p_i/q_i)` over a draw from `q`. The target-versus-BF16 KLD supplied in the brief measures a different distance. It was also measured before HQ prep; it does not validate the served FP8-cache distribution.

`codex/teacher_probe.py` streams dequantized body layers through the installed Transformers reference implementation, then runs the identical MTP head. Its alignment follows the installed `vllm/model_executor/models/qwen3_5_mtp.py:183`: concatenate normalized embedding and hidden state. `h[t]` plus `x[t+1]` produces the draft prediction for `x[t+2]`; compare that draft with target logits from `h[t+1]` (`codex/teacher_probe.py:222`). Both checkpoints see exactly the same frozen token histories.

The probe computes raw and reference-filtered overlap, using temperature `1`, `min_p=0.05`, `top_k=20`, and `top_p=0.95`. It saves full final hidden states and compares their cosine similarity and relative L2 distance. The frozen fixtures cover reasoning, tool arguments and prose; each scores M: 16 fixed positions. Their total token lengths are M: 94 / 129 / 74. Layer streaming avoids loading the whole dequantized body onto either GPU.

The probe isolates packed-body effects using BF16 KV. It does not reproduce Marlin, TP rounding, FP8 cache, grammar, long campaign histories, actual multi-step draft trajectories, or the production top-k kernel's tie behavior. Its filtered probabilities are a mathematical reference, not a claim of bit-identical vLLM logits. No sampling implementation is changed. A difference here would support a body-related distribution mismatch; equality on these short fixtures would not exclude a long-context or FP8-cache effect.

### Parent GPU item, not executed

`codex/gpu_ab.sh` uses one frozen interpreter/code snapshot and current checkpoint hashes. Arms `up` and `patch` select g32i8 and codexh. It interleaves fresh probe processes, checks GPU memory and Xid, stops only its own exact child PID, enforces a wall budget, and fails on missing/incomplete results. It starts no server. Default `DRY_RUN=1` never touches GPU utilities or runs a model. The actual probe uses the first reserved GPU for streamed reference execution; it is not a tensor-parallel performance comparison.

Freeze or refresh the local inputs with this CPU-only command:

```sh
CUDA_VISIBLE_DEVICES="" nice -n 19 python3 codex/freeze.py
CUDA_VISIBLE_DEVICES="" nice -n 19 bash codex/gpu_ab.sh
```

Parent-only queue invocation:

```sh
PARENT_GPU_RUN=1 DRY_RUN=0 bash codex/gpu_ab.sh
```

`CONTROL_IDENTICAL=1` runs both labels against g32i8 and fails if the overlap or hidden states disagree beyond its comparison tolerance. `OUT` and `BUDGET_S` are configurable. The script's default wall budget is a literal `1800` seconds, not a runtime prediction. Frozen runtime and model manifests are local generated files under `codex/frozen/`; regenerate them when transferring the topic branch to another checkout.

### Small CPU follow-up

After the parent captures body states, replay only the common MTP/head on CPU:

```sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice -n 19 \
  /home/xanewok/repos/HyperQwen/venv/bin/python codex/teacher_probe.py \
  --model /home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-g32-attn-int8-codexh \
  --tokens codex/frozen/tokens.json --hidden-from "$OUT/0-up.hidden.pt" \
  --device cpu --out codex/replay-up-hidden.json
```

Repeat with the codexh hidden dump. No body forward is needed; only the common MTP module and streamed head rows are evaluated. A token-file digest prevents mixing incompatible prefixes. This can show whether fixed, observed body-state differences change MTP predictions. It does not establish that every difference is harmful: the target prediction changes too. To isolate FP8 scales, the next serving comparison must hold packed weights fixed and vary only the loaded KV scales, or repeat both checkpoints with BF16 KV. That experiment remains parent work; no checkpoint rewrite or unverified recipe change is justified now.

## Verification and CPU measurements

All executed shell work used `CUDA_VISIBLE_DEVICES=""` and `nice -n 19`. Torch was limited to `8` threads; no server or CUDA binary was executed. Required background reports, prep recipe code, launch records and installed model sources were read locally.

- Full checkpoint audit: M: 44.820 seconds wall time, including CPU hashing and reads. This is audit time, not inference performance.
- `codex/selftest.py`: detects a changed MTP payload and a changed config file; M: 0.003228 seconds inside the final test body. The negative control would fail if payload changes were missed.
- `teacher_probe.py --self-test`: passed streamed tiny HF body equivalence across full and linear attention, packed int4/int8 decoding with group scales and row slices, identity overlap and the analytic overlap example. Negative controls check a changed draft distribution and a broken packed sign offset. This ran on CPU only.
- Actual checkpoint schema preflight: all body and MTP reference parameter names resolve to ordinary or packed tensors; no unsupported `weight_g_idx` or zero-point payloads.
- CPU comparator identity and changed-overlap negative control: passed.
- Python compilation and `bash -n codex/gpu_ab.sh`: passed. GPU item dry-run: passed frozen-input verification and printed the interleaved plan without GPU calls.

No llama.cpp source changed, so no CMake build was needed. The real checkpoint reference forward, GPU overlap values and CPU replay of real final hidden states remain unmeasured.

## Repository and open items

The supplied workspace initially contained only `codex/PROMPT.md` plus empty protected `.git`, `.agents`, `.codex` and `.aws` directories. A local shared object store under `.topic-git` was created from `~/repos/llama.cpp`, with topic `topic/codexh-acceptance` based on `68d99c67d`. Other repositories and branches were only read. Use `git --git-dir=.topic-git --work-tree=.` here; the protected `.git` directory cannot hold repository metadata.

Recommendation: do not alter the MTP prep recipe. Treat the large g32i8 r1 gain as contaminated by repetition. Queue the fixed-prefix reference probe, then isolate loaded FP8 KV scales if a residual difference needs a production explanation. The exact residual cause remains unknown; neither codex-heavy calibration harm nor a uniform acceptance regression has been proved.
