# HQ lookup with MTP: source study and offline estimate

## Decision

Do not add lookup to the current serving stack on this evidence. V2 cannot combine it with MTP by configuration today. The evaluated causal gate does not support the requested **+5% decode threshold (D, decision criterion)** on the current `g128`/`g32i8` campaigns. No active vLLM overlay was built; the serving-prototype condition was not met.

For the candidate gate, expected decode throughput change is **+0.2% to +1.3% (D/G)** with copy-aware MTP acceptance, before lookup overhead. A favorable constant-acceptance model gives **+2.8% to +4.0% (D/G)**. These are replay estimates, not GPU measurements or confidence intervals. More sophisticated suffix-cache policies remain untested.

The workspace contained no repository, and `.git` was an empty read-only placeholder. Topic commits use `.topic-git` inside this workspace:

    CUDA_VISIBLE_DEVICES='' nice -n 19 git --git-dir=.topic-git --work-tree=. log --oneline

This is a standalone HQ research repository, not a llama.cpp topic based on `68d99c67d`. All requested background reports were read. No network, GPU query, model server, CUDA executable, or external write was performed.

Labels throughout: **M** = measured here or in existing campaign records; **D** = calculated or inferred from source; **G** = an unmeasured assumption on HQ. Versions, identifiers, policy parameters, and command arguments are identifiers/configuration rather than performance measurements.

## What the installed source supports

Installed vLLM is `0.30.0`, generated commit marker `gced6857af`; HyperQwen HEAD was `2522ef9c059b5ce6706578b608a57b1185ea4c94`. [source_snapshot.json](source_snapshot.json) pins the inspected bytes. Source references below are relative to `/home/xanewok/repos/HyperQwen/venv/lib/python3.12/site-packages/vllm/`.

* `v1/worker/gpu/spec_decode/__init__.py:47` constructs `MTPSpeculator`; `:58` rejects unsupported methods. Neither `ngram`, `ngram_gpu`, nor `suffix` has a V2 construction branch. `config/speculative.py:459` and `:493` expose the standalone settings, not proposer composition. `v1/spec_decode/suffix_decoding.py:5` uses the older `gpu_input_batch.InputBatch`; its Arctic Inference cache adapter is not a V2 adapter.
* HQ `patches/series` includes `dflash2-lookup-drafting.patch`. Its hooks in `v1/worker/gpu/model_runner.py:322` and `:2190` inject request history and an optional proposal length. `v1/worker/gpu/spec_decode/dflash2/speculator.py:18` imports the lookup helpers. `mtp/speculator.py:13` inherits the ordinary autoregressive drafter and has no lookup integration. Enabling `VLLM_DFLASH2_LOOKUP` with MTP therefore does not compose them.
* The existing GPU lookup kernel can supply a proposal from request history. `dflash2/lookup.py:75` finds the longest match, breaks ties by recency, and `:101` limits continuation to already present tokens. It searches every suffix length; this replay's sparse length set must be mirrored if reusing that kernel, or the replay must be rerun with the kernel's exact tie-breaking semantics.
* V2's proposal length hook is batch-wide: `v1/worker/gpu/spec_decode/utils.py:40`. A single-request experiment is straightforward. Mixed batches need per-request lengths or a batch-wide fallback. The current campaigns use a single sequence.

The SGLang throughput comparison in the brief was not independently verified: network access was prohibited. It cannot establish this workload's benefit.

## Wrong-round cost and hybrid rollback

Cheap rollback is supported by source, not measured by this task. `third_party/flash_linear_attention/ops/fused_recurrent.py:106` loads the retained state at `num_accepted_tokens - 1`; `:170` stores recurrent state after each speculative position. `v1/worker/gpu/model_states/mamba_hybrid.py:340` records accepted counts, and `:386` performs aligned-cache postprocessing. `:196` prepares aligned-state migration; recurrent and convolution copy machinery uses the accepted-position offset.

Thus rejecting a suffix chooses an already retained GDN state rather than replaying the rejected token span. **Extra rollback replay rounds: zero (D, source inference)** for this path. Align-boundary state copies and ordinary next-step state handling still exist; their elapsed cost is unknown. This does not establish arbitrary long-block correctness on GPU.

An entirely wrong lookup still emits only **one token (D, rejection-sampling contract)**. For `K=12` versus MTP `k=5`, the charged step is:

    C_lookup = C_MTP5 + 7 * marginal_position_ms - skipped_MTP_ms + lookup_overhead_ms

Using the supplied cost slope, this is **40.7–48.4 ms (D)** for `g128` and **43.0–50.7 ms (D)** for `g32i8`, before overhead and without credit for skipped MTP. The baselines are **30.9 ms and 33.2 ms (M, campaign-derived step means)**. A wrong round loses most of the baseline's approximately **four output tokens (M/D)** even though state restoration is cheap. The slope is a combined serving-step measurement, not an isolated target-verification measurement; its transfer to this design is an assumption.

## Smallest exact design if revisited

Choose the drafter from committed request history, before drawing any MTP proposals:

1. Find the longest suffix among `{32, 24, 16, 12, 8, 6, 4, 3}`, choosing its most recent earlier occurrence with an observed continuation. Choose lookup only if match length is at least `12` and at least `12` source tokens are available. Otherwise use MTP `k=5`. These are policy parameters, not measured run lengths.
2. Keep configured maximum speculative capacity at the lookup length, and separate it from the MTP generation count. Schedule only the ordinary short proposal when lookup is absent. Capture both short and long verification shapes and both MTP catch-up shapes. `v1/worker/gpu/cudagraph_utils.py:292` currently adds the short graph only for DFlash's configuration; `autoregressive/speculator.py:145` initializes draft-prefill capture from the configured speculative count. Raising that count alone would make normal rounds draft too much or miss their graph.
3. The simplest safe prototype runs the ordinary MTP work even on lookup rounds, then substitutes the history-selected proposal and its law. A later optimization may retain the MTP prefill/catch-up routine while skipping its remaining autoregressive draft forwards. **Do not skip the MTP KV catch-up entirely**: `_prefill` populates the drafter's attention cache from the current target batch; later MTP proposals need that history. Long lookup verification must also fit the drafter's catch-up buffers and recurrent scratch capacity.
4. On lookup rows, completely clear the dense draft-logit row and write `0` at the proposed token, `-inf` everywhere else. The ordinary MTP cache is dense (`spec_decode/speculator.py:174`, `:374`, `:418`); DFlash's sparse-candidate clearing kernel is insufficient. The rejection sampler consumes that cache through `rejection_sampler.py:208` and `model_runner.py:1581`. Greedy verification remains target-authoritative.

For history-selected deterministic lookup, `q` is a point mass. Acceptance is `p(d)` and the rejection residual is `p` conditioned away from `d`; their sum is exactly `p`. This identity holds conditionally at every history, so choosing lookup or MTP from history preserves the full autoregressive law. It does **not** promise identical random tokens under the same seed; the brief permits an equivalent law.

Avoid replacing the MTP head based on whether its sampled tokens agree with lookup while retaining the original `q` on the other branch. The selection changes that branch's proposal distribution. The rational CPU counterexample yields **0.45 rather than 0.30 (D, exact enumeration)** for a target probability. Extending an unchanged MTP head with a conditional deterministic tail can be exact, but its agreement frequency and cost cannot be estimated from these records: HQ draft tokens were not logged. This is a design constraint, not a complete audit of HQ's DFlash fusion patch.

## Replay evidence

[replay.py](replay.py) re-tokenizes stored rollouts with the local Swift tokenizer. Its `Lookup` index contains only previously observed tokens. Source continuation never crosses into unobserved output; overlapping matches copy only the available history, without extrapolating a repeating period. `gate` sees only match length and available continuation. Future output is used to score acceptance, never to choose the branch.

Coverage: **15 campaigns, 108 task attempts, 1,683 retained calls, 1,304,401 re-tokenized output tokens (M)**. They represent **1,826,242 usage output tokens (M)**. One malformed, truncated tool-argument response was excluded: **one call and 32,768 usage tokens (M)**. Its omitted text is also absent from later reconstructed history. Replayed classes contain **879,360 reasoning, 72,926 prose, 307,942 argument, and 44,173 scaffold tokens (M)**. The sample has repeated runs of the same task set; it is not independent task-level validation.

The estimator evaluates a causal policy using a renewal recurrence, including completely wrong lookup rounds. Baseline MTP emits a geometric acceptance chain up to each task's recorded `k`. Its parameter is calibrated to that task's measured `accepted_per_draft + 1`, including response boundaries. Two assumptions bracket MTP:

* Constant conditional acceptance at every position (**G**, calibrated to measured task-level throughput).
* Copy-aware acceptance **0.975 (G on HQ)** where lookup correctly predicts the recorded next token, with other-position acceptance recalibrated to preserve the same task-level baseline. The rationale is the llama acceptance report's **97.5% (M on llama)** on copyable positions. This transfers an observation across engines and is a sensitivity, not a measurement of HQ.

The primary estimate credits no gain to missing output, including reasoning truncation. Baseline cost for that output is included in both arms. Reconstructed prompts omit the full rendered tool schema and cut reasoning; token boundaries are re-rendered, not raw server token IDs. Therefore this treatment is conservative in intent, **not a proven lower bound**. Missing text can affect both matches and MTP correlation.

The current deployment cohort is the **four campaigns / 32 attempts (M)** for `g128` and `g32i8`; `codexh` and earlier `k=4` campaigns are kept separately in the raw sweep. [estimate.json](estimate.json) contains every gate/cost/cohort result. [cohorts.json](cohorts.json) groups the main gates. Every percentage below is a throughput change (**D/G**), not time saved; `estimate.json` also records time saved.

| Gate (`n≥12`) | Extra-position slope (M supplied, ms) | Current, constant MTP (D/G) | Current, copy-aware MTP (D/G) | Older campaigns, copy-aware MTP (D/G) |
|---|---:|---:|---:|---:|
| `K=8` | 1.4 | +3.32% | +0.71% | +2.67% |
| `K=8` | 2.5 | +2.68% | +0.11% | +1.71% |
| `K=12` | 1.4 | +3.97% | +1.30% | +3.41% |
| `K=12` | 2.5 | +2.78% | +0.20% | +1.96% |
| `K=16` | 1.4 | +4.11% | +1.45% | +3.60% |
| `K=16` | 2.5 | +2.51% | −0.04% | +1.75% |

For `n≥12, K=12`, current-campaign ranges are **+0.84% to +1.98% (D/G)** at the lower slope and **−0.10% to +0.66% (D/G)** at the higher slope. Approximately **9.46% of selected lookup rounds are entirely wrong (D/G)**. The broad `n≥4, K=8` gate instead estimates **−3.43% to −5.15% (D/G)** across current campaigns at the lower slope under copy-aware MTP.

The favorable pooled sweep reaches **+5.43% (D/G)** by mixing older campaigns with the new ones and assuming constant MTP acceptance. That does not meet the deployment criterion: the current cohort is below it, and the copy-aware model is materially lower.

### Cost sensitivity

[sensitivity.py](sensitivity.py) evaluates `n≥12, K=12` on the current cohort with copy-aware acceptance. [sensitivity.json](sensitivity.json) keeps the results. Lookup overhead and MTP-skip credits in this table are **G**; throughput is **D/G**.

| Extra-position slope (M supplied, ms) | Lookup overhead per round (G, ms) | MTP work saved per lookup round (G, ms) | Expected decode change (D/G) |
|---:|---:|---:|---:|
| 1.4 | 0.1 | 0 | +0.99% |
| 1.4 | 0.1 | 4 | +1.57% |
| 1.4 | 0.1 | 8 | +2.16% |
| 2.5 | 0.1 | 0 | −0.10% |
| 2.5 | 0.1 | 4 | +0.47% |
| 2.5 | 0.1 | 8 | +1.04% |

With no lookup overhead, reaching the decode criterion requires **24.64–32.34 ms saved per selected lookup round (D/G)**. With **0.1 ms overhead (G)**, it requires **26.79–34.49 ms (D/G)**. These follow by linear interpolation of expected time versus skip credit, not by assuming a measured skip cost. Savings approaching an entire serving step are implausible for skipping only MTP work while still verifying a longer block.

## CPU checks and reproduction

Commands run under empty device visibility and low priority; Python used the existing HQ venv. Extraction used **two worker processes (M)**; numeric analysis used a single CPU thread. No llama.cpp build applies to this standalone Python research deliverable.

    CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false nice -n 19 /home/xanewok/repos/HyperQwen/venv/bin/python -B codex/replay.py --extract --workers 2
    CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 nice -n 19 /home/xanewok/repos/HyperQwen/venv/bin/python -B codex/sensitivity.py
    CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 nice -n 19 /home/xanewok/repos/HyperQwen/venv/bin/python -B -m unittest discover -s codex -p 'test_*.py' -v

Extraction caches live only in ignored `codex/cache/`. The extraction command resumes existing caches; add `--refresh` if changing tokenization or lookup semantics. Only passed jobs are included; the aborted campaign is excluded. Analysis from the cached corpus took **5.043 s wall / 4.654 s user CPU (M)**; this is analysis time, not drafter performance. [cpu-time.log](cpu-time.log) records the timing.

**Eight tests passed in 0.630 s (M)**: prefix-only lookup, self-match exclusion, a disabled-gate baseline, the cost of completely wrong rounds, exact rational proposal-law identities, and A/B metric/path handling. [tests.log](tests.log) records them. These tests validate the proposed probability contract and replay, not vLLM kernels.

Negative control:

    CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 HQ_TEST_STALE_Q=1 nice -n 19 /home/xanewok/repos/HyperQwen/venv/bin/python -B -m unittest discover -s codex -p 'test_*.py'

This deliberately keeps stale `q` after changing a proposal. It exited **1 (M)** and failed `test_exact_law_with_history_selected_proposal`; [negative.log](negative.log) records the mismatch. The sampler-law identity holds exactly in rational arithmetic, including zero-support target cases.

## Parent-only A/B scaffold

[gpu_ab.sh](gpu_ab.sh) defaults to a dry plan and launches nothing. Shell syntax and its dry run passed. It is **not queue-ready for an active lookup experiment**, because no active candidate overlay was justified or delivered. It intentionally requires the parent to supply two genuine frozen launch descriptors rather than fabricate a patch arm.

For a future candidate, each descriptor JSON must contain `argv` (absolute Python executable followed by `-m vllm.entrypoints.openai.api_server` and full arguments), `env` (the corresponding frozen overlay `PYTHONPATH` and HQ settings), `python_sha256`, and `sha256` (relative snapshot file paths mapped to their digest). Snapshot directories and their manifests must be inside this worktree. Freeze the installed Python source, baseline HQ overlay and host-staged AR library into both snapshots; replace only candidate files in the patch snapshot. Include all loadable snapshot bytes in the manifest, and use the same immutable dependency environment and model in both arms. Do not point `PYTHONPATH` at the mutable live overlay.

The baseline is ordinary MTP `k=5`; a future patch must implement the separate MTP/lookup lengths and capture both shapes. Preserve the serving checkpoint, host-staged AR, fast top-k, temperature/top-k/top-p/min-p, prefix-match unit, context, and single-sequence configuration. Parent evidence must establish the active gate and lengths from logs before interpreting timing. Freeze a JSON list of nonstreaming `/v1/completions` requests with fixed seeds and the same rendered agent prompts, and pin its SHA-256.

    PARENT_GPU_RUN=1 CUDA_VISIBLE_DEVICES=0,1 codex/gpu_ab.sh --run --up codex/frozen/up/launch.json --patch codex/frozen/patch/launch.json --cases codex/frozen/cases.json --cases-sha256 HASH --out codex/ab-runs/run1

This command is for the parent only and was not run. The script uses `up patch patch up up patch patch up` (**eight boots, planned**) with a warm-up on each boot, a **2,400 s wall budget (configuration)**, checksummed inputs, Xid checks before/after boots, compute-process exclusion, and termination of only the server process group it created from its exact launch PID. It never starts `llama-server`. It confines caches/output to this worktree and disables model downloads. Decode throughput comes from vLLM decode-time/generation-token metric deltas, not request wall time. Individual responses and paired boot results are retained.

Before an active candidate, the useful measurement is V2 per-round telemetry: record proposal source/length, match length/available continuation, accepted prefix, verification time, MTP catch-up time, subsequent draft-forward time, and align-boundary events. That separates actual skip savings from the supplied combined step slope and supplies HQ acceptance on copy spans. Full raw token histories are needed to resolve truncation bias. Until then, the active prototype and GPU A/B remain open; the available evidence supports a marginal opportunity, not the requested adoption threshold.
