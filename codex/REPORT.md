# Swift stack2 early-stop knob sweep

2026-10-04, Europe/Warsaw (CEST). Source: `0dba27b9002b8347c6f14a9f4a1651b3478065ef`. Topic branch: `topic/knob-sweep-pmin`. M = measured or directly observed; D = derived, including source-based inference; G = guess. No engine source changed, no GPU commands or inference binaries ran, and no network was used.

Implementation commits: `5649a72d9` (copied workloads), `d0dd554b0` (planner/harness/tests/matrix), `c8c71338c` (canary trace qualification). This report, generated item and CPU test evidence are a separate final commit.

## Verdict and semantics

`--spec-draft-p-min` is active on probabilistic MTP with the subset head and draft backend sampling (M, source). It compares the **largest probability in the actual draft proposal distribution**, not the sampled token's probability and not the full-vocabulary target top-1 probability. Keep every requested matrix cell; no sampling patch is needed.

Evidence on the supplied source:

- `common/speculative.cpp:1667`: probabilistic mode retains `result_q`. `:1673` retunes the host sampler; `:1728` samples; `:1731` requests sorted candidates; `:1740` selects the sampled token when `result_q` exists.
- `common/speculative.cpp:1743`: `cur_p->data[0].p < params.p_min` stops this sequence before accepting or appending that candidate. Equality continues drafting. The failed-confidence step has already executed the draft decode and sampling; the saving is the remaining draft steps and a shorter verify batch (D).
- `common/speculative.cpp:35`, `:64`: probabilistic retuning installs top-k 10 and the request temperature, with an independent mixed draft seed. `:1464` installs backend **top-k only**. Backend sampling here accelerates candidate extraction; the host still applies temperature and distribution sampling.
- `common/sampling.cpp:137`, `:148`: backend candidate logits become the host candidate array. `:545` installs `dist`; `src/llama-sampler.cpp:1185` normalizes its probabilities. `common/sampling.cpp:1064` sorts candidates by probability while retaining the sampled selection.
- `src/llama-context.cpp:1313`: the full-row fallback fills non-subset IDs with negative infinity. The proposal support is the best 10 entries **within the configured subset**, normalized over those 10 after temperature. This is neither softmax over the entire subset nor softmax over the full vocabulary (D).
- `common/speculative.cpp:1758`: each appended proposal records that exact candidate distribution. `common/sampling.cpp:984`, `:1004`: verification uses its sampled-token `q_x`, accepts with `min(1,p_x/q_x)`, and samples rejection residual `max(p-q,0)`. Target tokens outside the subset retain their target mass in that residual.
- `common/speculative.cpp:1808`: `n_min` clears the **entire** drafted token list when it is shorter than the minimum. It does not force that many steps and does not bypass the confidence test. `:1761` enforces `n_max`; Qwen's single MTP head can be reused for larger autoregressive draft lengths.

Exact-law inference for this matrix (D): `n_min=0` throughout. The top-q stop predicate is constant over the current proposal draw, so a retained draw still has distribution q. At any verified position, the accepted mass is `min(p,q)` and the rejection residual restores `p-min(p,q)`. Stopping before another proposal leaves the existing accepted prefix intact and returns to target sampling. Adaptive proposal length therefore preserves the existing verifier's target law; it does not promise identical tokens across different k/thresholds because RNG consumption changes. This is conditional on the existing sampler/verifier implementation, not a new proof of all backend numerics or grammar paths.

Do not extend that claim to `n_min>0`: post hoc discarding can condition retention of earlier proposals on later sampled prefixes. Do not replace top-q with sampled-token q while recording unconditioned q. The finite-law CPU control demonstrates that conditioned proposals with stale q change the law. The earlier greedy k=3 threshold slowdown in the brief does not establish the sampled larger-k outcome.

## Harness and generator

`gpu_harness.py` copies the graph-reuse harness's frozen validation, owned-PID lifecycle, HTTP/error recording and SSE assembly. It drops profiler modes and adds a matrix command hook. The source and copied-input provenance are in `inputs/PROVENANCE.json`; the original files were not edited.

`sweep.py plan codex/matrix.toml` emits `codex/items/knob-sweep-pmin.toml` and its cost plan. Arm flags are long option/value pairs; harness ownership, context and `n_min` cannot be overridden. A future measured `arms.cost.per_boot_s_M` replaces that arm's transferred cost prior. This is a llama F1 planner, not a scheduler or a general optimizer.

- `sweep.py:47`: first arm order followed by its reverse, reducing to ABBA for two arms. The first sweep has 7 arms and 14 full boots (D), each arm appearing twice, with the baseline at both ends.
- `sweep.py:152`, `:186`: existing `LLAMA_TRACE=1` acceptance lines map cumulative drafted counts to verification ordinals. Trace drafted/accepted/round totals must exactly match `/metrics`. Only adjacent mapped rounds enter steady timing, after the first 3 rounds and before the last k+1 output tokens (D). Partial SSE rows are skipped; gaps are excluded rather than inferred by dividing drafted tokens by k.
- `sweep.py:338`: per-request server-log capture waits briefly for the asynchronous logger's release line. Canary also requires at least one long request with nonempty verifications, so an entirely zero-proposal canary cannot qualify trace parsing. Checkpoint restore/replay accounting follows `server-context.cpp:4080`, `:4107`, `:4127`. Zero-proposal work between nonempty verifications remains in their elapsed cycle cost. All-zero arms use adjacent one-token SSE clocks.
- `sweep.py:362`, `:381`: tool and thinking requests at actual approximately 8k and 90k prompt depth; the deep archive is tokenizer-calibrated in the system message. The first deep request is cold; the second must retain at least 79k cached tokens (D, gate). Full boots add 8 sequential coding-agent follow-ups; canary adds 2 (D).
- `sweep.py:267`: every request is recorded before boot validation. Each long request needs at least 40 steady intervals and 2048 output tokens (D, gates); cold/cache/depth/counter/tool gates apply in canary and full runs. Follow-ups have 768-token headroom (D) and incomplete tool calls are rejected before replay.
- `sweep.py:323`: repeats of the same arm must have identical completion content, finish reasons, counters and prompt/cache/output token counts after normalizing random tool IDs. Different arms are not required to match sampled completions.
- `sweep.py:429`: request rows, per-arm means/ranges and paired baseline deltas retain acceptance, tokens/round, steady ms/round, authoritative emitted tokens/s, wall rate and fixed-cost estimates. Negative fixed-cost estimates are retained. Whole-generation tokens/round includes zero-proposal emissions; the selected-interval accepted+1 measure is named `verified_tokens_per_round_D`.
- `gpu_harness.py:90`, `:130`, `:173`: real memory-query format, word-bounded journal hazard checks, memory-based idle waiting, reusable port binding, exact PID plus start-time stop. Each boot copies verified libraries and the executable into its own directory and runs `sweep-srv`, never a process named `llama-server`.

Long file requests intentionally stream more content than the completion cap. Their truncated `write_file` content is measured but never replayed; complete follow-up arguments are required. That matches the long workload inherited from the A/B harness and avoids feeding cut tool calls back into the server.

## CPU evidence

Commands run with `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`, and no more than 8 threads:

```
export CUDA_VISIBLE_DEVICES=''
nice -n 19 env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s codex -p test_sweep.py -v
nice -n 19 env PYTHONDONTWRITEBYTECODE=1 python3 codex/sweep.py plan codex/matrix.toml
nice -n 19 bash -n codex/gpu_ab.sh
nice -n 19 env PYTHONDONTWRITEBYTECODE=1 /home/xanewok/repos/gpuq/bin/gpuq submit --check codex/items/knob-sweep-pmin.toml
```

Results: 25 tests pass (M), CPU suite wall time 2.234 s (M); shell syntax and gpuq manifest check pass (M). `test-results.txt` records the test output. No queue submission occurred. No CMake build or CPU-model inference was needed because the engine is unchanged.

Real artifacts: `orchestration/gpuq/fixtures/llama-sse/`, `xid-lines.txt`, `nvidia-smi-query.csv`, and graph-reuse's named `20261003-173851/1-up` captures. CPU parser replay of those previously recorded GPU results yields 337 tool and 341 thinking steady intervals (M), with 35.7391 and 35.6084 ms/round respectively (D). These are **fixture regression numbers**, not speed measurements from this worktree. The deliberately short real fixture yields 23 intervals (M) and fails the 40-interval gate (M).

Negative controls cover absent/null draft rows, reversed timing, counters that disagree, a missing interior round, insufficient rounds, cut follow-up arguments, bad server configuration/fallback, actual hazard log lines, changed frozen bytes, invalid GPU-memory rows, same-arm output divergence, excessive stage cost and execution without parent authorization. The old graph-reuse `steady()` function is extracted directly from the original source: it rejects the variable-length synthetic trace while this parser accepts it (M). A finite-law control fails when proposals are conditioned on sampled-token q without updating recorded q (M).

The trace parser is tested against the **real uninstrumented** server log and correctly rejects missing round coverage (M); its positive variable-length and checkpoint cases are synthetic. There is no recorded positive `LLAMA_TRACE=1` GPU artifact available in the searched task locations. Thus positive real trace qualification, deep-prefix retention, and actual model length gates remain for the mandatory canary. This limitation is explicit, not a claim that real trace parsing has already passed.

## Item, cost and decisions

Item: `codex/items/knob-sweep-pmin.toml`. Requested destination: `SP/overnight/items/knob-sweep-pmin.toml`. It was staged locally because the hard rule permits writes only inside this worktree; the parent must copy that manifest. Its script/pin paths already point here.

Arms: baseline k5/p0; k8/p0.5, p0.7, p0.8; k12/p0.7, p0.8; k5/p0.7. S5 flags use Swift IQ4_XS, np1, tensor split, q8_0 target KV, f16 draft KV, subset vocabulary, probabilistic proposals and draft backend top-k. Canary uses k12/p0.8 and one boot (D).

Cost: total reserved GPU budget 8100 s = 135 min (D), including canary ceiling 600 s (D). Full stage estimate 5237 s (D), full ceiling 7500 s (D); canary estimate 412 s (D). Both estimates fit the contract's 70% headroom (D). Inputs: boot 90 s (G), shallow request 22.95 s and generation 12.278937 s (M, recorded up capture), deep cold 122.45 s (D, transferred shallow prefill rate), follow-up allowance 3 s (G; recorded mean 1.0061 s, M). Larger k gets a pessimistic linear generation allowance (G). This is a cost estimate, not a GPU speed prediction.

PASS means measurements are valid and all same-arm repeat checks passed; it does not mean an arm is faster. Prefer an arm only if emitted tokens/s improves consistently at both depths beyond the observed paired boot spread, with no follow-up wall/fixed-cost regression. A lower ms/round with lower tokens/round or emitted rate is not a win. A depth-specific win implies a depth-specific setting, pending workload-weighted replay. If thresholds eliminate most proposals or worsen emitted rate, retain k5/p0. An INVALID trace/cache/length result means fix or replace the workload and rerun a fresh canary, not rank the resulting numbers. Hazard stops the item; budget expiry yields `budget`.

Open risks: trace logging overhead is unmeasured (G); repeated synthetic archive text is not a representative agent history; the second deep request assumes hybrid prefix retention; larger-k draft scratch/graph fit must be established on the real GPUs. This sweep ranks F1/F2 synthetic behavior and does not measure task solve rate, KLD, or production speed. No sampling change, upstream package, PR, push, or external posting was made.
