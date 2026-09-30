# Device draft continuation: HyperQwen and llama.cpp

## Conclusion

Graph reuse did help llama.cpp, but only within its host-build budget. Patch A saved D0.45 ms/round for think and D0.18 for tool against its own cache-disabled control. It did not replace meta's fragmented CUDA submission, the internal AllReduce host waits, input uploads, catch-up, or host rejection. Calling the result "nothing" hides a small measured effect and overstates what that patch could remove.

The device chains moved dependencies into a ggml graph without removing the rest of the round's control path. The original chain still crossed the CPU embedding backend; its repaired successor retained the full draft head and communication submission. The subset chain removed intermediate hidden-state transfers but added a device sampling graph, host shadow sampling, RNG snapshots and mismatch replay, while rebuilding alternating catch-up/chain shapes. Those are observed mechanisms; their individual shares of the GPU slowdown are unknown without a chain trace.

HQ's decisive contract is device-resident proposal state and continuation, including rejection results, rather than merely "a graph containing the draft loop." The supplied anatomy traces show a separate replay for each draft forward. Current installed HQ source additionally supports capturing the remaining decode loop when attention metadata can update on-device; its use in the patched k=5 measurement is not established by those traces.

Recommendation: obtain a post-subset, batched-upload stack trace first, then implement a trusted proposal-state contract and safe graph-plan ownership. Capture the draft communication path only after it can replay without host slot acquisition. Do not remove subset-chain host validation as a performance toggle: that changes its correctness contract.

No engine prototype was implemented. CPU-only work cannot validate the required CUDA communication/capture changes, and another CPU-passing unrolled graph would reproduce the earlier evidence gap. This branch contains a design, an independent rejection-law oracle, evidence extraction, and a parent-only A/B harness; it supplies no faster engine binary.

## Evidence and scope

M = existing measurement, observed source/log fact, or a result of this CPU audit. D = arithmetic or inference. G = forecast or unvalidated assumption. Source line numbers, commit hashes, model names, k, seed, flags and logger levels are identifiers/configuration, not performance estimates. Every duration, size, count and performance ratio below carries its label. Source behavior is distinguished from trace-time behavior.

Read before design: `/home/xanewok/repos/draft-anatomy/REPORT.md`, `SP/gpu-sampling/REPORT.md`, `SP/gpu-sampling/items/96b-attrib.md`, `SP/patchA/REPORT.md`, `SP/codex-reviews/patchA.txt`, and `C/codex/REPORT.md`. `codex/evidence.json` records paths and SHA256 hashes of the inspected live source and background evidence. It also independently extracts the cache and chain deltas from the original logs. No external source or network was used.

Path abbreviations in the source tables:

| Root | Exact location / provenance |
| --- | --- |
| B | This checkout, fork `68d99c67d958c28b6bff5fdc2db096f81c324ef9` |
| S | `/home/xanewok/repos/qwen38-weekend/llama.cpp-stack2`, `0dba27b9002b8347c6f14a9f4a1651b3478065ef` |
| C | `/home/xanewok/repos/llama.cpp-draft-chain`; implementation `0e29601422f35572757734544cfdad673df3f621`, artifact HEAD `937f72660` |
| A | `/home/xanewok/repos/llama.cpp-patchA`; review and experiment refer to `20fa77943` |
| V | `/home/xanewok/repos/HyperQwen/venv/lib/python3.12/site-packages/vllm`; live installed source, hash-pinned by inventory |
| HQ | `/home/xanewok/repos/HyperQwen`; local patches and serve logs, read-only |
| SP | `/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad` |

The local repository is a shared-object clone inside `left-out/llama.cpp`, on `topic/draft-loop-anatomy`, directly based on B. Its refs/index/config are local; the other repositories remain read-only. The root `REPORT.md` points here. No engine source changed.

### Comparable timing budget

The anatomy report's local idle-HQ median is M30.041 ms/round; its old full-head llama B pooled median is M34.522. Only M15 steady B rounds are available, versus M197 HQ periods. This is a cross-engine trace comparison, not an engine A/B. The full-head delta is D3.195 ms over k=3 draft positions; the brief's subset saving is M2.4-3.8 ms/round. Those describe substantially the same removed head work and cannot be added as separate opportunities.

After applying that saving to the old period, the residual would be D0.681-2.081 ms against idle HQ, assuming transfer to this workload. That assumption is unverified. No supplied post-subset stack trace or patched-HQ k=5 trace supports a tighter attribution. The brief-supplied M29.5 ms/step and M1.4-1.6 ms per extra draft for patched HQ are not isolated draft-forward measurements: verify batch size and acceptance also change with k.

| Old B-tool boundary, from `96b-attrib.md` | Total ms M | Host CPU ms M | Inputs ms M | Readback ms M | What it bounds |
| --- | ---: | ---: | ---: | ---: | --- |
| V -> C | 1.09 | 0.59 | 0.30 | 0.18 | Spec process, hidden rows, C preparation/build/plan, upload |
| C -> D1 | 1.22 | 0.81 | 0.30 | 0.09 | Exposed target sampling/bookkeeping, D1 build/plan, upload |
| D1 -> D2 | 0.24 | 0.06 | 0.15 | 0.01 | Remaining host draw, next inputs and synchronization |
| D2 -> D3 | 0.23 | 0.06 | 0.16 | 0.01 | Same dependency |
| D3 -> V | 0.49 | 0.17 | 0.29 | 0.01 | Final q/draft handoff, target batch and inputs |
| Sum | 3.27 D | 1.69 D | 1.20 D | 0.30 D | Launch tails and rounding account for the remainder |

These are mutually delimited host-gap segments. Do not additionally add the build, sampling or synchronization subtotals below to this sum. The earlier Patch A input subtotal D0.93 ms uses a different delimitation; it is not interchangeable with this table's D1.20 ms. The whole gap budget is not the budget of chaining only D1-D3: its internal boundaries sum to D0.47 ms in this older configuration, before batched uploads.

`patchA/REPORT.md` section 1 narrows the C/D1 build/alloc/replan opportunity to approximately D0.76 ms/round: C D0.37 plus D1 D0.39. It also attributes approximately M/D0.46 ms of target rejection beyond C's GPU work and M0.10-0.18 ms of accept/server processing. Removing C can expose target CPU work it currently overlaps; C's compute plus its whole gap is not a valid saving forecast.

## Mechanism comparison

The MTP path is `V/v1/worker/gpu/spec_decode/mtp/speculator.py:12`, which inherits `AutoRegressiveSpeculator`. The older `v1/spec_decode/llm_base_proposer.py` is not the anatomy report's active runner. The following costs belong to old B or the explicitly named background experiment; the present S stack already contains the subset, backend target candidates and batched input uploads.

| Mechanism | HQ/vLLM implementation | llama.cpp implementation | Cost / implication |
| --- | --- | --- | --- |
| Forward/head scope | `V/model_executor/models/qwen3_5_mtp.py:104`, `:181`, `:331`: device embeddings, MTP forward, compact head expanded to real IDs with `-inf` outside its support | `S/src/models/qwen35.cpp:519`, `:640`: host-backed input embedding lookup, MTP body, compact output head in S; B used full head | Old head difference D3.195 ms/round. Already addressed by subset. HQ draft body weight kernels M0.529-0.537 ms/step versus llama M0.336-0.337; no evidence of a faster HQ draft body |
| Persistent input/output ownership | `V/v1/worker/gpu/spec_decode/autoregressive/speculator.py:33`, `:282`, `:681`: persistent hidden/input/token buffers, device-to-device continuation | `S/common/speculative.cpp:1581`, `:1637`, `:1729`, `:1786`: fetch target/draft h on host, retain host `pending_h`, reconstruct next batch | Old per-step readback M0.01 ms with backend candidates; broader V -> C readback M0.18. Hidden transfer bytes and synchronization cannot be inferred to vanish just because tokens are device-selected |
| Proposal draw and q | `V/v1/worker/gpu/spec_decode/speculator.py:400`; `V/v1/worker/gpu/sample/gumbel.py:159`: Gumbel draw, cache pre-temperature logits in device proposal storage | `S/common/speculative.cpp:1728`, `:1758`; `S/common/sampling.cpp:789`: backend top-k followed by host temperature/dist, synchronous sample, host q vector | Old backend top-k GPU subtotal M0.026-0.0275 ms/step versus HQ Gumbel M0.0057-0.0058. Different proposal laws. Remaining host CPU M0.06 per internal D boundary includes more than sampling; no isolated cost for its draw |
| RNG contract | `V/v1/worker/gpu/sample/gumbel.py:16`, `:118`, `:177`: request seed, sample position, token-ID noise key, separate draft salt | `S/common/sampling.cpp:126`, `:943`; `C/src/llama-sampler.cpp:1446`: stateful mt19937 proposal/verify streams; peek double draws and convert to F32 for chain input | Independence and q consistency are correctness conditions. Same seed does not equate HQ's keyed Gumbel law to llama's host CDF trajectory |
| Next token, hidden state and positions | `V/v1/worker/gpu/spec_decode/autoregressive/speculator.py:903`: Triton stores selected token, hidden feedback, positions/lengths; final step skips unused continuation | `S/common/speculative.cpp:1719`, `:1785`; `S/src/llama-graph.cpp:94`: host batch rebuild, token/h/position input setters | Old D1/D2 inputs M0.15-0.16 ms/boundary, approximately M14 small uploads across devices. S batching is `ggml/src/ggml-backend.cpp:1721` and CUDA `:802`; it reduces this cost but still completes synchronously |
| KV slot/mask updates | `V/v1/worker/gpu/block_table.py:191`: slot-mapping kernel reads device positions/block tables; autoregressive `:638` updates slots and attention metadata inside fused capture | `S/src/llama-context.cpp:1513`: host memory context apply; `C/src/models/qwen35.cpp:675`: chain gets statically prepared per-step KV views and mask rows | Included in preparation/upload/build windows; no isolated measured mask/KV saving. A device token without device KV metadata is not autonomous continuation |
| Catch-up and first draft | Autoregressive `:240`, `:261`, `:460`: pad to target shape; device rejected count selects true query/last row; prefill graph computes catch-up plus first draw | `S/common/speculative.cpp:1571`, `:1610`, `:1681`: separate C on target rows, then D1 on corrected/bonus token and selected pending h; `:1824` selects h after accept | C GPU busy M0.42-0.44 ms in old B. V -> C and C -> D1 gaps above are much larger, but include other work. HQ's first graph does extra/padded work; "skip C" does not reproduce its trajectory |
| Graph reuse | Autoregressive `:157`, `:313`, `:545`: capture full forward/head/draw and replay compatible batch descriptor; persistent addresses | `B/src/llama-context.cpp:1462`, `:2479`; `S/src/llama-context.cpp:1527`, `:2564`: two output/non-output arenas but reuse only the active result/scheduler; C/D1 alternate and rebuild | Old opportunity about D0.76 ms/round, not all D3.27 ms gaps. CUDA capture already exists at `S/ggml/src/ggml-cuda/ggml-cuda.cu:4484`, `:4560` |
| Whole loop capture in current source | Autoregressive `:101`, `:184`, `:591`, `:623`: eligible attention backends capture steps after first draw in one graph; `V/v1/attention/backends/flash_attn.py:690`, `:1022` expose on-device metadata-update capability | `C/src/models/qwen35.cpp:661`: unroll all steps into one ggml cgraph; CUDA backend still captures only meta-dispatched subgraphs | Anatomy traces M1 draft graph replay per forward. Current fused path is source capability, not measured evidence of its use. It still leaves a separate first/catch-up graph |
| Tensor communication/submission | Trace evidence: HQ draft forwards include NCCL inside the per-forward graph; target M1 replay/GPU | `S/ggml/src/ggml-backend-meta.cpp:2061`, `:2526`, `:2549`; `S/ggml/src/ggml-cuda/allreduce.cu:364`: meta replans by one UID, submits per-device graphs around communication, acquires ring slot with host event wait | Old B M129 target and M3/4/5/5 C/D1/D2/D3 CUDA graph launches/GPU. M540 shared-host sync APIs/round; their M24.46 ms wait includes computing time, not removable idle. The ring prevents host runahead |
| Verify and rejection | `V/v1/worker/gpu/model_runner.py:1577`; rejection sampler `:180`, `:189`; utils `:661`, `:839`: target filters, ratio test and residual sampling on GPU, return accepted/rejected counts as tensors | `S/common/sampling.cpp:929`, `:936`, `:994`, `:1004`; server `:4063`: host target filters/grammar, compare against host q, residual draw, token acceptance and KV rollback | Old exposed target rejection M/D0.46 ms; HQ GPU target sampling M1.891 ms/GPU. GPU rejection can trade busy time for hidden host time; this is not a guaranteed net gain |
| Host overlap | `V/v1/worker/gpu/model_runner.py:2099`, `:2133`, `:2161`: launch output copy before device proposal; `async_utils.py:137`, `:168`: wait on output event only at get_output; `V/v1/engine/core.py:692` queues execution before waiting for previous result | `S/tools/server/server-context.cpp:3881`, `:4063`, `:3154`: process hook submits C, host verifier/process_token follows, next draft depends on host results; internal AR waits limit queued work | HQ round idle D0.521-0.528 ms/GPU versus B D3.695-3.739. Async scheduling can hide prep once dependencies live on device; it does not permit ignoring grammar, EOS or cancellation dependencies |

In the old anatomy trace, the GPU union-busy difference is D1.278 ms on GPU0, and the idle difference D3.218 ms. These explain approximately the D4.481 ms period gap; median columns need not sum exactly. Individual weight/norm/AR kernel-sum differences are not additive critical-path estimates. HQ verify norms/projections, its slower GDN/attention/communications, and its GPU target sampler all differ. The comparison does not isolate graph capture as their cause.

## Failure audit

### Original device chain and the embedding-copy repair

Observed: the problem described as "chain v1 (8c1828a85)" is the pre-repair behavior documented by that commit. `git show 8c1828a85` identifies the commit itself as the embedding-copy repair. Its message reports M3 meta splits for k=3, device token readback to CPU `get_rows`, embedding upload, and approximately M0.5 ms per meta rebuild on the GPU host. The input weight placement, not the token's device residence, dictates where lookup runs.

Added work: device selection nodes followed by cross-backend token/embedding transfers, repeated meta setup and copy synchronization. Not removed: separate C/verify host work, alternating graph builds, input preparation and communication submission. GPU work not shrunk: the full head and each autoregressive MTP block. Inference, high confidence: this was a misplaced device dependency; a ggml graph crossing CPU `get_rows` could not eliminate the host boundary it targeted.

### Repaired chain2b (`ebfc900d5`)

Observed: this commit copies a full quantized token table to each output device and checks architecture, capability and memory headroom. Its `src/models/qwen35.cpp:664` unrolls steps, `:692` still multiplies the full head; `common/speculative.cpp:1845` reads all chain output at once, `:1864` records the returned device q directly. It does not contain the subset chain's host shadow draw. Do not apply the later validation explanation retrospectively.

Added work: device draw/reduction and gather kernels, a larger graph and mirrored table; its commit message reports M0.26 s load-time copy and D644 MiB/GPU. The earlier repair message's D675 MiB estimate conflicts with this; neither is a new allocation measurement. The subset-chain metadata calculation is a more precise source for current table sizing.

Not removed: C and alternating plan work outside the unrolled region; host q publication and target rejection; meta per-communication submission and ring waits. Its own commit message records measured meta rebuild ratios M0.74 -> M0.62 per compute for chain, and M0.69 -> M0.37 with fused catch-up: not elimination. GPU work not shrunk: full draft head, MTP bodies, attention and AR; serially dependent steps cannot batch their weight projections into one independent GEMM.

The brief reports M1.01-1.06x, not statistically significant. The commit message describes different arm combinations and modest throughput ratios; these are not clean chain-only measurements. Inference, high confidence on structural limitation and low confidence on timing allocation: embedding repair removes a known regression but leaves only a small host boundary budget, while repeated full-head GPU work and plan costs remain. No evidence establishes that graph-cache misses alone explain the nonsignificance.

### Subset chain (`034e51d38` through `0e2960142`)

Observed: `C/src/llama-context.cpp:1305` supplies compact embeddings; `C/src/models/qwen35.cpp:665`, `:668` keeps later embeddings/h on device. Its chain samples global top-k candidates in `C/src/llama-graph.cpp:2534`, gathers logits/real IDs/compact indices, sorts, computes softmax/CDF, picks and packs rows (`:2559`). The table's D67.13 MiB/GPU and D124 bytes per returned step are metadata-derived in `C/codex/REPORT.md`, not peak resident memory measurements.

Added work: device softmax/CDF/index/gather/packing plus host sampler cloning (`C/common/speculative.cpp:1674`), RNG peek, dummy batch rows, and another host temperature/dist calculation for each result row (`:1702`). The recorded q is the host sampler's q, not the device softmax vector. A mismatch restores RNG and KV then reruns the original path (`:1719`). Correctness requires this validation in that implementation.

Not removed: C, host target sampling and server handoff, alternating C/chain graph/meta builds, internal AR waits, initial input upload and final round output consumption. Source correction to the brief: chain output D2H is batched once in `C/src/llama-context.cpp:2261`, and `get_mtp_chain` synchronizes once at `:1359`; the host walks rows afterward. There is no demonstrated per-step D2H within the chain. Duplicate host validation is real, but it cannot be charged as repeated GPU readbacks.

GPU work not shrunk: each compact head and MTP block still runs k times; top-k/gather/CDF/embedding nodes add work. A large ggml chain is still decomposed by meta at each communication boundary. Plan/property churn is plausible from the cache keys, but its timing, fallback frequency and kernel count are unmeasured in this A/B.

The independently parsed `SP/codex/chain-ab.log` gives these same-k changes. Units are ms/round D, computed from measured generation time and actual measured round counters, mean over M2 boots/arm:

| k | Tool chain - plain D | Think chain - plain D |
| --- | ---: | ---: |
| 3 | +1.140273 | +1.606604 |
| 4 | +2.027708 | +2.148625 |
| 5 | +1.888100 | +2.132500 |

Tool/think completion and counter identity passed in the log. The long k=3 summary failed `completion identity failed`, then aborted; no clean long-workload performance conclusion follows. The brief's M/D1.6-2.1 ms slowdown summarizes think, not all cells. CPU identity in C does not establish universal GPU identity.

Inference, high confidence: the subset chain trades at most the old internal D0.47 ms k=3 boundary budget, already reduced by batching, for extra device work and retains its host shadow draw. That is a poor performance trade before addressing ownership and communication. Unknown: how much of the observed slowdown comes from graph/meta rebuilds, GPU sampling/communication, or mismatch replay. A trace plus counters is required to separate them; assigning the entire slowdown to host validation would exceed the evidence.

### Patch A graph and meta-plan reuse

Observed: `B/src/llama-context.cpp:1462` only reuses the active graph; `:2479` chooses arenas by presence of outputs. C and D1 alternate. Patch A preserves multiple graph/scheduler results and UID-keyed meta plans to remove that alternation's builds. It is a host plan cache, not a backend-wide capture of verify, C and drafting.

Added work: cache lookup, validity/epoch tracking, shared scheduler families, pinned containers and rematerialization. Not removed: sampling, uploads, C execution, server work, AllReduce host waits or meta's per-boundary CUDA replay. GPU work not shrunk: every model op and communication payload. The old rebuild opportunity is approximately D0.76 ms, so a sub-millisecond effect is the expected class of result.

`99-patchA-graph-cache-ab.log` records cache-disabled base versus patch means: think M/D35.67 -> M/D35.22 ms/round, tool M/D36.88 -> M/D36.70, giving D-0.45 and D-0.18. Upstream's first boot was slower than its second, so using upstream's mean to claim the cache benefit confounds boot effects. This run used host draft sampling (`--no-spec-draft-backend-sampling` in the queue script); the sampler-binding defect is not an explanation for these particular small gains.

The review at `SP/codex-reviews/patchA.txt:10458` finds graph selection failing to restore sampler tensor bindings/RNG arity (`A/src/llama-context.cpp:1469`, `A/src/llama-graph.cpp:3855`, `A/src/llama-sampler.cpp:1359`), unpinned input-container slices, unchecked allocation epochs in public compute, shared-container races and rematerialization into pinned arenas. These are merge blockers, especially with backend draft sampling. Inference: CPU cache identity tests without installed samplers do not test the intended sampling workload. The observed saving below the forecast could reflect different host sampling arm, overlap and profiling/boot noise; no supplied patched trace apportions the shortfall.

## Design that removes the dependencies

### 1. Make proposal state authoritative

Introduce an internal proposal result owned by the draft context/backend: selected real IDs, candidate support, actual normalized q (or losslessly stored logits plus canonical normalization), per-step h/KV state, and RNG transaction state. Keep it device-resident until verification or a requested CPU fallback. Use the sampler/backend interfaces rather than a Qwen-specific float row protocol as the long-term ownership boundary. Compact IDs must never reach verifier/token APIs as real IDs; use typed integer IDs for control, not arithmetic AllReduce on IDs encoded in F32.

The sampling operator must both draw and record q from the same candidate order, temperature arithmetic, tie policy and normalization. A trusted implementation replaces shadow host sampling; its contract and offline conformance tests justify trust. Merely supplying matching mt19937 draws does not: `C/src/llama-sampler.cpp:1462` casts double draws to F32, and device softmax/CDF arithmetic differs from host sampling. Cache input bindings and RNG cardinality must belong to the selected graph, not the last graph built.

Prefer a reference-preserving implementation first: the CPU and device path share a documented candidate ordering/CDF construction and explicit uniform precision/endpoint behavior. Retain distinct draft and target/acceptance RNG streams. Preserve how many draws the legacy path consumes, including a p_min-failed attempt and request truncation; pre-generated unused suffix draws must not silently advance the committed RNG. Backend capability gates must fall back before mutating token, sampler or KV state. Do not promise GPU same-seed identity until kernel rounding, top-k ties and draw consumption have passed the engine tests below.

A new proposal law is mathematically permissible if the verifier uses that actual law: for any real token x, accepted mass is `q(x) min(1,p(x)/q(x)) = min(p(x),q(x))`; the rejected residual adds `max(p(x)-q(x),0)`, yielding p. This argument applies conditionally on each emitted history and requires independent acceptance/residual randomness. It does not require the proposal to equal the target or HQ's proposal. It is a real-arithmetic theorem; finite RNG grids and rounded CDF probabilities need their own defined contract. A general GPU Gumbel implementation is not automatically a proof of exact equivalence to the current host sampler.

If choosing a finite integer-mass proposal instead, define q from its exact draw interval masses, not rounded softmax weights, and use compatible exact Bernoulli/residual rules. That could prove a law on the finite sample space, but it is a separate sampling change with a substantial implementation/proof burden. This report recommends no such algorithm switch without those checks.

### 2. Keep the recurrent draft inputs and plans resident

Keep compact quantized embedding rows, hidden feedback, token IDs, positions, causal lengths, KV slot indices and masks in backend-owned persistent buffers. Resolve an initial corrected/bonus token outside the subset separately; copying only compact rows cannot embed it. Later picks are inside the subset and can use the existing compact table. The model's h feedback/final-norm convention and shifted token/h pairing must exactly match the current graph.

Pre-reserve and retain the plans for C/first-draft and continuation shapes; include active sampler graph bindings, outputs, adapter/model identity and memory/allocation generation in validity. Validate the allocation epoch at compute, not only cache lookup. Pin every referenced metadata owner, including leaf/source containers, under a clear cross-context lifetime/threading contract. Prefer backend-owned plan storage or an explicit narrow reusable-plan API over Patch A's global unverified container registry. Keep unsupported concurrent slots/adapters/backends on the existing driver.

Start with per-step CUDA replay and device successor inputs. The anatomy traces prove that this already permits low intra-draft idle in HQ; whole-loop capture is a later submission optimization. This separates proposal correctness from capture and makes failures attributable. A chain requiring host q validation cannot be the steady fast path.

### 3. Make internal AllReduce capturable before enclosing k steps

The non-NCCL meta route calls communication on the host between CUDA graphs (`S/ggml/src/ggml-backend-meta.cpp:2526`). The AR ring chooses slots/tokens with host `call_count`, then waits on both GPUs before reuse (`allreduce.cu:364`). CUDA capture in `ggml-cuda.cu:4560` encloses each simple-backend subgraph; it does not enclose the meta loop or communication callback. This is why one ggml graph is not one launch per GPU.

For a fixed draft plan, preassign persistent scratch/event/arrival storage per communication boundary and encode peer/copy/reduction dependencies on streams or graph nodes. Avoid slot reuse until all consumers finish. Arrival epochs/reset and replay generation must be device-safe: baking a host incrementing token into capture can make later replays see stale arrivals or deadlock. Capture paired per-device graphs with persistent addresses and explicit cross-device completion ordering, preserving the present AR rounding and operation order. Capture/replay must not call host event synchronization or allocate scratch in steady state. The larger copy-engine path and unsupported device/backend combinations need a normal fallback.

Increasing the ring size alone only postpones the wait and does not establish this contract. A permanently growing ring is unacceptable; size a finite plan and establish reuse after replay completion. This is a generic backend communication change and should be reviewed/tested separately from the proposal API.

Once those dependencies are capturable, capture the supported fixed-k draft segment, including draw, feedback, embedding, KV metadata and AR. A separate first/catch-up graph is acceptable; current HQ itself captures it separately. Capturing all k without C/first-draft ownership or communication changes repeats the previous chain experiment.

### 4. Fuse catch-up correctly, then move verification behind a capability gate

After V, device verification can return accepted-prefix length, corrected/bonus token and selected target h without a host read. Catch-up/first-draft uses these outputs to repair draft KV and construct shifted h/token rows. Pad/mask unused verification rows like HQ only if tests establish that they cannot enter future attention. Do not drop C, substitute an h row from the rejected suffix, or leave recurrent GDN state at an unaccepted position. C uses the draft's own attention KV; target rollback also includes hybrid recurrent state.

For device target rejection, apply the exact target filters and per-prefix sampler state, then consume the authoritative proposal q for ratio/residual computation. Sparse draft q is sufficient even when p's support is broader: absent draft IDs have q=0, so they retain all of p in the residual. Target top-k must not accidentally be reinterpreted as draft top-k. Grammar masking/renormalization, forcing budgets, penalties, EOS/stops and request limits retain current semantics.

Initially restrict the fully resident verify path to sampler chains with supported state/filters. Active grammar and forcing cases can wait at a round-level fallback and use the current verifier; correctness is preferable to an implicit approximate mask. Require a test where grammar actually rejects a draw and a budget forces, not only lazy grammar passthrough. HQ's model runner explicitly handles grammar input and can defer scheduling for pending structured-output tokens; its async implementation does not remove those dependencies.

When supported, enqueue device verification, request/KV state update and next draft before waiting for the completed token batch's asynchronous copy. Deliver accepted tokens to the server and prepare the next host bookkeeping while drafting runs. Use separate copy/event ownership so outputs and inputs cannot be overwritten by a subsequent replay. Streaming cancellation/EOS may waste already queued work; discard it without committing RNG/KV state beyond the accepted request boundary. Pipeline/multiple-context cases need explicit ownership rather than global cache assumptions.

### Risks, gates and realistic budget

The draft-only internal-boundary ceiling is the old D0.47 ms at k=3; S has already batched uploads. A safe cache addresses a different approximately D0.76 ms old build/plan budget. Full verify-to-next-draft residency addresses more of the old D3.27 ms gaps but can add target GPU work, change overlaps and invalidate simple addition. These are stale ceilings, not forecasts for current S. HQ's much lower idle demonstrates feasibility in its architecture, not guaranteed llama.cpp speed.

| Gate | Evidence required / negative control |
| --- | --- |
| Proposal law and same-seed path | Existing Qwen3.5-4B MTP model from `SP/patchA/run_cpu.sh`; compare selected IDs, real-ID support, q/logit bytes, RNG continuation and teacher-forced h for k=3/4/5, ties, temperatures, p_min and truncation. Wrong q, wrong compact-ID map and changed CDF boundary must fail; no inference from completion text alone |
| Correct prefix/KV | Cover every accepted-prefix length, first/last rejection, all-accept/bonus, initial token outside subset, rollback and context wrap. Compare draft KV and target GDN state. Deliberately shift h/token pairing or retain one rejected slot and require failure |
| Safe reuse | Alternate C/D shapes and different active sequence IDs with multiple output draws and backend samplers; grow allocation, evict a source owner, and compute old plans. A deliberately stale binding/epoch must fail or safely rebuild |
| CUDA communication/capture | Paired-rank replay, repeated arrival generations, actual kernel/graph counters, no host event wait inside resident segment, peak memory and cancellation. CPU meta tests cannot substitute for this gate |
| Net performance | Frozen, same-stack, same-k A/B with unchanged completions/counters for scheduling-only changes; if proposal law changes, require independent engine law validation rather than relaxing the same-seed gate casually |

## CPU work and handoff

`CUDA_VISIBLE_DEVICES='' nice -n 19 bash codex/run_cpu.sh` passes. All invoked commands used hidden GPUs and nice priority; the Python audit is single-threaded. There was no GPU query, engine execution, build, network access or process named `llama-server`. No new model CPU latency measurement is claimed. The queue harness is committed as `921175ee1`; this report and its reproducible evidence form the next topic commit.

The law oracle passes M441 exact rational p/q pairs including zero-support tokens; M4 controls reject wrong normalization, wrong ID mapping, wrong support and shared proposal/acceptance randomness. This proves the checked rational identity only; it does not certify a device sampler, logits, hidden state, finite RNG implementation or engine target law. `check_harness.py` passes frozen-copy/rename/summary/plan checks and M6 controls for changed bytes, completion, acceptance, missing boot and protected env. The actual existing vocab/chain frozen snapshots also passed CPU manifest/copy/rename verification, M2 snapshots; only file bytes were read/copied, no binary was executed. Python AST and Bash syntax checks pass. `evidence.py` records M37 source/report/log hashes and validates the cache arithmetic and long-chain failure.

Prior engine CPU results, read-only evidence: `C/codex/REPORT.md` reports M2700 single-CPU and M1080 meta comparison rounds, wrong-pick rollback controls, and CPU identity. Its k=3 CPU plain/chain draft means are M14.171/M13.562 ms; meta M16.679/M16.092. Those are not reproduced here and did not predict the GPU slowdown. Patch A's report likewise demonstrates CPU build-time savings but omits backend samplers in its graph-cache identity test; the review explains the coverage gap.

The new `codex/gpu_ab.sh` defaults to plan-only and performs no GPU query or HTTP call in that mode. Execute mode requires the parent's explicit `RUN_GPU_AB=1` and nonempty `CUDA_VISIBLE_DEVICES`. It accepts frozen roots with `SOURCE` and `bin/SHA256SUMS`, copies bytes into a fresh in-worktree OUT, verifies source manifests and shipped libraries, renames the executable `draft-loop-ab`, generates a new manifest, and rechecks complete file sets before boots and after the experiment. It uses `up,patch,patch,up`, checks selected-GPU idle memory and boot Xid, protects thread/library environment, checks port/process names, bounds wall time including exact-owned-PID shutdown, and records raw responses, commands, manifests, inputs, metrics and resident memory.

Both arms use the same compact head, probabilistic proposal, draft/target backend candidates, tensor split and request settings. `--log-verbosity 3` suppresses library INFO/CONT mapped to TRACE level 4 (`S/common/log.cpp:529`), retaining common/server INFO and warnings. No enablement test depends on library INFO; the existing chain diagnostic checks the common logger's `MTP subset chain enabled` marker. Performance uses the actual `spec_decode_num_drafts_total` delta, not `draft_n/k`; repeated identical requests collect at least the configured M20 observed rounds per case. This is not a steady-state nsys sample count. Same-seed messages/finish reasons and draft/acceptance/round counters must match; generated tool-call IDs alone are excluded. Chat response equality is not a proof of token-ID/q identity.

Smallest queued diagnostic: rerun the existing subset chain against its actual b8668a92e parent at k=3, target/backend candidates and uploads fixed. This reproduces a failure to attribute, not a proposed speedup. Use the existing frozen `vocab-ab` and `chain-ab` snapshots; supply the exact vocabulary and template used by the parent. Only the parent may execute:

```sh
RUN_GPU_AB=1 CUDA_VISIBLE_DEVICES=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f \
nice -n 19 bash codex/gpu_ab.sh --execute \
  --up /home/xanewok/repos/llama.cpp-mtp-draft-vocab/codex/frozen/stack --up-exe vocab-ab \
  --patch /home/xanewok/repos/llama.cpp-draft-chain/codex/frozen/chain --patch-exe chain-ab --patch-chain \
  --vocab /home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json \
  --template /home/xanewok/repos/qwen38-eval/results/session2/pair-5060ti-2060s/campaign-swift-fullsuite-20260915/swift-iq4-xs/attempts/66ebeeaec6a7043b-01/chat-template.jinja
```

The default G1200-second queue allocation is a bound, not a runtime prediction; use a fresh OUT for another run. Optional `--k 3 5 --budget-s 2400` is the follow-up scaling experiment. For a future trusted-proposal patch, freeze up as current S and patch as its single-change descendant; omit `--patch-chain` if it is not that implementation. No future patch snapshot is supplied by this branch.

For causal attribution, the parent should capture one otherwise identical up/patch think request after warmup and retain at least M20 steady rounds, logging graph reuse/misses, meta plan rebuilds, chain mismatch/replay counts, D2H bytes, CUDA graph launches, AR host waits and V/C/D phase ranges. Those counters need instrumentation before queueing; this harness intentionally supplies an unprofiled A/B only. First separate the existing failure's added CPU/device/rebuild work, then test trusted proposal, safe plan reuse and capturable communication as separate same-stack changes. A whole-loop capture alone should fail the design review if host validation or per-step AR slot waits remain.

Open: current-S steady host-gap attribution; patched-HQ fused-loop enablement trace; long-chain GPU identity failure cause; per-component chain slowdown; kernel-level finite-precision exactness; graph-plan lifetime under concurrent contexts; capturable non-NCCL arrival generation; target grammar/forcing device state; peak VRAM; actual execution validation of this parent-only harness. No GPU performance result was generated by this task.
