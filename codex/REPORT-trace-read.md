# First GPU round trace and AR pool start check

## Conclusion

D, high confidence: prioritize safe C/D1 graph and meta-plan reuse, then narrower input-lifetime work. This capture provides no case for committing weeks to a device-resident draft loop or capturable AR yet. AR pool 2 -> 8 remains a useful bounded diagnostic of verify backpressure, not an expected large draft optimization.

M: the pool A/B failed because backend INFO is filtered at the default verbosity, not because its message changed or the frozen library lacks it. The harness now forces `LLAMA_ARG_LOG_VERBOSITY=4` in both arms, records that environment, and requires matching target and draft pipeline initialization. Completion/counter exactness guards remain. No sampling, RNG, CUDA runtime or reduction code changed. GPU rerun is pending.

M = observed/measured; D = interval unions, partitions, averages or arithmetic; G = forecast. Source lines, versions, seeds, counts in configuration names and K are identifiers/configuration. Every timing table below is D ms per retained round unless explicitly marked M. No optimization GPU speed was measured in this task.

## Evidence and selection

Work branch: `topic/round-trace-read`, starting at `a92015dab`. Harness fix: `f673ba010`. Background report and review paths/hashes are in `codex/evidence.json`.

Input is the parent's run `codex/runs/20261001-214325/trace/`, frozen source `dfeef79179c2aa86994c9e369b5fa925b6f58a6a`, IQ4_XS, tensor split, NCCL off, target and draft backend sampling, subset draft vocabulary, K=3. This is a descendant of the requested base, not an A/B against `68d99c67d`. SHA256:

- `capture.sqlite`: `900fc16feb942e8f9d23b9a217a2af9bdf3af2d9b422bf4159d90ef85d39da50`
- `rounds.jsonl`: `e357db781a9fc202ca3f44efe40ff18cdd83a057850f6de8ce63615a6dbc49c7`

The saved analyzer retains M44 think and M41 tool rounds, with host V-to-V medians M31.070 and M31.013 ms. `codex/read_round_trace.py` reproduces that host selection, then also trims first-V-kernel to next-first-V-kernel periods to +/-3% of their median. That retains M42 think and M41 tool rounds. Think boundaries `253->254` and `258->259` are excluded: their kernel periods are M34.299 and M59.968 ms. The latter enters a next-V graph miss, including M17.228 ms inside `llama_build`; next-V startup is outside the host V-to-V selection window. The saved think D3->V mean D1.837 ms, including D0.639 ms graph builds, is therefore not a steady boundary estimate. All next-V batches in the retained host set have M4 tokens; batch size alone does not detect this contamination.

The offline reader uses the existing process-scoped runtime correlation and NVTX clock alignment (`codex/attrib.py:105`), includes graph-node kernels, and asserts that all executed kernels in each retained full phase envelope have an owner. The large saved `unmatched_kernels_M` counts include other requests outside that case's selected JSONL; they are not missing kernels in these retained windows. Activity is the union of KERNEL, MEMCPY and MEMSET intervals on either selected GPU, never the sum across GPUs. "Active" includes polling/stalled kernels and says nothing about SM efficiency. Idle-only host attribution intersects the event partition with the complement of that union; overlapping GPU work is removed before assigning categories. Results, round IDs, exclusions and checks are reproducible in `codex/trace-read.json`.

## Critical path at K=3

A phase envelope runs from its first kernel on either GPU to its last kernel on either GPU. Boundaries run from that last kernel to the next phase's first kernel. Copies within boundaries remain active time. These windows cover the kernel-to-kernel round exactly.

| Window | Think span D | Think any-GPU active D | Think both idle D | Tool span D | Tool any-GPU active D | Tool both idle D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V | 25.853 | 25.637 | 0.217 | 25.884 | 25.698 | 0.186 |
| V->C | 0.747 | 0.053 | 0.693 | 0.737 | 0.054 | 0.683 |
| C | 0.457 | 0.455 | 0.003 | 0.458 | 0.456 | 0.002 |
| C->D1 | 0.659 | 0.006 | 0.653 | 0.640 | 0.006 | 0.634 |
| D1 | 0.629 | 0.593 | 0.037 | 0.635 | 0.597 | 0.037 |
| D1->D2 | 0.193 | 0.008 | 0.185 | 0.165 | 0.008 | 0.157 |
| D2 | 0.611 | 0.578 | 0.033 | 0.612 | 0.578 | 0.034 |
| D2->D3 | 0.185 | 0.008 | 0.177 | 0.160 | 0.008 | 0.152 |
| D3 | 0.605 | 0.578 | 0.027 | 0.607 | 0.578 | 0.029 |
| D3->V | 1.114 | 0.087 | 1.027 | 1.057 | 0.087 | 0.970 |
| Round | 31.054 | 28.002 | 3.053 | 30.955 | 28.069 | 2.886 |

Per-device full-round active unions are D27.455/27.383 ms (think GPU0/GPU1) and D27.483/27.422 ms (tool). The union across GPUs is larger because their work overlaps imperfectly; neither summing device durations nor calling phase span "busy" gives the critical-path idle budget.

The tracer itself occupies a measured envelope before every next V: M timestamps give mean D0.582 ms think and D0.571 ms tool. `ggml/src/ggml-round-trace.cpp:50` opens the file, groups/sorts counters, formats every event and closes it synchronously; `:105` invokes this before the next phase timestamp at `:109`. All of this envelope is both-GPUs-idle in the retained set. It is included in residual below. Subtracting it leaves D2.471/2.315 ms of idle associated with the observed serving path, but that subtraction is not an unprofiled performance measurement: collection also perturbs CUDA API scheduling and event storage.

Idle-only categories, per boundary. Each cell is think/tool D ms; `sync` excludes waits nested in the input scopes. AR waits contribute D0.000 at every boundary. Categories sum to the both-idle column, up to rounding.

| Boundary | Graph builds D | Inputs D | Other sync D | Sampling D | Readback D | Other CUDA API D | Residual D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V->C | 0.418/0.411 | 0.061/0.060 | 0.037/0.038 | 0.000/0.000 | 0.000/0.000 | 0.023/0.022 | 0.154/0.152 |
| C->D1 | 0.432/0.432 | 0.059/0.059 | 0.021/0.019 | 0.000/0.000 | 0.000/0.000 | 0.024/0.025 | 0.118/0.099 |
| D1->D2 | 0.000/0.000 | 0.055/0.056 | 0.041/0.041 | 0.002/0.002 | 0.001/0.001 | 0.018/0.017 | 0.068/0.040 |
| D2->D3 | 0.000/0.000 | 0.055/0.055 | 0.041/0.041 | 0.001/0.001 | 0.000/0.000 | 0.017/0.017 | 0.062/0.037 |
| D3->V | 0.000/0.000 | 0.110/0.108 | 0.061/0.057 | 0.001/0.001 | 0.000/0.000 | 0.029/0.026 | 0.827/0.777 |

Full-round category totals include the idle inside phases:

| Idle category | Think D ms | Tool D ms |
| --- | ---: | ---: |
| Graph builds | 0.850 | 0.843 |
| Inputs | 0.340 | 0.338 |
| AR waits | 0.176 | 0.145 |
| Other sync | 0.314 | 0.318 |
| Sampling | 0.005 | 0.005 |
| Readback | 0.010 | 0.007 |
| Other CUDA API | 0.123 | 0.120 |
| Residual, including trace flush | 1.234 | 1.110 |

M: C and D1 each have exactly one `graph_miss`, one `llama_build` and one `meta_build` in every retained host round. V, D2 and D3 each hit once per round. `src/llama-context.cpp:1528` requires both `gf_res_prev_active == res` and `can_reuse`; C/D1 alternation breaks the active-result check. `ggml/src/ggml-backend-meta.cpp:2077` measures plan reconstruction. Their exposed build cost is D0.843-0.850 ms/round, chiefly V->C and C->D1. D2 also recaptures CUDA subgraphs every round, D0.185-0.191 ms host wall time in the saved sets, but only D0.000001-0.000070 ms is on the corrected idle path. Do not add that host wall duration to expected savings.

M: inputs are already batched in this stack. V submits M18 logical regions through M6 APIs; C M12/M4; each D M14/M4 across the pair. `ggml-cuda.cu:808` stages the batch; `:859` counts the batch API; `:870` retains the stream synchronization. The old "14 individually synchronized input APIs per D" budget does not apply. Remaining idle in input scopes is D0.338-0.340 ms/round; additional input-related fencing is partly in the D-boundary `sync` and CUDA-API columns. The device copies themselves total only D0.16-0.17 ms across the boundary windows and were subtracted from idle attribution.

D: target/draft sampling is almost entirely hidden by GPU work here. Exposed instrumented sampling is D0.005 ms/round, despite C-owned sampling host wall time of D0.086/0.132 ms in the saved think/tool sets. Decode submission finishes before the corresponding GPU envelope ends. Moving that host work to a device kernel is not a demonstrated net saving. Residual after the trace-flush envelope is D0.653/0.539 ms/round; it includes uninstrumented bookkeeping, allocation and CPU descheduling. This capture does not identify a specific server subroutine responsible for every residual interval.

## Internal AllReduce

M, saved host sets: V makes M128 logical AR calls and M128 wait API invocations per GPU per round. GPU0 wait p50 is M118-119 us and p95 M186 us; GPU1 p50 is M1.31-1.32 us. C and each D make M2 calls and M2 waits per GPU, normally M1-3 us per wait. Pool wrapping checks both ranks sequentially (`allreduce.cu:361`, `:370`); the first wait can consume the GPU work while the second returns almost immediately.

D, corrected full-round sets: total AR host-wait wall time is D17.957 ms think and D18.022 ms tool. Only D0.176/D0.145 ms overlaps both-GPUs-idle intervals, all inside V. D99.02%/99.20% overlaps activity on at least one GPU. The entire V idle budget is D0.217/D0.186 ms; there is no AR wait on the phase boundaries and no exposed draft AR wait.

D, high confidence: this establishes host backpressure in V, not an independently removable D18 ms delay. It does not establish that communication is free. AR-kernel active unions are D2.036/2.231 ms per round on think GPU0/GPU1 and D1.919/2.263 ms on tool; kernel durations include peer polling and overlap other work. Larger look-ahead could change peer skew or subsequent fences, so neither those kernel times nor the idle intersections prove a total speed ceiling for a protocol/capture redesign.

Keep the pool experiment because it is prepared and can cheaply separate backpressure from harmful submission starvation. Its likely direct idle opportunity is small, G0-0.2 ms/round at K=3; K=5 is unmeasured, G0-0.3 ms. Focus the trace comparison on V, total generation period and where waits move, not the already negligible draft wait histogram. Wait duration falling alone is not success. The harness's boundary-idle summary also omits within-V idle; inspect phase activity before drawing a capture conclusion. Unchanged period with smaller waits supports the backpressure interpretation; unchanged waits means this diagnostic did not change effective look-ahead. This A/B cannot separately falsify replay launch-overhead savings.

## Ranked changes

Forecasts are G ms per round saved; they are neither measurements nor additive. K=5 assumes similar input ownership and the same single-draft-head route, not identical verify cost.

| Rank | Concrete change | K=3 saving G | K=5 saving G | Evidence and upstream shape |
| --- | --- | ---: | ---: | --- |
| 1 | Retain valid C/D graph results and meta plans across alternation, with graph-local sampler bindings and allocation validity enforced at compute | 0.6-0.85 | 0.6-0.9 | D0.84-0.85 ms exposed. General reuse is upstream-shaped; the reviewed Patch A implementation is not ready. |
| 2 | Replace remaining synchronous input-batch lifetime fences with owned pinned staging slots and ordered compute-stream/event handoff; keep ordered fallback for overlapping destinations | 0.15-0.30 | 0.25-0.45 | D0.34 ms inside input scopes plus some separate fences. Batching already exists. Backend transfer ownership is upstream-shaped; deleting syncs is unsafe. |
| 3 | Profile and remove avoidable host preparation in the residual, especially V->C/C->D1 and the post-flush D3->V tail | 0.1-0.2 | 0.1-0.25 | D0.54-0.65 ms unclassified after flush is a ceiling, not all redundant work. Narrow batch/server bookkeeping changes can fit upstream after a subroutine is measured. |
| Diagnostic | Increase active AR ring capacity using the existing frozen env-only A/B | 0-0.2 | 0-0.3 | Mainly V backpressure. Keep DO NOT MERGE; this is causal evidence, not a proposed public tuning knob. |
| Defer | Device-resident D1..Dk token/hidden/q loop, after the simpler work | 0.2-0.4 | 0.4-0.7 | Interior D1->D2 and D2->D3 idle totals D0.309-0.363 ms; four such boundaries at K=5 give G0.62-0.73 ms before ownership changes. Partial input/reuse savings overlap this estimate. Full capture has weaker upstream fit and much larger validation cost. |

Exactness obligations for future changes: reuse must restore the selected graph's sampler tensor bindings and RNG draw sequence; staging must remain live until every consumer finishes; device drafting must preserve proposal IDs/q, hidden/token alignment and the rejection residual law, including grammar/ties/fallback. Require same-seed token/q evidence and negative controls for stale bindings, overwritten staging or stale replay epochs as applicable. No sampling-law change was made here.

Do not transplant Patch A as reviewed: `SP/codex-reviews/patchA.txt` identifies shared sampler bindings, unpinned meta input owners, an unchecked public compute epoch, shared container races and unsafe rematerialization. Resolve those contracts in a minimal design before pursuing its measured opportunity. No upstream issue status was refreshed under the no-network rule; "upstream-shaped" is a design assessment, not maintainer approval.

M, parent-reported comparison inputs: HQ g32i8 K=5 step M33.1 ms, K=8 M36.5 ms; stack2 IQ4_XS S5 round approximately M36.8 ms. Their difference is D3.4 ms across three extra draft positions and larger verify/catch-up batches, or D1.13 ms per increment averaged over that range. It is not isolated draft-loop overhead. Different quantization, context/request mix and trace instrumentation prevent applying these forecasts directly to those totals. A K=5 capture with the same pins must establish its phase budgets.

Recommendation, D: do not spend weeks on a full device loop/capturable AR now. The reduced vocabulary and backend sampling are already active, each D envelope is only D0.60-0.63 ms, and sampled host work is mostly hidden. Queue the repaired pool A/B; pursue correct graph reuse and input ownership first. Reconsider a narrow fixed-draft communication-capture prototype only if matched trace-off A/B and phase traces expose a remaining gain large enough to justify replay epochs, scratch ownership, paired-rank cancellation and GPU exactness work. This trace alone does not provide that evidence.

## Pool failure and fix

Failed boot: `codex/pool-runs/20261001-214628/off-1-pool2/`, source `1efa250254d0603c63dbac64bb6d49285c16da6b`. Its `server.log:2` reports verbosity `3`; `:6` and `:9` report NCCL unavailable and internal-AR fallback, but no initialized-pool line. SHA256 of that log is `f846b42001592ed05c4a1ec9ad0d94330571c64be511e488c51ae252259d69fd`. Static `strings` inspection of the frozen `libggml-cuda.so.0.25.3` finds both `GGML_CUDA_AR_POOL` and the exact initialization format from `allreduce.cu:539`.

M source chain: `common/log.cpp:532` maps `GGML_LOG_LEVEL_INFO` to `LOG_LEVEL_TRACE`; `common/log.h:26` makes that level `4`; `common/log.cpp:544` filters against the threshold. `common/arg.cpp:3949` binds `LLAMA_ARG_LOG_VERBOSITY`. `PoolHarness` strips inherited AR overrides, then sets the requested pool before `boot` (`codex/pool_harness.py:138`); `Harness.boot` passes `self.env` to `Popen` (`codex/gpu_harness.py:213`). There is no environment stripping between that assignment and launch. The failed `command.json` omitted the pool because it was recorded only after the old check; absence there is not evidence that the child lacked the environment. Successful GPU initialization at the requested pool remains unverified until the rerun.

The fix is harness-only (`codex/pool_harness.py:17`, `:98`, `:142`): common verbosity for all boots, environment persisted before validation, exactly two matching target/draft initialization lines. It does not accept generic fallback warnings as confirmation. Tests reject missing, single, wrong and mixed pool reports, inherited suppressed verbosity and completion/counter differences. Frozen bytes and GPU code are unchanged.

## CPU checks and parent queue

M: `codex/test_pool_harness.py`, `codex/test_harness.py` and `codex/test_attrib.py` pass on CPU. The latter also verifies overlapping device activity is unioned, complement clipping, and rejection of reversed activity; accounting and ownership assertions pass against all retained real-trace windows. No CPU inference or build was needed for the Python-only changes. Existing token/logit fixture results in `codex/REPORT-ar-pool.md` remain historical evidence, not rerun evidence. No GPU executable, GPU query, server process or network call ran in this task. CPU commands use `CUDA_VISIBLE_DEVICES=""`, nice priority 19 and single-process analysis.

```sh
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 codex/test_pool_harness.py
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 codex/test_harness.py
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 codex/test_attrib.py
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 codex/read_round_trace.py codex/runs/20261001-214325/trace --out codex/trace-read.json
CUDA_VISIBLE_DEVICES="" nice -n 19 bash -n codex/pool_ab.sh codex/gpu_ab.sh
```

Parent-only, not executed: queue `RUN_GPU=1 CUDA_VISIBLE_DEVICES=<pair> K=3 bash codex/pool_ab.sh --execute`; repeat separately with `K=5` if the first run passes. Same frozen bytes, pool order `2,8,8,2`, separate trace-off/on series, exact PID/start-time stop, frozen hashes, pinned requests, identity, memory and Xid checks remain; wall budget is G2400 s per series invocation. Both arms now use backend INFO verbosity. Avoid interpreting increased logging as a pool effect; it is identical across arms.

The existing `codex/gpu_ab.sh` supplies frozen up/patch/patch/up boots with the same ownership, identity and budget guards. `--trace-patch` can quantify tracing perturbation separately; default compares the disabled instrumentation against base. This is the existing instrumentation A/B, not a benchmark of this Python fix. No new binary freeze is needed for the repaired pool A/B. The parent's pre-existing `gpu_harness.py` Xid-window edit and untracked prompt/backup are preserved and excluded from the commits.

Open: successful pool startup on GPU, trace-off pool period and identity, proposal/q byte identity beyond completion/counters, boot variance, K=5 phase activity, residual subroutine attribution, and safe reuse/staging/replay contracts. The offline reader is intentionally specific to this K=3 capture; it does not replace the generic queue analyzer.
