# T-8: final-state GDN verification with accepted-prefix replay

## Status

Implemented under the owner's 2026-10-15 22:35 decision lifting the production-line cap. The prescribed CUDA build, CPU operator tests, new rollback/memory/meta tests, and generated-model state/logit identity gates pass after fixes. Implementation and CPU work are complete. This is uncommitted and not frozen or GPU-qualified.

Base `9dd7d8c9bc5cd5e61931ba5c41aa28706bba340b`, branch `topic/t8-gdn`. The pre-existing `codex/PURPOSE` edit is unrelated and untouched. No staging, commits, servers, GPU execution, or queue mutations were performed. The external handoff script owns committing, GPU gating, freezing, and submission.

## Queue overlap

The first build used only the requested fresh empty `running` check at startup. It ran from 00:00:39 to approximately 00:13:51. The read-only queue attempt record `/home/xanewok/.local/state/gpuq/items/fs2-np2-k-iq4-r2/a1/attempt.json` records that r2 started at 00:01:35.636, so its window overlapped the build. Its timing may be contaminated; the parent must assess or rerun affected legs. No queue item was modified.

The waiting test-build guard was stopped before it launched a compiler. The replacement requires two fresh idle observations at least 300 seconds apart before work, rechecks before each command, and monitors long commands every 300 seconds, pausing their process group if the queue becomes busy or stale. This reduces gap risk; a read-only status file is not an exclusive queue lease. The final build/test sequence started after fresh idle observations at 06:58:57 and 07:03:57 and finished at 07:11:30. No later gate observed a running item. The compact gate log preserves timestamps, file ages, and running IDs; its original form is retained at `/mnt/scratch/tmp-build/t8-gdn-gpuq-gates-raw.log`.

## Storage and replay contract

For a verify batch with `T = K = n_rs_seq + 1`, each eligible GDN layer runs the existing final-state (`K=1`) recurrence. The kernel also retains the exact raw post-convolution inputs `q,k,v,g,beta` and the initial recurrent-state bytes in a persistent backend buffer. Each logical sequence has a distinct retention slot, independent of physical cache rows and batch ordering. Attention output and the final recurrent state keep their existing formats. Convolution snapshots and the existing recurrent-cache snapshot allocation remain unchanged.

The packed F32 row is `[q[T], k[T], v[T], g[T], beta[T], initial_state]`. Its leaf metadata records `(S,Hq,Hv,G,T)`; `G` is 1 or S. The checked-size helper owns this layout. The CUDA kernel captures the initial state from its input shards and stores raw inputs before modifying them. This avoids separate input-retention launches. CPU copies the same bytes while processing each head. The CUDA snapshot specialization excludes retention writes with `if constexpr`; the recurrence source is unchanged.

An accept of `a` drafts retains `r=a+1` verify rows, including the first token. Full accept submits no replay from `seq_rm`. Partial `seq_rm` queues an ordinary final-state GDN (including its unused attention output to preserve the shared arithmetic) on the first `r <= k` retained input rows and the retained initial state for every affected layer. Its state output is copied to the existing rollback plane `depth=T-r`. Only after successful submission does bookkeeping update `rs_idx` and position. The existing single-use `s_copy` then selects the same plane for convolution history and GDN state. State serialization immediately after rollback therefore reads the rebuilt selected state without waiting for another decode.

Replay uses cached graph/scheduler plans keyed by logical sequence, physical row, depth, and layer set. Plans borrow the context's existing backend handles, so replay and the next model graph use the same backend streams. Graphs, views, persistent inputs, and output addresses remain alive across CUDA graph reuse. First use creates and allocates a plan on the host; subsequent use reuses it. No state/input data travels to the host to compute a rollback. First-use allocation or CUDA graph capture can still synchronize; its latency must be measured separately from steady replay.

Exceptional state operations materialize the omitted snapshot planes before changing the state contract: sequence copy, keep, position shift/division, or a shorter subsequent batch. Copy propagates a pending selected plane to the destination. A shorter batch can therefore add K-1 history replays even after a full accept; fixed-K verify loops avoid that transition. Removal/clear/restore invalidate retained records. Failed graph submission marks the record unusable; failed replay leaves rollback position and index unchanged. The context destroys replay plans while its backends are still alive.

## Eligibility and backend fallback

The path requires recurrent snapshots, F32 GDN state, a complete K-row batch, single sequence IDs per sequence group, unbroadcast q/k sequence dimensions, no extra recurrent rows in the active range, and the existing indexed-state getter. Backend support is probed before choosing it. `LLAMA_GDN_REPLAY=0` restores the snapshot path. CPU and CUDA support retained `src[7]` plus I64 sequence-slot `src[8]`; all other GDN backends explicitly reject the extension and retain their existing snapshot/gather path. The existing API and plain GDN op remain compatible.

Tensor-parallel storage partitions all six packet components by the same key-head groups and modulo value-head repetition used by the recurrent cache and GDN inputs. Replay views carry internal component metadata so the meta backend maps input views to axis 1 and state views to axis 2, including nonuniform splits. No Swift or GPU architecture constants enter the implementation. The packet uses the existing recurrent-head granularity calculation.

## Identity argument and break points

Replay starts from copied initial-state bytes and consumes copied raw input bytes. It executes the same sequence of per-token operations as snapshot verification through the accepted row, with the same head mapping and one final state write. Recomputing model inputs, storing exponentiated gates, changing reductions, or replaying from the final state would invalidate that argument; none is done here.

The tests compare each path within its own backend with `memcmp`, including signed zero. CPU and CUDA use different accumulation orders and are not required to match each other. CUDA's scalar-gate path scales the dot result after reduction; CPU scales state elements before its dot. Other places identity could break are compiler FMA/reassociation or register-spill changes between the K=1 and snapshot specializations, NS=1 versus joint NS=2 scheduling, incorrect TP head grouping or view offsets, overwriting pre-state during indexed fused cache writes, wrong logical slots after sequence reordering, and stale captured addresses or retained bytes across graph reuse. The new op tests, nonuniform CPU meta test, changing-input graph replay, and recurrent-memory tests target these classes. The CPU meta test alternates separate verify/replay schedulers over a nonuniform 1:3 head split. They do not prove equality against the separately compiled frozen fs2 CUDA binary; the parent must run the GPU checks and cross-arm token identity before accepting the change.

The indexed CUDA cache-fusion path still uses the existing joint-sequence kernel so all input shards for a head/column are loaded before any sequence writes that column. Retention and replay add no inter-head reduction. Supported S values remain 16/32/64/128 on CUDA; CPU follows the existing reference implementation.

## Changes and line counts

Final tracked counts, excluding the pre-existing `codex/PURPOSE`: production +550/-81 lines in 27 files (net +469); tests +612/-5 in three files; documentation +5/-1 in two files (`docs/ops.md` and two actual CPU capability rows in `docs/ops/CPU.csv`). Total +1167/-87 in 32 files. Untracked deliverables are this report (100 lines), `codex/READY`, and three files in `gpu-checks/`: 41 command-list lines and 114/115 spec lines. The owner lifted the original 250-line cap. Hybrid input setup/reuse now shares the existing recurrent implementation instead of duplicating it. Production changes are the retention/replay ggml helpers, CPU/CUDA capture, backend fallback, TP packet/view mapping, recurrent-memory lifecycle, and graph/context integration. Tests cover the same contract and cannot be shipped independently of the extension. `gpu-checks/` contains two parent-owned experiment inputs and a command list.

## Build and CPU evidence

Configuration succeeded with the exact frozen fs2 recipe in `build-cuda`; see `configure-t8-gdn.log`. Every build/test run is guarded by a fresh read of `/home/xanewok/.local/state/gpuq/status.json`: `running` must be empty and the file younger than 30 seconds. Busy/stale snapshots are polled every 300 seconds. All commands run under `taskset -c 6-11,18-23 nice -n 19`, with `CUDA_VISIBLE_DEVICES=""`, `GIT_NO_LAZY_FETCH=1`, and build temporary files under `/mnt/scratch/tmp-build`. Gate reads are logged in `gpuq-gates-t8-gdn.log`.

Passed: the initial prescribed `buildgate cmake --build build-cuda -j4 --target llama-server test-backend-ops`, exit 0 in 791.820 seconds (`build-t8-gdn.log`); final incremental main build, 25.128 seconds (`build-final-t8-gdn.log`); test targets, 11.031 seconds (`build-tests-t8-gdn.log`). After the hybrid fix, the prescribed main targets passed again in 25.917 seconds (`build-hybrid-fix-t8-gdn.log`) and test targets in 5.020 seconds (`build-tests-hybrid-fix-t8-gdn.log`). The final test-only capability addition rebuilt the prescribed main targets in 25.110 seconds (`build-support-tests-t8-gdn.log`). All compile commands exited 0. No result from the old binary is evidence that this implementation works.

CPU results: 1856/1856 GDN/replay/indexed/fusion cases passed in 60.821 seconds, including all 1792 new accept-pair cases and the changing-input byte-identity check (224 graphs x four runs with one-bit negative controls). The two additional single-op retention cases passed 2/2 in 0.765 seconds; they also produced the actual capability rows. Rollback bookkeeping passed; persistent memory/cached replay passed 880 cases; nonuniform meta replay passed 126 cases. The generated-model matrix passed all four processes: np1 has eight matching state/logit records per mode, np2 sixteen, with negative controls passing in both modes. Positive enabled/restored proof is present for np1 and np2. `test-integration-t8-gdn.log` confirms hash equality and exact template-shape preservation for both GPU specs. Final `git diff --check` passed (`test-final-diff-t8-gdn.log`).

Fixture: `qwen35-dense.gguf`, 5,557,728 bytes, SHA-256 `26c2b6f3ad745c5cf2cc6618038e50bf5c798132b407eb9441a17915675327c1`, generated with seed 2049007652 and stdev 0.1. This is a tiny random-weight integration fixture, not served-model quality evidence. Frozen fs2 has no recurrent-state test executable; the CPU reference is this build's unchanged snapshot path with `LLAMA_GDN_REPLAY=0`. Separately compiled frozen fs2 identity remains a parent GPU gate.

The CPU command matrix uses `LD_LIBRARY_PATH="$PWD/build-cuda/bin"`. `LLAMA_GDN_REPLAY=1` is enabled except for the explicit snapshot reference runs; replay logging is off except for enabled model proof runs.

| Gate | Command after the affinity/env prefix | Log |
| --- | --- | --- |
| Final main targets | `buildgate cmake --build build-cuda -j4 --target llama-server test-backend-ops` | `build-hybrid-fix-t8-gdn.log`, `build-support-tests-t8-gdn.log` |
| Test targets | `buildgate cmake --build build-cuda -j4 --target test-recurrent-state-rollback test-llama-archs` | `build-tests-hybrid-fix-t8-gdn.log` |
| CPU operators and byte identity | `build-cuda/bin/test-backend-ops test -b CPU -o 'GATED_DELTA_NET.*'` | `test-backend-cpu-t8-gdn.log` |
| Single-op retention capability | `build-cuda/bin/test-backend-ops test -b CPU -o GATED_DELTA_NET_RETAIN` | `test-retain-cpu-t8-gdn.log` |
| Rollback bookkeeping | `build-cuda/bin/test-recurrent-state-rollback --bookkeeping-only` | `test-bookkeeping-t8-gdn.log` |
| Persistent memory and replay plans | `build-cuda/bin/test-recurrent-state-rollback --gdn-replay-only` | `test-memory-t8-gdn.log` |
| Nonuniform TP views and scheduler switching | `build-cuda/bin/test-recurrent-state-rollback --gdn-replay-meta-only` | `test-meta-t8-gdn.log` |
| Generated model fixture | `build-cuda/bin/test-llama-archs -a '^qwen35$' -o /mnt/scratch/tmp-build/t8-gdn-models` | `test-fixture-t8-gdn.log` |
| Model state movers, four runs | `LLAMA_GDN_REPLAY={0,1} LLAMA_GDN_REPLAY_LOG={0,1} build-cuda/bin/test-recurrent-state-rollback --gdn-state-movers -m /mnt/scratch/tmp-build/t8-gdn-models/qwen35-dense.gguf -ngl 0 -np {1,2} -t 1 -tb 1 -c 256 -b 64 -ub 32 -fa off -lv 5` | `test-model-np{1,2}-mode{0,1}-t8-gdn.log` |
| CPU support rows | `build-cuda/bin/test-backend-ops support -b CPU -o GATED_DELTA_NET_RETAIN --output csv` | `support-cpu-t8-gdn.log` |

Fixes exposed by the CPU gates: the new CPU meta fixture initially used unsupported segmented uploads (axis 2, then 4-D repeated-head uploads) and omitted the scheduler's required terminal CPU backend. The corrected fixture uses the model cache's flattened storage/reshape pattern, flattened token/sequence input storage, and a CPU fallback. These three setup failures are retained as `test-meta-{first,second,third}-failure-t8-gdn.log`. The corrected test passed 126 cases, including prefix attention and state bytes.

The first enabled generated-model run exposed a production integration defect: hybrid input wrappers filled only the old state-copy indices, bypassing the recurrent input setter. Retention slots and records were not initialized (`test-model-first-failure-t8-gdn.log`). All three hybrid input variants now delegate recurrent input setup and reuse checks to `llm_graph_input_rs`; this also updates its memory-context pointer during graph reuse. The prescribed rebuild and complete np1/np2 enabled/disabled model matrix pass after this fix.

The initial support query returned no rows because the exporter omits whole-graph tests (`support-replay-excluded-t8-gdn.log`). Two non-indexed single-op retention cases now cover capability reporting and CPU-reference output comparison for scalar/KDA gates. The first integration-log check failed because library INFO proof lines require higher common-log verbosity; the final proof matrix uses `-lv 5` and passes. The failed proof assertion is retained in `test-integration-proof-failure-t8-gdn.log`. CUDA initialization messages in these logs are expected with `CUDA_VISIBLE_DEVICES=""`; no GPU op ran.

Prerequisite observations before implementation: frozen fs2 CPU `GATED_DELTA_NET` passed 36/36 (`baseline-t8-gdn-cpu.log`); an independent frozen-CPU prefix probe passed 448 cases and rejected 620 wrong-slot controls (`cpu-prefix-probe-t8-gdn.log`, source `/mnt/scratch/tmp-build/t8-gdn-cpu-prefix-probe.py`). These only establish that unchanged per-backend prefix arithmetic can be byte-identical.

## Parent GPU checks

`gpu-checks/COMMANDS.md` lists the operator gates on each GPU with CUDA graphs on/off, generated-model state/logit identity with retention on/off, and parent expansion/submission steps. The two window specs copy the exact requested np1 depth and np2 real ABBA shapes. Arms are fs2 (`frozen/stack11-fa-s2`) and `frozen/t8-gdn`, with cross-arm token identity on both. The np1 depths are 8192/32768; np2 real depths are 8192/32768/90112 with two joint-start streams. Both plans are fs2/t8-gdn/t8-gdn/fs2. Frozen SOURCE resolves the future commit instead of inventing a hash now.

Canaries require positive final-state retention and partial-prefix replay log lines for t8-gdn, plus the template's existing fs2 proofs. Main measurement boots do not enable replay logging. GPU gates and experiment expansion are intentionally not executed by this worker.

## Automation handoff

`codex/READY` selects `t8-gdn`, thread T-8, all six GDN operator-test names, `AB=yes`, `ENVS=LLAMA_GDN_REPLAY=1`, and `EXPECT=-0.3..0.05` (ms/round, negative faster). `IDENTITY=on` requires strict fs2 token identity; it does not claim that the still-pending CUDA/server identity gate has passed. The combined guess includes the np1 and np2 ranges below. `NEW_FILES` preserves this report and both GPU specs plus their command list when the script commits; no new C/C++ source file was introduced.

`TESTS` is empty because the existing model test binaries require fixture paths and arguments, while the handoff script runs extra binaries without arguments. Their GPU matrix remains in `gpu-checks/COMMANDS.md`. The operator gate includes the 224 changing-input graph cases. Its failure branch now prints `FAIL` so the script's text-based gate cannot overlook a returned graph-identity failure after the regular case-count summary. The marker-only test change rebuilt successfully in 24.825 seconds; its CPU recheck passed 1792/1792 cases plus 224 changing-input graphs x four runs in 61.913 seconds; see `build-ready-t8-gdn.log` and `test-ready-cpu-t8-gdn.log`.

## Cost and expected gain

Observed prior op probe (`/mnt/scratch/work/gdn-reprobe/a1/analysis/summary.txt:115-117`): np2 gross saving 1.316/1.328 ms per round, three-row replay 0.370/0.368 ms per sequence, net 0.820/0.834 ms using 0.67 partial-accept probability. That probe reused the same input tensors across 48 layers and did not measure persistent retention, real acceptance integration, or this implementation. Its number is an optimistic reference, not a measured result for this code or a formal upper bound.

For served S=128, Hq=8, Hv=24 per GPU, T=6, 48 GDN layers, the scalar-gate raw inputs occupy about 5.95 MB per GPU per sequence, and retained pre-state about 75.5 MB. These writes reduce the original six state writes to one final state plus one pre-state capture. Cached replay plans add persistent compute buffers and first-use allocation/capture overhead. Existing snapshot cache storage remains allocated. Input retention, extra pre-state traffic, replay launches, and memory pressure reduce the probe's ideal gain. Inference: six state writes become two, so four writes are saved instead of the probe's five; scaling its gross saving by 4/5 gives about 1.06 ms/round at np2, or about 0.56 ms after the same 0.50 ms expected replay cost, before raw-input stores and integration overhead. This scaling is a bandwidth approximation, not a measurement. Actual meta-buffer padding and cached-plan allocation can exceed the logical byte counts above.

Guess (G), pending actual GPU measurements: np2 saves roughly 0.1-0.3 ms/round; np1 saves 0-0.1 ms/round and could be neutral or regress. These ranges discount the earlier board calibration for the extra pre-state capture and unmeasured integration cost. The guess assumes steady six-row verification; frequent shorter target batches can erase the gain because they materialize omitted history. No performance claim is made for this code. Inspect free VRAM and cold replay latency as well as steady timing; a token mismatch blocks the change regardless of speed.

## Deliberate limits

No GPU execution, timing, freeze, queue submission, or commits. No new non-CPU/CUDA retention kernel: capability fallback preserves those backends. Pre-state capture was chosen over a new ping-pong row mapping to keep the existing shared convolution/GDN state movers intact. No recurrent-cache allocation shrink or convolution redesign: both would alter independent state-management contracts. No claim of cross-backend byte identity. First-use replay-plan setup is retained for simplicity and must be included when assessing cold latency. This private-fork implementation is not an upstream PR submission.
