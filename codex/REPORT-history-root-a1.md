# history-root a1 canary abort

Cause: tool completion-shim defect over a Meta planner limitation shared by both frozen engines. Switching to stack10 alone cannot fix it.

Evidence:
- M `/mnt/scratch/work/history-root/a1/canary/0-a-long-r0/status.json`: rc=-6, elapsed_s=106.857938, maps_checked=true, faults=0; gpuq's rc=8 is the runner's deterministic-failure classification.
- D the input has requests at depths 128, 256, 512. `history-root.cpp:507` prints completion after decode; the last request=1 line does not identify the failing call. M the native trace contains 55 complete target/prefill records plus an incomplete record, so the abort is in the captured final request.
- M archived executable `canary/build-stack9-p55/history-root`: the observer call is at ELF offset 0x9ec1 and its return at 0x9ec4 (`evidence/a1-observer-call.txt`). D frame #4, 0x56aaa42eeec4, resolves there with PIE base 0x56aaa42e5000. The pass-through path tail-jumps instead. This establishes the augmented-graph branch, not an untraced decode failure.
- `45d90d3e4:ggml/src/ggml-backend-meta.cpp:2466` skips views of plain host roots before checking the final-node boundary; lines 2503-2505 require that boundary to advance `i_start` to `n_nodes`. The old shim (`e5ebabb21:codex/history-root/history-root.cpp:41-62`) appends an observer with the same host root. D a host-only callback and observer are both skipped: i_start=0, n_nodes=2. A mixed graph with a skipped suffix likewise lacks its final boundary.
- `a588f2a44:ggml/src/ggml-backend-meta.cpp:2582-2621` has the identical skip and assertion. Its distinct capacity fix at lines 2654-2664 recreates `max_subgraphs` after arena reset, after this assertion. D the same callback path fails on stack10 too; M the CPU Meta negatives reproduce that shared source path. No native stack10 execution is claimed.
- M both generated `meta-layout.h` files equal their pinned-source definitions, with identical private-layout SHA256 `e375272bbd87d21cc28f23130e5b07b8002773d47d65f2d84847006420541c4f`; `test_history.py:test_frozen_meta_layouts` checks this. The failing shim uses graph/tensor interfaces, not the private tensor-map fields.
- D normal serving has no diagnostic interposer and `common/common.h:501` defaults `cb_eval` to null; scheduler callback partitioning is at `ggml/src/ggml-backend.cpp:2074-2093`. This failure does not establish a served-path defect. The exact offending node name was not recorded before abort.

Fix:
- `history-root.cpp:30-86` recognizes only VIEW/RESHAPE/PERMUTE/TRANSPOSE over non-Meta host roots. Pure host-view graphs return success after the existing synchronization/input-observation hook; there is no arithmetic or device work to perform.
- Mixed graphs with such a suffix retain every original node and append the completion view of the last other node, closing the planner boundary. CPU callbacks, Meta-backed host views, CPY and SET_ROWS are not bypassed. Graph UID is reset on augmentation; original graphs are unchanged.
- All original arms, engines, depths, repeats and readings remain. No engine source, sampling, queue execution, or frozen library changes.
- `fingerprint-test.cpp:59-138` covers traced/untraced mixed and host-only graphs, size changes, host-byte identity, input capture, write exclusion and AllReduce. The two mutant modes bypass the fix and must abort with this exact assertion. `validate_cpu.sh` runs both controls and accepts a fresh `HISTORY_CPU_OUT`.

Validation:
- M queue first line was `gpuq: IDLE` before validation. All commands hid CUDA; builds used nice 19, affinity 6-11,18-23, cpugate/buildgate and -j4. CPU model runs used one serial process, -t 8 and -c 4096.
- M `evidence/fingerprint-test.log`: `meta completion controls passed: full=32 bare=16 completed=32`; `meta host suffix controls passed: traced/untraced mixed/host-only graphs, result=32, host bytes equal`.
- M `evidence/host-suffix-mutant.log`: both `--host-suffix-mutant` and `--host-callback-mutant` report `expected SIGABRT and i_start assertion`.
- M `evidence/python-tests-final.log`: `Ran 10 tests`; `OK`. Existing parser controls use recorded GPU KLF bytes, recorded journal/device fixtures, and rejected trace/logit mutations.
- CPU replay, render and submit-check results: pending completion of the gated validation run.

Next GPU attempt:
- Tests the original stack9-p55 canary, then the original stack9-p55 ablations and stack10 arm, for request-order tensor/logit dependence with the repaired instrument. Preserves the cross-engine comparison; no arm substitution confounds it.
- Does not establish GPU speed, sampled-token identity, probabilistic proposal/rejection behavior, or the served four-row verify path. Callback partitioning and completion views still perturb execution. Native GPU validation remains the parent's next canary; stack9's separate subgraph-capacity defect remains a possible later failure.
- The supplied SP background reports were absent at their named paths. No GPU binary, queue submit, retry, network, or external file mutation was performed.
