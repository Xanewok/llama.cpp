# stack7 + package 55 build window and GPU items

Prepared on `topic/stack7-p55`. No compile, CPU model run, GPU query/execution, network operation or queue submission occurred. Production source is unchanged.

## Deliverables

Window: `codex/bw-stack7-p55.toml`, payload `codex/bw-stack7-p55.sh`. It is a non-preemptible filler with `host_sensitive = false`, priority 19. Output is `work/build-window-1007/stack7-p55-aN/` under the worktree collection. The existing `bw-run.sh` supplies the window marker and hides GPUs; only the deferred correctness commands restore GPU visibility, under `GPUQ_ACTIVE=1`.

| Item | Priority | Experiment | Rendered item | Identity |
|---|---:|---|---|---|
| `p55-depth-iq4` | 27 | `codex/experiments/p55-depth-iq4.toml.in` | `codex/items/p55-depth-iq4.gpuq.toml` | within-arm |
| `p55-np2-real-iq4` | 27 | `codex/experiments/p55-np2-real-iq4.toml.in` | `codex/items/p55-np2-real-iq4.gpuq.toml` | off, explicit arrival/batching exception |
| `p55-depth-j5q` | 15 | `codex/experiments/p55-depth-j5q.toml.in` | `codex/items/p55-depth-j5q.gpuq.toml` | within-arm |

The templates are committed. Rendering writes their `.toml` counterparts, generated items and immutable harness snapshots inside `codex/`; these generated files are ignored. The parent can copy the items after the window succeeds. Current items are drafts: the absent p55 freeze means its complete library inventory cannot yet be pinned. The window renders again using the actual frozen `SOURCE` and checks each resulting item without submission.

All plans are `up patch patch up`, comparing `frozen/stack7` with `frozen/stack7-p55`. Both arms use identical serving environments: chain/fold on for np1, off for np2; input staging and KV pad retained. The copied workload is k=5, not the historical k=3 attribution workload. Each run and canary starts with the pinned `bash work/aa-load/ccx01.sh` wrapper and has `host_sensitive = false`. J5q uses the stack7 template in this worktree because `harness/experiments/stack7-depth-j5q.toml` is absent.

## Window and canary gates

`codex/bw-stack7-p55.sh` (`preflight`) rejects a dirty tracked tree, an existing p55 freeze or the wrong branch/base. CUDA configuration follows `frozen/stack6/RECIPE.txt`: CUDA 13.2, NCCL off, architecture 120a, CCCL v3.4.3 from the existing local checkout, disconnected FetchContent and shared ccache. The task-specific stack6 recipe takes precedence over the generic architecture 120 example.

The private CMake wrapper `codex/p55-test/CMakeLists.txt` changes only the generated test source's `N_THREADS` to 8: `std::thread::hardware_concurrency()` ignores taskset affinity. All compiles are deferred through `nice -n 19 bin/buildgate`, `-j4`. No CPU/Mac stages are included. The test target is enabled in addition to the stack6 recipe's server target.

The window requires the build stamp and `DeviceTopK` count of 25, then `test-backend-ops -o MUL_MAT` and `-o MUL_MAT_ID` on each GPU through cpugate, nice and an eight-CPU affinity. SQL gates reject empty/skipped/failed/wrong-commit results and require the changed quant types at columns 5 through 8 (`codex/p55_checks.py:9`). Existing tail cases are at `tests/test-backend-ops.cpp:10692`. Kernel journals are scoped to correctness start and stop the window on a fault or unreadable journal. Freeze is reached only after correctness passes; `bin/wt freeze` supplies `$ORIGIN` RUNPATH and checksums.

Each measurement canary is one patch boot at depth 8192, capped at 600 s, with the source workload's size, overlap, native-token and engagement gates. `codex/gpu_p55.py:14` verifies frozen checksums and `SOURCE`, then matches `/proc/<exact PID>/exe` to the arm executable. It saves `binary-proof.json` with arm, source, executable and SHA256 and emits `frozen_binary_proof` in the item log. This proves which p55 binary ran without inventing a kernel engagement marker. Stack7's inherited runtime guards retain exact owned-process cleanup, scoped fault checks and boot-budget expiry.

## Identity decision and expected outcomes

Evidence: `harness/experiments/mmvq-pertype-depth-iq4.toml:24,63` and `mmvq-pertype-depth-j5q.toml:24,63` explicitly use within-arm identity because w1r4 changes the per-row K split. The kernel item `mmvq-pertype-ab` is a correctness/performance probe, not a serving identity proof; its recorded first canary timed out. The successful rerun is `mmvq-pertype-ab2`, with output `~/repos/mmvq-w2r4/codex/runs/pertype2/full-a1/`. `docs/ledger/both-hardware.md:27` records non-bitwise behavior and the prior depth/KLD evidence.

The np1 items therefore compare native tokens and draft counters across repeated boots of each arm. The np2 source already disables identity for arrival-dependent ubatches; its exception now also states that package 55 changes reduction order. Neither within-arm reproducibility nor tolerance-based CUDA correctness proves an equivalent sampling law. This is a timing/no-regression evaluation, not authorization to claim exactness or adopt the new serving binary.

G: IQ4 may retain the prior approximately 0.3-0.6 ms/round benefit on stack7; np2 transfer is unknown. G: J5q should show no material gain or regression. These are hypotheses, not measurements of this stack. The ledger's earlier IQ4 replication gives D deltas of -0.32/-0.31/-0.58 ms/round at its three depths; this work did not reproduce them.

G: each full boot takes 900 s; D: four boots give 3600 s, below 70% of the 5550 s full-run budget. D: total budget is 6300 s including the 600 s canary and cleanup. G: canary work takes 300 s or less, below 70% of its cap. G: the build window takes 4200 s; D: this is below 70% of its 7200 s budget. Actual build, correctness-suite and canary durations remain unmeasured.

## Local evidence and prerequisites

M PASS: `bash -n` for the new shell files; `python3 -m py_compile` for the new Python files; render and render-equality checks via `PYTHONPATH="$PWD/codex/render" python3 -m harness.mkitem`; `gpuq submit --check` reports `valid` for the window and all three drafts; `git diff --check`. Every command used `CUDA_VISIBLE_DEVICES=""`, `GIT_NO_LAZY_FETCH=1` and nice 19. Python caches stay in the ignored local build directory.

Rendering replays real SSE, counters, overlap, memory and kernel-journal artifacts through `gpu_stack7.fixture_checks()`, plus the recorded per-type CUDA SQL at `codex/p55_checks.py:6`. Negative controls reject changed native tokens, insufficient steady intervals, missing engagement, empty/failed SQL, wrong GPU/commit/op and missing quant coverage. Additional within-arm mutation testing uses the recorded `results/2026-10-05-stack6-depth-j5q/full-a1/1-s6/depth-8192.json`. Logs are under `codex/checks-p55/`. There are no new CPU performance numbers.

The pre-existing tracked modification to `codex/PROMPT.md` was preserved and excluded from the commit. The parent must resolve it before queueing the clean-tree window. No window/preflight payload was executed here. The private CUDA test configuration, live executable proof and real canaries remain parent-queue validation. After the window, copy the regenerated items, not today's draft snapshots.
