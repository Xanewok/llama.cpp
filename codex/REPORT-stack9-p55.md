# stack9-p55: one llama.cpp build for all four Swift serving profiles

Labels: M measured, D derived, G guess. No GPU, no CUDA binary, no `gpuq submit` (only `--check`). Branch `topic/stack9-p55`, base `cd776e104` (frozen/stack9).

## Source changes (package 55 only)

Cherry-picked the four source commits of `wt/stack7-p55` (not its codex/evidence commits), in order. No conflicts (M): git auto-merged `tests/test-backend-ops.cpp` and `tests/CMakeLists.txt`.

| new | original (stack7-p55) | subject |
|---|---|---|
| d61d26275 | 990cd60f3 | cuda : bind MMVQ launch dimensions to kernel parameters (new `ggml-cuda/mmvq-params.h`, `tests/test-mmvq-params.cpp`) |
| c7855eda7 | 2851d6ad7 | cuda : use one-row MMVQ blocks for partial tails |
| f94389ef7 | 07cf46670 | cuda : reject unavailable MMVQ parameter tables |
| 6f81b879c | 0302404cb | cuda : select Blackwell MMVQ launch parameters by quant type |

Files vs stack9: `ggml/src/ggml-cuda/{mmvq-params.h,mmvq.cu,mmvq.cuh}`, `tests/{CMakeLists.txt,test-backend-ops.cpp,test-mmvq-params.cpp}`. `mmvq*`, `test-mmvq-params.cpp`, `CMakeLists.txt` are byte-identical to `stack7-p55@0302404cb` (cmp, M); the `test-backend-ops.cpp` hunk (5 quant types x n 5..8 x m 5..7 MUL_MAT cases) is identical text. Stack9's own changes (gated_delta_net, llama-context/graph, speculative) do not touch these files, so stack9-p55 = stack9 + p55 exactly. `bw-stack9-p55.sh` preflight enforces that: `git diff --name-only cd776e104 HEAD -- . ':!codex'` must list exactly those six files.

Codex commits on top (harness/window only): 0ac388097 (Mac gate), 18da28f2e (window, items, renderer, mutant link fix), dcc13b114 (Mac evidence a2), then this report.

## Mac gate (M)

`bin/bw-mac wt/stack9-p55 codex/bw-stack9-p55-mac.sh stack9-p55` at 18da28f2e: **PASS rc 0**, `work/build-window-1007/stack9-p55-mac-a2` (copied to `codex/stack9-p55-evidence/`). Stages (s): 20 cpu-build 149, 36 mmvq-table-mutant 2, 21 unit-tests 541, 29 np2-chain-identity 299, 31 recorded-parser-controls 0. First run a1 (0ac388097) failed stage 36 only: the mutant did not link (`ggml_abort`); fixed by linking `libggml-base` and requiring rc 134 (SIGABRT), so a compile/link failure cannot pass as a caught mutant.

- Added to the stack9 payload `codex/bw-stack9-mac.sh`: build + run `test-mmvq-params` in stage 21 (`M: 1840 configurations passed; D: 1241 distinct MMVQ specializations; 612 CUDA specializations`, equal to FACTS.md) and stage 36 (negative control: drop IQ4_XS from the Blackwell n>=5 table in a copy of `mmvq-params.h`; the test aborts, rc 134, `mmvq-table-mutant.log`). `source_files` hash now covers `mmvq*`, `test-mmvq-params.cpp`, `tests/CMakeLists.txt`.
- **The CPU gate cannot exercise p55.** `mmvq.cu` is not compiled without CUDA. CPU coverage = the header-only table logic (`test-mmvq-params`) and that the merge builds and runs the stack9 CPU stages. Kernel correctness is the window's `test-backend-ops` MUL_MAT / MUL_MAT_ID on both GPUs.
- Stack9 stages 23, 24, 32-35 (np1 chain 56 min, fold-mseq 25 min, np2 old control, candidates, getter) were not rerun: each is far over the 10 min gate and tests sources p55 does not touch (D; the preflight diff allowlist enforces it). Their stack9 evidence stays in `codex/stack9-evidence/`.

## Window `bw-stack9-p55` (not submitted)

- `codex/bw-stack9-p55.toml` / `.sh`: filler, priority 20, preemptible=false, host_sensitive=false, est 4200 s (G), budget 7200 s (D: est < 0.7 x budget), GPUQ_ACTIVE=1.
- `gpuq submit --check codex/bw-stack9-p55.toml` -> `codex/bw-stack9-p55.toml: bw-stack9-p55: valid` (M). `bash codex/bw-stack9-p55.sh --preflight` -> `PASS preflight` (M; includes the Mac-evidence source hash and the diff allowlist). `bash -n`, `gpuq.shlint`, `py_compile`: pass.
- Stages: 40 cuda-build (same flags as bw-stack9: 120a, CUDA 13.2, NCCL OFF, CCCL 3.4.3, buildgate -j4, ccache; **deviation**: `-S . -DLLAMA_BUILD_TESTS=ON` and target `llama-server test-backend-ops`, as the stack7-p55 window froze it), 41 stamps (DeviceTopK == 25, `libggml-base` stamp, clean tree), 42 correctness (`test-backend-ops -o MUL_MAT` and `-o MUL_MAT_ID`, `-b CUDA0/CUDA1`, GPU UUIDs, cpugate + taskset 0-3,12-15, `p55_checks.py` quant x n=5..8 coverage and wrong-commit/skipped/failed rejection, kernel-journal Xid gate; MUL_MAT_ID added beyond the brief because p55 changes the shared MMVQ launch), 43 `wt freeze stack9-p55` + RECIPE.txt, 44 render and `gpuq submit --check` the four items into `codex/items/rendered/` (renderer `codex/render_s9p55_items.py`, ccx01 prefix, sibling pins, harness snapshot).
- Only stage 42 sees the GPUs (two test-backend-ops processes, sequential).

## Items (`codex/items/s9p55-*.toml.in`; rendered by the window after the freeze)

The rendered items do not exist until the window freezes `frozen/stack9-p55`. I dry-rendered all four against `work/s9p55-dry/frozen/stack9-p55` (symlink to frozen/stack9, SOURCE = cd776e104; a stand-in, never submit these) with the real renderer logic and `gpuq submit --check` (M): `s9p55-depth-iq4: valid`, `s9p55-np2-real-iq4: valid`, `s9p55-duo-soak-iq4: valid`, `s9p55-depth-j5q: valid` (`codex/items/dry/*-contract.log`, gitignored). Mac stage 31 also checks the templates (priority, canary plan, budget <= 0.7 x budget_s, regexes). Common: harness defaults (joint np2 start, `--cache-ram 0`), ABBA (2 boots per arm, interleaved), 8k/32k/90k, k 5, seed 1729, ignore_eos true, CCX0+1 via ccx01.sh, host_sensitive=false, canary on the stack9-p55 arm only. Served flags match `serve.sh command` for `swift15-iq4` / `swift15-iq4-duo` (-ub 512, -sm tensor 50,50, k 5), except the chat template path (harness convention `swift-leading-instructions-v1.jinja`, as every earlier item).

| item | prio | arms (first = baseline) | identity | expected (D) | falsifier |
|---|---|---|---|---|---|
| s9p55-depth-iq4 | 28 | s7p = frozen/stack7-p55 (6feab35f9) vs s9p = frozen/stack9-p55; np1, CHAIN=1 FOLD_C=1 | cross-arm | -0.9..-0.3 ms/round | delta > -0.1 or < -1.2 at any depth; any cross-arm token/counter difference (rc 6) |
| s9p55-np2-real-iq4 | 28 | s7 = frozen/stack7 (CHAIN=0 FOLD_C=0, today's duo) vs s9p (CHAIN=1 FOLD_C=1); np2 -ub 512 | off (arrival-dependent, p55 K split; reason in file) | -2.2..-0.4 ms/round | delta > -0.2 at 32k/90k or > +0.15 anywhere; canary lacks `mtp_fold_decode: engaged n_seq=2` / chain-fallback line |
| s9p55-duo-soak-iq4 | 27 | A = frozen/stack9-p55, -ub 512, 2 x 102400, CHAIN=1 FOLD_C=1 | n/a (campaign) | 8..8 tasks solved | < 8/8, server error/retry, runaway, min free VRAM < 1500 MiB (stack7 duo had 2167/2280) |
| s9p55-depth-j5q | 27 | s9 = frozen/stack9 (cd776e104) vs s9p; np1 J5q | within-arm | -0.2..0.0 ms/round | delta > +0.1 or < -0.35 at any depth; within-arm token difference |

Reasoning behind the expectations:
- (a) p55 is in both arms, so its gain cancels. What remains is stack8's candidate-path gain under the harness `ignore_eos` bias (J5q -0.55/-0.57/-0.81 on stack7f, harness-only: ~0 in serving, notes 10-08 04:48) plus stack9 np1 null (+0.23/-0.03/+0.03, 16:44). IQ4 rounds are shorter, so the range is wide (G on the IQ4 transfer).
- (a) identity: cross-arm tokens are expected identical (D). Evidence: stack7f vs stack8 cross-arm identity held on GPU (04:48), stack9 vs stack8 token-identical (16:44), candidate path exact on the Mac (cand identity + mutant, bw-stack9 a1) and under -sm tensor (01:16). The candidate path changes where top-k runs, not the sampled order. Both arms run the same p55 kernels, so the K-split change cannot separate them. If rc 6 fires, rerun within-arm and read ms/round.
- (b) J5q stack9 vs stack8 at np2 was -0.43/-0.94/-1.33 (fold -0.45/-0.73/-1.13); stack7 -> stack8 at np2 assumed ~0 in serving (not measured) plus the harness-only candidate term at np2 (unmeasured, G); IQ4 transfer of fold C is the open question. p55 expected inert at np2 (G): the 12-column verify goes to MMQ (MMVQ max 8); p55 on stack7 np2 was null (a3 -0.12/+0.15/-0.34 +- 0.2).
- (c) shape of `stack9-duo-soak-j5q` (8 tasks, 2 Codex agents, min_drafts 200) with IQ4 model and `-ub 512` (serve.sh: only j5q-duo keeps 128). One campaign, a soak, not an A/B.
- (d) p55 on stack7 J5q np1 was -0.18/-0.02/-0.07 (04:17); cross-arm tokens may differ (w1r4 changes the K split), judge ms/round only.

## Decision-relevant uncertainty

- Items (a) and (b) run IQ4_XS under stack9 for the first time (no IQ4 measurement of stack9 exists). A single `s9p55-duo-soak-iq4` pass is the only fit check at 2 x 102400 with fold C.
- The window deviates from bw-stack9 in `LLAMA_BUILD_TESTS=ON` and `-S .`: the frozen `llama-server` should match stack9's, but that is not proven bitwise (libs unchanged except mmvq objects expected, G). `test-backend-ops` lands in frozen/stack9-p55/build/bin and is hashed by freeze.
- Pins: items and window pin the stack9-p55 codex scripts. Do not edit `codex/*.py|*.sh`, the templates, or the p55 source files while the window or an item is queued.
- `work/s9p55-dry/` is a scratch stand-in (symlink + SOURCE copy); delete after the window freezes the real tree.
