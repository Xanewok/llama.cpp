# stack10 integration report (wave 2)

2026-10-09 03:05 CEST. Tree `~/repos/qwen38-eval-worktrees/wt/stack10`, branch `topic/stack10`, base `def4a4e20` (= frozen/stack10-cand2 source;
K5 already in). gdn4 is NOT in this stack (coordinator 10-09: moved to wave 3). Nothing here ran on a GPU; every CUDA claim is a window gate (G until then).

## 1. Result

- 11 components integrated, one series each, general commits first, one `local:` switch commit per component grouped after them
  (order p55, K7, K4, K6a, K6b, K2, K10, K9, K8(+L1), K11a, K11b; K9 precedes K8 so K8's launches use K9's `ggml_cuda_kernel_launch`).
- Mac CPU payload (`codex/stack10/bw-stack10-mac.sh`): 50 stages, all rc 0, 1812 s wall (M), at HEAD `0128ceadf`; no source defect, no compile error.
- Drop map `codex/DROPMAP.md` + recipe `codex/stack10-drop/`: all 11 single drops, 6 multi-component subsets and drop-all (tree == base) checked (M).
- Window `codex/stack10/bw-stack10.{toml,sh}` (`COMPONENTS=` subset) and six A/B items: written and validated on CPU only; not queued, not run.

## 2. Commits (def4a4e20..HEAD, oldest first; component, general or local)

| # | hash | component | kind | subject |
|---|---|---|---|---|
| 1 | `b28f11784` | p55 | general | cuda : bind MMVQ launch dimensions to kernel parameters |
| 2 | `311452e04` | p55 | general | cuda : use one-row MMVQ blocks for partial tails |
| 3 | `96217fbd5` | p55 | general | cuda : reject unavailable MMVQ parameter tables |
| 4 | `cc0bfd1b4` | p55 | general | cuda : select Blackwell MMVQ launch parameters by quant type |
| 5 | `20ac6ec57` | k7 | general | cuda : run paired RMS norm and scale chains in one launch |
| 6 | `386282191` | k4 | general | ggml: GGML_OP_SSM_CONV_RS, indexed conv-state read with snapshot writes (CPU, CUDA, Meta) |
| 7 | `6ac5f5897` | k4 | general | llama: read the recurrent conv state by id through ggml_ssm_conv_rs, gathered fallback |
| 8 | `30afc9758` | k6a | general | cuda : reuse shared matmul activations of up to 16 rows within a graph |
| 9 | `1376b053b` | k6b | general | cuda : fuse residual add with RMS normalization of up to 16 rows |
| 10 | `fac1254d2` | k2 | general | cuda : scan the flash-attention mask once per graph evaluation |
| 11 | `4bf1c2ea3` | k10 | general | cuda : batch top-k over rows in one launch set with a deterministic order |
| 12 | `cd72032e8` | k9 | general | cuda : launch the MMQ kernel and its stream-k fixup with PDL |
| 13 | `b33022516` | k9 | general | cuda : launch the MMQ q8_1 quantize kernel with PDL |
| 14 | `c15713fb1` | k9 | general | tests : add an MMQ producer/consumer chain with byte hashes and replay checks |
| 15 | `88fca1a7b` | k8 | general | cuda : shape-gated small-I MMQ tiles for 16 and 24 columns on Blackwell |
| 16 | `a66f33bc4` | k8 | general | tests : cover MMQ small-column tile selection |
| 17 | `498383fa1` | k8 | general | cuda : MMQ stream-k partitions in default-tile units for narrow tiles |
| 18 | `4c3d32429` | k8 | general | cuda : take SMALL_I MMQ tiles on stream-k launches too |
| 19 | `f13e859cd` | k8 | general | tests : cover MMQ small-I tiles on stream-k shapes |
| 20 | `c696e7d7f` | k11a | general | llama : skip getter synchronize() when no device work is pending |
| 21 | `e37639ef7` | k11b | general | meta : stage mirrored host inputs once and upload slowest link first |
| 22 | `cf55cb8cc` | k7 | local | local: K7 paired RMS norm and scale switch, mutants, engagement log |
| 23 | `509d4cf39` | k4 | local | local: K4 SSM_CONV_RS off switch, mutants, engagement log |
| 24 | `c22042231` | k6a | local | local: K6a shared-activation reuse switch, engagement log, stale-cache mutant |
| 25 | `b303083b7` | k6b | local | local: K6b residual add + RMS norm fusion switch, engagement log, numerical mutant |
| 26 | `7950ebbbc` | k2 | local | local: K2 shared KV-max scan off switch and negative controls |
| 27 | `7378321b9` | k10 | local | local: K10 top-k batching off switch, engagement log, negative-control mutants |
| 28 | `23abb0b21` | k9 | local | local: K9 MMQ PDL off switch, wait-removal mutants, engagement log |
| 29 | `dfe49db88` | k8 | local | local: K8/L1 MMQ tile A/B switch, auto-drop and stream-k partition mutants, mmq-hash tool |
| 30 | `886be9958` | k11a | local | local: K11a getter sync elision switch, debug check, mutants |
| 31 | `3cdbc3dfb` | k11b | local | local: K11b input staging A/B controls and negative-control mutants |
| 32 | `027931d98` | - | codex | codex : component tools copied from the p55, K2/K4/K6/K7/K8(+L1)/K9/K10/K11a/K11b trees |
| 33 | `538499d47` | - | codex | codex : stack10 drop map and checked drop recipe |
| 34 | `0dd7c15df` | - | codex | codex: combined stack10 Mac CPU payload and per-component stage libraries |
| 35 | `651eab15b` | - | codex | codex: stack10 Mac payload reads the component filter from the OUT_NAME |
| 36 | `c20793622` | - | codex | codex: stack10 Mac payload phase and component filter words |
| 37 | `60bd39445` | - | codex | codex: p55 item templates where gpu_stack9.py globs them |
| 38 | `e7b53f589` | - | codex | codex: p55 helpers back at the codex root where gpu_stack9.py reads them |
| 39 | `8ef47987c` | - | codex | codex: stack10 Mac payload without gdn4; K6 split into k6a and k6b; COMPONENTS filter |
| 40 | `9fabde78b` | - | codex | codex: stack10 combined CUDA window, gate extraction, stock-MMQ reference, A/B item templates and renderer (not run) |
| 41 | `5344fcce9` | - | codex | codex: stack10 window accepts topic/stack10-* branches |
| 42 | `ca07df958` | - | codex | codex: stack10 window keeps the test-only gdn4 timing commit K8 depends on |
| 43 | `3e75a7cde` | - | codex | codex: stack10 item fixtures compare normalized wave-1 fields |
| 44 | `9ffb429a3` | - | codex | codex: stack10 Mac payload runs k4 op stages; full-run evidence at ade74ab90 |
| 45 | `10eb69f82` | - | codex | codex: stack10 window rejects both gdn4 commits, reads mac-evidence, toml without gated_delta_net.cu |
| 46 | `0128ceadf` | - | codex | codex: stack10 stock tree reverts must not run git apply inside the worktrees repo |
| 47 | `06ebc1d4f` | - | codex | codex: stack10 full Mac run evidence at 0128ceadf |

`local:` commits carry every env switch, engagement log and mutant (`GGML_CUDA_*` etc.); the general commits contain no env var. The
engagement-chain edits of `tests/test-backend-ops.cpp` live in the general commits.

## 3. Conflicts and source changes beyond a clean cherry-pick

Detected by comparing patch-ids of each applied commit with its source (`cherry picked from`); identical patch-ids are not listed.

| commit | conflict or adaptation | resolution |
|---|---|---|
| K4 `386282191` | `tests/test-backend-ops.cpp`: the `test_backend` return chain is edited by every identity-check component | kept all checks; chain order K6a, slice, K6b, alpha, K4 conv, K7, K2 (the canonical text is rebuilt by `resolve-drop-conflicts.py --normalize` on drops) |
| K6a `30afc9758`, K6b `1376b053b` | same chain; adjacent hunks in `ggml-cuda.cu` / `common.cuh` (q8 block next to K2's scan) | both added; `ggml_cuda_q8_cache` and `ggml_cuda_fattn_kv_max_cache` locals are declared q8 first so the pool frees in LIFO order (documented in the code) |
| K2 `fac1254d2` | `common.cuh` forward declaration, test chain | merged; K2's mask scan and K6a's cache are separate `if (... && stream_ctx.concurrent_events.empty())` blocks. The K2 local first coupled the q8 and kv_max switches: the general commit was restructured into independent ifs (fixup + autosquash) and the locals replayed. Invalidate calls for both caches sit in the fused-group loop and the per-node loop; both pointers reset after the node loop |
| L1/K8 `88fca1a7b`, `498383fa1`, `4c3d32429` | `mmq.cuh` launch lines: K9 (applied before) turned `<<<>>>` launches into `ggml_cuda_kernel_launch` / `ggml_cuda_mmq_kernel_launch` | ported K8's `tiles` template argument and `launch_mul_mat_q<type,J,fallback,tiles,prec_src1>` onto K9's launch form |
| K8 `use_small_i` (incidental, L1/K8 general commits) | general form `return args.nrows_x % config_default.I == 0;` after the smem check | removed the unused `plan`/`nsm` locals (would warn under -Werror) |
| K8 prerequisite L1 | L1 was in neither stack9 nor stack10-cand although K8 builds on it | L1's 2 commits are part of the K8 component (see DROPMAP: one drop removes both) |
| K9 local `23abb0b21` | `mmq.cuh` 8 launch lines | `tiles` argument added |
| K8 local `dfe49db88` | the squash of 275ad4038, 09c80fd7f, fef62ffd3 (topic/l1-k8) landed in the wrong function (`max_rows` undeclared) and its cold-weight perf hunk assumed the gdn4-only `perf_nodes()` | tile switch block moved into `mmq_tiles_hash`; `GGML_TEST_PERF_COLD_MB` rotation rewritten against the original perf loop (`node_slack`, each rotated copy re-added in the duplicate loop). Syntax check only (`check-drop.sh`); the perf mode is not a gate |
| K11b `e37639ef7` | `tests/CMakeLists.txt` | both tests (test-fattn-kv-max-cache, test-meta-input-stage) registered |
| p55, K7, K9 generals, K10, K11a | none (identical patch-id to source) | |
| tool/payload copies | shared files `codex/gpu_stack9.py`, `codex/bw-stack9-mac.sh`, `codex/bw-np2-driver.cpp` carry several components' hunks | applied by patch; p55 helpers kept at the `codex/` root (gpu_stack9.py globs `codex/items/s9p55-*.toml.in`, `codex/bw-stack9-p55.toml`) |

Incidental fixes found by the checks (all in codex/, none in ggml/src): `stock_tree.py` ran `git apply -R` inside `work/` (inside the worktrees
repo), which silently skips paths with rc 0, so the stock-MMQ reference would still have contained K8/K9 and the K8/K9 gates would have compared
a build against itself (fixed with `GIT_CEILING_DIRECTORIES`, `0128ceadf`); `gpu_s10.py --fixtures` had four fixture bugs (`ade74ab90`);
`k4` was missing from the Mac `CHEAP_ORDER` (`9ffb429a3`).

## 4. Drop map

`codex/DROPMAP.md` (component to hashes, recipe, checks). Summary: `codex/stack10-drop/drop-component.sh k7,k4` on a scratch branch drops any
subset (k6a implies k6b); `check-drop.sh` is the compile check of the host-compilable files. M: 11 single drops, `k7,k4` `k2,k10` `k8,k9`
`p55,k8` `k6a,k11b` `k4,k6b,k11a`, and all 11 dropped: all complete and pass `check-drop.sh`; drop-all tree equals `def4a4e20` outside
`codex/` except 3 lines (`alpha_ok` form of the identity chain). `ggml-cuda.cu` of the k6a drop diffed by eye: removes exactly the q8 block.
CUDA files are not compiled on this host: leftover CUDA references would be found by the window's CUDA build (`COMPONENTS=` then selects the gates).
The window does not build a dropped tree by itself: drop on a branch, then run the window on that branch (`topic/stack10-*` is accepted).

## 5. Exactness (cross-arm token identity vs frozen/stack9-p55)

All components are designed exact; CUDA exactness is unproven until the window gates run (nothing compiled). The np1 depth items assert cross-arm identity.
Exceptions and caveats: K10 output order is deterministic (key desc, column asc) where CUB's was unspecified, so byte identity to CUB is impossible by
definition; tokens unchanged because both consumers use the indices via `get_rows` with the same index tensor and the serving chain falls back to the full
path on any tie among the k+1 best (argument, G; the window's canonical TOP_K cases and the depth items test it). K4: FMA contraction identity vs the old
fused `ssm_conv_state_f32` is assumed (window byte gate decides). K8: bitwise equality to default tiles is a Mac model result (5256 shapes, 0 differ) plus the
window hash gate; the L1 11520-row inexactness is removed. K6a/K6b, K7, K2, K9, K11a, K11b, p55: exact by construction (same per-row arithmetic, same
quantization cache, scheduling only), each with a byte-identity gate in the window.

## 6. Mac stages (M, wall s; `bw-mac` run `stack10-final-mac-a1`, HEAD `0128ceadf`, source_sha256 `457696a066e1...`, tree clean)

Evidence `codex/stack10/mac-evidence/`. Total 1812 s; longest gates k11a mutants 409 s, k11a identity 379 s (< 10 min). Mac load 1.2 at start; no ninja/llama.

| code | stage | wall s (M) |
|---|---|---|
| 10 | build-cpu-build | 49 |
| 140 | ops-mmvq-mul-mat-small-n | 3 |
| 141 | ops-mul-mat-id-small-n | 0 |
| 142 | ops-mul-mat-id-shared-ops | 1 |
| 30 | p55-mmvq-table-test | 0 |
| 31 | p55-mmvq-table-mutant | 1 |
| 33 | p55-tp-sampling | 16 |
| 40 | k7-alias-guard-mutants | 1 |
| 41 | k7-pair-identity | 0 |
| 42 | k7-neighbour-op-tests | 0 |
| 50 | k4-ssm-conv-rs-ops-identity | 1 |
| 51 | k4-ssm-conv-rs-ops-mutants | 1 |
| 60 | k6a-q8-cache-contract | 1 |
| 61 | k6a-q8-cache-contract-mutants | 1 |
| 62 | k6a-mul-mat-shared-identity | 2 |
| 65 | k6b-add-rms-norm-matcher | 1 |
| 66 | k6b-add-rms-norm-matcher-mutants | 2 |
| 67 | k6b-add-rms-norm-identity | 1 |
| 70 | k2-kv-max-cache-unit | 1 |
| 71 | k2-shared-mask-identity | 1 |
| 72 | k2-kv-max-unit-mutants | 6 |
| 73 | k2-shared-mask-identity-mutants | 11 |
| 80 | k10-emu-build | 1 |
| 81 | k10-emu-kernels | 88 |
| 82 | k10-emu-mutants | 23 |
| 83 | k10-topk-unit | 93 |
| 84 | k10-topk-cpu-ops | 1 |
| 85 | k10-canonical-control | 1 |
| 90 | k9-mmq-chain-harness | 4 |
| 100 | k8-stream-k-partition-model | 2 |
| 101 | k8-mul-mat-cases | 19 |
| 102 | k8-mmq-hash-plumbing | 17 |
| 120 | k11b-meta-input-stage-unit | 0 |
| 121 | k11b-meta-glue-mutant | 0 |
| 122 | k11b-cuda-sim | 3 |
| 123 | k11b-cuda-sim-mutants | 8 |
| 20 | base-payload-lint | 0 |
| 21 | base-sampling-and-alloc-units | 1 |
| 22 | base-softplus-and-fa-q8-ops | 2 |
| 23 | base-recorded-parser-controls | 1 |
| 32 | p55-recurrent-state-rollback | 82 |
| 34 | p55-recorded-parser-controls-stack9 | 1 |
| 35 | p55-np2-chain-identity | 103 |
| 43 | k7-real-graph-adjacency | 1 |
| 52 | k4-ssm-conv-rs-model-gate | 85 |
| 53 | k4-ssm-conv-rs-np2-driver | 126 |
| 24 | base-rotation-identity | 262 |
| 110 | k11a-sync-elide-identity | 379 |
| 111 | k11a-sync-elide-mutants | 409 |

## 7. Negative controls (all killed, M, Mac)

Each message is the failure that makes the mutant stage pass (stage exits nonzero for the mutant run):
- p55: MMVQ table mutant rc 134 (stage 31).
- K7: alias mutants 1, 2, 4, 8 each on its own `matcher accepted: ...` (stage 40).
- K4: ops mutants 1, 2: `FAIL SSM_CONV_RS byte identity` (stage 51); model mutants 1, 2: `logits differ` (stages 52, 53).
- K6a: stale-cache and bound mutants rc 1 (stage 61). K6b: cap, norm, resid mutants rc 1 (stage 66).
- K2: 10 unit mutants, each a `FAIL tests/test-fattn-kv-max-cache.cpp:NN`; 2 identity mutants `FAIL FLASH_ATTN_EXT shared mask` (stages 72, 73).
- K10: emulator mutants 1-6 each `FAIL narrow-...`; canonical control: CPU order fails `canonical=1` cases (stages 82, 85).
- K9: corrupt control `mmq-chain: FAIL 2 recomputes differ` (stage 90). No mutant exists for the wait in `quantize_mmq_q8_1` (byte identity only).
- K8: sim fixup mix-up caught 3/3 (stage 100; the CUDA SK mutant `GGML_CUDA_MMQ_SK_MUTANT` is a window gate).
- K11a: mutants 1-3 caught by tokens or the debug check in all 3 combos (stage 111). K11b: glue mutant, no-slot-wait, rank-0-only wait, ascending order, shifted source (stages 121, 123).
- Base: rotation mutant `rotation matrix bytes differ from the Hadamard reference` (stage 24).
- Window negative controls (not run): SASS gate fails on `frozen/stack10-cand` libs; stock-MMQ tree must differ from HEAD in the 6 MMQ files; K8/K9 CUDA mutants; K6 compute-sanitizer memcheck; k11a sync-count off/on.
- Drop recipe: `check-drop.sh` fails on leftover conflict markers and undeclared identifiers (caught k7/k4/k6a before the normalize and dependency rules, M).

## 8. Window

`codex/stack10/bw-stack10.toml` (generated by `mk_window_toml.py`) and `bw-stack10.sh`. Filler, priority 19, `host_sensitive=false`, `preemptible=false`,
est_s 34800 (G, sum of the component windows), budget_s 69600 (D, 2x), 124 pins (`check_pins.py`: none written by the window). GPU UUIDs and recipe as
`bw-stack10-cand`: 120a, CUDA 13.2 on PATH, NCCL OFF, CCCL 3.4.3, DeviceTopK count 25 read whole. `COMPONENTS=` (default all 11) selects the gates and the canaries; build
the dropped branch first (section 4). Entry needs `GPUQ_ITEM=bw-stack10`, `GPUQ_BUILD_WINDOW=1`, OUT `codex/window/stack10-a[0-9]*`.

Stages (exit codes: 3 hazard, 4 transient, 5 insufficient_data, 6 integrity, 7 timeout, 8 deterministic; gates are collected, worst code wins, Xid stops at once):
40 cuda-build, 41 cuda-stamps, 42 stock-mmq-build (HEAD with the 7 K8/K9 ggml commits reverse-applied, only if k8 or k9), 50 base-ops, 52 k7, 53 k4, 55 k6a,
56 k6b, 57 k6-memcheck, 58 k2, 59 k10, 60 k9, 61 k8, 62 k11b, 64 fafix-sass (instances == stack9; stack10-cand lib is the negative control), 65 fafix-identity,
66 full-suite (`test-backend-ops` both GPUs, p55-style flake handling), 63 k11a-sync-count (last, only if no gate failed), 70 freeze (`bin/wt freeze` as frozen/stack10),
71 render-and-check-items. Component gates call the component's own script functions (`shfn.py` extracts them by name; table `gates_table.py`).

Checks run on the CPU (M): `env -i HOME=$HOME PATH=/usr/local/bin:/usr/bin:/bin bash codex/stack10/bw-stack10.sh --preflight` -> `bw-stack10: PASS preflight` (also with
`COMPONENTS="k7 k4"`); `check_pins.py` PASS; `gpuq submit --check codex/stack10/bw-stack10.toml` -> `bw-stack10: valid`; `bash -n` + `python3 -m gpuq.shlint` on all 21 .sh.
Preflight also asserts that neither gdn4 commit is in HEAD and that the Mac evidence hashes HEAD's `src ggml tests common`.

## 9. Items (templates `codex/stack10/items/s10-*.toml.in`; rendered by window stage 71 / `render_stack10_items.py`; dry renders in `work/s10-dry-a2/items-full/`, all six `gpuq submit --check` valid)

Arms `s9p` (frozen/stack9-p55, wave-1 serving) vs `s10` (frozen/stack10), interleaved, harness defaults (`--cache-ram 0`, joint np2 start), run and canary wrapped in
`bash ~/repos/qwen38-eval-worktrees/work/aa-load/ccx01.sh`, candidate env == base env (no switches), canary at `-lv 5` plus canary log envs.

| item | prio | est_s | budget_s | boots | note |
|---|---|---|---|---|---|
| s10-depth-j5q | 26 | 4200 | 6300 | 2+2 | np1, 8k/32k/90k, cross-arm token identity |
| s10-depth-iq4 | 27 | 4200 | 6300 | 2+2 | same |
| s10-np2-real-j5q | 26 | 6300 | 9000 | 3+3 | round-trace on (per-phase STEP/SAMP/KV/FWD deltas) |
| s10-np2-real-iq4 | 27 | 4200 | 6300 | 2+2 | |
| s10-duo-soak-j5q | 28 | 6000 | 15300 | 2+2 | port 8081; VRAM falsifier below |
| s10-duo-soak-iq4 | 28 | 6000 | 15300 | 2+2 | |

(The 3+3 boots are everywhere in the J5q np2 item, not per depth; per-depth counts were not available in the harness.)
Engagement canaries (strings in `engagement.py`, selftest checks each literal exists in source): base `persistent rotations backend=`, `alpha_gate_fusion: device=`; k7 `rms_norm_scale_pair_fusion: device=`;
k6a `q8_reuse: device=`; k6b `add_rms_norm: device=`; k2 `fattn_kv_max: device=`; k4 `ssm_conv_rs engaged:` (`GGML_SSM_CONV_RS_LOG`); k10 `top_k_batched: engaged` (`GGML_CUDA_TOPK_LOG`);
k9 `mmq_pdl: kernel=mmq|quantize ... pdl=1` (`GGML_CUDA_MMQ_PDL_LOG`); k8 `mmq_tiles: ... small_i=1` (`GGML_CUDA_MMQ_TILES_LOG`); k11a `sync_elide: debug check engaged` and no `skipped sync waited`
(`LLAMA_SYNC_ELIDE_DEBUG=2000`, weak); k11b boot line `one pinned copy per input, uploads slowest host link first`. No observable line: p55 (proof = full-suite MUL_MAT gate);
k8 does not engage at np1 (6 columns) so only the np2 item proves it. Some K7/K6a/K6b/K2/alpha-gate patterns may not appear in Swift graphs (G): a missing line then means "not engaged", not "broken".

## 10. Expected numbers and falsifiers

Component estimates, np2 8k, ms/round (from each component's report): K11a D -0.20 (G -0.12..-0.20), K4 D -0.30..-0.35, K11b D -0.20 (G -0.10..-0.20), K6a+K6b D ceiling -0.30 (G -0.09..-0.30),
K10 G -0.10..-0.25, K9 G -0.05..-0.15, K7 D -0.03..-0.06, K8 0..-0.55 (parity possible), K2 no own estimate. Sum of the D point estimates without K8 about -1.3 (D); with K8 best case about -1.9.
Combined np2 8k expectation: 0.5x the sum, about **-0.65 ms/round (G)**; kernel-level D estimates realise 0.34-0.8x, i.e. -0.45..-1.1; the item templates state -1.0..-0.3 (G). np1 smaller: -0.3..0.0 (G; K9's PDL win is already in at np1).
Depth-flat for K4, K7, K9, K10, K11a (counts do not depend on KV); K11b about 0.75x at 90k; IQ4 prefill unchanged.
- Falsifier (brief): combined np2 gain < 50 % of the sum of the components that passed their own gates -> interaction; bisect with `codex/DROPMAP.md` (drop half, rebuild, repeat).
- Identity: any cross-arm token difference in np1 depth falsifies exactness (D) -> bisect by drop map; K10 first (tie order).
- Engagement: a component whose canary line is absent while its switch is default-on is not credited with any gain.
- VRAM (relative, per profile): J5q duo min free must not fall below the stack9-p55 base minus 100 MiB at each depth (stack9 had 367/480 MiB free); the same relative rule for IQ4 duo against its own base arm.
  K6a's activation cache, K2's and K10's scratch add VRAM (G); if the J5q duo falsifier trips, bisect by dropping `k6a,k6b`, then `k10`, `k2`.
- Per-component falsifiers are in `codex/<component>/REPORT-*.md` (K4: < 0.1 ms with `ssm_conv_rs engaged`; K11a: StreamSync > 60/round; K10: `auto_over_off` > 0.6 at 12 rows; K8: `auto - default` not below A/A spread on 3072/5120/6144 x 12/24).

## 11. State of the repository

- Real branch `topic/stack10` history was rewritten once after the coordinator dropped gdn4 (backup branch `backup/stack10-pre-nogdn` = pre-rewrite tip `53b980155`, still containing gdn4).
- Scratch refs in `wt/stack10-dropdry` (my own): `scratch/*`; stand-ins by the window agent: `wt/stack10-pf1` (obsolete), `wt/stack10-pf2`, `work/s10-dry-a1`, `work/s10-dry-a2`, `work/s10-stock-chk-a1`, `work/s10-stock-chk-a2`. None deleted; each has a PURPOSE file; removal needs the owner's go-ahead.
- Not done, needs a GPU or a build: CUDA build, `sass_gate.py` on the frozen libs, every component's CUDA gates, the full suite, the freeze. First CUDA compile of K2, K4, K6a/b, K7, K8, K9, K10, K11b code happens in window stage 40.
- K2 has no own gpuq window result; its est_s (2400) is a guess (G).
- The 5-minute Mac load average spiked to 133 during one earlier cheap run (cause unknown; 1-min load was 1.2-4 at each gate start).
