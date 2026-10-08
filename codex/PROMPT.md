# Brief for Codex (gpt-6.1-sol) working on the Swift 1.5 / 2x RTX 5060 Ti llama.cpp stack

Project: serve UkisAI Swift 1.5 (Qwen3.8-27B hybrid GDN + full attention, MTP head) on 2x RTX 5060 Ti 16 GB with
`-sm tensor` (meta backend, internal AllReduce, NCCL off), speculative MTP k=3, probabilistic draft + rejection verify.
Base of all work: fork commit 68d99c67d (branch `stack-resync-fmq/2026-09-30` on Xanewok/llama.cpp).
Background reports (read-only, read them before designing): `SP/gpu-sampling/REPORT.md`, `SP/gpu-sampling/items/96b-attrib.md`,
`SP/patchA/REPORT.md` (sections 1 and 6 break down the host time per MTP round), `SP/codex-reviews/patchA.txt`.
SP = /home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad

The MTP round (stack, `-sm tensor`, measured, trimmed steady rounds): V (verify, 4 rows) -> C (catch-up) -> D1 -> D2 -> D3,
~34.5 ms. Host gaps with both GPUs idle ~3.3 ms/round: C->D1 1.22, V->C 1.09, each D gap ~0.24 (0.15 of it ~14 small H2D input
copies + per-GPU syncs). Every host ms is exposed (AllReduce pool of 2 stops the host from running ahead).

## Hard rules (no exceptions)
1. Never run anything on a GPU. Keep `CUDA_VISIBLE_DEVICES=""` for every command. Do not run llama-server, llama-cli,
   test-backend-ops, compute-sanitizer, nvidia-smi or any CUDA binary. Building CUDA code with nvcc is allowed.
2. Never start a process named `llama-server` (the benchmark harness checks that name). Rename a test server binary if needed.
3. CPU work under `nice -n 19` with at most 8 threads (`-j8`, `-t 8`). Do not compile or run CPU tests while a gpuq depth/perf A/B runs (`gpuq status --brief` shows `RUN` with `depth` or `perf` in the id): host load fakes A/B results; wait for it to end, never bypass a gate. Wrap every compile in `~/repos/qwen38-eval-worktrees/bin/buildgate` with `-j4` (e.g. `~/repos/qwen38-eval-worktrees/bin/buildgate cmake --build build -j4`): other agents build in parallel; it waits for free RAM. ccache is shared across worktrees (keep `GGML_CCACHE=ON`).
4. Write only inside your worktree. Never delete or modify files outside it. No `rm -rf` outside `build*/` dirs of your worktree.
5. No network: no git push, no GitHub/API calls, no posting anywhere, no downloads.
6. Commit to your topic branch only (small, reviewable commits, messages in llama.cpp style, no AI attribution lines).
   Never rewrite other branches.
7. Exactness: any change to sampling must keep the output distribution exact (bit-identical tokens for the same seed, or a
   proven-equivalent law). Add a negative control that fails when the change is broken.
8. Builds: `cmake -B build-cpu -DGGML_CUDA=OFF` for tests; CUDA build `-DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2
   -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120`. CPU test model: Qwen3.5-4B MTP GGUF, see `SP/patchA/run_cpu.sh`.
9. Label every number: M measured, D derived, G guess. Do not claim GPU speed; propose a GPU A/B script instead.

## Deliverables
- Commits on your topic branch.
- `codex/REPORT.md` in your worktree: conclusion, design, evidence (file:line, tests, commands), CPU numbers, risks, open items.
- `codex/gpu_ab.sh`: a GPU A/B script for the parent to queue (NOT run by you): same frozen-binary pattern as
  `SP/queue/done/99-patchA-graph-cache-ab.sh` (up vs patch, interleaved boots, Xid check, exact-PID stop, budget).
- Final message: only decision-relevant results (conclusion, evidence, failures, uncertainty).

## GPU scripts for the parent (gpuq items)
Any script the parent will queue on the GPUs must satisfy `/home/xanewok/repos/qwen38-eval/orchestration/gpuq/HARNESS-CONTRACT.md`.
Test every parser against a real recorded artifact from `/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/` (or a
run dir the task names), not only CPU-fixture output. Provide a canary mode (one boot, smallest workload that hits every parser and gate, ≤ 10 min).

- Upstream packages live only in `~/repos/qwen38-eval/patches/upstream/` (index `UPSTREAM.md`). Never create a second package tree. Put package files you author under `codex/upstream/<engine>/<NN>-<slug>/` in your own worktree; the parent merges them. `~/repos/upstream-patches` is a read-only archive.

## Where you work (project rule)
- The parent names your worktree; it lives under `~/repos/qwen38-eval-worktrees/wt/` (created by `bin/wt new`). Read
  `~/repos/qwen38-eval-worktrees/AGENTS.md` and `~/repos/qwen38-eval/AGENTS.md`. Never create `~/repos/<name>` dirs or new
  worktrees yourself; never `git gc|prune|worktree prune` (18 repos borrow `~/repos/llama.cpp` objects).
- Frozen builds: `~/repos/qwen38-eval-worktrees/bin/wt freeze` (writes `frozen/<name>/build/bin`, RUNPATH `$ORIGIN`).

**Commit messages (owner rule, overrides the repository's AGENTS.md "Assisted-by" instruction):** never add `Assisted-by:`, `Co-authored-by:` or any other AI/assistant trailer to a commit message. The owner writes the upstream commit messages and the PR disclosure themselves; local commits carry no attribution.

**CPU model runs (RAM rule):** every CPU run of a model (tests, servers, tools) passes an explicit small context (`-c 4096` unless the task needs more; `n_ctx = 0` means the trained context, 262144 for Qwen3.5 → ~8 GB KV per context). At most one CPU model process at a time, started only while MemAvailable ≥ 10 GiB.

**Heavy CPU work:** start CPU model runs, OMP tests and other multi-core jobs through `~/repos/qwen38-eval-worktrees/bin/cpugate CMD...` (waits while a gpuq measure item runs: host load fakes GPU A/B regressions).

# Task stack10-cand: HQ-gap code candidates on stack9, window ready to queue (Codex write mode)

Worktree (yours): `~/repos/qwen38-eval-worktrees/wt/stack10-cand`, branch `topic/stack10-cand`, base `topic/stack9` cd776e104 (= `frozen/stack9`). `GIT_NO_LAZY_FETCH=1` on every git command; never gc/prune.

## Read first
- `~/repos/qwen38-eval/docs/proposals/hqgap8-experiments.md` (entries K1–K7 and the "Review outcome" section: it overrides the table where they differ) and `hqgap8-experiments-codex-review.md`.
- `~/repos/attrib-pack/runs/hqgap8-causes/{A,D}-REPORT.md` (evidence for K1–K5, K7).
- `wt/stack9/codex/REPORT-stack9.md` and its window (`codex/bw-stack9.sh`, `.toml`, `render_stack9_items.py`): reuse that structure.
- `~/repos/qwen38-eval/patches/upstream/llama.cpp/54-grouped-q8-one-row-fa/` (series, EVIDENCE.md).

## Content (one commit per component, general upstreamable form: backend-portable, default-on only where exact, exact automatic fallback, multi-sequence, no env var; any A/B switch only in a separate `local:` commit on top)
1. **pkg 54** grouped one-row q8_0 FA (apply the series; resolve conflicts in general form). Enables K1 (q8_0 draft KV) as a flags-only A/B: `--spec-draft-type-k/v f16` vs `q8_0` in the same binary.
2. **K4** indexed conv-state read for `GGML_OP_SSM_CONV`: separate state and token inputs, indexed state reads by sequence id, snapshot writes, replacing the `build_rs` conv gather + concat (`src/models/delta-net-base.cpp:463-520`, `ggml/src/ggml-cuda/ssm-conv.cu:250-260`); same pattern as stack9's indexed GDN getter (`cfce3d732`). CPU and CUDA paths; gathered path stays as fallback. Tests: accepted-slot rollback, overlapping reads/writes, extra cached states, swapped/non-contiguous seq ids, unsupported-backend fallback; byte-identical outputs and states vs the gathered path.
3. **K3** persistent device-resident Hadamard rotation tensors (`k_rot` and `v_rot`): upload once per allocation, not per eval (`src/llama-kv-cache.cpp:1951-1957`), with device/backend ownership and allocation fallback.
4. **K2** FA mask → KV-max scan shared across layers that use the same mask within one evaluation and device (`ggml/src/ggml-cuda/fattn-common.cuh:1163-1177`; masks `src/llama-graph.cpp:2893/3010`): key on mask tensor data/geometry/strides/tile/KV extent; must stay correct under CUDA graph replay (rescan when the mask changes) and capture-safe buffer lifetime.
5. **K5** fuse only the alpha gate chain ADD(dt_bias) → SOFTPLUS → MUL(ssm_a, broadcast) in CUDA (`src/models/qwen35.cpp:488-492`; the shape guard at `ggml/src/ggml-cuda/ggml-cuda.cu:3674` blocks it today). Exact: same per-element op order and rounding as the unfused ops; prove byte identity in `test-backend-ops`.
6. **K7** q/k RMS-norm + scale (`src/models/models.h:15-17`) as one paired launch for q and k with the same reduction order; byte-identical; no fold into GDN.
Skip any component you cannot make exact or general within the limits; say why in the report rather than forcing it.

## Window (do not submit; the parent submits)
Reuse `wt/stack9/codex/` window structure. CPU stages on the Mac via `~/repos/qwen38-eval-worktrees/bin/bw-mac TREE PAYLOAD NAME` (CPU build `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod`, unit tests and new op tests; Mac model runs ≤ 4 threads, ≥ 4 GiB free gate, ≤ 10 min per gate then fail). Local gpuq window manifest (filler, `host_sensitive = false`, `preemptible = false`, `priority = 19`, `CUDA_VISIBLE_DEVICES` = the two GPU UUIDs from `wt/stack9/codex/bw-stack9.toml`): CUDA build with the frozen/stack8 recipe (`-DCMAKE_CUDA_ARCHITECTURES=120a`, CUDA 13.2, NCCL OFF, CCCL 3.4.3; `strings libggml-cuda.so | grep -c DeviceTopK` read whole, never `| grep -q` under pipefail), **build `test-backend-ops` too and run the new/changed op cases on CUDA inside the window** (it may use the GPU: the window is a gpuq item), `bin/wt freeze` as `frozen/stack10-cand`, item render + `gpuq submit --check` (never submit). Run `python3 -m gpuq.shlint` (in `~/repos/gpuq`) on every `.sh`.

GPU items (shape: `wt/stack9/codex/items/`; explicit `priority`; each run/canary cmd prefixed with `bash ~/repos/qwen38-eval-worktrees/work/aa-load/ccx01.sh`, wrapper pinned, `host_sensitive = false`; harness defaults give `--cache-ram 0` and joint np2 start — do not hard-code them; ≥ 2 boots per arm, interleaved):
  a. `s10c-np2-real-j5q` priority 27: frozen/stack9 vs frozen/stack10-cand, np2 real prompts, served `swift15-j5q-duo` flags, 8k/32k/90k; plus drop-one arms via the `local:` switches if you add them (each switch one component).
  b. `s10c-q8dkv-np2-real-j5q` priority 26 (K1): frozen/stack10-cand, draft KV f16 vs q8_0 (flags only), np2 real prompts, 8k/32k/90k, ABBA; report tokens/round and acceptance per arm.
  c. `s10c-depth-j5q` np1 priority 26: stack9 vs stack10-cand, cross-arm identity expected (every component in this binary is exact; K1 is not part of this item).
Canaries prove engagement for each component (debug log lines at `-lv 5` or trace counts).

## Resources and limits
Sandbox writes only inside your worktree: `BW_MAC_OUTPUT_BASE=$PWD/build-bw-mac-results`; window manifest and rendered items under `codex/`. All compiles and model runs on the Mac (network only for ssh/scp to mini-builder; never push or post). On this host: edits, git, `bash -n`, `python3 -m py_compile` only. Do not touch other worktrees, any `frozen/` tree, `bin/`, `~/repos/gpuq`, `~/repos/qwen38-eval` (read only). Code must compile with GCC 13 and clang. No gpuq submit/cancel. Never load a file > 200 MB into memory.

## Deliverable
`codex/REPORT-stack10-cand.md`: commits (hash, subject, component), conflicts and resolutions, exactness evidence per component, Mac stage results (M, wall time), negative controls (each test fails on the unfixed/mutated path), window manifest path, item paths and expected outcomes (D/G) with falsifiers. **List every source change, including incidental compile fixes.** Final message: decision-relevant results only.
