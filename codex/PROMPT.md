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

# Task stack7-p55: build window + IQ4 items for stack7 + package 55 (per-type MMVQ)

Worktree (yours): `~/repos/qwen38-eval-worktrees/wt/stack7-p55`, branch `topic/stack7-p55` = `topic/stack7` (frozen/stack7 = 48b82b0d1; ddedffacb only adds codex/ files) + package 55 cherry-picked clean (990cd60f3 2851d6ad7 07cf46670 0302404cb; origin `wt/mmvq-pertype-s6` 848391a78..3d12d526f on stack6) + 3c47eb68c (PURPOSE). `GIT_NO_LAZY_FETCH=1` on every git command; never gc/prune.

Package 55 is CUDA-only (MMVQ launch parameters per quant type on Blackwell). Earlier result (M): IQ4_XS −0.3…−0.6 ms/round, J5q none. Prior A/B item and its identity expectation: `orchestration/gpuq/items/drafts/` and gpuq history `mmvq-pertype-ab` (`gpuq show mmvq-pertype-ab`, read-only), tree `wt/mmvq-pertype-s6/codex/`.

## Do

1. Window (copy the shape of `wt/stack7/codex/bw-stack7.toml` + `bw-stack7.sh`, CUDA stages only, no Mac stages): preflight (clean tracked tree, no existing `frozen/stack7-p55`), CUDA build with the stack6 RECIPE (`-DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF`, CCCL 3.4.3 `-DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL=…`, `strings libggml-cuda.so | grep -c DeviceTopK` = 25), a GPU correctness stage `test-backend-ops -o MUL_MAT` (and `MUL_MAT_ID` if the model uses it) must pass, `bin/wt freeze` as `frozen/stack7-p55`, item render + `gpuq submit --check` (never submit). Manifest: `kind = "filler"`, `host_sensitive = false`, `preemptible = false`, `priority = 19`. Window output under `work/build-window-1007/<name>-aN/` as the other windows.
2. GPU items (rendered from experiments with explicit `priority`; each run/canary cmd prefixed with `"bash", "/home/xanewok/repos/qwen38-eval-worktrees/work/aa-load/ccx01.sh",` and that wrapper pinned, `host_sensitive = false`; exact shape: `work/stack7-items/stack7-depth-iq4-ccx01.gpuq.toml`):
   a. `p55-depth-iq4` [27]: copy `~/repos/qwen38-eval/harness/experiments/stack7-depth-iq4.toml`; arms up = frozen/stack7 (serving env CHAIN=1 FOLD_C=1), patch = frozen/stack7-p55 same env; plan `up patch patch up`. Set `identity` from the evidence of the earlier per-type A/B (state which); canary must prove the patch arm is the p55 binary (SHA256SUMS / a log line if one exists; else say so).
   b. `p55-np2-real-iq4` [27]: copy `stack7-np2-real-iq4.toml` the same way (chain/fold 0, `-ub 512`).
   c. `p55-depth-j5q` [15]: J5q no-regression np1 (copy `stack7-depth-j5q` shape; up = stack7, patch = stack7-p55, serving env).
   Put the experiments under `codex/experiments/`, the rendered items under `codex/items/`; the parent copies them out.

## Limits

On this host only edits, git, `bash -n`, `python3 -m py_compile`, `python3 -m harness.mkitem` renders and `gpuq submit --check` (read-only). No compiles, no model runs, no gpuq submit/cancel. Do not touch other worktrees, `frozen/`, `bin/`, `~/repos/gpuq`, `~/repos/qwen38-eval` (read only).

## Deliverable

`codex/REPORT-stack7-p55.md`: window manifest path, item paths, identity decision with evidence, expected outcomes (D/G). Final message: decision-relevant results only.
