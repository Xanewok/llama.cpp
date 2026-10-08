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

# Task np2-gate-fix: make the multi-sequence getter gate backend-numerics independent, then re-arm the window

Tree (yours): `~/repos/qwen38-eval-worktrees/wt/np2-kernels` (HEAD includes 0951e4ad3 gate-diag). `GIT_NO_LAZY_FETCH=1`. No compiles/tests/model runs on this Linux host; no gpuq submit/cancel. Mac via `bin/bw-mac` or `ssh mini-builder` (bounded: RAM wait ≤ 10 min for ≥ 4 GiB, job ≤ 60 min, one model job at a time).

## Facts (M)
`work/build-window-1007/np2-gate-diag-a1/diag.txt` (x86 Ryzen 3900X AVX2, repack/llamafile OFF): every two-sequence batch phase (verify batch, decode both seqs, seq beside an idle slot, swapped cells, extra state) differs from the single-sequence context by nmse up to 1.1e-3 (> the 1e-4 bound in `tests/test-recurrent-state-rollback.cpp` ~:218), and the CONTROL build (stack7 gather path, `n_seqs == 1` mutation) gives bit-identical DIAG values to the new by-id build in all 124 positions for t4, t1 and FA-off. Repeat run identical. So the failure is x86 batched-vs-single numerics, not the by-id read. The Mac (arm64, i8mm off) stayed under 1e-4.

## What
1. Change the gate so that it tests what the patch changes: the by-id read must give logits bit-identical (or nmse 0) to the gathered read for the SAME batch. Preferred general form for upstream: the test runs each multi-sequence phase twice in one binary — once with the by-id read and once with the gathered read forced (a test-only hook or an existing graph option; no env in the root series) — and compares them exactly; the single-sequence comparison stays as information with a bound that holds across backends (justify it) or is dropped. Keep the structure checks (by-id node counts, gathered 0) and the mutants (they must still be caught).
2. Update `codex/np2/cpu-gate.sh` accordingly (it already builds a ctl; a ctl-vs-new exact comparison of the DIAG/logits output is acceptable as the window gate if the in-test comparison is not feasible).
3. Mac rerun of the full cpu-gate (bounded) → PASS required. Then confirm `codex/np2/bw-np2-getter.toml` (or the parent's pinned copy shape `~/repos/qwen38-eval-worktrees/work/stack7-items/bw-np2-getter-ccx01.toml`) still `gpuq submit --check` valid with any new pins.
Commit everything (window refuses a dirty tree).

Deliverable: `codex/REPORT-np2-gate-fix.md`. Final message: decision-relevant results only.
