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

# Task stack8: stack7 + np2 chain fallback fix + backend candidates under -inf bias, window ready to queue

Worktree (yours): `~/repos/qwen38-eval-worktrees/wt/stack8`, branch `topic/stack8`, base `topic/stack7` 48b82b0d1 (= `frozen/stack7`, serving `swift15-j5q[-duo]` since 2026-10-07). `GIT_NO_LAZY_FETCH=1` on every git command; never gc/prune.

## Content

1. Source commits from `topic/chain-np2`: da787bb40 (`common : restrict MTP draft chain to single-sequence slot sets`) and its check commit 374a465b8 (strict np2 identity + negative control). Report: `wt/chain-np2/codex/REPORT-chain-np2.md`.
2. The cand-neginf source change from `topic/cand-neginf-mac` (base stack6 ac7465885): the commit `common : sampling: keep the backend candidates under a -inf logit bias` (062667905 or its descendant; find it with `git log ac7465885..topic/cand-neginf-mac -- common/ src/ include/`) plus 935c302b9 (candidate LOG_DBG) and the unit tests. Only source/tests; not that tree's `codex/` window scripts unless needed for stage reuse. GPU result (M): −0.65/−0.26/−0.08 ms/round np1, tokens identical (`results/2026-10-07-cand-neginf-depth-j5q/full-a1`).
3. Resolve conflicts in general upstreamable form; one commit per component, plus a separate commit for any stack-integration glue.

## Window (do not submit; the parent submits)

Reuse `wt/stack7/codex/bw-stack7.sh` structure (`S7_ONLY` stage selection, `BW_*` env convention of `~/repos/qwen38-eval-worktrees/bin/bw-mac` header) — do not copy per window beyond what is needed:
- CPU stages on the Mac via `bin/bw-mac TREE PAYLOAD NAME` (results land in `work/build-window-1007/<name>-mac-aN`): CPU build (`-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod`), unit tests, stage 23 np1 chain identity, the np2 chain check now strict (chain on == chain off at np2, plain + Meta, with the old-binary negative control), cand-neginf CPU identity + mutant. Mac model runs: ≤ 4 threads, 4 GiB free gate; another job (`~/qwen38-chain-np2/mac-stage23.sh`) may be running: max 5 build jobs while it does.
- Local gpuq window manifest (filler, `host_sensitive = false`, `preemptible = false`, explicit `priority = 20`): only the CUDA build (`-DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF`, stack6 RECIPE incl. CCCL 3.4.3 `-DGGML_CUDA_CCCL_VERSION=v3.4.3 -DFETCHCONTENT_SOURCE_DIR_CCCL=…`, `strings libggml-cuda.so | grep -c DeviceTopK` = 25), `bin/wt freeze` as `frozen/stack8`, item render + `gpuq submit --check` (never submit). Preflight refuses a dirty tracked tree or an existing `frozen/stack8`.
- GPU items (rendered from experiments with explicit `priority`; each run/canary cmd prefixed with `bash ~/repos/qwen38-eval-worktrees/work/aa-load/ccx01.sh` and that wrapper pinned, `host_sensitive = false` — see `work/stack7-items/stack7-depth-j5q-ccx01.gpuq.toml` for the exact shape):
  a. `stack8-depth-j5q` np1: s7f (frozen/stack7, CHAIN=1 FOLD_C=1) vs s8f (frozen/stack8, same env), ABBA, 8k/32k/90k, identity cross-arm expected only if cand-neginf does not change tokens (state your expectation and set `identity` accordingly; harness requests use `ignore_eos`).
  b. `stack8-np2-chain-j5q` np2 real prompts: s8 CHAIN=0 FOLD_C=0 vs s8 CHAIN=1 FOLD_C=0 (the fix must make them equal in speed; canary must prove the fallback logs at np2).
  c. `stack8-duo-soak-j5q`: duo serving flags with CHAIN=1 (and FOLD_C as decided by the parent; template the value).

## Resources and limits

Your sandbox writes only inside your worktree: run `bw-mac` with `BW_MAC_OUTPUT_BASE=$PWD/build-bw-mac-results` (gitignored via `/build*`), and write the window manifest and rendered items under `codex/`; the parent copies them out. Items must not pin paths inside `build-bw-mac-results`.

All compiles and model runs on the Mac (network only for ssh/scp to mini-builder; never push or post). On this host: edits, git, `bash -n`, `python3 -m py_compile` only — GPU timing A/Bs run here all night. Do not touch `wt/stack7`, `wt/chain-np2`, `wt/cand-neginf-mac`, any `frozen/` tree, `bin/`, `~/repos/gpuq`, `~/repos/qwen38-eval` (except reading). Code must compile with GCC 13 and clang (`json::array({a, b})`, no vexing parses). No gpuq submit/cancel.

## Deliverable

`codex/REPORT-stack8.md`: commits (hash, subject, origin), conflicts and how resolved, Mac stage results (M, per-stage wall time), negative controls, window manifest path, item paths and their expected outcomes. **List every source change you made, including incidental compile fixes.** Final message: decision-relevant results only.
