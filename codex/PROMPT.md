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

# Task stack9: stack8 + np2 getter + multi-sequence fold C, window ready to queue

Worktree (yours): `~/repos/qwen38-eval-worktrees/wt/stack9`, branch `topic/stack9`, base `topic/stack8` 5df0dbffc (source = `frozen/stack8` 41d2954a2, measured today). `GIT_NO_LAZY_FETCH=1` on every git command; never gc/prune.

## Content

1. np2 getter (multi-sequence indexed GDN state read + race-free joint kernel) from `topic/np2-kernels` (base stack7): source + test commits only (`git log 48b82b0d1..topic/np2-kernels -- src/ ggml/ common/ include/ tests/`; includes 011df3795, 858599c30, 96774329b, 5a855ecc3; docs 980e7bb13 optional). Not its `codex/` scripts. GPU result (M/D): np2 real J5q −0.18/−0.57/−0.40 ms/round vs stack7 (`results/2026-10-08-np2-getter-real-j5q/full-a1`). Report: `wt/np2-kernels/codex/REPORT-np2-gate-fix.md`.
2. Multi-sequence fold C from `topic/fold-mseq` (base stack7): source + test commits (`git log 48b82b0d1..topic/fold-mseq -- src/ common/ ggml/ include/ tests/ tools/`), including the M-RoPE fix e61e6bb31 and fixture fixes fe34c623a, 186e41a52. Report: `wt/fold-mseq/codex/REPORT-fold-mseq.md`. Its np2 GPU A/B is queued today; keep fold-mseq as its own commit(s) so the parent can drop it if that A/B is null.
3. Owner rule: default-on where exact, exact automatic fallback, multi-sequence capable, no env var in the upstream form. Any A/B switch (e.g. `LLAMA_MTP_FOLD_C=0` to disable) lives only in a separate local commit on top, clearly named `local:`.
4. Resolve conflicts in general upstreamable form; one commit per component, plus a separate commit for stack-integration glue. Conflicts likely in `src/llama-context.cpp`, `common/speculative.cpp`, `ggml/src/ggml-cuda/gated_delta_net.cu`: both levers touch the MTP / GDN state paths. State how each conflict was resolved.

## Window (do not submit; the parent submits)

Reuse `wt/stack8/codex/` window structure (`bw-stack8.sh`, its Mac stage selection and item renderer); do not copy beyond what is needed.
- CPU stages on the Mac via `~/repos/qwen38-eval-worktrees/bin/bw-mac TREE PAYLOAD NAME`: CPU build (`-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod`), unit tests, np1 chain identity (stage 23), np2 getter gate (by-id vs gathered on the same batch, byte-identical; mutants caught), fold-mseq gates (24 identity pairs + controls, `wt/fold-mseq/codex/mseq/matrix.py`), stack8's np2 chain fallback check, cand-neginf identity. Mac model runs: ≤ 4 threads, ≥ 4 GiB free gate, ≤ 10 min per gate then fail.
- Local gpuq window manifest (filler, `host_sensitive = false`, `preemptible = false`, `priority = 19`): only the CUDA build (`-DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF`, stack6 RECIPE incl. CCCL 3.4.3, `strings libggml-cuda.so | grep -c DeviceTopK` = 25 — read the whole input, never `| grep -q` under pipefail), `bin/wt freeze` as `frozen/stack9`, item render + `gpuq submit --check` (never submit). Preflight refuses a dirty tracked tree or an existing `frozen/stack9`. Run `python3 -m gpuq.shlint` (in `~/repos/gpuq`) on every `.sh` you write.
- GPU items (explicit `priority`; each run/canary cmd prefixed with `bash ~/repos/qwen38-eval-worktrees/work/aa-load/ccx01.sh`, wrapper pinned, `host_sensitive = false`; shape: `wt/stack8/codex/items/`). The harness now defaults `--cache-ram 0` in every arm (qwen38-eval 3d18039); do not hard-code it. Canary engagement lines that are `LLAMA_LOG_DEBUG` need `LLAMA_ARG_LOG_VERBOSITY = "5"` in the canary env.
  a. `stack9-depth-j5q` np1, priority 30: s8f (frozen/stack8, CHAIN=1 FOLD_C=1) vs s9f (frozen/stack9, same served flags), ABBA, 8k/32k/90k; state the identity expectation (fold-mseq and the getter should be token-identical at np1; set `identity` accordingly).
  b. `stack9-np2-real-j5q` np2 real prompts, priority 30: arms s8 / s9 / s9-nofold (local switch), served `swift15-j5q-duo` flags (k5, `-ub` as served), plan of 6 boots balanced (e.g. A B C C B A), 8k/32k/90k. Canary proves the getter path and `mtp_fold_decode: engaged n_seq=2` at `-lv 5`.
  c. `stack9-duo-soak-j5q` priority 29: duo serving flags on frozen/stack9, fold default-on.

## Resources and limits

Your sandbox writes only inside your worktree: run `bw-mac` with `BW_MAC_OUTPUT_BASE=$PWD/build-bw-mac-results` (gitignored via `/build*`); write the window manifest and rendered items under `codex/`; the parent copies them out. Items must not pin paths inside `build-bw-mac-results`.

All compiles and model runs on the Mac (network only for ssh/scp to mini-builder; never push or post). On this host: edits, git, `bash -n`, `python3 -m py_compile` only — GPU timing A/Bs run here. Do not touch `wt/stack8`, `wt/np2-kernels`, `wt/fold-mseq`, any `frozen/` tree, `bin/`, `~/repos/gpuq`, `~/repos/qwen38-eval` (except reading). Code must compile with GCC 13 and clang. No gpuq submit/cancel. Never load a file > 200 MB into memory.

## Deliverable

`codex/REPORT-stack9.md`: commits (hash, subject, origin), conflicts and how resolved, Mac stage results (M, per-stage wall time), negative controls, window manifest path, item paths and expected outcomes. **List every source change you made, including incidental compile fixes.** Final message: decision-relevant results only.
