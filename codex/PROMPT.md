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

# Task stack7: integrate six changes on stack6, add the GDN out-proj 2D fix, write ONE build window and the GPU items (do not compile now)

Worktree (yours): `~/repos/qwen38-eval-worktrees/wt/stack7`, base stack6 `ac7465885` (detached or create branch `topic/stack7`). `GIT_NO_LAZY_FETCH=1` on every git command; never gc/prune.

**Do not compile, link, or run tests or models now**: GPU measure items run for the next hours and builds/CPU tests disturb them. No `buildgate`, `cpugate`, `cmake --build`, `g++ -fsyntax-only`. Static checks only (`bash -n`, `python3 -m py_compile`, `git diff --check`). All compiling and testing happens inside the window you write.

## Components (all based on ac7465885; take source commits only, not each branch's `codex/` scratch)

| Change | Branch (tree `~/repos/qwen38-eval-worktrees/wt/<tree>`) | Frozen evidence | CPU check to re-run in the window |
|---|---|---|---|
| Gated GDN getter (n_seqs == 1 only) | `topic/gdn-getter-gate` 3c97c4af8 (`gdn-getter-gate`) | `frozen/gdn-getter-gate` | its gate tests (`codex/bw-getter-gate.sh` stage 1, incl. mutants) |
| Host trims v2 (indexed `seq_rm`, bulk MTP verify rows) | `topic/host-trims-v2` 81f954b2a source; ee32a870d adds items only (`host-trims-v2`) | `frozen/host-trims-v2` | CPU np2 exactness + mutants (`codex/bw-host-trims-v2.sh` CPU stages; a mutant crash counts as caught; mutants must log engagement) |
| Draft chain M1 (`--spec-mtp-chain` / `LLAMA_MTP_CHAIN`) | `topic/draft-chain-spike` 0c5c3d451 (`draft-chain-spike`) | `codex/frozen/draft-chain-m1` there; memcheck clean | chain-on vs chain-off token identity, engagement line (`codex/REPORT.md` there) |
| Round-trace writer thread | `topic/trace-writer` abe4f6ad2 (`trace-writer`) | `frozen/trace-writer` | sync vs async byte identity + its negative control (`codex/bw-trace-writer.sh` stage 1) |
| NH1 fold (`LLAMA_MTP_FOLD_C=1`) + Meta planner fix | `topic/nh1-fold-c` 186226ceb (`nh1-fold-c`; fix 376e5c8d9, test c73714737) | none (window never ran with the fix) | `codex/bw-nh1.sh` stage 1: plain AND Meta k3/k5 fold-on vs fold-off token identity + engagement (`on_mtp_fold` ≈ 1/round), the engagement negative control, `test-alloc` |
| NEW: GDN out-proj 2D input | write it (below) | — | new CPU check (below) |

1. Integrate in that order (cherry-pick or merge; one commit series, llama.cpp style messages, no AI trailers). Resolve conflicts by intent; list every conflict and its resolution with file:line. Default switches must match each component's tested default (getter on but gated; chain/fold per their branches: state each default explicitly, and keep env switches so the GPU A/B can toggle NH1 and the chain in one binary).
2. GDN out-proj fix: at stack6 `src/models/qwen35.cpp` (~473-481) `final_output` is 3D `[head_v_dim*num_v_heads, n_seq_tokens, n_seqs]`, so `build_lora_mm(ssm_out, final_output)` runs as n_seqs batched matmuls and the CUDA path reads the `ssm_out` weights once per sequence (MMVQ 6 cols × 2 at np2; measured Q6_K 67 µs vs 38 µs, IQ4 46 vs 25). Reshape to 2D `[d, n_seq_tokens*n_seqs]` before the matmul (the existing reshape after it then becomes a no-op or goes). Apply the same to every model with the identical pattern (check `qwen35moe.cpp`, `qwen3next.cpp`, any other `ssm_out`/linear-attention out-proj with a 3D input; list them). One separate upstream-form commit. Check LoRA (`build_lora_mm` with adapters, `ssm_out_s` scale) still gets correct shapes. CPU check in the window: np2 (two sequences) logits/tokens identical before vs after on the 4B fixture (`/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf`), plus a `test-backend-ops`-style or graph check proving the matmul input is 2D at n_seqs = 2 (negative control: the check must fail on stack6).
3. Window payload `codex/bw-stack7.sh [OUT | --preflight]` + manifest `codex/bw-stack7.toml`, cloned from the existing payloads (`wt/trace-writer/codex/bw-trace-writer.sh` is the newest; wrapper `~/repos/qwen38-eval-worktrees/work/build-window-1007/bw-run.sh`, read it: it writes to `work/build-window-1007/<name>-aN/`). Stage 1: CPU build via `buildgate -j4`, then every CPU check in the table + the new one, each through `cpugate`, each as its own `run_stage` with a distinct exit code. Stage 2: CUDA build with `~/repos/qwen38-eval-worktrees/frozen/stack6/RECIPE.txt` (sm 120a, CCCL v3.4.3 flags, `-DGGML_CUDA_NCCL=OFF`, `-DCUDAToolkit_ROOT=/usr/local/cuda-13.2`), `strings libggml-cuda.so | grep -c DeviceTopK` = 25, ggml stamp, `wt freeze` as `stack7`. Pipes: never `grep -q`/`head` after a producer under `pipefail`; use `| grep -x PAT >/dev/null`. Exit codes: 0 pass, distinct per failing check, never 3/4. Manifest: id `bw-stack7`, `kind = "filler"`, **priority 25**, budget = estimate × 1.5, pins = wrapper, payload and every script it runs. The clean-tree check fails on uncommitted tracked edits: commit everything.
4. GPU items (generate from the window's last stage like `bw-spec-tf` does, pin the freeze and the window's real `work/build-window-1007/stack7-aN/summary.json`, never `codex/bw-results/...`), main harness `~/repos/qwen38-eval` (`python3 -m harness.mkitem`, real-prompt support merged), priority 30, all untraced (no `LLAMA_ROUND_TRACE`), J5q, serving flags of `~/.claude/skills/local-qwen-server/bin/serve.sh` profile `swift15-j5q` (np1) and `swift15-j5q-duo` (np2; `PROFILE=... serve.sh command`; read-only):
   a. `stack7-depth-j5q`: np1 k5, 8k/32k/90k real prompts (corpus `tokens64k.bin`), arms `s6` = `frozen/stack6`, `s7` = `frozen/stack7` NH1 off, `s7f` = same binary NH1 on; balanced order (e.g. s6 s7 s7f s7f s7 s6), cross-arm token identity gated where the harness supports it. Canary proves each component engaged in `s7f` (getter, chain, fold, trace writer not needed untraced, out-proj 2D not at np1).
   b. `stack7-np2-real-j5q`: np2, same depths, `up` = stack6, `patch` = stack7 (NH1 setting: whichever wins in (a) is unknown now, so use NH1 off and say so), plan `up patch patch up`, wall 900 s per boot. Canary proves host trims + out-proj 2D engaged.
   c. `stack7-duo-soak-j5q`: clone `~/repos/qwen38-eval/harness/experiments/stack6-duo-soak-j5q.toml` with stack7, priority 20.
   `gpuq submit --check` each; read its text: any "contract violated" line is a failure even if rc is 0 (gpuq bug being fixed). Do not submit.

## Deliverables
`codex/REPORT-stack7.md`: commit list (source → stack7 commit), conflicts and resolutions, defaults and switches, the out-proj change with the list of models touched, window stages + exit-code table + estimate, item paths, risks (interaction between host trims and the fold/chain at np2; getter gate vs fold; trace writer shutdown). Final message: decision-relevant results only.
