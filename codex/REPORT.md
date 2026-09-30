# Weekend Swift campaigns and offline draft coverage

## Conclusion

The requested stack at `0dba27b9002b8347c6f14a9f4a1651b3478065ef` is built and frozen at `frozen/stack2`. Its sources and history are unchanged. The weekend chain defaults to dry mode; no GPU campaign, inference server, nvidia-smi invocation, or memory-clock guard was run here. GPU throughput, acceptance, quant fidelity, and the best draft depth remain unknown.

**Policy failure:** the initial CMake configuration implicitly executed `CMakeDetermineCompilerABI_CUDA.bin`. `CUDA_VISIBLE_DEVICES` was empty, and its output was `No CUDA devices found.`, but this still violated the prohibition on executing CUDA binaries. `/usr/share/cmake-3.28/Modules/CMakeTestCUDACompiler.cmake` invokes `cmake_cuda_architectures_native`, which executes the ABI binary even with an explicit architecture. `codex/build.sh` now sets `CMAKE_CUDA_COMPILER_FORCED=ON` to skip that path. A fresh guarded configuration has no CUDA ABI binary; its compile/link result is recorded in `codex/cuda-guard-build.log`. No `nvidia-smi`, llama executable, or GPU benchmark was directly executed.

Labels: M = measured, D = derived, G = guess. Operational constants below come from the brief, not performance measurements. There is no GPU speed claim.

## Offline coverage

M: the requested `campaign-*swift15*/**/model-calls.jsonl` glob returned zero files. The llama campaigns instead retain Codex rollouts. `codex/coverage.py:16` and `:33` extract assistant `response_item` reasoning, prose, and raw tool arguments, excluding history, tool results and duplicate event records. Failed/retried attempts are included. Inputs and SHA256 provenance are in `coverage.json`; per-campaign counts and the most frequent excluded tokens are in `coverage.md`.

M: 187 files, 27 campaigns, including the September 30 llama campaigns; 1,853,467 reconstructed output tokens; 63,837 outside the 25,879-ID subset. M: all 248,077 tokenizer IDs match the IQ4_XS GGUF token spellings; its remaining 243 entries are padding placeholders. Tokenization uses the installed CPU `tokenizers` package and the supplied Swift tokenizer.json, without added delimiters.

| Region | Tokens (M) | Outside (M) | Outside fraction (D) |
|---|---:|---:|---:|
| Reasoning | 1,212,685 | 26,244 | 2.164123% |
| Prose | 109,929 | 2,529 | 2.300576% |
| Tool-call arguments | 530,853 | 35,064 | 6.605218% |
| All | 1,853,467 | 63,837 | 3.444194% |

D: subset acceptance is bounded by `sum min(p,q) <= p(S)`. The empirical support ceiling is 96.555806%, and support alone forces 3.444194% loss relative to a perfect full-vocabulary drafter. **This is not an upper bound on incremental loss against the actual full drafter:** renormalization changes q within the subset. That incremental loss needs the GPU campaigns. Parsed content is reconstructed rather than original wire token IDs; normalization, region boundaries and stripped template tokens limit the estimate.

M: coverage wall time was 28.163 s under competing build load. This is CPU analysis timing, not model inference speed.

## Build and freeze

`codex/build.sh` records the full recipe: CUDA 13.2, NCCL off, architecture 120 (llama CMake translates this to 120a), Release, ccache off, FA types `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`, no UI download, work-local TMPDIR, nice 19, `-j8`. Both configurations build only target `llama-server`. No CPU model inference or sampling-source change was required by this task.

`codex/freeze.py:17` copies the server and shared libraries, sets RUNPATH `$ORIGIN`, creates a local shared/no-checkout git clone with detached HEAD at the requested commit, archives gguf-py from that commit, and writes SOURCE and SHA256SUMS. M: 24 frozen file pins. `codex/frozen-ldd.log` verifies every libggml, libllama and libmtmd resolves within `frozen/stack2/build/bin`; system and CUDA runtime libraries remain external. The frozen executable was never run. `weekend-srv` has identical bytes to the registry's `llama-server` file.

## Campaign design

`run_weekend.sh` supplies nice 19, thread caps, offline environment, local TMPDIR and `DRY_RUN=1` by default. Results stay under `results/<RUN_ID>` in this work directory; `weekend.log` is the shared log. Default RUN_ID includes a timestamp and random suffix. Supplying a prior RUN_ID permits only fully verified reuse; partial or foreign directories refuse.

- `codex/weekend_common.py:57` imports the read-only runner rather than forking its safety logic. `SWIFT_DRAFT_MAX` alone supplies k; SWIFT_EXTRA enables probabilistic draft, draft backend sampling, target backend sampling, and the subset for S3/S4/S5. The original eight tasks, tensor 50,50, context and q8_0/q8_0 target KV remain in use.
- `codex/weekend.py:31`, `:44`, `:74` enforce readable/clean boot journal, idle process inventory, no unapproved compute jobs, aggregate GPU memory below 1500 MiB, free port 8081, and the deadline. Memory-clock guard exit zero is required before and after a new campaign. Missing or timed-out telemetry fails closed.
- Phase 1 follows S0 S3 S4 S5, S5 S4 S3 S0, S0 S3 S4 S5. Names encode phase, arm, pass/pair, model and digest. Phase 2 alternates A and B using the highest campaign-mean decode tok/s arm when all three passes are eligible; otherwise WINNER_ARM defaults to S3.
- `codex/weekend_common.py:34` selects a complete J5q with sidecar, otherwise fdown6mid; MODEL_B can override it. J5q completed during preparation and the final dry run selected `/mnt/scratch/work/models/j5/Swift-1.5-Qwen3.8-27B-J5q-bf16.gguf`, M: 19,996,051,744 bytes, SHA256 `f2c850ea3cd818b9d669fbd1e18767ef669ce012e614d819cce5ab6809e3a774`. Registration uses the runner's LOCAL_BUILDS metadata and the sidecar digest.
- `codex/campaign.py:25` executes only `weekend-srv`, preserves registry pins, and seals the actual executed command into launch.json. It refuses model downloads. `:42` opens existing global lock files read-only and retains their locks; new runtime files stay local.
- `codex/weekend_common.py:105` validates complete reuse, every passed job, the exact queue/model/flags/binary pins, target KV in launch.json and available server log lines, exactly one k flag, and the alternate executable name.
- The deadline is October 3 at 06:00 Europe/Warsaw. A campaign needs the full 3600 s reservation. Its runner gets 3480 s and exact-PID cleanup reserves 120 s; no new campaign starts with less than the full reservation. A failed or capped campaign stops the chain, as in item 98-a. The runner owns child process groups and its existing parent-death protection remains active.

`codex/summarize.py:12` uses actual `metrics_delta.drafts` verification counters rather than `draft_tokens/k`. It reports solved, wall, output tokens, decode tok/s, round ms, acceptance, tokens/round and max depth; campaign means +/- SEM by arm and pinned model; and within-pass deltas against S0. Incomplete campaigns remain visible but cannot enter means or winner selection. `codex/gpu_ab.sh` runs phase 1 only, the real-traffic baseline/subset A/B with interleaved campaign boots and the same frozen checks, Xid checks, exact-PID ownership and cap. Its baseline is the requested stack2 without subset, rather than repeating the parent's already completed older-stack A/B.

## Evidence and commands

M: `codex/cuda-build.log` passed compile/link. M: `codex/contract-tests.log` passed seven CPU tests in 1.020 s. Controls reject a foreign manifest, f16 KV, duplicate k flags, incomplete winner samples, faulty/unreadable journal, and a round calculation that relies on fixed draft length. Extraction excludes user/tool output and duplicated reasoning. The dry-mode test fails if GPU checks or Popen are reached. A real `hillclimb.initialize` invocation in a work-local fixture verifies the runner manifest contract without launching a server.

M: `DRY_RUN=1 RUN_ID=weekend-20260930-prepared bash run_weekend.sh` passed. `codex/dry-run.log` records the full phase-1 plan, model hashes, frozen engine, flags, phase-2 choice, paths and budget. It checks the models, frozen files, subset, existing lock paths, assets/tool binaries and 876 pinned runner/asset inputs; it does not execute GPU checks. Shell syntax and Python contract checks passed. `llama.cpp-stack2` is clean at the requested commit.

Reproduce CPU analysis with `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 /home/xanewok/repos/qwen38-eval/.cache/exl3-venv/bin/python codex/coverage.py`. Run contract checks with `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice -n 19 python3 codex/test_weekend.py`.

Parent launch, not run here: `DRY_RUN=0 RUN_ID=weekend-20260930-main bash run_weekend.sh`. Phase 1 only: `DRY_RUN=0 RUN_ID=weekend-20260930-ab bash codex/gpu_ab.sh`. Regenerate summaries with `CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/summarize.py results/weekend-20260930-main --out results/weekend-20260930-main/summary.md`.

## Commits and remaining risk

Artifact commits are on `topic/weekend-campaigns`, in the independent git directory `codex/.git` with this work directory as its working tree. Use `git --git-dir=codex/.git --work-tree=. log` to inspect them. This preserves the explicit instruction not to commit to or change `llama.cpp-stack2`.

The live campaign adapter, idle telemetry, lock acquisition, fault handling and signal cleanup were not exercised with GPU jobs. A campaign that reaches its cap or fails safety checks halts unattended work; it is not silently resumed. Sparse task repetitions cannot establish quant fidelity by themselves; output volume and solved counts must be read alongside wall time, depth and acceptance. The frozen git clone depends on the existing read-only object store; deleting that store invalidates source identity checks. Re-pinning inputs after an upstream edit is an explicit preparation step, not automatic adaptation during the run.
