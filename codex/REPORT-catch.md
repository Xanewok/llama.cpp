# Prefix repetition capture (DO NOT MERGE)

Conclusion: `codex/catch.sh` is ready for parent GPU execution on `codex/ra-catch`. CPU identity, strict replay, interventions, onset analysis and negative controls pass. No GPU executable or GPU query was run. No new live loop, attractor verdict, base-model loop rate or GPU performance result is available yet. Confidence is high for the tested CPU contracts; GPU transport and long-history replay remain unverified.

Labels: M = measured/observed, D = derived/planned, G = unmeasured time scenario. Literal settings, identifiers, hashes and source locations are configuration, not performance estimates. Production sampling code is unchanged.

## Engine and base model

The reused capture engine is `8a15734ca7f8e1315007da7813e83b9893f10f92`, based on stack2 `0dba27b90`. `CAPTURE-PINS.json` pins the renamed `capture-srv` and its libraries. The read-only reference is `/home/xanewok/repos/ra-capture/stack/codex/frozen/capture/build/bin`; all pinned bytes were verified and copied to this worktree's `build-catch-frozen`. Parent boots copy the same bytes into the run directory, verify them before every boot, and never start an executable named `llama-server`. Reuse avoids another CUDA build. The reference records CUDA toolkit `/usr/local/cuda-13.2`, architecture `120`, NCCL off and stack2's scheduler/subset features.

`BASE-METADATA.json` records observed GGUF metadata: `general.name=Qwen3.8 27B`, `general.basename=Qwen3.8`, architecture `qwen35`, file type `30` (IQ4_XS). M: file size 15,475,951,328 bytes; M: 866 tensors. `qwen35.nextn_predict_layers=1` and preserved NextN tensors are present; this is not a missing-MTP file. The comparison deliberately uses no-spec. Identity is established from local metadata, not an external provenance audit; parent `RUN.json` additionally hashes both weight files.

Both selected histories were written by Swift and reconstructed by the existing study. They omit production tool schemas, grammar and budget state. A base-vs-Swift difference measures behavior conditional on those selected Swift histories, not unconditional task reliability or a general checkpoint ranking. Capturing this reconstructed study makes its own generated history replayable; it does not recover the original campaign state.

## Run order and parent commands

Use one output directory across phase invocations. Resume with the same experiment settings and source; `RUN.json` rejects changed binaries, models, fixtures, sampler inputs or drivers. Budget and selected phases can change. Existing completed trials are skipped; interrupted requests are pending, never negative outcomes.

```sh
export CUDA_VISIBLE_DEVICES=""
PLAN=1 bash codex/catch.sh --phases 0 1 2
RUN_PARENT_GPU=1 PLAN=0 bash codex/catch.sh --phases 0 --budget 10800
RUN_PARENT_GPU=1 PLAN=0 bash codex/catch.sh --phases 1 --budget 21600
RUN_PARENT_GPU=1 PLAN=0 bash codex/catch.sh --phases 2 --budget 86400
```

These live commands are for the parent only. Repeat the phase invocation after a wall-budget stop. Defaults use `build-ra-catch`; `--out` must remain inside this worktree. To use the locally verified copy, add `--frozen-dir "$PWD/build-catch-frozen"` consistently. `codex/gpu_ab.sh` remains the previous continuation experiment; this task's GPU comparison entry point is `codex/catch.sh`.

| Phase | Registered work (D) | GPU time scenario (G) |
|---|---|---|
| 0 | Base IQ4_XS no-spec; prefix 5 and prefix 4; 24 trials each; 8,192 retained-token horizon | About 2 hours |
| 1 | Swift IQ4_XS prefix 5; no-spec and S3; each stops at 3 loops or 64 trials; 8,192-token horizon | Up to about 5 hours |
| 2 | Every retained phase-1 loop; onset minus 0, 64 and 512 generated tokens; 16 continuations per intervention; 4,096-token suffix | Up to about 48 hours |

Those times are guesses for queue sizing, not GPU measurements. D: phase 0 has 48 trials; phase 1 has at most 128. D: phase 2 has at most 1,152 continuations when both arms catch their full loop quota, plus 18 strict boundary checks. The scenario assumes G: about 150 seconds per original trial including amortized boot/prefill, and G: about 150 seconds per branch including teacher forcing and amortized boot. D: the corresponding trial-only totals are 2, 5.33 and 48 hours; actual duration depends strongly on onset position, prefill, early EOS, capture overhead and retained loop count. Default D: invocation budget is 86,400 seconds with 30 seconds reserved for exact-child cleanup. Phase 2 can require multiple invocations.

Boot order rotates/reverses blocks of eight trials, including branch interventions. Before every boot the parent checks idle selected GPUs, active compute processes, benchmark locks, occupied port and an Xid/fallen-off-free kernel journal. Exact owned `Popen` children receive TERM, then KILL if necessary; Xid is checked after cleanup. Shared locks are opened read-only; all new files and the run owner lock stay inside this worktree. HTTP targets loopback only, with proxies disabled.

`LLAMA_CAPTURE_PROBS=1` is on for every original and branch request. Original full captures are retained only for loops and the first three non-loop controls per model/arm/prefix cell. Branch captures are removed after validation and suffix evidence are saved. Original completion summaries remain available for every completed trial. D: output limit is 5,000,000,000 bytes; the watchdog checks once per second and stops with 20,000,000 bytes of headroom, also requiring 100,000,000 bytes free. Partial captures from interruptions remain evidence. This is a polling guard with headroom, not a filesystem quota; very large unanticipated writes can overshoot it.

## Probability and onset contract

`catch.py` reuses `experiment.py`'s prefix construction, sampler request and server flags: tensor mode with 50/50 split, target q8_0 KV, draft f16 KV, target/draft backend sampling requested, temperature 1, top-k 20, top-p .95, min-p .05, and S3 probabilistic subset MTP with rejection verification. No generated tools are executed. Evaluated prefix IDs are saved and checked for consistency between boots.

The existing capture records normalized, filtered target `p`, emitted token ID and exact piece bytes, plus round proposal IDs and complete draft `q`. `onset.py` writes `per-token.jsonl` with sampled-token target probability, filtered top-20 IDs/probabilities, entropy in nats, support size, min-p cut size when certified, phase-aligned repeated-token probability, and corresponding draft data. These are target probabilities, including for tokens obtained through residual rejection sampling; they are not residual probabilities or raw softmax probabilities.

With this unmasked default chain, top-k bounds support to 20, and the stages after min-p do not delete support. Thus final support equals the min-p cut size. The analyzer checks normalization/omitted mass, grammar absence, positive temperature, XTC/adaptive/mirostat settings and chain suffix before making that derivation. Grammar captures report cut size as unknown. Positive emitted-token support establishes survival through top-k, top-p and min-p. Earlier ranks, cutoff margins and pre-filter probabilities were not captured and remain explicitly unknown; final rows cannot identify a single responsible filter or recover raw-model entropy.

The existing detector registers the same sustained low-diversity outcome as the rate study. Its character span can include pre-loop text. `find_onset` searches within that span for a sustained exact model-token period, chooses its earliest start and extends it backward/forward by token equality. Nonperiodic detector hits retain their evidence but cannot supply an exact-period branch. The report includes the entering token's probability/rank and a D: 512-token window on each side. A low entry probability followed by high repeated-token probabilities is evidence consistent with tail entry into an attractor; the trajectory itself is retained, without an automatic causal verdict. Controls use the loop's generated-token positions and period phase, not their own detected onset; this matches positions, not semantic state.

## Branch interventions and fidelity

`catch_replay.py` is a local copy of the audited reference replay helper, with original requests allowed to omit `__capture_force`. It preserves prompt/generated roles, penalties, original request grammar fields and reasoning-budget transport, checks retained forced IDs, waits for final capture flush and checks every expected row. Strict fidelity uses the original arm at each selected boundary, D: absolute tolerance `1e-6`, before any continuation at that boundary.

The interventions are:

- `served`: original no-spec or S3 arm and sampler; every branch remains strict.
- `no-spec`: same target sampler with speculation disabled.
- `temp07`: original arm, temperature `.7`, same top-k/top-p/min-p.
- `presence`: original arm, `presence_penalty=1.0`, `repeat_last_n=102400`, repeat penalty `1.0`, frequency penalty `0.0`, DRY multiplier `0.0`. Equivalent built-in CLI settings are `--presence-penalty 1.0 --repeat-last-n 102400`; the REST schema rejects `repeat_last_n=-1`.

Changed arms retain numerical row-drift counts against the source capture; budget disagreement is still fatal. Temperature/penalty settings also apply while rebuilding their teacher-forced history, so changed-row comparison is diagnostic, not a fidelity equality claim. At the boundary a fresh sampler accepts the same prompt/generated roles and uses an independent explicit branch seed. Matched branch seeds do not imply matched speculative RNG streams. Only the generated suffix is scanned for re-looping; forced repetitions are excluded. Results group by source loop, offset and intervention, with completed denominators and incomplete groups marked.

Observed limitation: the frozen server cannot reset the branch sampler at generated position zero, because the first output is sampled during prompt processing before its generating-loop branch check. Positions at or below zero are recorded as unavailable, never clamped or counted as negatives. Strict mismatch, context shifts, checkpoint restores, unsupported sampler state, missing rows or incomplete captures fail the run. Default teacher forcing remains within the original response's generated history; missing history is not replaced with prefilled prompt tokens.

Phase 1's stopped fractions are descriptive because stopping depends on loop count. The base fixed-N result is compared with the supplied Swift no-spec M: 2/24 at each prefix; it is not compared inferentially with the optional-stopping phase-1 fraction. Branches from selected loop histories are conditional diagnostic experiments and share their source history.

## CPU evidence and reproducibility

The CPU executable is copied from the reference `build-cpu/bin/llama-server` to a worktree-local `catch-cpu-srv` with CPU-only libraries. Reference CMake cache confirms `GGML_CUDA=OFF`, `GGML_CPU_REPACK=OFF`, Release. No C++ source changes or rebuild were needed. Qwen3.5-4B fixture: `/home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf`; inference uses four threads, nice 19 and CUDA hidden. `CPU-CATCH.json` records binary identity and results.

| Check | Result |
|---|---|
| Capture off/on, ordinary seeded sampling | M: 16 emitted IDs bit-identical |
| Capture off/on, probabilistic MTP k3 | M: 16 emitted IDs bit-identical; 13 proposals, 10 accepted in each run |
| Strict teacher-forced branch from both source arms | M: 9 matching rows per arm at generated position 8; 18 total; zero differing rows |
| Temperature `.7` | M: 9 compared rows per arm; 2 differing rows per arm |
| Presence intervention | M: requested setting captured by both arms; 9 compared rows per arm with zero differences on this short fixture |
| Corrupt expected probability | Server rejects `capture replay distribution mismatch`; M: zero branch events |
| Complete CPU smoke elapsed | M: 31.199 seconds, including boots and diagnostic branches; not an overhead benchmark |

`test_catch.py` passes exact onset mapping, low-probability entry and entropy arithmetic, matching positions, damaged-row and damaged-period negative controls, replay history/budget transport, intervention settings, unsupported zero-position rejection, frozen-byte corruption, the two-variable GPU gate, and plan-mode subprocess/HTTP sentinels. The analyzer also processed M: 3 complete original captures from the read-only earlier CPU fixture, including a tool grammar, and refused replay captures as original inputs.

```sh
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
nice -n 19 python3 codex/test_catch.py
nice -n 19 python3 codex/test_experiment.py
nice -n 19 bash -n codex/catch.sh
PLAN=1 nice -n 19 bash codex/catch.sh --phases 0 1 2
CPU_CATCH_OUT=build-catch-cpu-fresh OMP_NUM_THREADS=4 nice -n 19 python3 codex/cpu_catch.py
```

The sandbox initially blocked loopback binding; the CPU smoke completed with approved socket escalation. All writes stayed inside this worktree; the reference checkout was only read. Parent GPU execution was not attempted.

Selected dry output, observed from `PLAN=1`:

```text
"run_order": [0, 1, 2]
"phase0_trials_D": 48
"phase1_max_trials_D": 128
"phase1_stop": {"loops_D": 3, "max_trials_D": 64, "arms": ["no-spec", "S3"]}
"phase2_max_continuations_D": 1152
"offsets_tokens_D": [0, 64, 512]
"variants": ["served", "no-spec", "temp07", "presence"]
"continuation_tokens_D": 4096
"length_tokens_D": 8192
"disk_cap_bytes_D": 5000000000
PLAN: no GPU query, server, subprocess, or HTTP request executed
```

The complete dry output is `build-catch-check/PLAN.txt`.

## Source evidence and open items

Local evidence: `catch.py:33` (GPU gate), `:42` (base metadata), `:57` (frozen bytes), `:113` (guarded boots/cleanup), `:178` (bounded original trials and retention), `:229` (interventions), `:238` (strict boundary checks and branches), `:304` (completed denominators), `:385` (budget/disk watchdog); `onset.py:23` (exact period), `:58` (complete support), `:71` (min-p inference), `:83` (token metrics), `:99` (matched trajectories). Reference contracts are `tools/server/server-capture.h:74` (filtered rows), `:117` (draft q), `:131` (emitted piece bytes), `common/sampling.cpp:893` (ordinary observer), `:1024` (target p before rejection), and `tools/server/server-context.cpp:3157` (branch rebuild), all in the read-only capture checkout. Its `codex/REPORT.md` and CPU evidence were inspected. The host attribution/patch-A reports and review were read as background; this task does not introduce their graph cache changes.

The parent must establish actual loop probability trajectories, GPU strict replay fidelity and base conditional loop rates. Capture overhead on the GPU is unknown. Existing checkpoint/RNG defects and unsupported capture transitions remain unchanged; no sampling-correctness clearance follows from this harness. Branch suffixes have a shorter horizon than the original trial and can miss later re-entry. The polling disk guard is not a quota. Raw sampler-stage margins would require a separately audited capture extension; they cannot be inferred from these final rows.

User-reported context only: the parent capture campaign found no onset in M: 13 full reruns. B's verify-rows script in `~/repos/ra-verify-rows/stack2/build-gpu-verify-1238` reportedly stopped after M: 35 seconds with missing `schedule.json` and rc `0`; it was not investigated or rerun here.
