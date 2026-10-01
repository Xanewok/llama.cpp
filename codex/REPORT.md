# Production request capture and replay (DO NOT MERGE)

Worktree: `/home/xanewok/repos/ra-capture/stack`.

Conclusion: the env-gated capture and replay prototype is implemented on `codex/request-capture`, based on stack2 `0dba27b9002b8347c6f14a9f4a1651b3478065ef`. CPU end-to-end checks pass with real probabilistic MTP, an activated lazy tool grammar, and reasoning-budget transitions. The CUDA build is frozen at `codex/frozen/capture`; no GPU binary was executed. Production runaway capture, long-context fidelity and arm comparisons remain parent work. This does not clear or fix the repetition bug.

Labels: M = measured/observed, D = derived/planned, G = guess. Commit hashes, source lines, versions, enum values and literal configuration identifiers are not estimates.

## Design and contract

`LLAMA_CAPTURE_DIR` enables per-request JSONL files. `LLAMA_CAPTURE_PROBS=1` adds filtered target rows. `LLAMA_CAPTURE_LIMIT` clamps the effective response limit even when a request explicitly asks for more. Without the directory variable, no capture observer is installed and the private replay transport is rejected.

- `tools/server/server-capture.h:51` records the original parsed HTTP JSON, the converted completion JSON, effective sampling settings, resolved target seed, exact prompt IDs and reasoning-budget token sequences. `server-context.cpp:4043` also records the actual evaluated prompt after cache/tokenization handling. Original tools and parser/grammar configuration remain in the request representations.
- `server-capture.h:117` records each sampling round, actual draft token IDs and their complete candidate probabilities. `server-context.cpp:4151` records the retained verify result and accepted proposal count. `server-capture.h:131` records every emitted ID, its round, and exact token-piece bytes in hex; this preserves split UTF-8 tokens without invalid JSON.
- `common/sampling.cpp:1024` observes the verifier's normalized, grammar-masked target p before acceptance or residual sampling. Ordinary/match/bonus rows use `common_sampler_observe` at `sampling.cpp:893`: candidate and grammar copies isolate diagnostics from live sampler state. Captured rows are filtered target rows, not residual distributions. With grammar-after-sampling, this is the row reached by the actual sampling path; it is not an independent proof that the existing grammar algorithm has the intended unconditional law.
- `server-context.cpp:4071` teacher-forces generated IDs through the normal decode path. The original prompt is accepted with `is_generated=false`; generated history uses `true`, preserving penalties, lazy trigger/grammar acceptance and reasoning budgets. `server-context.cpp:3157` constructs a fresh branch sampler, replays those two history roles, and uses an independent explicit branch seed for target and draft streams. Model/draft contexts have already processed the forced history through the existing MTP process hook.
- `server-capture.h:103` validates ordered IDs, probabilities, omitted mass and budget state before emission. The default absolute probability tolerance is D: `1e-6`. Strict mismatch stops the request with `capture replay distribution mismatch`. The helper refuses missing rows or top-20 truncation exceeding that tolerance.

Replay is text-only, without LoRA, on an exclusive server slot. Mirostat, adaptive sampling and XTC are rejected because accepting token history does not reconstruct their sample-dependent state. Terminal IDs cannot occur inside the forced prefix. Captures containing context shifts, checkpoint restoration, dynamic reasoning control, incomplete responses or missing evaluated prompts are refused. These limits are explicit; those transitions are never silently discarded. The known checkpoint acceptance-RNG bug is unchanged.

## CPU evidence

Fixture: `/home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf`; target CPU, `GGML_CPU_REPACK=OFF`, FA enabled, renamed `capture-cpu-srv`, nice 19, D: four inference threads. No test executable named `llama-server` was started. Full captures and server logs remain in `build-capture-test5` and `build-capture-budget2`; committed results are in `codex/CPU-EVIDENCE.json`.

| Check | Result |
|---|---|
| Capture disabled/enabled, ordinary seeded sampling | M: 24 emitted IDs bit-identical |
| Capture disabled/enabled, probabilistic MTP at configured k=4 | M: 22 emitted IDs bit-identical; termination occurred before the requested limit |
| Tool-calling request with lazy grammar | M: 300 prompt IDs; `<tool_call>` triggered; tools, converted grammar and triggers captured |
| Explicit request limit versus environment clamp | M: 48 generated IDs despite a requested limit of M: 200 |
| Tool response speculative rounds | M: 10 verifies; M: 37 proposals, all M: 37 accepted |
| Teacher forcing plus independent branches | M: 13 rows per run, M: 39 matching rows total; M: two branches of eight suffix tokens each |
| Probability agreement | M: maximum absolute error zero on those positive validation rows |
| Negative control | Corrupting a captured probability fails with `capture replay distribution mismatch`; M: zero branch events in the failed request |
| Reasoning budget | M: COUNTING/FORCING/DONE observed; M: seven replay rows validate across the transition |
| Tool fixture elapsed time | M: 75.230 seconds, including capture/replay/branches and the negative control, excluding initial server boot and identity fixtures |

The elapsed time is a loaded CPU fixture duration, not an overhead benchmark. The budget fixture exercises actual forcing, rather than only transporting configuration. The tool fixture's budget state is DONE while its grammar constrains tool arguments.

`codex/test_capture.py` passes prompt/generated role transport, grammar and budget fields, banned logit biases, split UTF-8 bytes, the final-capture-flush race, unsupported-transition rejection, a changed-library negative control and plan-mode sentinels. Sentinels reject subprocess, HTTP or GPU queries in default plan mode. Shell syntax and Python compilation pass.

Reproduce in fresh output directories:

```sh
export CUDA_VISIBLE_DEVICES="" TMPDIR="$PWD/build-tmp" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
nice -n 19 cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CPU_REPACK=OFF -DGGML_CCACHE=OFF -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_APP=OFF -DCMAKE_BUILD_TYPE=Release
nice -n 19 cmake --build build-cpu -j4 --target llama-server
CAPTURE_CPU_OUT=build-capture-fresh nice -n 19 python3 codex/cpu_e2e.py
CAPTURE_BUDGET_OUT=build-budget-fresh nice -n 19 python3 codex/cpu_budget.py
nice -n 19 python3 codex/test_capture.py
nice -n 19 bash codex/gpu_ab.sh
```

The sandbox blocked the initial loopback socket. Authorized CPU HTTP runs completed after socket escalation; all inference still had CUDA hidden. Initial build/test client failures were corrected: explicit nested `common_json` conversion, and treating model-loading HTTP responses as retryable. No network download was attempted by the configured builds.

## CUDA build and frozen artifact

M: CUDA configuration and both initial/final builds exited successfully. Configuration: CUDA toolkit `/usr/local/cuda-13.2`, `GGML_CUDA_NCCL=OFF`, `CMAKE_CUDA_ARCHITECTURES=120`, Release, ccache/curl/UI downloads off, D: four build jobs. The compiler is forced to avoid running a CUDA compiler probe. Selected FA instances match the supplied stack build.

```sh
export CUDA_VISIBLE_DEVICES="" TMPDIR="$PWD/build-tmp"
nice -n 19 cmake -B build-cuda -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DCMAKE_CUDA_COMPILER_FORCED=ON -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 '-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16' -DGGML_CCACHE=OFF -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
nice -n 19 cmake --build build-cuda -j4 --target llama-server
nice -n 19 python3 codex/freeze.py
```

Frozen engine source: `8a15734ca7f8e1315007da7813e83b9893f10f92`. `codex/frozen/capture/build/bin/capture-srv` and local libraries use `$ORIGIN` RUNPATH. M: 24 SHA256 pins validate, recorded in `codex/FROZEN.json`; the renamed executable equals the pinned server bytes. Only host tools read or modified these ELF files. Freeze is intentionally outside git; the source, harness and pin manifest are committed. System CUDA libraries are the installed toolkit, as in the existing frozen-stack pattern.

## Parent execution

`codex/gpu_ab.sh` defaults to planning. Only `RUN_PARENT_GPU=1 PLAN=0` executes. The default starts an interleaved frozen up/capture/capture/up greedy identity smoke, then reruns the original `rust/alphametics` and `rust/nucleotide-codons` tasks through read-only weekend harness modules under S4 with a D: 40,000-token per-response cap. No reconstructed historical chat or tool simplification is substituted.

The detector consumes generated token pieces only, maps its first hit to a captured token boundary, and selects D: 2,000 generated tokens before onset. A strict S4 replay must validate first. Continuations then rotate/reverse S0, S4 and no-spec boot order, default D: four continuations per arm, D: 2,048 suffix tokens. Changed arms log numerical row differences explicitly; budget-state disagreement remains fatal. S4 remains strict. Extra supported arms are S3, S5, S4-f16 and S4-convert. The latter sets `GGML_CUDA_FA_MMA_Q8_0=0`; it does not eliminate the padded-row dispatch.

```sh
cd /home/xanewok/repos/ra-capture/stack
RUN_PARENT_GPU=1 PLAN=0 bash codex/gpu_ab.sh --budget 7200 --capture-budget 3600 --out "$PWD/build-parent-capture"
```

Both budgets are D: planned seconds, with D: 30 seconds reserved for cleanup. A fresh output directory is required. To branch from an already completed capture, pass `--capture /absolute/path/request-....jsonl`; to extend the matrix, pass `--arms S0 S4 no-spec S3 S5 S4-f16 S4-convert`. `--tolerance` changes the recorded numerical tolerance, not grammar/budget state checks. The standalone `replay.py` targets an already running capture-enabled arm; its help documents position, continuation count and suffix length.

The driver checks frozen bytes on every boot, idle GPUs and existing Xid/fallen-off logs, occupied ports and shared benchmark locks; it stops exact owned Popen children, reserves cleanup time and checks Xid after boots. The existing task runner also owns exact process groups and sets parent-death cleanup. External downloads are blocked and Rust tests remain offline. Outputs and local locks stay inside this clone; existing shared locks are opened read-only. The parent manifest pins model, subset, template, detector and local drivers. Interrupted requests, no replayable onset, a fidelity mismatch or an exhausted budget are not negative continuation outcomes.

## Remaining uncertainty

- No GPU boot, production task rerun, long-context replay or alternative-arm continuation was executed here. Parent process/transport integration is source-audited and contract-tested, not GPU-validated.
- Fresh teacher forcing can expose differences from a production prompt cache, speculative rollback or CUDA batch shape. A failed strict S4 replay is evidence to retain and investigate, not permission to call the prefix faithful. The default GPU tolerance is uncalibrated.
- Capture changes host timing through copies, JSON encoding and per-event flushes. Its GPU overhead and effect on a timing-sensitive defect are unknown. The cap cannot test onset after its horizon.
- Top-20 rows cover the supplied top-k setting. Other target chains can have omitted support and are refused by strict replay. Filtered-row agreement does not independently prove the existing grammar-after-sampling law.
- This prototype does not serialize arbitrary adaptive sampler state, multimodal/LoRA inputs or checkpoint/control/context-shift transitions. It records those unsupported transitions where applicable and refuses replay. The campaign uses text requests and disables context shifting.
