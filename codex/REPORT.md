# graph-reuse-tandem

## Conclusion

Ready for the parent-only GPU canary and three-arm A/B. CPU token, candidate ID, proposal probability/q, sampled-logit and final-target-logit bytes match up on the tested fixtures. Whether graph reuse pays off with input staging is **unknown**; no GPU executable or GPU query ran.

Labels: M measured/observed, D derived, G estimate. Revisions, paths, seeds, K, dimensions and command options are configuration identifiers.

## Integration and provenance

Topic: `topic/graph-reuse-tandem` in `codex/local.git`, based on `75a8a897d`. The outer `.git` is untouched. Local, offline object import from `~/repos/input-staging/codex/local.git` supplied the staging commits. No external repository was written and no network operation occurred.

- `19759fb85`: exact cherry-pick of `5f73e984d`, owned staging and ordered CUDA uploads.
- `6918e7773`: exact cherry-pick of `ec17de5fb`, synchronous host producers and safe graph-alternation fence bypass.
- `15f29401c`: inherited CPU fixtures/controls, rerun against the combined source; separate output names preserve the earlier prefill/cached evidence.
- `9d37533ea`: balanced three-arm harness and real-artifact replay controls.
- `561be5151`: restrict backend discovery to the frozen snapshot and flush boot gate progress.
- Final artifact commit (`HEAD`): freeze manifests, normalized up metadata, gpuq draft, dry plans and CPU evidence.

M: `git --git-dir=codex/local.git range-diff 7fc0244d7..ec17de5fb 75a8a897d..6918e7773` marks both production commits `=`. No merge conflicts or additional production edits. The design combines the existing spare meta arena/hash traversal with the existing two pinned input slots; it adds no sampling arithmetic or RNG draw.

Source contracts: `ggml/src/ggml-backend-meta.cpp:1933,2156,2679` retains arena capacity while rebuilding topology and releasing container ownership; `ggml/src/ggml-backend-impl.h:171` defines owned transfer; `ggml/src/ggml-backend.cpp:1730,1767,2220` selects host producers, fallback and allocation/capability checks; `ggml/src/ggml-backend-meta.cpp:2044` preflights/drains ranks; `ggml/src/ggml-cuda/common.cuh:1452` owns slots; `ggml/src/ggml-cuda/ggml-cuda.cu:703,2557,2607,2650` waits before slot reuse/free and orders compute -> upload -> compute; `src/llama-context.cpp:1535` retains fences for unsupported alternation.

The requested `codex/frozen/up` uses `0dba27b9002b8347c6f14a9f4a1651b3478065ef`; its initially missing local binaries were copied from the original read-only `~/repos/round-trace/llama.cpp/codex/frozen/up`, with every original binary digest preserved. Its legacy top-level binary manifest was normalized to the current SOURCE/bin-SHA manifest layout; source and binary bytes did not change. CPU up is the existing `codex/BASE`/`build-base-cpu` source `2b645a655`, the instrumentation descendant used by ab3. Those intervening production changes add tracing, not sampling arithmetic. The graph-prefill snapshot remains unchanged at `7dd59ccc7`; production code equals the prefill topic through `75a8a897d`.

## CPU evidence

All commands used `CUDA_VISIBLE_DEVICES=''`, nice priority 19, builds `-j8`, and at most eight compute threads. Meta uses four threads per rank. Tests link the CPU server context in process; none launches a process named `llama-server` or opens an HTTP listener. Both CPU build trees have `GGML_CUDA=OFF`.

M PASS: eight synthetic seeds and four real-model seeds, K=3/K=5, CPU/Meta(CPU,CPU), up/tandem/tandem/up, twelve rounds per run. All raw token/ID/q/probability/logit bytes match. Per backend/K/arm: synthetic M 16 runs and M 144 retained rounds; real M 8 runs and M 72 retained rounds. Tandem retains C/D1 hits with D 0 build-union time after the first three rounds.

D medians of per-run mean decode wall, ms/round; these exclude sampling readback and trace flush and establish no GPU speed:

| Backend / K | Synthetic up D | Synthetic tandem D | Real up D | Real tandem D |
| --- | ---: | ---: | ---: | ---: |
| CPU / 3 | 0.950 | 0.819 | 151.917 | 152.399 |
| CPU / 5 | 1.281 | 1.168 | 190.769 | 190.303 |
| Meta(CPU,CPU) / 3 | 2.743 | 1.707 | 238.623 | 240.167 |
| Meta(CPU,CPU) / 5 | 3.480 | 2.425 | 324.631 | 324.711 |

M PASS: the test-only CPU owned hook matches up bytes on four seeds per real/synthetic fixture and K. Each run has M 18 staged C/D1 alternations and M 0 host fences. The real rejection fixture matches M 48 completion tokens and every final-target-logit byte at K=3/K=5, seeds 1729/2718.

M PASS: binding/UID/epoch/borrowed-container contracts, `test-llama-archs --tp-sampling`, and `test-sampling`. Negative binding and epoch mutants exit M 1 with the intended contract failures. Production CUDA upload code, compiled into CPU delayed queues, passes ownership/ring/overlap/refusal/allocation-fallback/teardown checks; removing each of the three event-ordering waits exits M 1. These controls remain in `codex/negative_reuse.py` and `codex/test_input_staging.py`.

M PASS: repeated cold prefill, K=3, numeric grammar, M 534/555 prompt tokens. All up/tandem/unpooled/reuse-disabled token and final-logit bytes match. Target prefill has M 17/18 ubatches and M 5 misses per request in up/tandem/unpooled. Tandem creates M 0 target arenas; unpooled creates M 5 and requests M 87,516,000 arena bytes per request, failing the same allocation contract. Target build unions are up D 56.687/47.697 ms, tandem D 61.392/52.052 ms, unpooled D 60.305/53.816 ms. This single boot per arm does not resolve total prefill speed. The reuse-disabled control removes hits and preserves bytes.

M PASS: K=5 cached up/tandem/tandem/up, two requests per boot, M at least 40 interior SSE intervals per request, cached prefixes and exact token/final-logit bytes. The disable control fails the C/D1-hit contract while keeping bytes.

Model: existing `~/models/Qwen3.5-4B-UD-Q4_K_XL.gguf`. TP target uses the inherited worktree-only untied-head adaptation, copying embedding blocks byte-for-byte; drafting retains the original compact MTP head. CPU results are neither unmodified tied-head TP coverage nor quantitative Swift GPU predictions.

Evidence: `codex/{synthetic,real,owned,server}-input-results.json`, `codex/input-contract-results.json`, `codex/negative-results.json`, `codex/tandem-prefill-summary.json`, `codex/tandem-cached-cpu-results.json`; raw records/logs stay under ignored `codex/input-results/` and `codex/reuse-results/{tandem-prefill,tandem-cached}/`.

Reproduce, sequentially:

```sh
CUDA_VISIBLE_DEVICES='' nice -n 19 bash codex/run_input_cpu.sh
CUDA_VISIBLE_DEVICES='' CPU_CAMPAIGN=tandem-prefill CPU_CACHE_PROMPT=0 CPU_ARMS=up,patch,unpooled,disabled PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 codex/prefill_cpu.py
CUDA_VISIBLE_DEVICES='' CPU_CAMPAIGN=tandem-cached CPU_TARGET_MODEL="$PWD/build-tmp/reuse-untied-target.gguf" PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 codex/run_ab2_cpu.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 codex/test_tandem_harness.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 codex/test_harness.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 codex/test_attrib.py
```

The unpooled allocation control depends on the existing saved CPU executable/libraries in `codex/reuse-results/prefill-unpooled/bin`, not a toggle on the candidate.

## Parent harness and budget

`codex/gpu_ab.sh` now invokes `gpu_harness.py tandem`; the older `ab`/`trace` entry points remain available directly. Full run: K=5 then K=3, each up/prefill/tandem/tandem/prefill/up. `--k 5` or `--k 3` selects one configuration. All arms are trace-off; frozen bytes and source manifests are checked and copied per boot. External backend search paths are removed.

Workloads/seeds match ab3: original cold tool after warm-up; long tool/think, seeds 1/2, fixed literal-file prompts and 2048 max tokens; eight sequential cached tool requests with 768-token headroom. Canary: one boot per arm at K=3, one long tool and think at seed 1, one cold tool, two short requests to reach a true cached follow-up, plus warm-up. It exercises the same size/round/cache/counter/content/hash/Xid/PID/budget gates and all summary parsers; K=5 has the same gates in the full run.

The names cold-10k/cold-1k identify the inherited request classes: recorded ab3 prompts are M 8,412/8,436 and M 602 uncached tokens, not newly padded prompts. Summary retains actual prompt/cache counts. Output: `kK-ab-results.json`, per-boot request/error/validation records, `kK-identity.json`, `ab-summary.json` and `verdict.json`. Summaries report each arm's steady ms/round, cold/cached prompt time, generation excess, fixed-cost estimate, and net request wall; compare each arm's first/second boot against the corresponding up boot, retaining pair differences, ranges and per-arm boot spread. These ranges are not confidence intervals. Short excess uses the same boot's long-tool steady rate; ramp/tail and dispatch costs remain included.

Failures record requests before boot validation, preserve partial SSE/error bodies, stop the exact PID/start time, and produce INVALID/hazard/budget without a performance summary. Memory release is waited for; port probing uses SO_REUSEADDR. GPUq handles RAM, foreign/desktop ownership, boot-ID and debounced-idle gates. No resume identity is reused: `retry=fresh`, new outputs per attempt, complete hash inventory in `pins.files`.

M PASS: `gpuq/fixtures/llama-sse/` parser regression (including its insufficient-round control), decisive ab3 K=5/K=3 request replays, complete three-arm canary and full-run control-flow replays, plus identity/steady/cache/HTTP/partial-timing/frozen/PID-reuse/Xid/idle/budget failure controls. All launches, GPU commands and sockets are mocked in execution-flow tests. Authoritative metric values and device memory values come from recorded artifacts; HTTP Prometheus formatting and utilization polling use mocked transports. The actual transport remains a real-canary check.

Stage estimates use recorded up request wall M 22.891-23.080 s (K=5) and M 26.375-26.755 s (K=3) and model-ready M 5.558 s from the K=3 server log. Conservative estimates:

| Stage per boot | Estimate |
| --- | ---: |
| idle/load/snapshot | G 20 s |
| warm-up | G 5 s |
| cold tool | G 3 s |
| each long request | G 30 s |
| each short request | G 3 s |
| stop/Xid/hash | G 10 s |

Initial input hashing: G 20 s per invocation. Canary: D 3 boots x G 124 s + G 20 s = G 392 s, below D 420 s (0.7 x configured 600 s). Full: D 12 boots x G 182 s + G 20 s = G 2,204 s, below D 2,520 s (0.7 x configured 3,600 s). Combined estimate G 2,596 s is below D 2,940 s (0.7 x manifest budget 4,200 s). Estimates are not runtime measurements and do not predict a GPU speedup.

## Frozen build and manifest

M PASS: CUDA compile and freeze at source `561be515111e66c98cb93f4d9002def11a8ad4d5`, stored in `codex/frozen/graph-tandem`. Toolkit 13.2, architecture 120 (mapped to 120a), NCCL off, compiler forced from configure to suppress the CUDA device/ABI probe. `GGML_OPENMP_FETCH=OFF` and `LLAMA_BUILD_UI=OFF`; no download. M 23 binary/library files per arm verify against both manifest levels; the server binary is named `round-trace` in each freeze. SOURCE and manifests are committed; local binaries remain ignored and must be preserved for the queued run.

Build command (compile/link/copy only):

```sh
CUDA_VISIBLE_DEVICES='' FREEZE_PATCH_ONLY=1 PATCH_FREEZE_DIR=codex/frozen/graph-tandem nice -n 19 bash codex/freeze_reuse.sh
```

Draft: `codex/gpuq-graph-reuse-tandem.toml`, kind measure, ab=true, retry=fresh. M 90 pinned files exactly equal `tandem_pins()`'s complete hash inventory, including all frozen SOURCE/manifests/binaries, model, vocabulary, inputs, wrapper, harness and imported attribution code. Separate canary/full outputs live under `codex/runs/graph-reuse-tandem/`, with attempt-specific names. Fresh retry only voids this item's output tree. M PASS: installed gpuq schema validation and `submit --check`; no submission occurred. Dry plans for both K values, selected individual K values and canary pass without launch or socket calls. The earlier up manifest needed only format normalization; its binaries were not rebuilt.

Exact parent commands, not submitted here; gpuq supplies selected GPUs and `XID_SINCE`, runs the real canary, and starts measurement only on its success:

```sh
CUDA_VISIBLE_DEVICES='' nice -n 19 /home/xanewok/repos/gpuq/bin/gpuq submit --check /home/xanewok/repos/graph-reuse/codex/gpuq-graph-reuse-tandem.toml
CUDA_VISIBLE_DEVICES='' nice -n 19 /home/xanewok/repos/gpuq/bin/gpuq submit /home/xanewok/repos/graph-reuse/codex/gpuq-graph-reuse-tandem.toml
```

Harness references: `codex/gpu_harness.py:132,168,251` implements idle wait, exact cleanup and persistent requests/errors; `:321,353,423` parses SSE/steady rounds and validates all requests; `:466,558` summarizes boot spread and gates identity; `:587,596,605` defines the pin inventory, estimates and tandem execution. CPU/harness/build logs are retained as ignored `codex/tandem-*.log`, `codex/build-cuda-safe.log` and `codex/cmake-cuda-safe.log`. No CUDA binary was executed in this task.

## Risks and open items

GPU payoff, proposal/q/logit byte identity, CUDA capture/replay, cross-context stress and pinned-slot backpressure remain untested. GPU identity compares normalized completion/finish reason, authoritative rounds/draft/accept counts and prompt/cache/output counts; it is not a GPU q-byte capture. The combined source changes transport/storage lifetime, not arithmetic or draws; CUDA event correctness still needs runtime validation.

Pinned slots retain up to twice their maximum input payload per backend, and pooling retains one high-water host arena per meta backend. CPU fallbacks and allocation controls do not measure pinned-memory high water, CUDA scheduling or total GPU request overhead. Two boots per arm/K can expose boot spread but may not resolve small differences. Parent adoption requires the real canary and exactness-gated full A/B.
