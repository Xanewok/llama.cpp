# Vanilla upstream versus served stacks

## Conclusion

D: upstream `08246a28f6000100433d297c4e037c02e9d2d464` implements Qwen35 MTP and probabilistic rejection verification, so the planned vanilla arm uses MTP with draft k=5 (D). No plain-decode fallback is needed on the source evidence. Actual Swift GGUF loading, long-context fit, speed, and vanilla request-history dependence remain unmeasured; the parent must run the GPU canaries and items.

D: this is a closest-supported configuration comparison, not a patch-only speed attribution. Vanilla lacks the draft-vocabulary restriction and the local host/kernel changes. Under tensor split, vanilla explicitly rejects backend sampling and falls back to CPU. Cross-arm token identity is disabled with a recorded reason; the history item requires identical tokens and counters within each request-order alias. No engine source or sampling law was changed.

M: CPU-only configure and compile smoke passed. Real recorded SSE, timing, overlap, cache, token, history classification, and negative controls passed. All D six dry manifests pass `gpuq submit --check`. No GPU binary, model inference, network operation, or queue mutation was performed. The CPU-only server was copied as `build-cpu/bin/vanilla-cpu-help` and executed only with `--help`.

## Feature map

| Served feature | Vanilla source observation | Planned difference |
|---|---|---|
| `--spec-type draft-mtp`, draft k=5 (D) | `common/speculative.cpp:1390`, `:2600`; `src/models/qwen35.cpp:504` implement a separate MTP context using the target model. | Same explicit speculative type, maximum/minimum/p-min and f16 draft KV flags. Model loading is gated by the real canary. |
| `--spec-draft-vocab` | Absent from `common/arg.cpp` and CPU `--help`. | Vanilla projects/samples the full vocabulary; stacks retain the served subset JSON. |
| `--spec-draft-sampling probabilistic` | `common/arg.cpp:4233`; MTP retuning at `common/speculative.cpp:1673`; rejection verification at `common/sampling.cpp:735`. | Same requested probability settings and seed. Different draft support can change acceptance and token trajectories. |
| `--backend-sampling`, `--spec-draft-backend-sampling` | Flags exist at `common/arg.cpp:2309`, `:4224`, but `src/llama-context.cpp:1316` returns false for tensor split; draft shares the target model at `common/speculative.cpp:2642`. | Flags stay explicit in vanilla; both chains fall back to CPU. Its canary requires the fallback warning. Stack candidate/backend behavior is an effective difference. |
| `-sm tensor`, split 50,50 (D) | `common/arg.cpp` registers tensor split; `src/llama-arch.cpp:1205` permits Qwen35. | Same split and internal AllReduce configuration; NCCL OFF (D). Runtime is unverified here. |
| `LLAMA_ARG_KV_PAD=4096` (D) | No matching argument/env in vanilla. | Omitted in vanilla; set in both stacks. Vanilla retains its native padding policy. |
| `GGML_SCHED_INPUT_STAGING=1` (D) | No matching scheduler env in vanilla. | Omitted in vanilla; enabled in stacks. |
| `LLAMA_MTP_CHAIN=1`, `LLAMA_MTP_FOLD_C=1` (D) | Neither local switch exists in vanilla. Upstream's multi-trained-head `chain_heads` is a different mechanism. | Omitted in vanilla; enabled in stacks. Chain falls back for concurrent sequences; stacked fold remains available. |
| `GGML_CUDA_Q8_DECODE` / package 54 | Switch absent in vanilla. | IQ4 sets `0` (D), J5q sets `1` (D), in stack arms. The switch is a no-op in stack9-p55; package 54 is in stack10. |
| `-ub 512` (D) | Supported by vanilla. | Same for np1 and IQ4 np2; J5q np2 uses the served `128` (D) fit setting. |

D: all arms share the same Swift GGUF per quant, target q8_0 KV, flash attention ON, GPU-layer setting `99` (D), batch `2048` (D), template, temperature `1` (D), top-k `20` (D), top-p `0.95` (D), min-p `0.05` (D), and CPU threads `8` (D). Slot context is `102400` (D); np2 total context is `204800` (D). Host cache is explicitly zero (D) in every arm, including np1: a deliberate difference from the single-slot serve.sh default, matching the existing timing/history harness.

D: the frozen stacks are `stack9-p55` (`45d90d3e4`) and `stack10` (`a588f2a44`); vanilla engine sources are upstream master, while stack sources originate from the earlier fork base. Consequently upstream changes between those bases are also part of the comparison. The future vanilla `SOURCE` names the topic commit containing these codex files; preflight and renderer require no engine-source differences from upstream. The freeze name retains the upstream engine revision.

D: requests use the existing real-session KLDT corpus, fixed seed `1729` (D), output cap `512` (D), and `ignore_eos=true`. This forces useful length but differs from production requests: EOG bias can force stack candidate fallback. The report must not interpret those timings as unbiased production speed or isolate individual stack gains. All stack10 component log instrumentation and round tracing are disabled for timing; vanilla has no local round-trace writer.

## Build window and provenance

Submit `codex/bw-vanilla.toml` first. It is a filler at priority `17` (D), `host_sensitive=false`, with the existing CCX wrapper. gpuq requires a nonempty GPU selector in the manifest; its payload immediately hides CUDA devices before every command and keeps them hidden for every stage; it never executes the CUDA server. Configure/build run under `nice -n 19`, affinity `6-11,18-23`, shared ccache, and `buildgate -j4` (D). The no-fetch recipe matches stack10b: CUDA `13.2` (D), architecture `120a` (D), CCCL `3.4.3` (D), NCCL OFF (D), locally supplied CCCL, disconnected FetchContent, UI fetching disabled.

The window stamps the current source and CMake settings, copies the server to `round-trace`, and calls `bin/wt freeze` for `frozen/vanilla-08246a28f`. The helper creates `SOURCE`, `$ORIGIN` RUNPATHs, and `build/bin/SHA256SUMS`. The payload adds `RECIPE.txt`, `STAMPS.txt`, and the lightweight tag `refs/tags/frozen/vanilla-08246a28f`. Stage `49` (D) renders the real items into `codex/rendered/` and runs contract checks only. Nothing is submitted automatically.

G: window estimate `2400 s`, based on one server CUDA build with shared ccache and no CUDA test suites. D: its wall cap is `7200 s`; this reserves queue time but executes no GPU workload. A cold ccache can exceed the estimate. Freeze is intentionally non-retryable; a failed post-freeze render must be rerun by the parent against the existing freeze, rather than rebuilding or overwriting it.

## Items and bounded GPU budget

| Manifest after stage 49 (D) | Workload | est_s (G) | budget_s (D) |
|---|---|---:|---:|
| `codex/rendered/vanilla-depth-j5q.gpuq.toml` | np1 real corpus at 8192 / 32768 / 90112 tokens (D) | 1380 | 2000 |
| `codex/rendered/vanilla-depth-iq4.gpuq.toml` | Same depths (D), IQ4 pkg54 OFF in stacks | 1380 | 2000 |
| `codex/rendered/vanilla-np2-real-j5q.gpuq.toml` | Same depths (D), sequential cold prefills then joint cached decode | 1620 | 2350 |
| `codex/rendered/vanilla-np2-real-iq4.gpuq.toml` | Same np2 design (D), IQ4 pkg54 OFF | 1620 | 2350 |
| `codex/rendered/vanilla-history-iq4.gpuq.toml` | Vanilla only: short / long / long / short (D) | 900 | 1360 |
| `codex/rendered/vanilla-cold-prefill-iq4.gpuq.toml` | One cold 8192-token request (D) per boot, same arms and MTP flags | 450 | 740 |

D: comparisons use `vanilla,s9p,s10,s10,s9p,vanilla`, two fresh boots per arm. Each canary is one vanilla boot at the smallest depth, with the same output/steady/overlap gates as the full run. History's one-boot canary runs the full long order so its real 90k record exercises history classification and repeat negative controls without another GPU boot. Canary caps are `240 s` (D) for comparisons, `300 s` (D) for history, and `180 s` (D) for cold prefill.

G basis: recorded stack10 long-order np1 work was approximately `151 s` (M, identity2 report), with the middle request approximately `36 s` (M). Allow `180 s` (G) per np1 boot, `220 s` (G) per np2 boot, `150 s` (G) per average history boot, and `45 s` (G) per shallow prefill boot; estimates include canary and transitions. Full-vocabulary CPU sampling may make vanilla slower than these estimates. Item timeout is an explicit budget result, never silently converted to a number.

D: total measurement wall caps sum to `10800 s` = `3 h`, including all canaries and cleanup allowances. G: aggregate estimate is `7350 s` = approximately `2.04 h`. Adding the GPU-idle build estimate gives `9750 s` (G), approximately `2.71 h` of queue occupancy; its separate wall cap is not GPU measurement time. Every stage estimate plus canary is at most `0.7` (D) of its item's wall budget.

The reused harness reports steady full-k interior ms/round and excludes warmup, tail, missing boundaries, and nonpositive periods. `vanilla_runner.py:finish` adds per-arm means and SE across boots for ms/round, token rate, and cold prefill, plus paired deltas against vanilla for every arm. D: the independent unit is the boot; the two np2 streams share batch steps and are averaged before calculating SE. With two repeats (D), SE measures boot noise on this fixed prompt/seed, not workload variability. Different drafts/counters are reported, not treated as exact optimization attribution.

The history item compares `8192->90112` versus `8192->32768->90112` (D), repeats both in reverse order, and requires each alias to repeat exactly. Both target requests must report `cache_n=0`, `prompt_n=90112` (D), enough native tokens, and at least `40` (D) steady intervals. `history-diagnostic.json` reports `order_dependent`, token/counter equality, and first differing token. A same-order repeat mismatch is integrity failure, not a verdict. The vanilla verdict is currently **unknown**.

## CPU evidence and checks

Commands ran with `CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1` and the required nice/affinity. Evidence lives in `codex/evidence/`; text captures have trailing whitespace removed.

- M: `buildgate cmake -S . -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DGGML_OPENMP_FETCH=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF` passed (`cpu-configure.log`). The cache confirms CUDA OFF and ccache ON. CMake reports unused FetchContent-disconnect variables in the CPU configuration because no content is fetched there.
- M: `CCACHE_READONLY=1 CCACHE_TEMPDIR=$PWD/build-cpu/ccache-tmp buildgate cmake --build build-cpu --target llama-server test-arg-parser -j4` passed (`cpu-build.log`). Read-only ccache permits shared hits without sandbox writes outside this worktree. `test-arg-parser` was compiled but not executed: its download tests would violate the no-network rule. No CPU model timing number was collected.
- M: renamed CPU `--help` confirms tensor split, MTP, probabilistic drafting, and both backend flags; subset-vocabulary and kv-pad flags are absent (`cpu-help.txt`). Source establishes the tensor backend-sampling fallback; help alone cannot prove GPU engagement.
- M: parser replay uses `orchestration/gpuq/fixtures/llama-sse/long-tool-s1-1.json`, GPU CSV/journal fixtures, recorded native np1/np2 depth records, and real short/long stock 90k artifacts from `wt/s10-identity`. Null/partial counters, insufficient length, changed token, reused cache, disjoint overlap, repeat mismatch, and history classification controls pass (`parser-controls.log`). The same native records exercise summary/SE code. No parser is tested only on a CPU-generated response.
- M: dry renderer uses a labelled, non-executable stand-in rooted inside `codex/evidence/stand-in-vanilla/`; its manifests have `dry-` IDs and titles `DO NOT SUBMIT`. The runner explicitly rejects that root for canary/execute. The real renderer accepts only the intended frozen destination. Dry contract logs are in `dry-render/`; captures are in `capture/`.
- M: shell syntax, `gpuq.shlint`, window preflight, window contract check, clean-environment window check, all dry contracts including clean-environment checks, and capture assertions passed; detailed commands/status are in `validation.json`; reproduce with `nice -n 19 taskset -c 6-11,18-23 python3 codex/test_validation.py` after the labelled dry render. Capture uses an explicit port to avoid opening a socket in the sandbox.

## Risks and open work

GPU execution remains entirely with the parent. Vanilla MTP support is established by source, not by a Swift-model boot; every measurement starts with that real canary. Vanilla np2 J5q fit is not known. The diagnostic's full vocabulary, CPU sampling, native padding, and different upstream base prevent a causal claim about one stack patch. Timeout or fit failure is not speed evidence. Neither upstream identity nor a GPU speed gain is claimed here.

`gpuq status --brief` cannot acquire its state lock in this sandbox. Read-only `~/.local/state/gpuq/status.json` showed the acceptance item `spec-tf-j5q` running (M), not a depth/perf timing A/B, before local compilation. No lock or queue state was modified. The requested SP reports and old frozen-A/B script are absent at the supplied paths; the retained build-window scripts, source, harness contract, history report, and actual recorded artifacts were used instead. No writes were made to other trees, the parent repo, or frozen builds.
