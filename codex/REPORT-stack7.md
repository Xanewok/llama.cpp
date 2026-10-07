# stack7 integration

## Conclusion and validation status

Source integration is complete on `topic/stack7`, based on stack6 `ac7465885`. No build, link, project test, model process, GPU tool, parser replay, GPU submission, or network operation ran in this session. Only shell/Python/TOML syntax checks and `git diff --check` ran. CPU measurements and integrated runtime exactness remain unknown until the build window passes; GPU speed is unknown.

The requested background reports were read from the supplied scratchpad: `gpu-sampling/REPORT.md`, `gpu-sampling/items/96b-attrib.md`, `patchA/REPORT.md` and `codex-reviews/patchA.txt`. Their host-gap attribution motivates the components; this integration does not extrapolate a GPU gain from it.

## Source commits

Only source/test paths outside the component branches' `codex/` trees were replayed, in the requested order. The source-tip hashes in the brief often name scratch/test follow-ups rather than the feature's first commit. Source mapping is also in `codex/source-map.tsv`.

| Source | stack7 | Subject |
|---|---|---|
| `c68ec4329` | `29e71fa0b` | ggml : read gated delta net state by row id |
| `8d8d2ed1c` | `3637dda66` | llama : use a recurrent state getter for GDN |
| `0a473e886` | `6727f3cb1` | cuda : trace GDN getter engagement |
| `743acb55f` | `0a3054288` | llama : remove the replaced stack6 GDN gathers |
| `f61fb1bfb` | `9a3af300a` | virtgpu : do not claim GDN state ids support |
| `5c7e31094` | `4ba349250` | llama : read GDN state by id only for single-sequence ubatches |
| `8f91e4816` | `8c013b9bb` | tests : check the GDN state getter gate by ubatch sequence count |
| `3b9585e43` | `a59f806e4` | cuda : log GDN cache fusion per captured graph |
| `7ef7b5873` | `8646cbd0a` | tests : roll back the multi-seq split replay within the last ubatch |
| `3c97c4af8` | `ba8bc3fa0` | tests : GDN getter gate tolerance above the batch-shape noise floor |
| `36cb09139` | `447a47613` | llama : remove KV sequence ranges through the position index |
| `3b5176efd` | `89637899d` | speculative : copy MTP verify hidden rows in one readback |
| `f51358cc0` | `f61bee3fa` | host-trims : report exercised paths for local A/B checks |
| `81159be0d` | `d7bc08c2c` | host-trims : env switches LLAMA_HOST_TRIMS_SEQRM and LLAMA_HOST_TRIMS_BULKH restore the stack6 paths |
| `f8a00adc5` | `89cff0ccf` | host-trims : read the target nextn pointer after the draft catch-up decode; BULKH=2 keeps the hoisted pointer as a control |
| `5e7968aee` | `f27ce2576` | host-trims : test-only LLAMA_HOST_TRIMS_VERIFY cross-checks and LLAMA_HOST_TRIMS_MUTANT negative controls |
| `81f954b2a` | `a7f31a570` | speculative : log and offset the bulk row negative control |
| `06bfba82f` | `151ab739b` | graph : split build_sampling into per-row helpers |
| `ac4f39358` | `96e025496` | llama : load a compact token embedding table for the draft vocabulary |
| `26e8945eb` | `d298d3dfd` | qwen35 : add a device-side MTP draft chain |
| `27f71a822` | `aa037232c` | sampler : add a top-k that sorts its backend candidates |
| `74c460137` | `21ba77aa8` | speculative : draft the MTP steps of a round in one decode |
| `1e51d8c1f` | `4be92a4fa` | speculative : add negative controls and a draft dump for the MTP chain test |
| `0c5c3d451` | `d92da203f` | speculative : attach a fresh sampler each time the MTP chain is switched |
| `74618074e` | `9afc6714e` | ggml : move round trace writes off the decode thread |
| `8a0a5e786` | `6e99847f9` | ggml : restore static trace kinds and strengthen writer tests |
| `7f1fe261f` | `ffdb7d6e8` | llama : fold the MTP catch-up K/V into the target decode (LLAMA_MTP_FOLD_C) |
| `cd4a9d49a` | `e99e52ef3` | speculative : add a draft dump and a fold mutant for the MTP fold test |
| `376e5c8d9` | `14b23db7c` | ggml : include host-backed Meta views in graph planning |
| New | `ce555540a` | models : flatten SSM output projection inputs across sequences |
| New | `d72722a99` | llama : report stack7 canary and mutant engagement |

The artifact commit `codex : add the stack7 build window and GPU item templates` adds the build window, its test fixtures, item templates, queue entry and this report. Existing component test drivers were copied individually and adapted to this tree; no component scratch tree was imported wholesale. `ee32a870d` and the scratch-only changes in `abe4f6ad2`/`186226ceb` were excluded. The engagement control from `c73714737` is carried as `codex/test_nh1_meta_engagement.py`, not as an unrelated source commit.

## Conflicts and resolutions

`7f1fe261f` conflicted at `common/speculative.cpp:19`: the chain already supplied `<cstdio>` while fold added `<cstdlib>`; retain both once. It also conflicted at `src/llama-ext.h:103`: retain the chain, sorted-top-k and fold APIs together. Automatic merges preserved independent chain/fold fields, graph-reuse keys and context initialization.

`cd4a9d49a` conflicted at `common/speculative.cpp:1596`: retain sampler cleanup and close both dump files with separate braces. At `common/speculative.cpp:2042`, retain both step-loop dump calls. Fold tests run with the chain off, and chain tests use their own dump; the combined control uses the chain dump and fold trace counter. No other patch required manual conflict resolution.

## Defaults and switches

| Component | Default | Switch and effective gate |
|---|---|---|
| GDN getter | On when eligible | `src/models/delta-net-base.cpp:552`: `keep && cparams.offload_kqv && n_seqs == 1`, then backend support probe. Multi-sequence ubatches retain the gather. |
| Indexed `seq_rm` | On | `LLAMA_HOST_TRIMS_SEQRM=0` restores the full scan. |
| Bulk verify hidden rows | On | `LLAMA_HOST_TRIMS_BULKH=0` restores per-row access; `=2` is the historical early-pointer control. The default fetches the pointer after catch-up at `common/speculative.cpp:1733`. |
| Draft chain | Off | `--spec-mtp-chain` or `LLAMA_MTP_CHAIN=1`; needs probabilistic/backend draft sampling, zero p_min, reduced vocab, one active drafting sequence, supported k and model. Multiple active sequences fall back. |
| NH1 fold | Off | `LLAMA_MTP_FOLD_C=1`; one configured sequence, separate draft KV and one head. The target additionally checks one supported ubatch, no LoRA and compatible caches/backends. |
| Trace writer | Async when tracing enabled | `LLAMA_ROUND_TRACE=PATH` enables tracing; `GGML_ROUND_TRACE_SYNC=1` selects synchronous output. |
| SSM projection flattening | On | No runtime switch. The old graph is the stack6 CPU negative control and GPU reference. |
| Local engagement diagnostics | Off | `LLAMA_STACK7_PROOF=1` logs an actual folded decode and a multi-sequence flattened projection once per process. `LLAMA_HOST_TRIMS_PROOF=1` and `GGML_CUDA_DEBUG_GDN_STATE=1` supply the existing path proofs. |

Test-only switches for host trims, chain uniforms/embedding table and fold pairing remain intact. The new `seqrm_skip` message records actual mutant engagement before a crash can be accepted as caught.

## Projection design and coverage

`ce555540a` changes `ssm_out` inputs from `[d, n_seq_tokens, n_seqs]` to `[d, n_seq_tokens * n_seqs]` before `build_lora_mm`. Each column still holds the same token and sequence; the projection weights are shared across every column. The former post-projection reshape is removed where the function can return the 2D result directly. No data copy or new operator is introduced.

| File | Flattened input / projection |
|---|---|
| `src/models/qwen35.cpp:575` | GDN gated norm, `ssm_out` with `ssm_out_s` |
| `src/models/qwen35moe.cpp:478` | Same GDN pattern |
| `src/models/qwen3next.cpp:549` | Same GDN pattern; existing call has no scale argument |
| `src/models/qwen4exp.cpp:1140` | Same GDN pattern |
| `src/models/clef.cpp:461` | GDN output; this builder fixes `n_seqs = 1` |
| `src/models/mamba-base.cpp:143` | Mamba SSM scan/gating output; Mamba2 already flattens at `:296` |
| `src/models/plamo2.cpp:415` | SSM scan/gating output |

Other GDN consumers (`bailingmoe3`, `glm5-next`, `kimi-k3`) already flatten before their `wo` projection, so they need no shape edit. The scan found no remaining `ssm_out` call with this unflattened pattern.

LoRA shape reasoning (static, not a measured adapter test): `src/llama-graph.cpp:1579` keeps the base matmul, scale multiply and `B * (A * input)` adapter path. Flattening changes the number of columns, not their content or channel dimension. Both adapter matmuls therefore produce the corresponding 2D columns, and channel/scalar `ssm_out_s` broadcasting remains valid. The calls and scale tensors are unchanged. `plamo2`'s GLU result is allocated contiguously; the old view only grouped its head dimensions. Mamba's shared final reshape remains to cover its other branch and is a no-op on the changed path.

`codex/outproj-cpu.cpp:20` probes real `GGML_OP_MUL_MAT` nodes whose weights contain `ssm_out.weight`. It also requires a real GDN node with two sequences, so a split into single-sequence graphs cannot make the check pass silently. Each run writes every np2 output-logit row and greedy token. The window builds stack6 from a local `git archive` inside `build-stack6-source/`, requires its rank check to exit `42`, requires the stack7 rank check to pass, then compares the output files byte for byte. No new git worktree or branch is created.

## One deferred build window

Queue entry: `codex/bw-stack7.toml`, id `bw-stack7`, kind `filler`, priority `25`. It calls the supplied wrapper with `work/build-window-1007/stack7-a{attempt}`. Payload: `codex/bw-stack7.sh [OUT | --preflight]`. It refuses execution without `GPUQ_BUILD_WINDOW=1` and rejects an output path outside the real wrapper attempt directory. Everything must be committed: preflight, post-mutant restore and pre-freeze checks reject uncommitted tracked changes or moved HEAD.

All compiles go through `codex/build.sh`, which runs `nice -n 19 bin/buildgate`; all CMake builds use `-j4` and shared ccache. Every CPU check goes through `cpugate`; model runs are sequential with explicit context limits. The fixture context is `4096`; the getter's own smaller contexts are preserved. Plain CPU uses at most eight compute threads; each Meta CPU backend uses four. The trace stress has four producers and one writer. No CUDA executable runs in the window.

| Payload code | Stage / gate | Estimate, seconds (G) |
|---|---|---:|
| 8 | Unexpected setup failure / not in build window | Included below |
| 9 | Interrupted | Included below |
| 10 | Preflight: tools, inputs, source ancestry, clean tree, syntax | Included below |
| 20 | CPU current + archived stack6 builds, fixtures, trace oracle/mutant builds | 1800 |
| 21 | Full getter rollback test, getter gate, row-zero/no-gate/inverted-gate controls, restore and rerun | 1800 |
| 22 | Plain + Meta np2 host-trims exactness, cached prompts, verify hooks and engaged mutants; crash can count as caught | 3600 |
| 23 | Chain off/on plain + Meta k3/k5 token/counter/ID identity, engagement, off/off repeat and uniform/table controls | 2400 |
| 24 | Trace old/sync/async byte identity, missing-final-enqueue control, ownership, queue pressure and exit drains | 300 |
| 25 | Fold off/on plain AND Meta k3/k5 token/counter/draft-probability identity and >= 0.9 fold per round (D gate) | 2400 |
| 26 | Meta silent-disengagement control: identical output must still fail engagement | 60 |
| 27 | `test-alloc`, including the host-backed Meta view regression | 60 |
| 28 | Chain plus fold, plain and Meta: same-seed tokens/counters/draft IDs; actual folds and chain rounds | 900 |
| 29 | np2 fold rejection plus chain request/fallback token and counter identity | 600 |
| 30 | Stack6/stack7 np2 logits/tokens byte identity and real graph-rank negative control | 300 |
| 31 | Main harness parsers on real SSE/native-token/np2/log/memory/journal artifacts and controls | 30 |
| 40 | CUDA stack6 recipe, server build and renamed copy | 900 |
| 41 | DeviceTopK count exactly 25, ggml stamp, stable HEAD and clean tree | 30 |
| 42 | `wt freeze` as `stack7`, verify frozen SOURCE, record recipe in freeze and real attempt directory | 60 |
| 43 | Render and contract-check all three GPU items; never submit | 60 |

Total estimate: 15300 s (G). Budget: 22950 s (D, estimate x 1.5). RAM waits/cache misses and CPU model speed can consume the margin. No payload path returns reserved queue codes `3`/`4`; the host comparator's internal `3` only records that its historical GPU-only early-pointer defect did not reproduce on CPU. Both deliberate host mutants must still engage and fail. The wrapper maps other payload failures to its deterministic code.

CPU configuration is `-DGGML_CUDA=OFF`, with llama common/tools/server/tests explicitly on in the fixture parent project. CPU repacking/llamafile are disabled consistently for the archived and integrated projection oracle. CUDA configuration copies `frozen/stack6/RECIPE.txt`: CUDA 13.2, `120a`, CCCL v3.4.3 from the existing local checkout, NCCL off, disconnected FetchContent, shared ccache. Both the DeviceTopK count and ggml stamp use consuming pipelines, never a short-circuit grep/head after a producer.

## GPU items for the parent

The committed `.toml.in` files are templates. Stage 43 renders the final experiment `.toml` and `.gpuq.toml` beside each template after the freeze exists. `codex/make_stack7_items.py` calls the MAIN `python3 -m harness.mkitem`, checks its canonical rendering, then adds priority and replaces the entry with `codex/gpu_ab.sh`. It calls `gpuq submit --check` for each result and treats any `contract violated` text as failure even with rc zero. These checks have not run yet.

| Experiment/item prefix under `codex/items/` | Plan / serving shape | Priority |
|---|---|---:|
| `stack7-depth-j5q` | `s6 s7 s7f s7f s7 s6`; np1 k5, real corpus depths 8192/32768/90112; chain on in both stack7 arms, NH1 off/on respectively; cross-arm tokens AND draft counters gated | 30 |
| `stack7-np2-real-j5q` | `up patch patch up`; np2, same depths, boot wall 900 s (G cap), NH1 off pending np1 winner; chain requested, falls back when multiple sequences draft | 30 |
| `stack7-duo-soak-j5q` | Clone of the main stack6 duo campaign with stack7; NH1 off, chain requested, same tasks and serving shape | 20 |

The depth flags follow the read-only `serve.sh` source for `swift15-j5q` and `swift15-j5q-duo`: total contexts 102400/204800, np1/np2, ub512/ub128, tensor 50/50, q8_0 target KV, f16 draft KV, k5, p_min zero, probabilistic rejection verification, backend sampling, draft vocab, leading-instructions template and default host prompt cache. Depth requests set `ignore_eos=false` to preserve the serving sampler path; all size gates remain active if the model ends early. np2 GPU identity stays off because arrival-dependent ubatches do not repeat reliably; the deferred deterministic CPU np2 gates carry exactness evidence.

Every item pins the entire frozen binaries and the actual `work/build-window-1007/stack7-aN/summary.json`, plus local entry scripts and real recorded parser inputs. No `codex/bw-results` evidence path is used. Each canary has one boot, one smallest depth for depth items, and a 600 s maximum (G cap). The depth canary's steady/size/identity parser gates are the same as the full run. np1 `s7f` additionally requires getter, host trims, chain, initialized fold AND an actual folded decode; np2 patch requires host trims and a multi-sequence 2D out-projection. Trace-writer proof is deliberately absent in untraced jobs.

`codex/gpu_stack7.py` imports the main harness. Its narrow local hooks remove the harness's automatic canary trace environment, retain expected-environment checks, enforce boot work deadlines with cleanup headroom, and reject unreadable journal windows or new Xid lines. Campaign child invocations use the same local entry. Server ownership, exact-process-group cleanup, input/freeze checks, request records, identity/steady gates and scoped journal capture remain the main harness's implementation. The canary timer reserves cleanup inside its queue cap; full depth boot work reserves 60 s (G) inside its wall cap.

Parser replay is written but deferred. It uses real `fixtures/llama-sse`, the memory/journal fixtures, recorded stack6 native-token depth data, a recorded llama.cpp np2 depth artifact, and real getter/chain GPU logs. The new fold/projection log markers have no GPU artifact before the window: the replay checks that their absence in a real log is rejected; the queued canary must prove positive engagement. This is an explicit validation limit, not a completed GPU check.

## Risks and open items

Integrated compilation and every CPU/contract/runtime gate remain unverified. The next action is for the parent to queue this single build window; a passing stage 43 produces reviewable, checked GPU items. No CPU timing or GPU speed number is claimed here.

Host trims must fetch target hidden rows after catch-up even when the chain switches sampler/graph state; NH1 skips catch-up only after the target reports the exact batch folded. np2 rejects NH1 at construction, while the chain can engage with one active slot and release its sampler when another joins. The window tests these interactions explicitly. A GPU failure must stop the experiment, not relax exactness or engagement gates.

Getter safety is per ubatch, independent of folding the attention-only MTP head. The target's recurrent verify still writes rollback snapshots. The combined Meta fold/chain test and np1 getter canary protect this interaction, but GPU verification is pending. np2 must retain gathered state while the projection independently flattens its columns.

The existing chain CPU evidence reports equal finite-seed tokens/IDs/counters with candidate-probability differences up to 1.79e-7 (M, original branch artifact). The carried test allows 2e-7 (D bound from that evidence), requires exact tokens/IDs/counters and catches intentional controls; it is not an all-seed bit-identity proof. Rejection verification continues to use the actual candidate probabilities. No stronger distribution claim is made without the deferred integrated evidence and a mathematical audit; chain remains off by default.

Only the Qwen3.5 fixture is available for measured projection exactness; the other model edits and LoRA compatibility have static shape evidence. A different backend may choose a different reduction/kernel for the wider matrix, so GPU cross-arm np1 identity and np2 serving stability still need measurement; np2 has no cross-arm bitwise-logit GPU oracle in these items.

The trace worker must drain producer TLS tails and join before stream destruction (`ggml/src/ggml-round-trace.cpp:181`, `:206`, `:242`). The copied byte/ownership/backpressure/exit tests cover this in the window. No new TSan run is included; the requested stage-1 writer checks are included. GPU items remove tracing throughout their canary and full runs, so they do not measure writer shutdown.
