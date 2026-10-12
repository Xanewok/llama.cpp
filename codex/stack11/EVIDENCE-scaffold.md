# stack10 → stack11-base: scaffolding audit, gate needs, dead/neutral evidence

Tree: `~/repos/qwen38-eval-worktrees/wt/stack10b` at `a588f2a44` (HEAD), range `050439614..a588f2a44`. All reads are read-only
(`GIT_NO_LAZY_FETCH=1`). Line numbers marked "HEAD" are in `a588f2a44`. Path shorthand: `T` = the stack10b tree,
`C` = `T/codex`, `N` = `~/repos/qwen38-eval/research/2026-09-28-dual-5060ti/notes`, `P` = `~/repos/qwen38-eval/patches/upstream/llama.cpp`,
`H` = `~/repos/qwen38-eval/harness`.

Method for "default-off identical": I read each diff (non-codex paths) and checked what runs with every new env var unset.
"Identical" means identical outputs and control flow. It does not mean identical instructions: several commits add a per-call
`getenv`, an extra kernel argument or a `__constant__` read. Those are listed per row. The perf effect is G (negligible, not measured).

Method for textual dependency: I extracted the touched files from HEAD into `ev-scaffold/tree/`, with the patches in `ev-scaffold/patches/`.
I reverse-applied the 32 commits newest-first with `git apply -R --reject` and kept the rejected hunks in `ev-scaffold/rej/<commit>/`.
Then `git log -S<symbol>` over the range found the production commits that touch scaffolding symbols.

## Task A: verdict per commit

| commit | subject (short) | verdict | default-off evidence / production hunk |
|---|---|---|---|
| ac6db6790 | DO NOT MERGE GGML_CUDA_FA_MMA_Q8_0 knob | scaffolding (A/B env switch) | `fattn.cu` HEAD:264-271: unset → `enabled=true`, no early return. Reverts alone cleanly at HEAD. |
| 8dc66b1b7 | trace MTP round submission (DNM) | scaffolding (trace) | `ggml-round-trace.h`: `ggml_round_trace_enabled()` caches `getenv("LLAMA_ROUND_TRACE")`. begin/end/scope/count return early when it is unset (`ggml-round-trace.cpp` HEAD:211-233). Build side effect: `ggml/src/ggml-cuda/CMakeLists.txt` HEAD:74-77 gives ggml-base the CUDA include dir, `GGML_ROUND_TRACE_NVTX` and `${CMAKE_DL_LIBS}`. NVTX push/pop runs only inside the enabled branch. **Later production commits call its API (see dependencies).** |
| e486ceb19 | tests: trace prefill ubatches and meta plan allocations | scaffolding (trace) | Adds trace scopes only. It wraps the meta retention body in `{ ggml_round_trace_scope trace_retention(...); ... }` (re-indent only) and splits `properties_changed` into a scoped block. Same logic. |
| ac7465885 | report convolution snapshot fusion coverage | scaffolding (engagement log) | Moves the existing `GGML_CUDA_DEBUG_SSM_CONV` check (from production 0d307c896) into `ggml_cuda_debug_ssm_conv()`. Counting and logging run only `if (debug_ssm_conv)`. |
| 6727f3cb1 | trace GDN getter engagement | scaffolding (engagement log + trace) | `std::getenv("GGML_CUDA_DEBUG_GDN_STATE")` at every capture/eager submit; it is not static, so it costs one getenv per capture. The `gdn_gather_skipped` trace counter sits under `ggml_round_trace_enabled()`. Not consumed by any gate (see B). |
| a59f806e4 | log GDN cache fusion per captured graph | scaffolding (engagement log) | Everything is gated by `ggml_cuda_debug_gdn_state()` (static). It prints `gdn_cache_fusion:` (HEAD ggml-cuda.cu:5289). **Consumed by the np2 canaries.** |
| f61bee3fa | host-trims: report exercised paths | scaffolding (engagement log) | `static proof` lambdas only log under `LLAMA_HOST_TRIMS_PROOF`. |
| d7bc08c2c | host-trims env switches SEQRM/BULKH | scaffolding (A/B env switch) | Unset → `indexed=true`, `bulk_h=1`: the same as the packaged trims. |
| **89cff0ccf** | **read target nextn pointer after draft catch-up; BULKH=2 control** | **carry the pointer move (production change), drop the switch** | Default (`bulk_h==1`) behaviour changes from 89637899d (pkg trims, b115f8180 form). 89637899d took `h_tgt = llama_get_embeddings_nextn(ctx_tgt)` **once before the catch-up decode** and reused it for the bulk copy. 89cff0ccf reads it inside the catch-up block and again after it: `common/speculative.cpp` HEAD:1683 `const float * h_tgt = h_early ? h_early : llama_get_embeddings_nextn(ctx_tgt);` and HEAD:1730-1731 `// read after the catch-up decode: the bulk copy must not depend on a buffer address captured before other work ran` / `const float * h_all = bulk_h == 0 ? nullptr : (h_early ? h_early : llama_get_embeddings_nextn(ctx_tgt));`. The old hoisted form survives only as `BULKH=2`, labelled "historical hoisted-pointer defect" (`C/check-host.sh:39`; `C/bw-np2-compare.py:114`, "the CPU test does not reproduce the GPU defect"). The defect was not reproduced on CPU: `N/2026-10-07.md:587` "Legacy b115f8180 is still equal on CPU (rc 3 ...)". It is a contract fix (G that it fixes a live bug, D that it is safe). Squash the pointer move into 89637899d's package. |
| f27ce2576 | host-trims VERIFY cross-checks and MUTANT controls | scaffolding (mutant/test-only) | Both are static getenvs and unset is a no-op. The `beg` variable is assigned from `i_batch_beg` with no change when the mutant is unset. |
| a7f31a570 | log and offset the bulk-row negative control | scaffolding (mutant) | Changes only the mutant branch. |
| 4be92a4fa | MTP chain negative controls + draft dump | scaffolding (mutant + dump) | `LLAMA_MTP_CHAIN_DUMP` (fopen only when set), `LLAMA_MTP_CHAIN_DEBUG_SHIFT_EMBD` (`i_src = i` when unset, `llama-model.cpp`), `LLAMA_MTP_CHAIN_DEBUG_REVERSE_U` (`llama-sampler.cpp`). `dist_backend_set_input` now draws all uniforms first, then sets the tensors: the same RNG sequence and the same per-tensor values, plus one small vector allocation per call. |
| 9afc6714e | move round trace writes off the decode thread | scaffolding (trace) | Only `ggml-round-trace.cpp` (writer thread). Inert when `LLAMA_ROUND_TRACE` is unset. |
| 6e99847f9 | restore static trace kinds, writer tests | scaffolding (trace) | Header comment plus writer code. |
| e99e52ef3 | draft dump + fold mutant for the fold test | scaffolding (dump + mutant) | `LLAMA_MTP_DRAFT_DUMP` fopen only when set. `LLAMA_MTP_FOLD_DEBUG_MUTANT` sets view offset `0*nb1`. Production c19629b30 later deleted the mutant hunk (see dependencies). |
| d72722a99 | report stack7 canary and mutant engagement | scaffolding (engagement log) | `LLAMA_STACK7_PROOF`: one non-static `std::getenv` per fold decode (`llama-context.cpp`, since replaced by c19629b30) and per graph build with n_seqs>1 (`src/models/qwen35.cpp` HEAD:572). It also adds an INFO line on the seqrm_skip mutant branch. |
| 2ecda894d | codex: stack8 Mac checks (+1 production log line) | **diagnostic log line in production code** | `common/speculative.cpp` HEAD:1497-1499: `if (this->params.mtp_chain && n_seq != 1) SPC_INF("MTP chain disabled: needs one sequence (n_seq=%u)\n", ...)`. It has no behaviour effect. **The np2/duo canaries consume it** (B). It is a reasonable user-facing log: carry it folded into ad843282f (chain restricted to single-sequence), or keep it in the scaffolding layer. |
| 519069f3e | local: stack9 fold + GDN negative controls | scaffolding (A/B switch + mutants) | `LLAMA_MTP_FOLD_C`: unset → `fold = llama_mtp_fold_init(...)`, the same as c19629b30 default-on (`common/speculative.cpp` HEAD:1570-1571). This is the **only off-switch for fold**. `GGML_CUDA_GDN_JOINT_MUTANT` is static and unset → `joint` unchanged. `LLAMA_MTP_FOLD_DEBUG_LAYOUT` is a non-static getenv **every decode** (`llama-context.cpp` HEAD:2137). `LLAMA_MTP_FOLD_DEBUG_SWAP_SEQ` is a non-static getenv every fold `set_input` (`llama-graph.cpp` HEAD:548). Both are no-ops when unset. |
| bd251ce1b | local: stack10 component controls | scaffolding (A/B switches + mutants + DEBUG logs) | Unset: `GGML_CUDA_Q8_DECODE`→enabled, `GGML_CUDA_ALPHA_GATE`→fusion on, `LLAMA_KV_ROT_PERSIST`→persist, `LLAMA_KV_ROT_ALLOC_FAIL`/`_MUTANT` no-ops. `add_softplus_mul_f32` gains a `bool mutant` kernel argument, `getenv` per launch, false → same arithmetic. It adds `GGML_LOG_DEBUG` `fattn_q8_decode:` (fattn.cu HEAD:450) and `alpha_gate_fusion:` (ggml-cuda.cu HEAD:4923). **Gates and canaries consume them.** |
| f8a54c243 | local: alpha alias negative control | scaffolding (mutant) | Non-static `getenv("GGML_CUDA_ALPHA_ALIAS_MUTANT")` per can_fuse match. Unset → the original alias check. |
| a4a3a8c63 | local: K7 switch, mutants, engagement log | scaffolding | Commit body: "Env switches and negative-control mutants only; no default behaviour change." Static env reads with neutral defaults. Adds `GGML_LOG_DEBUG rms_norm_scale_pair_fusion:`. |
| 6565c43db | local: K4 SSM_CONV_RS off switch, mutants, log | scaffolding | CPU `supports_op` adds `case SSM_CONV_RS: return !getenv(...)`; the default branch was `return true` (`ggml-cpu.cpp` HEAD:488-489). CUDA supports_op is ANDed with `!getenv`. The kernel gains an `int mutant` argument; 0 → same. The `ssm_conv_rs engaged` log is under `GGML_SSM_CONV_RS_LOG`. Non-static getenv per supports_op. |
| b4efb51a0 | local: K6a switch, log, stale-cache mutant | scaffolding | `q8_reuse` defaults on and `q8_stale_mutant` false. Adds `GGML_LOG_DEBUG q8_reuse:` (HEAD:5283). |
| 31e27ee33 | local: K6b switch, log, numerical mutant | scaffolding | `add_rms_norm` defaults on. The kernel gains a `bool mutant` argument (getenv per launch). Adds DEBUG `add_rms_norm:` (HEAD:4865). |
| d045e5d49 | local: K2 off switch + negative controls | scaffolding | `kv_max_share` defaults true and `kv_max_mutant` 0. Adds DEBUG `fattn_kv_max:` (HEAD:5170). |
| 9b2ea143f | local: K10 off switch, log, mutants | scaffolding | Unset → the same selection. `topk_mutant` is a `__constant__` int, default 0, read in 6 hot kernel sites; no memcpy when the env is 0. Not SASS-identical. Test-only hooks in `test-backend-ops.cpp` (TOPK_CANONICAL_ANY) and `test-topk-batched.cpp` (TOPK_FAILFAST, TOPK_MUTANT). |
| 7491039b9 | local: K8/L1 tile switch, mutants, mmq-hash | scaffolding | Unset → `MMQ_TILES_AUTO` → `use=true`, the same as before (`nrows_x % I == 0`). Cost: `mmq_get_stream_k_plan` is now computed on every SMALL_I candidate launch (host). Test-backend-ops gains the `mmq-hash` mode and `GGML_TEST_PERF_COLD_MB`; `node_slack` divides by `perf_outputs.size()==1` by default. |
| a434dc77b | local: K11a switch, debug check, mutants | scaffolding | Defaults `on=true, debug=false, mutant=0` → the same elision. Adds `n_sync_elided++` and two counters. |
| a8eed261a | local: K11b A/B controls + mutants | scaffolding | Defaults: stage enabled, forward rank order, no mutant, measured order kept. |
| 330ab2cb8 | local: K11a mutant 4 + caller in log | scaffolding | State I/O calls `synchronize(sync_elide_env().mutant != 4)`, which is `synchronize(true)` by default (these forced syncs come from production ecb1f2524). New `const char * caller = __builtin_FUNCTION()` default argument (`llama-context.h` HEAD:63); no caller passes it explicitly. |
| 5f1825dfa | local: K11a query-based debug check | scaffolding, **but adds a public ggml API** | New `GGML_API int ggml_backend_query()` (`ggml-backend.h`) and a new `ggml_backend_i::query` member (`ggml-backend-impl.h`). Implemented for CUDA (`cudaStreamQuery`), meta and CPU (NULL). Other backends zero-initialise the trailing member. It is called only from the `LLAMA_SYNC_ELIDE_DEBUG` path. Iface/ABI change; default behaviour identical. |
| c3d17d5ef | local: GGML_META_SUBGRAPH_MUTANT | scaffolding (mutant) | Unset → `n_main = max_subgraphs` (the c5087a10f fix). |

Every reverse patch was checked one by one: 15 of 32 revert cleanly alone at HEAD. Newest-first, the sequence c3d17d5ef…2ecda894d, 6e99847f9,
9afc6714e, 4be92a4fa, a7f31a570, f27ce2576 and ac6db6790 reverse cleanly. The others hit rejected hunks: d72722a99, e99e52ef3,
89cff0ccf, d7bc08c2c, f61bee3fa, a59f806e4, ac7465885, e486ceb19 and 8dc66b1b7. All the rejects come from the production
dependencies below. None comes from a scaffolding-on-scaffolding cascade.

### Dependencies of production commits on scaffolding

1. **Round trace (8dc66b1b7 + e486ceb19 + 9afc6714e + 6e99847f9): compile dependency.** These production commits call `ggml_round_trace_*`.
   After all scaffolding is reverted, 23 call sites remain (in `ev-scaffold/tree`):
   - e07f827ce: `src/llama-context.cpp` `ggml_round_trace_count(allocated ? "graph_incompatible" : "graph_stale_alloc")`
   - 95141e1a5: `ggml-cuda.cu` `input`/`input_owned`/`input_slot_wait`/`input_api`
   - 6d86af379: `src/llama-context.cpp` `ggml_round_trace_count(staged ? "input_fence_staged" : "input_fence_sync")`
     (the harness staging proof reads these, `H/lib.py:655-661`)
   - 21ba77aa8: `common/speculative.cpp` `ggml_round_trace_begin("DC")`, `count("mtp_chain")`, `end`
   - ffdb7d6e8: `src/llama-context.cpp` `ggml_round_trace_count("mtp_fold")`
   - d5b2af7c9: `ggml-cuda.cu` `input_stage`/`input`/`input_owned`/`input_slot_wait`/`input_api`

   The `#include "ggml-round-trace.h"` lines come from 8dc66b1b7. Dropping it breaks compilation of these 6 commits unless their
   trace lines are stripped. Textually, a075f2cb0 (production, "hash visited tensors when retaining meta plans") edits inside the
   `{ ggml_round_trace_scope trace_retention ... }` block that e486ceb19 added (`git log -L` over that block). The meta
   reject of 8dc66b1b7 (`meta_build` scope) and the `graph_hit`/`graph_miss`/`llama_build` rejects in `process_ubatch` are
   context conflicts with e07f827ce, 6d86af379 and d719e9acc.
2. **Host-trims switches (d7bc08c2c/89cff0ccf): textual.** ffdb7d6e8/c19629b30 (fold) built the catch-up block on the switch
   variables: HEAD:1668 `h_early`, :1674 `if (!is_mem_shared && !folded)`, :1683 `h_early ? h_early : ...`. Dropping the switches
   means rewriting this block by hand: `h_tgt` read inside catch-up, `h_all` read after it (the 89cff0ccf contract).
3. **Fold proof/mutant (d72722a99, e99e52ef3): textual.** Production c19629b30 deletes their lines. It replaces
   `if (std::getenv("LLAMA_STACK7_PROOF")) ... "mtp_fold_decode: engaged"` with an unconditional
   `LLAMA_LOG_DEBUG("mtp_fold_decode: engaged n_seq=%u\n", n_seq)` plus a `mtp_fold_logged` bitmask (llama-context.cpp HEAD:2361).
   It also removes the `LLAMA_MTP_FOLD_DEBUG_MUTANT` view offset (qwen35.cpp). Dropping d72722a99/e99e52ef3 means removing
   those deletion hunks from c19629b30.
4. **Fold default and off-switch: functional.** ffdb7d6e8 made fold opt-in (`LLAMA_MTP_FOLD_C=1`). c19629b30 made it always on and
   removed the env read. 519069f3e re-added `LLAMA_MTP_FOLD_C=0` as an off-switch. Without 519069f3e, `LLAMA_MTP_FOLD_C`
   is ignored: the served `=1` is unaffected, but the gates that set `=0` change meaning (B).
5. **GGML_CUDA_DEBUG_SSM_CONV**: production 0d307c896 owns this debug env and its log. ac7465885 only refactors it. Context
   rejects for ac7465885/a59f806e4 come from 30afc9758/fac1254d2 (they added `q8_cache`/`fattn_kv_max_cache` resets next to it).
6. False positives checked: `GGML_CUDA_Q8_REUSE_MAX_ROWS`, `GGML_CUDA_ADD_RMS_NORM_MAX_ROWS`, `GGML_CUDA_FATTN_KV_MAX_STRIDE`,
   `GGML_CUDA_TOPK_BATCHED` (a macro) and `GGML_CUDA_MMQ_TILES_DEFAULT/SMALL_I` (enum) in production commits are constants, not env.
   `ggml_backend_query`, `mtp_chain_dump_step`, `draft_dump_step`, `LLAMA_SYNC_ELIDE*`, `GGML_META_INPUT_*`, `GGML_CUDA_INPUT_STAGE*`
   and `GGML_META_SUBGRAPH_MUTANT` appear only in scaffolding commits.

## Task B: which scaffolding the step-1 window and A/B items consume

Window: `C/stack10/bw-stack10b.sh` + `gates.sh`. The component gates are extracted by `gates_table.py` from each component window;
my extracted copies are in `ev-scaffold/gates/<comp>.sh`. Items: `C/stack10/items/*.toml.in`, rendered by `s10items.py` +
`engagement.py`; rendered copies are in `C/stack10/rendered/*.toml`.

### Window gates (string or env → producing commit → consumer)
| consumed thing | producer | consumer |
|---|---|---|
| strings `libllama.so` has `backend busy at a skipped sync` | 5f1825dfa | `gates.sh:132` (stamps, k11a); `C/k11a/sync_count.sh:81` |
| `GGML_CUDA_MMQ_SK_MUTANT` string in libggml-cuda (>0 k8-bin, ==0 stock) | 7491039b9 | `gates.sh:167,171`; `C/stack10/k8_gpu_patch.py:20`; `C/k8/gpu/k8-gpu.sh:34` |
| `mmq-hash` mode in test-backend-ops | 7491039b9 | `gates.sh:168`; `k8-gpu.sh:229,233` |
| `GGML_CUDA_MMQ_TILES={default,auto,auto-l1,auto-drop}`, `GGML_CUDA_MMQ_SK_MUTANT=1`, `GGML_CUDA_MMQ_TILES_LOG=1` (+`GGML_TEST_PERF_COLD_MB` in perf mode) | 7491039b9 | `k8-gpu.sh:164-237` (checks), :262-289 (perf) |
| `GGML_META_SUBGRAPH_MUTANT=1` (must die) | c3d17d5ef | `np2_smoke.sh:122-126` (hard); `gates.sh:189` (recorded only) |
| `LLAMA_MTP_FOLD_C=0`, `LLAMA_SYNC_ELIDE=0` in the smoke env | 519069f3e, a434dc77b | `np2_smoke.sh:68` (no assertion: without the switches the smoke silently runs fold and elision on) |
| `LLAMA_MTP_FOLD_C=0` + 0 lines `mtp_fold_decode: engaged`/`MTP chain: k=` | 519069f3e (switch); c19629b30 (line) | `sync_count.sh:23,145` (**hard: fails without 519069f3e, since fold engages at np2 after c19629b30**) |
| `LLAMA_SYNC_ELIDE=0`, `LLAMA_SYNC_ELIDE_DEBUG=1`, `LLAMA_SYNC_ELIDE_MUTANT=3` | a434dc77b | `sync_count.sh:94-98,134-137` |
| `sync_elide: debug check engaged (backend query exact)`, `sync_elide: missed: backend busy`, `sync_elide: elided=N missed=0 idle_slow=N` | 5f1825dfa (a434dc77b, 330ab2cb8 caller) | `sync_count.sh:138-157`; `C/k11a/count.py:64-67` |
| base: `GGML_CUDA_ALPHA_GATE_MUTANT=1`, `GGML_CUDA_ALPHA_ALIAS_MUTANT=1`, `GGML_CUDA_Q8_DECODE=0`; logs `alpha_gate_fusion:`, `fattn_q8_decode:` | bd251ce1b, f8a54c243 | `gates/base.sh:6-20` (from `C/bw-stack10-cand.sh` ops) |
| fafix: `GGML_CUDA_Q8_DECODE=0`, `fattn_q8_decode:` (hard); `GGML_CUDA_FA_MMA_Q8_0=0` (informational arm, rc still under set -e) | bd251ce1b; ac6db6790 | `gates/fafix.sh:25-30` |
| k7: `GGML_CUDA_RMS_NORM_SCALE_PAIR=0`, `GGML_CUDA_RMS_NORM_SCALE_PAIR_MUTANT=1,2`, `GGML_RMS_NORM_SCALE_PAIR_ALIAS_MUTANT=1,2,4,8`, `rms_norm_scale_pair_fusion:` | a4a3a8c63 | `gates/k7.sh:13-42` |
| k4: `GGML_SSM_CONV_RS_OFF=1`, `GGML_SSM_CONV_RS_LOG=1`, `GGML_SSM_CONV_RS_MUTANT=1,2`, `ssm_conv_rs engaged` | 6565c43db | `gates/k4.sh:21-53` |
| k6a: `GGML_CUDA_Q8_REUSE=0`, `GGML_CUDA_Q8_REUSE_STALE_MUTANT=1`, `q8_reuse:` | b4efb51a0 | `gates/k6a.sh:16-25` |
| k6b: `GGML_CUDA_ADD_RMS_NORM=0`, `GGML_CUDA_ADD_RMS_NORM_MUTANT=1`, `add_rms_norm:` | 31e27ee33 | `gates/k6b.sh:16-25` |
| k2: `GGML_CUDA_FATTN_KV_MAX_SHARE=0`, `GGML_CUDA_FATTN_KV_MAX_MUTANT=1,2` (FAIL texts from test in fac1254d2) | d045e5d49 | `gates/k2.sh:29-34` |
| k10: `GGML_CUDA_TOPK_BATCHED=0`, `GGML_CUDA_TOPK_MUTANT=1..6`, `GGML_CUDA_TOPK_LOG=1`, `top_k_batched: engaged` | 9b2ea143f | `gates/k10.sh:13-28` |
| k11b: `GGML_CUDA_INPUT_STAGE_ORDER=fwd/rev`, `GGML_CUDA_INPUT_STAGE_MUTANT=1`, `GGML_META_INPUT_STAGE_MUTANT=1` | a8eed261a | `gates/k11b.sh:17-35` |
| `round-trace` binary name (= `cp llama-server round-trace`) | window itself, not a commit | `gates.sh:121`; preflight `bw-stack10b.sh:119` (`$P55_LIB/round-trace`); `np2_smoke.sh:14`; `sync_count.sh:20`; every item `exe` |

Not consumed by the window: `LLAMA_ROUND_TRACE` (the window never sets it), `GGML_CUDA_DEBUG_GDN_STATE`, `gdn_state_indexed:`,
`LLAMA_HOST_TRIMS_*`, `LLAMA_MTP_CHAIN_DUMP`/`DRAFT_DUMP`/`DEBUG_*`, `LLAMA_MTP_FOLD_DEBUG_*`, `GGML_CUDA_GDN_JOINT_MUTANT`,
`LLAMA_KV_ROT_*`, `LLAMA_STACK7_PROOF`. Their consumers were the stack7–stack9 Mac/CPU stages under `C/`.

### A/B item canaries (rendered `canary_log` + `[canary.env]`)
Canary env that every depth item sets in the template (np1, np2): `LLAMA_ARG_LOG_VERBOSITY=5`, `LLAMA_HOST_TRIMS_PROOF=1` (f61bee3fa),
`LLAMA_STACK7_PROOF=1` (d72722a99) and `GGML_CUDA_DEBUG_GDN_STATE=1` (6727f3cb1/a59f806e4). Of these, only GDN_STATE's output is matched,
and only at np2. Env added per component (`engagement.py:34-64`): `GGML_SSM_CONV_RS_LOG=1` (6565c43db), `GGML_CUDA_TOPK_LOG=1` (9b2ea143f),
`GGML_CUDA_MMQ_TILES_LOG=1` (7491039b9, np2 only) and `LLAMA_SYNC_ELIDE_DEBUG=1` (a434dc77b/5f1825dfa).

| canary pattern (rendered) | shapes | producer | class |
|---|---|---|---|
| `MTP chain: k=5` / absence | np1 / np2, duo | 21ba77aa8 | production |
| `mtp_fold_decode: engaged n_seq=1|2` | all | c19629b30 (DEBUG) | production |
| `set_logits: using backend candidates ...`, no `set_sampler: ...using CPU` | np1 | before the range (stack9 base) | production |
| `gdn_cache_fusion: graph device=0/1 ... seqs=2 indexed=N gdn=N fused=N` | np2 | **a59f806e4** (+ `GGML_CUDA_DEBUG_GDN_STATE=1`) | scaffolding |
| `MTP chain disabled: needs one sequence (n_seq=2)` | np2, duo | **2ecda894d** | log line in a codex commit |
| `n_kv_pad += 4096` | duo | 52fc3fd2c | production |
| `persistent rotations backend=` | base (none of the rendered items list base; engagement.py:22) | 758c93a1c | production |
| `rms_norm_scale_pair_fusion:` | all | **a4a3a8c63** (DEBUG) | scaffolding |
| `ssm_conv_rs engaged: ... mutant=0` | all | **6565c43db** (needs LOG env) | scaffolding |
| `q8_reuse: ... skipped=[1-9]` | all | **b4efb51a0** (DEBUG) | scaffolding |
| `add_rms_norm: device=` | duo | **31e27ee33** (DEBUG) | scaffolding |
| `top_k_batched: engaged` | all | **9b2ea143f** (needs LOG env) | scaffolding |
| `mmq_tiles: ... small_i=1` | np2 | **7491039b9** (needs LOG env) | scaffolding |
| `sync_elide: debug check engaged (backend query exact)` + no `sync_elide: missed:` | all | **5f1825dfa** + a434dc77b (needs DEBUG env) | scaffolding |
| `one pinned copy per input, uploads slowest host link first` | all | d5b2af7c9 | production |
| K2 `fattn_kv_max:` | none (`shapes=()`, engagement.py:47-51, after the 10-09 park) | d045e5d49 | scaffolding, not consumed |
| `alpha_gate_fusion:` (base entry) | engagement.py lists it but base is not a component of the rendered items | bd251ce1b | scaffolding |

### Round trace at run time
- Harness: `H/depth.py:165-167` sets `LLAMA_ROUND_TRACE=<boot>/round-trace.jsonl` for **every canary boot** and for every timed boot of an
  item with `depth.counters`. `H/campaign.py:628` sets it for every campaign (duo) canary boot.
- Consumers: `H/depth.py:305,335,407` (`lib.trace_window`, which raises `insufficient_data` when no round falls in the window, `H/lib.py:633-652`);
  `H/depth.py:429` / `H/campaign.py:~660` staging proof (`lib.check_staging`, input_fence counters from 6d86af379), but only when
  `"staging" in canary.proofs`; `H/report.py:180-184` hard-fails `trace file/env mismatch` when env.json has LLAMA_ROUND_TRACE and no file exists.
- Stack10 items: all six templates have `proofs = []`. Only **`s10-np2-real-j5q`** has `depth.counters = ["graph_hit","graph_miss","llama_build","compute_submit","set_inputs","sync","sampling"]`
  (template :76; `s10items.py` TRACE_COUNTERS; `gpu/gpu_s10.py:5,93` "round trace only on the np2 J5q item").
  Those kinds come from 8dc66b1b7 (graph_hit, graph_miss, llama_build, sync, sampling) and e486ceb19 (compute_submit, set_inputs), with the writer from 9afc6714e.
  **So s10-np2-real-j5q needs the round trace at run time in both arms**: counters in the canary and timed boots, and the report's
  file/env check on timed boots. np1 depth, np2-real IQ4 and both duo soaks set LLAMA_ROUND_TRACE in canaries only and never read
  the file (`proofs=[]`, no counters); I found no file-existence check on canary dirs (`report.py` iterates the `full-a*` boot dirs, `report.py:103-109`).
- Every item runs `exe = "build/bin/round-trace"`: a copy of llama-server under that name (gates.sh:121). The name must exist; the trace feature is not needed for it.

## Task C: dead / neutral evidence (full quotes in `taskC.md`; spot-checked lines are marked ✓)

1. **K2 fac1254d2 (pkg 74): never engages, M.** ✓ `N/2026-10-09.md:466-468`: "## 19:50 stack10 A/B canaries: K2 never engages, K6b only in duo (M) …
   K2 0 lines in np1 depth, np2 joint start and duo … K2's shared scan needs Q ne[1] ≥ 1024 or ne[3] > 1 (`fattn-kv-max-cache.h:31`);
   served -ub 128/512 and these canaries never meet it (inference …)". Raw logs: `~/repos/qwen38-eval/results/2026-10-09-s10-*/canary-a1/*/server.log`.
   The subagent recounted them: 0 `fattn_kv_max:` lines in all 6. Positive control in the same logs: K7 `rms_norm_scale_pair_fusion:` 1056–2976 lines at -lv 5.
   ✓ `C/stack10/engagement.py:47-51` (`'k2': dict(shapes=(), ...)`).
   - Coverage (✓ item args): np1 J5q ub512, np1 IQ4 ub512, np2-real J5q **ub128**, np2-real IQ4 ub512, duo J5q **ub128**, duo IQ4 ub512.
     This covers all four served profiles (np1/np2 × J5q/IQ4) at their served ub. Limits: canaries ran only depth 8192 (`depths=[8192]`);
     np1 at ub 128 and prefill ub ≥ 1024 were not run.
   - Why: ✓ `ggml/src/ggml-cuda/fattn-kv-max-cache.h:30-33`, `ggml_cuda_fattn_kv_max_scan_wanted`: `mask && mask->type == F16 && K->ne[1] % 256 == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)`.
     It mirrors the stock dense-scan condition, so np1 decode/verify (ne[1] ≤ 6) and ub ≤ 512 prefill never scan. At np2 with non-unified KV, Q ne[3] = 2 passes
     this gate (D, subagent: `llama-graph.cpp:2717`, `llama-kv-cache.cpp:1377`). The second gate, ✓ `fattn-kv-max-cache.h:99` `if (g.uses < 2 ...) continue;`,
     drops a mask scanned by only one FA per evaluated graph. Under -sm tensor the meta backend splits subgraphs at every AllReduce
     (`ggml-backend-meta.cpp:~2590`, D, not traced), so each CUDA graph holds one attention layer. The notes' written reason covers only gate 1 and is incomplete for np2.
   - Timing: no M. `P/74-*/EVIDENCE.md:19` "K2's gain is unmeasured"; expected D −0.11 at np2.
2. **K7 20ac6ec57 (pkg 78): D only.** `P/78-*/EVIDENCE.md:19-20` "D −0.03..−0.06 ms/round … No measured ms/round"; `N/2026-10-09.md:249`; `P/ORDER.md:128`.
   Engages (M, canary line counts). Only kernel-count evidence exists: `W/work/attrib-s10/DIGEST.md:37-38` (rms_norm calls 309→213, one nsys boot per arm; ":44 single-axis deltas near 0.2 ms are not resolved").
   There is no M inside the neutral band: it is not proven dead and needs a measurement.
3. Other components (no isolated A/B inside stack10 exists for any K-component; the combined A/Bs are `docs/ledger/llama-cpp.md:108-109`):
   - K4 (75): D −0.30…−0.35 (`P/75-*/EVIDENCE.md:37`). M kernel time, np1 net 0 (−0.23/+0.22) and np2 ≈ −0.06 (`DIGEST.md:37-38`, 1 boot, not an A/B).
   - K6a (76): M precursor np1 −0.483/+0.005/−0.019 (`P/76-*/EVIDENCE.md:25-27`); neutral at 32k/90k on a different form; the 12-row form is not measured.
   - K6b (77): engages only in duo (M, `N/2026-10-09.md:468`, engagement.py:42-46); D ≤ −0.19; no timing M.
   - K8/L1 (80): L1 M −0.153/−0.186/−0.149, SE 0.054/0.088/0.050, 2 boots (`docs/ledger/llama-cpp.md:93`, `P/80-*/EVIDENCE.md:38`). This is at the edge of the band, and it was measured for L1 on stack8, not K8. Engages at np2 only.
   - K10 (81): M microbench; D ≤ 0.03 ms/round (in band) (`N/2026-10-09.md:209`); no end-to-end measurement.
   - K11a (83): M syncs/round 216.2→12.0; nsys period 41.39→41.26, "not a timing claim" (`N/2026-10-09.md:204`).
   - K11b (84): M mechanism (GPU1 lag 77→15 µs); D −0.20 (`N/2026-10-09.md:144-149`).
   - 758c93a1c rotations: M drop-one n=2 on a regressed build, −0.20/−0.08/−0.07 (`N/2026-10-08.md:524-525`). Confounded; neutral-ish except at 8k.
   - 3ebadf53a softplus: M drop-one n=2, −0.10/−0.05/+0.03, no SE (`N/2026-10-08.md:524,526`). Weak neutral M.
   - mask-np2 b91f5e291: M −0.87 (32k), −2.45 (90k), 8k unchanged (`docs/ledger/both-hardware.md:32`). Positive.
   - 48: M −1.24 at 90k, 8k −0.16 inside the boot spread (`P/48-*/EVIDENCE.md:22`). 50: M −1.06 at 90k, 8k −0.07 with the gate off by design (`P/50-*/EVIDENCE.md:18-19`).
   - 69: D −1.13/−1.01/−1.28 integrated (`P/69-*/EVIDENCE.md:9`). 70/71: D combined, 32k matched −0.33 (`P/70-*/EVIDENCE.md:9`, `P/71-*/EVIDENCE.md:9`).
   - 13: M −5.3±4.3 (8k), −19.8±12.1 (32k), n=3 (`P/13-*/EVIDENCE.md:33-35`). 15: M think −0.47/tool −0.31, no SE, superseded by 52 (`P/ORDER.md:124`).
   - 59: M prefill −9.3% (`P/59-*/EVIDENCE.md:19`). 06/07: M checkpoint 153→34 ms (`P/07-*/EVIDENCE.md:34-36`). 47: crash fix, not a perf lever.
   - Only K2 has a never-engages proof. No component has a neutral-band M from an isolated A/B.
