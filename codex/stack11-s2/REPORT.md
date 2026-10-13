# stack11-s2 (wave 3 step 2): stack11-base + 87, 88, 91, 90, then K7 and K10 dropped

Branch `topic/stack11-s2` (base `7e7b03962` = `frozen/stack11-base`, master `abee0c847`), `GIT_NO_LAZY_FETCH=1` throughout:

| commit | subject | what |
|---|---|---|
| `713d8b9de` | WIP s2 87 | MTP draft hidden-state carry reset on a reused slot: `common/speculative.cpp`, `src/llama-context.cpp`, `src/llama-ext.h`, test in `tests/test-llama-archs.cpp` |
| `55211a880` | WIP s2 88 | spec prompt built on demand (`uses_prompt()`), `common/speculative.{h,cpp}`, `tools/server/server-context.cpp`, `tests/test-speculative.cpp` |
| `ace6f148d` | WIP s2 91 | tests only: Meta subgraph-uid reuse tests in `tests/test-alloc.cpp` (no production hunk, see below) |
| `a2f920c53` | WIP s2 90 | `CUDA_Host_Mapped` buffer type, Meta host-source branch, `tests/test-host-mapped.cpp` |
| `edeb96dfd` | WIP s2 drop K7 | rows 154 + 168 of stack11 DROPMAP (RMS norm scale pair) |
| `a1b490c78` | WIP s2 drop K10 | rows 160 + 173 (batched top-k) |

then codex commits (Mac gate, window). `wt/stack11-s2-k` (`topic/stack11-s2-k`) = the 90 commit plus a copy of the Mac payload: it is the
with-K7/K10 variant (frozen/stack11-s2); the main tree is the dropped variant (frozen/stack11-s2-nok). The drop is not decided here.

## Conflict resolutions (file:line in the final tree)

**87 form: abee0c8 content, not `master-77c4540ce`.** The trimmed 77c4540ce patch applies textually but does not compile on stack11: it clears the
carry with `std::fill(pending_h[seq_id].begin(), ...)`, and stack11 has the flat `pending_h` ([n_seq][n_embd], stack10's chain/fold layout). It also
does not know the fold. The change is therefore the abee0c8 logic re-expressed on the flat carry:
- `common/speculative.cpp:1616-1624`: at the start of the catch-up branch `if (!is_mem_shared && !folded)`, a sequence whose draft memory is empty
  (`llama_memory_seq_pos_max(ctx_dft, seq_id) < 0`) gets its carry row zeroed. Keeps stack10's behaviour for non-empty memory (carry across batches
  of one prompt, across rounds), changes only the "new request on a reused slot" case.
- `src/llama-context.cpp:2504` (`mtp_fold_prepare`): `if (p0 == 0) return nullptr;` after `p0 = kv->seq_pos_max(seq) + 1`. The fold arms the previous
  run's last row for the first token of a sequence; a fresh sequence must not see it, and returning nullptr sends that batch to the catch-up path
  above, which zeroes it. Without this the fold (default in all served profiles: `LLAMA_MTP_FOLD_C`) bypasses the reset. Contract in `src/llama-ext.h`.
- Both hunks are needed: `p87-nofold` (fold refusal reverted) and `p87-nospec` (speculative.cpp reset reverted) both fail the test (MAC.md).
- `tests/test-llama-archs.cpp` (`check_mtp_draft_reset`, from 77c4540ce) plus a `common_log_flush(common_log_main())` before the return:
  the async logger lost the last lines in about half the runs under load (MAC.md).

**88:** `git am` applied clean on stack11-base (the touched files carry no stack10 component hunks).

**91: tests only, no production hunk (evidence, not choice).** `git am -3` of `master-abee0c8` conflicts in `ggml/src/ggml-backend-meta.cpp` and
`tests/test-alloc.cpp` with pkg 86 (Meta subgraph realloc) and stack11's retained Meta split plans. Reading stack11-base: `ggml_backend_meta_graph_plan`
(`ggml-backend-meta.cpp:2050`) is a per-split plan created by `ggml_backend_sched_alloc_graph` for every Meta split and kept (`spare_plan` at :2085,
`plan_create` at :2933), and each plan caches its subgraphs by uid; so the "several Meta splits evict each other's plan, simple backends see new
uids" failure of 91 cannot occur on the served path. The package's three tests (`test_meta_subgraph_reuse_{two_splits,ten_splits,same_graph}`,
`tests/test-alloc.cpp:1456`) PASS on unmodified stack11-base code (M, `p91-test-alloc`), and a uid-churn mutant (`backend_ctx->uid = 0`) fails them
(`p91-uidchurn` rc 134). The commit carries the tests (a regression guard for the retained-plan design) and drops the duplicate
`test_meta_split_state_mirrored` that stack11 already defines. There is no served-path gain to expect from 91 on stack11-base (D); the answer to
"does stack11-base have the Meta subgraph fix 91 needs": yes (86 + retained plans).

**90:** applies with three adjustments in `ggml/src/ggml-backend-meta.cpp`:
- `:182-202` `ggml_backend_meta_device_supports_host_buft`: the Meta device accepts a host buffer type that every sub-device accepts (the extra
  buft path of 90), tagged `[TAG_META_HOST_MAPPED]`.
- `:1385` the 2-arg `ggml_backend_meta_get_split_state` returns MIRRORED for a non-Meta buffer (host weight read in place).
- `:1299` (stack11-specific, no counterpart in the package): stack11's split-state resolver is the iterative driver
  `ggml_backend_meta_get_split_state(stc, tensor, assume_sync)`, which walks the sources and reads `src->buffer->context` as a Meta buffer context.
  A `CUDA_Host_Mapped` source would be read through the wrong context type (segfault). The branch resolves a non-Meta source through the 2-arg
  entry point. 90's hunk in the old recursive `calculate_split_state` is dead on stack11 and was dropped. New test
  `test_meta_host_mapped_weight` (`tests/test-alloc.cpp:1492`) builds Meta(CPU,CPU) with a fake host buft and a GET_ROWS on a foreign-buffer weight:
  without the `:1299` branch it segfaults (`p90-nodriver` rc 139), without the Meta acceptance it aborts (`p90-nometa` rc 134).
- `src/llama-model.cpp:1113,1267,1705`: `tok_embd_buft_list` only for the token embeddings (`find_device_mapped_host_buft`).
- Build fix: `static enum ggml_backend_dev_type` in the new test (C++ elaborated type needed by the Mac clang), amended into the 90 commit.

**90 vs "fix 2" (embd-dev `1d2e2dc7a`, `build_tok_rows`):** fix 2 is not in stack11-base or stack10 (`grep build_tok_rows src` empty) and is not
ported here, so in this tree only 90 serves. At J5q/IQ4 np1/np2 (all `-sm tensor`, so every profile has Meta) the token-embedding GET_ROWS runs on
the device from `CUDA_Host_Mapped` memory (zc placement proof on master: GET_ROWS on Meta, 0 CPU splits before Meta; bitwise equal to the device read).
If fix 2 were stacked it would be dead on that path (its trigger, the CPU dequant plus F32 upload of the table, no longer exists) and only matter on
the fallback (`GGML_CUDA_NO_PINNED=1` or a rejected pinned allocation: `test-host-mapped` NO_PINNED arm shows `fallback=1 get_rows=CPU`). Evidence:
`patches/upstream/llama.cpp/ORDER.md` row 12 ("supersedes 89 where pinned host memory exists"), `90 EVIDENCE.md` placement proof. Label D until the
placement proof is rerun on this tree (open question 3).

**Drops (K7, K10):** two revert commits. Conflict: `tests/test-backend-ops.cpp` (K7 revert) resolved by removing the K7 test function and its call and
keeping `alpha_ok`/`conv_ok` in the final `return n_ok == tests_run && slice_ok && alpha_ok && conv_ok;`. K10 revert removes `top-k-batched.cuh`,
`tests/test-topk-batched.cpp` and its CMake entry; `top-k.cu` returns to master's per-row path.

## Mac gates

See `MAC.md` (commands, rc, fail-before/pass-after table). nok tree: 69 stages, 0 FAIL. K tree: 68 stages, one inherited-test timeout
(`k10-topk-unit-run`, 600 s alarm at Mac load 150-240; rerun passes in 178 s; stack11-base measured 106 s).

## CUDA window and A/B drafts (not submitted)

- `window/bw-stack11-s2.sh` + `window/bw-stack11-s2.toml` (render with `window/mk_window_toml.py`): builds `frozen/stack11-s2` in `wt/stack11-s2-k` and
  `frozen/stack11-s2-nok` in `wt/stack11-s2`; np2 joint-replay smoke, CPU-device unit tests, `test-host-mapped` (bitwise per type, scheduler placement,
  NO_PINNED fallback) on both GPUs for both builds, touched ops (GET_ROWS, TOP_K, RMS_NORM*, SCALE...) on both GPUs vs `frozen/stack11-base`'s
  binary, served-flag health boots (4 profiles on K, 1 on nok; engagement line `load_tensors: CUDA_Host_Mapped model buffer size` gates), freeze both,
  render + `gpuq submit --check` the items. The stack11-base kernel gates and full `test-backend-ops` are not repeated (87/88/91 touch no kernel,
  90 touches GET_ROWS dispatch and buffer types). est_s 8220 (G), budget 24000. Preflight needs the two Mac evidence sets (present).
- Items (all vs `frozen/stack11-base`, CCX0+1 via `work/aa-load/ccx01.sh`): `s2-np2-real-{j5q,iq4}`, `s2-depth-{j5q,iq4}` (np1, cross-arm token
  identity at 8k/32k, base canary shows `CPU_Mapped` and no `CUDA_Host_Mapped` as the negative control), `s2-duo-soak-iq4`, plus
  `s2-drop-np2-real-j5q` (frozen/stack11-s2-nok vs frozen/stack11-s2; prices the drop, does not decide it). Drafts in
  `orchestration/gpuq/items/drafts/s2-*.gpuq.toml` carry a DRY banner (the frozen builds do not exist; stand-in `frozen/stack11-base`); the window
  stage 72 renders the real ones into `window/rendered/`. Expected deltas are priced from the single-package A/Bs on master (90: J5q np2 -0.30/-0.28/-0.56
  ms/round, IQ4 -0.5; 88: up to -0.14 at 90k np2; 87, 91: none).

## Open questions for the owner

1. 91 ships as tests only on stack11 (its production hunk is redundant with retained Meta plans). Keep the tests in the stack, or drop the commit?
2. Drop decision for K7/K10: `s2-drop-np2-real-j5q` is drafted; the earlier switch-off A/B (s11-drop) had s2 minus nok = +0.43/+0.06/+0.16 ms/round
   with SE 0.58/0.14/0.22, i.e. not resolved.
3. Run the GGML_SCHED_DEBUG placement proof (`work/zc-pr-window/zc_proof.py`) on `frozen/stack11-s2`? It would turn the "90 serves the token GET_ROWS on
   the device inside stack11's iterative Meta driver" claim from D into M; the canary line proves only the buffer type, np1 cross-arm identity proves
   exactness.
4. No KLD item drafted for s2 (90 is bitwise-equal by test, 87/88/91 do not change the target numerics). Add one anyway?
5. 90 costs +644 MiB pinned resident host RAM per server process (J5q; memory available stays >= 8 GiB by the AGENTS.md rule).
