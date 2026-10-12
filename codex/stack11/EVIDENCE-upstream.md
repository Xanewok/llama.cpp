# stack10 (a588f2a44, base 050439614) vs upstream master abee0c847: drop/keep evidence

Read-only. Labels: M measured (cited path), D derived from code read, G guess. "S:" = stack10 tree path:line *after the
named commit* (`git -C wt/stack10b show <commit>:<path>`); "S10:" = at a588f2a44. "M:" = master abee0c847
(`wt/upstream-master-abee0c8/<path>:line`). Nothing compiled or run by this task.
Scratch artifacts: `gdn/` (3-way merges of gated_delta_net.cu), `ports/`, `sweep.py`/`sweep.txt` (content sweep),
`pidsweep.py`/`pidsweep.txt` (per-file patch-id sweep).

---------------------------------------------------------------------------------------------------------------------
## 1. df5471ab7 sampling: greedy → vocab ids + top-k candidates 1-d  vs d0b490f25 (#29797)

Verdict: PARTIALLY LANDED. Residual = top-k hunk (+ its test `top_p_on_top_k`). Residual NOT needed by stack10.

Greedy hunk — landed, identical behaviour:
- S: src/llama-sampler.cpp:1089-1093 (df5471ab7)
  `if (data->candidates) { ... = ggml_reshape_2d(ctx, data->candidates, 1, ggml_nelements(data->candidates));`
  `curl = ggml_get_rows(ctx, candidates_rows, curl); ggml_set_name(curl, "greedy_sampled_token");`
- M: src/llama-sampler.cpp:1089-1093
  `if (data->candidates != nullptr) { struct ggml_tensor * candidates = ggml_reshape_2d(ctx, data->candidates, 1, ggml_nelements(data->candidates));`
  `curl = ggml_get_rows(ctx, candidates, curl); ggml_set_name(curl, "greedy_sampled_token");`
- Differs only in variable name/comment. Test: S tests/test-backend-sampler.cpp:1974 `test_backend_top_k_greedy_sampling`
  (top-k(8)+greedy == argmax) ≙ M tests/test-backend-sampler.cpp:422-441 `test_greedy_filtered` (top-k 1 and 8 + greedy,
  4 steps, asserts sampled == ids[argmax(filtered logits)]). M's test is a superset.

Top-k hunk — NOT on master:
- S: src/llama-sampler.cpp:1497-1500
  `ggml_reshape_2d(ctx, data->candidates, 1, ggml_nelements(data->candidates));`
  `data->candidates = ggml_reshape_1d(ctx, ggml_get_rows(ctx, candidates_rows, top_k), sctx->k);`
- M: src/llama-sampler.cpp:1495-1498 (unchanged from base)
  `ggml_reshape_2d(ctx, data->candidates, 1, data->candidates->ne[0]);`
  `data->candidates = ggml_get_rows(ctx, candidates_rows, top_k);`   → output shape [1, k]
- Consequence on master (D): input candidates 1-d → fine; input [1,n] (from an earlier top-k) → reshape_2d(.,1,ne[0]=1)
  nelements mismatch → abort. Output [1,k] → M top-p `ggml_sort` asserts `ggml_nrows(a) == 1` (M :1644) — this second
  half is fixed by stack10 ff9c8c416 (top-p flattens; S10 :1701-1703), which the master port keeps as
  topic/tp-bsamp-master bc7a6a808 (README `work/tp-bsamp-window/README.md`: "0001 dropped… 0002 test moved").
- Consumers of candidate shape in stack10 (all shape-agnostic, D):
  - greedy S10:1091, dist S10:1335, temp S10:2094, penalties S10:3113 use `ggml_nelements`; top-p S10:1702 flattens.
  - aa037232c sorted top-k S10:1531 `ggml_reshape_2d(ctx, data->candidates, 1, sctx->k)` — works for [k] and [1,k].
  - fffb753bc (TP): candidates enter the chain as a 1-d `ggml_view_1d(cand, n_cand, …)` cast (fffb753bc diff,
    llama-graph.cpp build_sampling) → leading top-k receives 1-d input → master form OK.
  - e849b7504 / c579a9127 `backend_candidates` chain = one `llama_sampler_init_top_k(candidate_k + 1 [+ n_ban])`
    (S10 common/sampling.cpp:619-620) — single top-k, no top-k→top-k.
  - 06b2752b4, d298d3dfd, 151ab739b: no candidate reshape (d298d3dfd only calls `llama_sampler_chain_leading_top_k`).
  - Host readback M src/llama-context.cpp:1695 `n_elements = ggml_nelements(tensor)` — shape-agnostic.
  - No stack10 chain contains top-k after top-k. Only S tests `top_p_on_top_k` exercises top-p→top-k→top-p, and that
    passes with ff9c8c416 alone (top-p flattens on input and output).
- Residual is a latent upstream bug (top-k after candidates-producing top-k), worth a small separate PR; not needed for
  the lift. Gain: correctness only, no perf.

## 2. 5fa55f5f7 ggml-meta: no host-noop passthrough for views of meta buffers  vs e60eff95f (#30217)

Verdict: SUPERSEDED (production hunk). M evidence exists.
- S: ggml/src/ggml-backend-meta.cpp:2087-2088 (5fa55f5f7)
  `if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer) &&`
  `        !ggml_backend_buffer_is_meta(node->view_src->buffer)) {  bcj.nodes[i] = node; continue; }`
- M: ggml/src/ggml-backend-meta.cpp:2044-2049
  `if (!ggml_backend_buffer_is_meta(node->buffer)) { // [TAG_META_HOST_VIEWS]`
  `    GGML_ASSERT(ggml_backend_meta_is_host_view(node));  bcj.nodes[i] = node; continue; }`
  plus :1179-1180 `ggml_backend_meta_is_host_view` and :1188-1192 get_split_state returns AXIS_NONE for non-meta host views.
- Difference: master keys on the node's own buffer (a view of a meta buffer has buffer = meta → mapped, same as S), drops the
  `view_src->op == GGML_OP_NONE` condition and asserts that any non-meta node is a host view (stricter).
- M: patches/upstream/llama.cpp/51-meta-host-views/master-abee0c8/RESULT.md — master + test config only:
  `test-llama-archs -s 42` 241/241 pass incl. 160 Meta(CPU,CPU) rows; negative control (meta.cpp = e60eff95f^) rc 139 in
  set_rows. Test-config commit survives as 66's 0001.

## 3. 14b23db7c ggml: include host-backed Meta views in graph planning (pkg 73)  vs #30217

Verdict: NOT SUPERSEDED (still needed for CPU-Meta; no GPU effect). #30217 changed only the node-mapping loop.
- S: ggml/src/ggml-backend-meta.cpp:2427 (get_i_delayed lookahead) and :2467 (subgraph split loop)
  `… && ggml_backend_buffer_is_host(next->view_src->buffer) && !ggml_backend_buffer_is_meta(next->view_src->buffer)) { continue; }`
- M: ggml/src/ggml-backend-meta.cpp:2179 and :2218 still the base form
  `if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) { continue; }`
- Consequence on master (D): with CPU devices the meta buffer is host, so a VIEW/CPY/SET_ROWS into a meta KV view is
  skipped by the split loop; when it is the graph's last node, i_start never reaches n_nodes →
  M :2256 `GGML_ASSERT(i_start == cgraph->n_nodes)`. pkg 73 EVIDENCE.md: "CPU Meta fixture planning aborts before the fix".
  GPU meta buffers are not host → no behaviour change on the served path.
- Test: S tests/test-alloc.cpp:1125 `test_meta_host_view` (6 cases) uses `ggml_backend_graph_plan_create` (stack-only;
  M ggml-backend-meta.cpp:2511/2514 plan hooks = nullptr) and helper `test_meta_split_state_mirrored` (stack-only; M
  test-alloc.cpp has no meta tests at all). Port must use 3× graph_compute with one uid (pkg 73 EVIDENCE.md "Mac test-port
  correction"). Upstream form = 2-line change at M :2179/:2218 + test.

## 4. b4017eeb2 MTP heads emit h_nextn for output rows only when masked (pkg 05)  vs f0c41e016 #30017, 448147d42 #30097, 1a3011cc0 #30020

Verdict: NOT LANDED; no textual conflict (D: `git apply --check` of b4017eeb2 on abee0c847 succeeds for all 9 files,
offsets −9..+1).
- #30017 handles the *main* graph only: M src/llama-graph.h:1063-1071 `crop_before_nextn`/`crop_after_nextn`; used at
  M src/models/qwen35.cpp:178-180 (crop before t_h_nextn when masked) and :213-216.
- MTP head graph still sets t_h_nextn before the out_ids crop: M src/models/qwen35.cpp:646-648
  `res->t_h_nextn = cur;` … `cur = ggml_get_rows(ctx0, cur, inp_out_ids);` (no masked re-set).
- S: src/models/qwen35.cpp:648-651 (b4017eeb2) `cur = ggml_get_rows(ctx0, cur, inp_out_ids); if (cparams.embeddings_nextn_masked) { res->t_h_nextn = cur; }`
- Master copy semantics unchanged: M src/llama-context.cpp:2034-2046 masked → copies first n_outputs rows of t_h_nextn.
  So the bug (masked MTP decode with non-leading output rows returns row-0 h) persists on master (D).
- #30097 (flags sharing) / #30020 (re-reserve on flag change) don't touch the head-graph crop.
- Upstream note: master has 7 more MTP heads not covered by b4017eeb2 (deepseek4, glm5-next, mimo2, nemotron-h-moe,
  qwen3next, qwen4exp, step35: `graph_mtp::graph_mtp` in src/models/*.cpp). Gain: correctness; needs-measure (none found).

## 5. Test hunks in tests/test-recurrent-state-rollback.cpp  vs fa3c2fab3 (#29923)

- 8646cbd0a: SUPERSEDED (alternative fix of the same failure in `test_multi_seq_split_replay`). Both keep the rollback
  anchor in the tail's ubatch.
  - S: :169 `decode_prefill` (all seqs, one batch), :181-182 `ok = decode_prefill(ctx_roll.get(), n_prompt); … decode_prefill(ctx_ref.get(), p0);`, n_prompt = 19 (:133).
  - M: :134 `n_prompt = 20`; :179-183 decode [0,p0-1) on both, then ctx_roll [p0-1,n_prompt), ctx_ref [p0-1,p0).
  - Textual conflict: same lambda/loop hunk. Drop 8646cbd0a (incl. its comment-only nmse hunk at S ~:211-216).
    Not proven: that master's form stays within nmse 1e-4 on the real Qwen3.5-4B fixture with stack10's GDN getter
    changes (needs a CPU test run on the lifted tree).
- ba8bc3fa0 (nmse_eps 1e-4→3e-3 in `test_getter_gate`), 8c013b9bb (`test_getter_gate`, S10 :383), 7130bcdc9
  (`test_state_movers`, S10 :824), cfce3d732 test hunk: stack-only tests, NOT redundant with #29923 (master has no
  getter/state-mover tests; M functions: test_multi_seq_split_replay :129, test_rollback :313, test_shared_seq_reserve :482).
  Expect textual conflicts in main()/run_tests with master's c173a53bd (`test_shared_seq_reserve`).

## 6. GDN getter (29e71fa0b 3637dda66 0a3054288 4ba349250 cfce3d732)  vs 24e41838e (#30087)

Verdict: NOT REPLACED (kernel thread-mapping change only; M gated_delta_net.cu has no `state_indices` at all).
Textual conflicts (3-way merge base 050439614, `git merge-file --diff3`, files in `gdn/`):
- 29e71fa0b alone vs master: 0 conflicts; merged kernel is consistent (state_row offset applied to curr_state base, master
  `r * lanes_per_col + lane_in_col` indexing) — `gdn/merged-29e71fa0b.cu:58-72`. Its other 18 files apply cleanly on master
  (`git apply --check --exclude=…gated_delta_net.cu`). Compile/test unverified.
- cfce3d732 (NS multi-sequence block) vs master: 15 conflict regions, all in `gated_delta_net_cuda` (template header,
  block/col/lane mapping, s_shard[NS][…] vs s_shard[rows_per_lane], state load loop, per-token update/attn reduction,
  snapshot writes, final state writes) + 1 in `launch_gated_delta_net` grid sizing.
  - S: gated_delta_net.cu:7 `template <int S_v, bool KDA, bool keep_rs_t, int NS>`, seq0/NS loops, `GGML_ASSERT(NS == 1 || n_seqs <= NS); dim3 grid_dims(H, NS == 1 ? n_seqs : 1, …)`
  - M: gated_delta_net.cu:4 `constexpr int gdn_cols_per_warp = 4;`, :40 `lanes_per_col`, :47 col mapping, :68/88/126/140/162
    `i = r * lanes_per_col + lane_in_col`, :196 `while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm)`.
  - Needs a re-design of the NS path on the 4-columns-per-warp mapping (not a mechanical merge). 519069f3e (local) adds
    15 more lines on top. 3637dda66/0a3054288/4ba349250 fail apply only on stack-predecessor context (stack6 GDN gathers),
    not on master changes (D: master commits touching delta-net-base.cpp are unrelated model additions).
- ORDER.md:123 pkg 85 "Drop on rebase"; :126 pkg 67 "0002 conflicts with #30087".

## 7. 4bf1c2ea3 batched deterministic top-k (K10/pkg 81)  vs 71ad0590f (#28713)

Verdict: ORTHOGONAL / partially overlapping on speed; NOT superseded (determinism absent on master).
- S: ggml/src/ggml-cuda/top-k.cu:240 `if (topk_batched_supported(ncols, nrows, k) && (nrows > 1 || ncols <= TOPK_SMALL_NCOLS)) { … topk_batched(…); return; }`
- M dispatch ggml/src/ggml-cuda/top-k.cu:298-344 (served build uses -DGGML_CUDA_CCCL_VERSION=v3.4.3 →
  CUB_TOP_K_AVAILABLE, NROWS_THRESHOLD = 2 (:41-43); BITONIC = 512 (:10), BITONIC_FEW_ROWS = 1024 (:23); CUDA 13.2 alone
  ships CCCL 3.2.0 (`/usr/local/cuda-13.2/include/cccl/cuda/std/__cccl/version.h:17`), so without the CCCL pin the
  radix/argsort paths apply instead):
  - ncols 124160 or 248320 ≫ 1024 → bitonic paths off (:301, :305-307).
  - nrows 1..2 → `for (i < nrows) top_k_cub(…)` per-row DeviceTopK (:324-327) — same as base/stock.
  - nrows 3..6 → `top_k_radix_cuda` (:318-321): 1 init + 4×(histogram+select) + reset + gather per chunk, output order
    from `atomicAdd` (:201, :204) = nondeterministic order (and tie choice).
- Stack10 (M, pkg 81 EVIDENCE.md:17,20 / notes 2026-10-09.md:209) measured only vs stock per-row CUB: [124160,6] 50 vs
  168 µs, [248320,12] 95 vs 470 µs; [124160,1] 1.19-1.25× slower; [124160,2] k=41 1.9× slower (noisy). Master's radix at
  3..6 rows is a different baseline → perf gain vs master: needs-measure. Order guarantee: survives only with pkg 81.
  K10 served value was D ≤ 0.03 ms/round for 1-2 rows (EVIDENCE.md:20).

## 8. 758c93a1c retain attention rotations on the cache backend (K3)  vs 210791069 (#28498)

Verdict: NOT LANDED; conflict (re-port needed). #28498 = save/restore rotation metadata, orthogonal behaviour.
- M src/llama-kv-cache.cpp:362-363 still `// pre-compute the haramard matrices and keep them in host memory` /
  `// TODO: … copies in the backend buffers to avoid host -> device transfers`; :1467-1476 `build_input_k_rot` creates a
  fresh `ggml_set_input` tensor sized `n_rot_k`; :1828-1834 `set_input_k_rot` memcpy from host each graph.
- S: src/llama-kv-cache.cpp:364-391 (persistent `ctx_rot`/`buf_rot` on the layers' buft, log :386), build_input_*_rot
  return `attn_rot_tensors.find(nrot)` (:1542, :1567), set_input_* early-return (:1991, :2005).
- Conflict: #28498 replaced the derived `nrot` loop with members `n_rot_k/n_rot_v` (M src/llama-kv-cache.h:281) and renamed
  `attn_rot_k` → `n_rot_k` in the ctor; `git apply --check` fails at llama-kv-cache.cpp:342. Port: look up
  `attn_rot_tensors.find(n_rot_k)`. Meta hunk (child-alloc failure cleanup) and test hunk separate.
- Gain: M drop-one, notes 2026-10-08.md:522-529 (s10c-np2-real-j5q): −K3 arm +0.43 vs s10 +0.23 ms/round at 8k → K3 ≈ −0.2
  at 8k, ≈ −0.08/−0.07 at 32k/90k (2 boots/arm). Survives on master by code (host upload still present); needs-measure.

## 9. 3ebadf53a fuse F32 add+softplus+broadcast mul (K5)  vs 08246a28f (#29781), ff30363a0 (#29687)

Verdict: NOT LANDED on CUDA. ff30363a0 = SYCL-only same fusion (M ggml/src/ggml-sycl/ggml-sycl.cpp:6391); 08246a28f =
unary striding, no fusion.
- M CUDA has only UNARY(SOFTPLUS)+MUL: ggml/src/ggml-cuda/ggml-cuda.cu:4379-4383 → `ggml_cuda_op_unary_mul`; no
  `{ GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL }` anywhere in ggml-cuda/.
- S: ggml-cuda.cu:3653-3683 (can_fuse ADD,UNARY,MUL SOFTPLUS) and :4572-4575 (`ggml_cuda_op_add_softplus_mul`), unary.cu:663-691.
- Apply: code files apply on master with offsets (`--exclude=docs/ops.md`); docs/ops.md conflicts; uses
  `ggml_ssm_conv_overlaps` (stack-only helper, absent on master) → prerequisite or inline replacement needed.
- Gain: notes 2026-10-08.md:526,529 −K5 ≈ −0.1 ms/round at 8k (drop-one). Survives; needs-measure on master.

## 10. 0f22f5ffb (pkg 06), ff9906f60 + 24334ff46 (pkg 07)  vs 033df86b6 (#26004)

Verdict: NOT LANDED; #26004 is slot save/restore of checkpoints (orthogonal). Conflicts expected (neighbouring code).
- M tools/server/server-context.cpp:2575-2576 `cur.update_tgt(ctx_tgt, …PARTIAL_ONLY); cur.update_dft(ctx_dft, …PARTIAL_ONLY);`
  (unconditional draft copy); M common/common.cpp:2389-2391 `data_tgt.resize(ckpt_size)` (fresh buffer each time).
- M src/llama-kv-cache.cpp: no PARTIAL_ONLY handling (grep: only iswa/dsa/hybrid/dsv4 caches check it) → the draft-KV
  dump premise still holds.
- S: server-context.cpp:873 `struct server_ckpt_buffers`, :2576 `cur.data_tgt = ckpt_bufs.take(…)` (0f22f5ffb);
  :2505 `if (ctx_dft_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_PART)` (ff9906f60); :2506 `… || llama_model_n_swa(llama_get_model(ctx_dft)) > 0)` (24334ff46).
- Note: 0f22f5ffb assigns `cur.data_tgt` before `update_tgt`; master update_tgt (common.cpp:2391) resizes — works with a
  reused buffer (D). Gain M: pkg 07 EVIDENCE.md:35-43 (create 114→10 ms; follow-up 578/1109→399/433 ms at 8k/24k);
  pkg 06 EVIDENCE.md:35 (76-119 ms per 150 MiB). Survives; needs-measure on master.

## 11. 590912a1d preserve buffered token IDs in streaming (pkg 57)  vs 18b5f8b18 (#29876)

Verdict: NOT LANDED; bug persists at abee0c847 (re-verified).
- M tools/server/server-context.cpp:1972-1974 `if (slot.task->params.return_tokens) { slot.generated_tokens.push_back(result.tok); }`;
  :2006 `slot.add_token(result);` only inside `if (!incomplete)`; :2194 `res->tokens.assign(1, tkn.tok);`; :2208-2210
  `res->prob_output = tkn;` → tokens/probs of UTF-8-buffered steps are never streamed.
- S: :281 `size_t n_sent_tokens = 0;`, :2009 `return_tokens || …stream`, :2012-2013 add_token before buffering,
  :2234 `res->tokens.assign(slot.generated_tokens.begin() + slot.n_sent_tokens, …end())`.
- Conflict: #29876 replaced `slot.generated_text` with `slot.generated` (M :1971 `slot.generated.append(token_str, result.tok)`,
  :2193 content substr) → rework, as UPSTREAM.md:103 says.

## 12. 151ab739b split build_sampling into per-row helpers, fffb753bc, 7d0c39624  vs a518119d3 (#30223)

Verdict: 151ab739b NOT superseded, conflicts; fffb753bc already re-ported onto #30223; 7d0c39624 still needed.
- M src/llama-graph.cpp:3975-4069: `for (i < cparams.n_outputs_max_per_seq)` chains per sampler, inactive → row 0,
  `ggml_build_forward_select(…, i_out)`; no helpers.
- S (151ab739b): src/llama-graph.cpp:3903 `build_sampling_row(…)`, :4030 `for (uint32_t i = 0; i < rows.size(); ++i)`
  (one chain per output row; varies with rows) — header S10 src/llama-graph.h (begin/cand/row/end) used by d298d3dfd
  (in-graph MTP draft chain).
- #30223's goal (static topology) and 151ab739b's per-row loop collide in the same loop: re-split the helpers on top of
  the n_outputs_max_per_seq loop (pkg 21 port).
- fffb753bc → topic/tp-bsamp-master 5a487435b: M-port src/llama-graph.cpp:4051-4076 static loop with `cand_ids`.
- 7d0c39624: S src/llama-graph.cpp:3833 `if (sampling_n_shards > 0 && ggml_nelements(res->t_logits) == 0) return;`;
  port 6c4a579d9 keeps it (tp-bsamp-master llama-graph.cpp:3982). work/tp-bsamp-window/README.md: Mac mutant with the guard
  off aborts `ggml-backend-meta.cpp:880 src_ss[i].axis != UNKNOWN` on master+#30223 (M per README).

## 13. 5aee04704 prompt-shift remap lambda `-> size_t`

Verdict: N/A on master (no lambda). The lambda is introduced by stack10 a48fd4290 (S tools/server/server-context.cpp:1927
`const auto remap = [&](size_t pos) {`; fix S :1930). M: no "only in spelling"/remap in server-context.cpp.
In the master port topic/port-c the lambda exists in a0c6f75e0 (:1921, unfixed) and is replaced by feb6e815d with
`size_t server_resync_remap_pos(…)` (port-c server-common.cpp:863; server-context.cpp:1923,1926) → fix subsumed at port-c HEAD.

## 14. 2d3511bf7 + c5087a10f (meta subgraph realloc)  vs pkg 86 master port

- `git patch-id --stable`: 1cdaba045 (topic/meta-subgraph-fix, parent 08246a28f) = 86afcf77e (topic/port-a, parent
  abee0c847) = patches/upstream/llama.cpp/86-meta-subgraph-realloc/master-abee0c8/0001-*.patch = f9d7c194035b. Same patch.
  Package patch header says `From 86afcf77e…`.
- In-process: yes. The package test (test-alloc.cpp `test_meta_subgraph_realloc_count/size`) has no fork/waitpid/_WIN32
  guard; uses `ggml_backend_graph_compute` (no plan API). So 2d3511bf7's purpose is built in.
- Production hunk differs from c5087a10f: master port changes BOTH loop bound (`n_subgraphs` → `max_subgraphs`) and
  capacity (`cgraph->n_nodes` → `max_nnodes`); stack10 c5087a10f changes only the bound because stack10 fc1bbaf12 had
  already set `max_nnodes` capacity (S10 ggml-backend-meta.cpp:2654-2658). M RESULT.md: each hunk necessary (count/size
  controls; ASAN UAF at meta.cpp:2307 on base).

## 15. Master ports vs stack10 commits

- d1022e454 vs topic/port-a 5f96b0b83: identical (patch-id a4c5f5ad55d5 both). Adds nothing.
- 52fc3fd2c vs topic/kvpad-master da5727df0: identical (patch-id e34b3be2068e both); later branch commits are `gate:` only.
- 825fd114b vs topic/port-f (c38ec6624, d31ee6e1d, 70b1b5cf2): concat.cu +/- lines byte-identical (diff rc 0). Tests:
  same correctness cases + v-flag comment; adds perf cases (conv-state shapes, ne0 5..512 sweep, long-row controls).
- a48fd4290 vs topic/port-c (5 commits): a0c6f75e0 code hunks identical to a48fd4290 (server-common.cpp/.h,
  server-context.cpp, server-task.cpp +/- lines equal). Changes: drops the C++ unit test tests/test-server-tokens.cpp (188
  lines) + CMake hook, replaces with python tests in test_completion.py (dddc19c14, 63eea7bc4, c2bb3e1a9: non-canonical
  split reuse, n_keep inside a reused span, invalid token ids); feb6e815d CHANGES behaviour: positions (n_keep, message
  span starts) that fall inside a replaced span move to the span END (stack10 kept the index → arbitrary cached token),
  via `server_resync_span` / `server_resync_remap_pos` (also subsumes 5aee04704).

## 16. Sweep for other stack10 commits already on master

- Per-commit content sweep (`sweep.py`): added lines (len ≥ 25, not in base file) of every non-codex/non-local stack10
  commit looked up in the master file. Max overlap 9 % (151ab739b 4/45, 8c013b9bb 7/83); all others ≤ 4 %. No commit
  substantially present on master.
- Subject sweep over the 162 master commits: no further duplicates. Near-area but different changes: 50e3e3e48 (SSM_SCAN
  snapshot copy fusion; stack10 0d307c896 is SSM_CONV), 8a1a9b512/fc9ce6b9d (MMQ; expect conflicts with stack10 015a0cbc7,
  3f0814173, c5baada6d small-I tiles), 9d3aba6b5 (MMVF thin), d89651a7b (FA scheduling), 8e2d31e0e (embedding get_rows
  reorder), c173a53bd (test-recurrent-state-rollback main()).
- `git cherry` impossible (blob:none clone; master blobs missing, lazy fetch off). Per-file patch-id sweep (`pidsweep.txt`):
  0 MATCH. Coverage gap: 163 master file-diffs on stack10-touched paths (73 of the 162 master commits) unreadable
  (missing blobs), listed as UNREADABLE there; the content sweep above covers them (it reads master's final files).
