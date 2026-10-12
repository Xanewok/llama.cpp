# stack10 → stack11-base drop map

stack10 = frozen/stack10 a588f2a44 (base 050439614), 233 commits. stack11-base = topic/stack11-base, base master abee0c847,
head 2f0fb6706 when this file was generated. One row per stack10 commit, in stack10 order.

Classes: carry (in stack11, same change or master port), landed (on master), superseded (master or an upstream-form package
does the same job), dead (measured to do nothing on the served shapes), scaffolding (codex/, logs, dumps, switches, mutants
not needed by any stack11 gate), local (kept as a local commit). needs-measure marks a carried row whose stack11 benefit is
unproven: it rides the optional drop arm. Evidence: S = stack10 tree, S10 = a588f2a44, M = master abee0c847; [U§n] = section n
of the upstream-equivalence audit, [EVIDENCE-scaffold.md:line] = the scaffolding audit (both kept beside this file as EVIDENCE-*.md).

Counts: carry 97, landed 1, superseded 8, dead 2, scaffolding 104, local 21; needs-measure 2.

Commits in stack11 with no stack10 source: ebf25ecef ggml : a shared scheduler inherits the parent's copy callback; c5baf5f84 speculative : the MTP draft chain passes the hidden state as the state embedding; 152959f6c codex: stack11 Mac CPU payload (from the stack10 component libs, k2/k9 and parser stages removed, embd_state identity added); 75fd1821c codex: stack11 Mac payload: executable scripts, mfix gates on the pkg 86 test names; 0dcebe53d codex: stack11-base CUDA build window, A/B and KLD items (validated by DRY_RUN=1, not run); 2f0fb6706 codex: stack11 Mac payload: test-backend-sampler takes --model.

| # | stack10 | subject | class | stack11 | evidence |
|---|---|---|---|---|---|
| 1 | df5471ab7 | sampling : map backend greedy to vocab ids and keep top-k candidates 1-d | landed | - | greedy hunk = #29797 d0b490f25: S llama-sampler.cpp:1089-1093 = M :1089-1093. Residual top-k 1-d hunk not needed: no served chain has top-k after top-k, consumers use nelements (S10 common/sampling.cpp:620, M llama-context.cpp:1695), top-p flattens (ff9c8c416) [U§1] |
| 2 | ff9c8c416 | sampling : backend top-p takes and returns its candidates flat | carry | b4605f9d1 | pkg 04 master port (wt/tp-bsamp-master, work/tp-bsamp-window/README.md); Mac tp-bsamp-mac-a2 rc 0 |
| 3 | c5cec2075 | ggml-meta : add ggml_backend_meta_gather, an exact all-gather | carry | 065e0c47a | pkg 04 master port (wt/tp-bsamp-master, work/tp-bsamp-window/README.md); Mac tp-bsamp-mac-a2 rc 0 |
| 4 | 46532391f | ggml-meta : device-local TOP_K and GET_ROWS on dim-0 split tensors | carry | 56bc8a8f0 | pkg 04 master port (wt/tp-bsamp-master, work/tp-bsamp-window/README.md); Mac tp-bsamp-mac-a2 rc 0 |
| 5 | fffb753bc | llama : sample from per-device top-k candidates when logits are vocab-split | carry | 701686188 | pkg 04 master port (wt/tp-bsamp-master, work/tp-bsamp-window/README.md); Mac tp-bsamp-mac-a2 rc 0 |
| 6 | 7d0c39624 | llama : no backend sampling nodes for vocab-split ubatches without outputs | carry | 365d8089c | pkg 04 master port (wt/tp-bsamp-master, work/tp-bsamp-window/README.md); Mac tp-bsamp-mac-a2 rc 0 |
| 7 | 0ad18ee77 | llama : allow backend sampling with SPLIT_MODE_TENSOR | carry | 14c323568 | pkg 04 master port (wt/tp-bsamp-master, work/tp-bsamp-window/README.md); Mac tp-bsamp-mac-a2 rc 0 |
| 8 | 5fa55f5f7 | ggml-meta : do not pass views of meta buffers through as host noops | superseded | - | #30217 e60eff95f: S ggml-backend-meta.cpp:2087-2088 vs M :2044-2049 (+:1179,:1188), master asserts is_host_view; M pkg51/master-abee0c8/RESULT.md: master 241/241 Meta(CPU,CPU), #30217 reverted rc 139 [U§2] |
| 9 | fcafdc81e | tests : backend sampling under SPLIT_MODE_TENSOR | carry | 5a9016e0c | pkg 04 master port (wt/tp-bsamp-master, work/tp-bsamp-window/README.md); Mac tp-bsamp-mac-a2 rc 0 |
| 10 | 491a20f5a | llama : TP backend sampling only when every vocab slice holds top-k rows | carry | 58337f5b1 | pkg 04 master port (wt/tp-bsamp-master, work/tp-bsamp-window/README.md); Mac tp-bsamp-mac-a2 rc 0 |
| 11 | 543907c61 | ggml-meta : document the contract of ggml_backend_meta_gather | carry | 31a570892 | pkg 04 master port (wt/tp-bsamp-master, work/tp-bsamp-window/README.md); Mac tp-bsamp-mac-a2 rc 0 |
| 12 | b4017eeb2 | models : MTP heads emit h_nextn for output rows only when masked | carry | 0913207ff |  |
| 13 | ff9906f60 | server : skip the draft KV in context checkpoints when the draft can seq_rm partially | carry | b7e677edb |  |
| 14 | 24334ff46 | server : keep the draft KV in context checkpoints of a sliding-window draft | carry | d4f62f051 |  |
| 15 | 0f22f5ffb | server : reuse and prefault the host buffers of context checkpoints | carry | 4647f6d41 |  |
| 16 | 06b2752b4 | common : sampling: take the chain's top-k straight from the logits | carry | d2057ce04 |  |
| 17 | bca823e3d | common : sampling: keep the top-k prefilter under a lazy grammar and the reasoning budget | carry | 6db0a3fcf |  |
| 18 | 10af6d7ca | common : sampling: the top-k prefilter falls back to the full row on ties | carry | 04ac1df38 |  |
| 19 | 2b53ebdca | ggml-cuda : FA K/V to f16, one contiguous conversion per KV stream | carry | 3a8b36663 |  |
| 20 | 7e1f24e3f | ggml-cuda : FA per-stream K/V conversion only for dense, aligned streams | carry | 47809d0e2 |  |
| 21 | a48fd4290 | server : reuse the cached tokens when the prompt only differs in spelling | carry | 0cf5d3d68,2f3d77a71,8bf672a52,d63fdd4ea,9678b56c4 | pkg 08 master port (port-c): code identical to a0c6f75e0; C++ test replaced by python tests; feb6e815d moves n_keep/span starts inside a replaced span to the span end (a fix) [U§15] |
| 22 | cf2580acc | ggml-cuda : FA MMA kernel dequantizes q8_0 K/V tiles for small batches | carry | 73e5d42c5 |  |
| 23 | 64730fd0d | tests : FA with q8_0 K/V at speculative decoding batch sizes | carry | 0ba9fbb28 |  |
| 24 | ac6db6790 | DO NOT MERGE: ggml-cuda : GGML_CUDA_FA_MMA_Q8_0=0 restores the f16 K/V copy (A/B knob) | local | d417a1c3b | fafix informational A/B arm; reworded local: |
| 25 | a655c25fa | ggml-cuda : spread q8_0 FA dequant stores over rows | carry | 30f48f751 |  |
| 26 | 4234745d2 | ggml : batch scheduler host input uploads | carry | 699635863 |  |
| 27 | e849b7504 | sampling : verify target rows from backend candidates | carry | 43727d77d |  |
| 28 | f1c04c71f | llama : add reduced vocabulary for Qwen3.5 MTP drafts | carry | 4adaef31a |  |
| 29 | af326cc40 | llama : handle small MTP draft vocabularies | carry | c02d9730b |  |
| 30 | ec8428ce2 | llama : retain scheduler when replacing backend samplers | carry | b5066ddcd |  |
| 31 | 45433b551 | llama : retain scheduler when detaching backend samplers | carry | 367cbf3b8 |  |
| 32 | c7d253b35 | ggml-meta : resolve split states without recursion | carry | edca49076 |  |
| 33 | 8dc66b1b7 | llama : trace MTP round submission (DO NOT MERGE) | local | 285679aa9 | round trace (DO NOT MERGE, pkg 27 local by design) kept in place: 23 production call sites depend on it |
| 34 | a78159b91 | ggml : retain split plans on shared compute storage | carry | e63bf62ea |  |
| 35 | e07f827ce | llama : reuse output and catch-up graphs across MTP rounds | carry | 6be3a28b4 |  |
| 36 | d719e9acc | llama : fetch deferred logits before graph invalidation | carry | c759b6f00 |  |
| 37 | 0116f803e | ggml : declare shared meta storage helpers | carry | cca692d6c |  |
| 38 | e486ceb19 | tests : trace prefill ubatches and meta plan allocations | local | 6cb4ca12f | round trace (DO NOT MERGE, pkg 27 local by design) kept in place: 23 production call sites depend on it |
| 39 | 83950697b | ggml : reuse one spare meta plan arena | carry | fbab69565 |  |
| 40 | a075f2cb0 | ggml : hash visited tensors when retaining meta plans | carry | abfea9c13 |  |
| 41 | fc1bbaf12 | ggml : keep recycled meta graphs at their retained capacity | superseded | - | pkg 86 port allocates every cgraph_main with max_nnodes for i < max_subgraphs: stack11 ggml-backend-meta.cpp:2682 vs S10 :2662 (this commit's capacity change) [U§14] |
| 42 | 95141e1a5 | ggml : own staged host inputs until uploads complete | carry | 2eb2a7b3b |  |
| 43 | 6d86af379 | ggml : stage inputs from synchronous host producers | carry | 04973e130 |  |
| 44 | d40158761 | llama : gate retained graphs and owned input staging | carry | aa57de464 |  |
| 45 | b60579ec1 | cuda : grow owned input slots geometrically | carry | 8e83cf368 |  |
| 46 | 47c53a6f5 | tests : check stack3 CPU identity and allocation contracts | scaffolding | - | codex/ only (28 files) |
| 47 | f11b64cec | tests : prepare a frozen stack3 GPU identity gate | scaffolding | - | codex/ only (16 files) |
| 48 | ac0f5003b | tests : harden stack3 harness cleanup and memory checks | scaffolding | - | codex/ only (3 files) |
| 49 | ae5d03d06 | docs : record stack3 checks and frozen GPU item | scaffolding | - | codex/ only (6 files) |
| 50 | ce19806d9 | kv-cache : fill dense sequence masks from position indexes | carry | a494bd6f4 |  |
| 51 | 77464fdba | kv-cache : take the dense-prefix mask path for M-RoPE positions | carry | b3f6a84e1 |  |
| 52 | 24ac966c7 | kv-cache : build visible mask prefixes on the meta backend | carry | 97ad8fc98 |  |
| 53 | 0fe95ad25 | tests : CPU mask byte-identity, compact-prefix and stale-reuse checks for the stack3 mask fixes | scaffolding | - | codex/ only (8 files) |
| 54 | cbca255eb | tests : freeze script for the stack3 mask build | scaffolding | - | codex/ only (1 files) |
| 55 | 08736c1f9 | tests : CPU checks for the stack3-mask rebase | scaffolding | - | codex/ only (10 files) |
| 56 | 98c695111 | tests : freeze script for the stack3-mask rebase build | scaffolding | - | codex/ only (1 files) |
| 57 | 8a1001e26 | cuda : four rows per block for 5-8 column MMVQ on Blackwell | local | 17b21e66a | sm_120 table data (pkg 56/0001) with pkg 55's per-type rows, one local commit on top; mmvq-params.h and test-mmvq-params.cpp byte-identical to S10 (wt/stack11-mmvq codex/stack11-mmvq/PORT.md) |
| 58 | b91f5e291 | kv-cache : allow compact masks for multiple streams | carry | 5a3d74a7f |  |
| 59 | 590912a1d | server : preserve buffered token IDs in streaming completions | carry | 72cb95542 |  |
| 60 | 562f6bd36 | ggml-cuda: allreduce copy path: optional separate H2D stream per GPU | superseded | 92a1fe061 | pkg 59 upstream form = this knob form at its default (2 streams) minus the knob, shared-stream aliasing and log line (patches/upstream/llama.cpp/59-ar-copy-dual-stream/apply.txt:7) |
| 61 | 8b758c95e | ggml-cuda: allreduce copy path: default to a separate H2D stream | superseded | 92a1fe061 | pkg 59 upstream form = this knob form at its default (2 streams) minus the knob, shared-stream aliasing and log line (patches/upstream/llama.cpp/59-ar-copy-dual-stream/apply.txt:7) |
| 62 | 52fc3fd2c | kv-cache : make the n_kv padding a context parameter | carry | ff7fcd54c | pkg 61 master port kvpad-master da5727df0, patch-id equal [U§15] |
| 63 | 7130bcdc9 | llama : keep recurrent convolution updates together | carry | 0e9647bdb |  |
| 64 | 201506802 | cuda : fuse recurrent convolution state updates | carry | f8bb7a90c |  |
| 65 | 825fd114b | cuda : flatten non-contiguous concat for small rows | carry | 88bd9bb01,97288571a,82f1f3537 | pkg 62 master port (port-f): concat.cu byte-identical, adds perf cases [U§15] |
| 66 | 0d307c896 | cuda : fuse convolution snapshots after concat | carry | e208d313c |  |
| 67 | ac7465885 | cuda : report convolution snapshot fusion coverage | carry | cf076f70d |  |
| 68 | 29e71fa0b | ggml : read gated delta net state by row id | carry | d60d67df4 |  |
| 69 | 3637dda66 | llama : use a recurrent state getter for GDN | carry | 05e741f2c |  |
| 70 | 6727f3cb1 | cuda : trace GDN getter engagement | scaffolding | - | engagement log / dump / mutant, default no-op, consumed by no stack11 gate or canary [EVIDENCE-scaffold.md:24; Task B tables :99-150] |
| 71 | 0a3054288 | llama : remove the replaced stack6 GDN gathers | carry | 3b43549c4 |  |
| 72 | 9a3af300a | virtgpu : do not claim GDN state ids support | carry | 3a38c89c6 |  |
| 73 | 4ba349250 | llama : read GDN state by id only for single-sequence ubatches | carry | 9f72b161a |  |
| 74 | 8c013b9bb | tests : check the GDN state getter gate by ubatch sequence count | carry | 162f266cb |  |
| 75 | a59f806e4 | cuda : log GDN cache fusion per captured graph | local | 32336c9a1 | engagement log consumed by the np2 canaries; reworded local:, 6727f3cb1 context dropped |
| 76 | 8646cbd0a | tests : roll back the multi-seq split replay within the last ubatch | superseded | - | #29923 fa3c2fab3 fixes the same anchor/ubatch case: S test-recurrent-state-rollback.cpp:169,:181-182 vs M :134,:179-183; CPU check = Mac stage p55-recurrent-state-rollback (test_multi_seq_split_replay) [U§5] |
| 77 | ba8bc3fa0 | tests : GDN getter gate tolerance above the batch-shape noise floor | carry | e450ac69e |  |
| 78 | 447a47613 | llama : remove KV sequence ranges through the position index | carry | 3e0c1db0d |  |
| 79 | 89637899d | speculative : copy MTP verify hidden rows in one readback | superseded | c4fde33c4 | pkg 71 upstream form: the v2 default path (one post-catch-up getter + single memcpy), knobs/counters/mutants excluded (patches/upstream/llama.cpp/71-mtp-verify-bulk-read/NOTES.md:15) |
| 80 | f61bee3fa | host-trims : report exercised paths for local A/B checks | scaffolding | - | host-trims proof log / env switch / mutant, default no-op; replaced with the pkg 71 form [EVIDENCE-scaffold.md:26] |
| 81 | d7bc08c2c | host-trims : env switches LLAMA_HOST_TRIMS_SEQRM and LLAMA_HOST_TRIMS_BULKH restore the stack6 paths | scaffolding | - | host-trims proof log / env switch / mutant, default no-op; replaced with the pkg 71 form [EVIDENCE-scaffold.md:27] |
| 82 | 89cff0ccf | host-trims : read the target nextn pointer after the draft catch-up decode; BULKH=2 keeps the hoisted pointer as a control | superseded | c4fde33c4 | its production change (nextn pointer read after the draft catch-up) is the pkg 71 form's default path; its BULKH=2 control is scaffolding [EVIDENCE-scaffold.md:28] |
| 83 | f27ce2576 | host-trims : test-only LLAMA_HOST_TRIMS_VERIFY cross-checks and LLAMA_HOST_TRIMS_MUTANT negative controls | scaffolding | - | host-trims proof log / env switch / mutant, default no-op; replaced with the pkg 71 form [EVIDENCE-scaffold.md:29] |
| 84 | a7f31a570 | speculative : log and offset the bulk row negative control | scaffolding | - | host-trims proof log / env switch / mutant, default no-op; replaced with the pkg 71 form [EVIDENCE-scaffold.md:30] |
| 85 | 151ab739b | graph : split build_sampling into per-row helpers | carry | 701fcc0ad |  |
| 86 | 96e025496 | llama : load a compact token embedding table for the draft vocabulary | carry | 530f13ec8 |  |
| 87 | d298d3dfd | qwen35 : add a device-side MTP draft chain | carry | 40e898e5e |  |
| 88 | aa037232c | sampler : add a top-k that sorts its backend candidates | carry | 2afe48b1f |  |
| 89 | 21ba77aa8 | speculative : draft the MTP steps of a round in one decode | carry | 65c2cffa6 |  |
| 90 | 4be92a4fa | speculative : add negative controls and a draft dump for the MTP chain test | scaffolding | - | engagement log / dump / mutant, default no-op, consumed by no stack11 gate or canary [EVIDENCE-scaffold.md:31; Task B tables :99-150] |
| 91 | d92da203f | speculative : attach a fresh sampler each time the MTP chain is switched | carry | 9b9ba4d02 |  |
| 92 | 9afc6714e | ggml : move round trace writes off the decode thread | local | d39793ced | round trace (DO NOT MERGE, pkg 27 local by design) kept in place: 23 production call sites depend on it |
| 93 | 6e99847f9 | ggml : restore static trace kinds and strengthen writer tests | local | 920fea241 | round trace (DO NOT MERGE, pkg 27 local by design) kept in place: 23 production call sites depend on it |
| 94 | ffdb7d6e8 | llama : fold the MTP catch-up K/V into the target decode (LLAMA_MTP_FOLD_C) | carry | 5cd40cf0f |  |
| 95 | e99e52ef3 | speculative : add a draft dump and a fold mutant for the MTP fold test | scaffolding | - | engagement log / dump / mutant, default no-op, consumed by no stack11 gate or canary [EVIDENCE-scaffold.md:34; Task B tables :99-150] |
| 96 | 14b23db7c | ggml : include host-backed Meta views in graph planning | carry | 00f268e44 |  |
| 97 | ce555540a | models : flatten SSM output projection inputs across sequences | carry | 760398e3b |  |
| 98 | d72722a99 | llama : report stack7 canary and mutant engagement | scaffolding | - | engagement log / dump / mutant, default no-op, consumed by no stack11 gate or canary [EVIDENCE-scaffold.md:35; Task B tables :99-150] |
| 99 | 3a77937b4 | codex : add the stack7 build window and GPU item templates | scaffolding | - | codex/ only (31 files) |
| 100 | d98212aef | codex: bw-stack7 manifest: GPU UUIDs as other windows, pin mmq.sh | scaffolding | - | codex/ only (1 files) |
| 101 | ce3c9554a | codex: bw-stack7 host overrides and Mac mini CPU runner | scaffolding | - | codex/ only (11 files) |
| 102 | 5aee04704 | server : give the prompt-shift remap lambda an explicit size_t return type | superseded | - | fixes the remap lambda of stack10's a48fd4290 (S server-context.cpp:1927); the pkg 08 port has no lambda: server_resync_remap_pos returns size_t (stack11 tools/server/server-common.cpp:863, used at server-context.cpp:1997,2000) [U§13] |
| 103 | d2aab42b9 | codex: Mac runner uses Ninja | scaffolding | - | codex/ only (1 files) |
| 104 | 592646955 | codex: bw-stack7 S7_CPU_EXTRA | scaffolding | - | codex/ only (1 files) |
| 105 | 9c44fcfb4 | codex: Mac runner builds without i8mm | scaffolding | - | codex/ only (1 files) |
| 106 | 4924ce0f2 | codex: Mac runner links ggml interposable for the row0 shim | scaffolding | - | codex/ only (1 files) |
| 107 | 7af8518de | codex: row0 shim via dyld interposing on Darwin | scaffolding | - | codex/ only (3 files) |
| 108 | 19314f46c | codex: getter check S7_SKIP_ROW0 diagnostic switch | scaffolding | - | codex/ only (1 files) |
| 109 | 5a0bcda09 | codex: bw-np2-driver Meta device list needs its nullptr terminator | scaffolding | - | codex/ only (1 files) |
| 110 | 3e759849b | codex: drop the non-working Darwin row0 interposing | scaffolding | - | codex/ only (4 files) |
| 111 | 48b82b0d1 | codex: bw-stack7 a2: stage 29 chain check report-only, np2/duo items chain off, rerun only missing stages | scaffolding | - | codex/ only (4 files) |
| 112 | ee9cf84c8 | codex: stack7 items pin every codex script (gpuq library-pin contract) | scaffolding | - | codex/ only (3 files) |
| 113 | ddedffacb | codex: stack7 items: harness snapshot migration, ignore_eos true for depth/np2 | scaffolding | - | codex/ only (6 files) |
| 114 | ad843282f | common : restrict MTP draft chain to single-sequence slot sets | carry | bbd245f4c |  |
| 115 | c579a9127 | common : sampling: keep the backend candidates under a -inf logit bias | carry | 7bf2e9249 |  |
| 116 | 2ecda894d | codex : integrate stack8 Mac identity checks and controls | carry | 7b779a589 | codex commit; its one production hunk (the 'MTP chain disabled' log line, a canary of the np2/duo items) is carried as a new commit, the rest is scaffolding [EVIDENCE-scaffold.md:36] |
| 117 | 8774e3ea0 | codex : prepare stack8 CUDA window and GPU item templates | scaffolding | - | codex/ window files plus a .gitignore line |
| 118 | 9cf7e13b3 | codex : document stack8 source and prepare final unit recheck | scaffolding | - | codex/ only (7 files) |
| 119 | 02ae1f30a | codex : replay stack8 fallback and candidate proof controls | scaffolding | - | codex/ only (6 files) |
| 120 | 0c0dc0d72 | codex : require nonempty banned backend candidate proofs | scaffolding | - | codex/ only (2 files) |
| 121 | 9054a6ee2 | codex : preserve stack8 build and contract failure gates | scaffolding | - | codex/ only (2 files) |
| 122 | 0e6826c8d | codex : validate stack8 experiment and window structures on Mac | scaffolding | - | codex/ only (1 files) |
| 123 | e45e46d84 | codex : run stack8 template rechecks without repeating model stages | scaffolding | - | codex/ only (2 files) |
| 124 | 63e659d71 | codex : prevent stack8 unit gate starvation and preserve controls | scaffolding | - | codex/ only (10 files) |
| 125 | 41d2954a2 | codex : record stack8 Mac validation and queue readiness | scaffolding | - | codex/ only (21 files) |
| 126 | 5df0dbffc | codex: stack8 items pin gpu_ab.sh's sibling scripts (gpuq library rule) | scaffolding | - | codex/ only (1 files) |
| 127 | cfce3d732 | llama : read multi-sequence GDN states by id | carry | 6c3b809cf |  |
| 128 | c19629b30 | llama : fold MTP catch-up for several sequences | carry | 4756040eb |  |
| 129 | 519069f3e | local: add stack9 fold and GDN negative controls | local | 53f7cb08c | local layer on top |
| 130 | 0dae1d784 | tests : integrate stack9 CPU identity gates | scaffolding | - | codex/ only (8 files) |
| 131 | 925c8dd82 | codex : prepare stack9 build window and GPU items | scaffolding | - | codex/ window files plus a .gitignore line |
| 132 | fd58c11eb | codex : match the soak proof to sequential canary requests | scaffolding | - | codex/ only (4 files) |
| 133 | 68514aae4 | codex : scope the soak check to fold engagement | scaffolding | - | codex/ only (1 files) |
| 134 | dc22c96d8 | codex : keep only the Mac parser dependencies | scaffolding | - | codex/ only (9 files) |
| 135 | 3baca78d8 | codex : record stack9 Mac gates and queue handoff | scaffolding | - | codex/ only (20 files) |
| 136 | e645e7afc | codex : build stack9 for sm_120a like frozen stack8 | scaffolding | - | codex/ only (1 files) |
| 137 | cd776e104 | codex : give the stack9 window the GPU set gpuq requires | scaffolding | - | codex/ only (1 files) |
| 138 | a1fff47fc | cuda : add grouped one-row q8_0 flash attention | carry | a29d0f11d |  |
| 139 | 758c93a1c | llama : retain attention rotations on the cache backend | carry | 8d4c05a8a |  |
| 140 | 3ebadf53a | cuda : fuse F32 add softplus and broadcast multiply | carry | 339ebe49c |  |
| 141 | bd251ce1b | local: add stack10 component controls | local | c87b8f692 | local layer on top |
| 142 | 0a0e2e0a0 | codex : prepare stack10 CPU gates and GPU window | scaffolding | - | codex/ only (21 files) |
| 143 | 7e4e8d661 | codex : record stack10 CPU evidence and validation limits | scaffolding | - | codex/ only (17 files) |
| 144 | 421d429e3 | codex : seal the final stack10 payload evidence | scaffolding | - | codex/ only (6 files) |
| 145 | f8a54c243 | local: add the stack10 alpha alias negative control | local | 583836fba | local layer on top |
| 146 | 0cfc0723e | codex : record the final stack10 alias gates | scaffolding | - | codex/ only (13 files) |
| 147 | 2efa6f8dc | cuda : keep the multi-query q8_0 MMA FA path free of single-query decode code | carry | dc9ebd52a |  |
| 148 | ebb2dfe01 | codex : prepare the stack10-cand2 verify-FA fix window and np2 item | scaffolding | - | codex/ only (12 files) |
| 149 | def4a4e20 | codex : fa-identity: q8_0 nb=1 belongs to the decode group, not other | scaffolding | - | codex/ only (1 files) |
| 150 | b28f11784 | cuda : bind MMVQ launch dimensions to kernel parameters | carry | 4e137cf14 |  |
| 151 | 311452e04 | cuda : use one-row MMVQ blocks for partial tails | carry | f0157366d |  |
| 152 | 96217fbd5 | cuda : reject unavailable MMVQ parameter tables | carry | 3e73c158e |  |
| 153 | cc0bfd1b4 | cuda : select Blackwell MMVQ launch parameters by quant type | carry | 3a7bea264,17b21e66a | split: general part (type parameter) upstream form, its Blackwell table data in the local commit |
| 154 | 20ac6ec57 | cuda : run paired RMS norm and scale chains in one launch | carry (needs-measure) | e986cc7a9 | K7: only a derived gain (-0.03…-0.06 ms/round, D, patches/upstream/llama.cpp/78-cuda-rms-norm-scale-pair/EVIDENCE.md:19-20, 'No measured ms/round'); drop arm s11-drop-np2-real-j5q |
| 155 | 386282191 | ggml: GGML_OP_SSM_CONV_RS, indexed conv-state read with snapshot writes (CPU, CUDA, Meta) | carry | aae2163c5 |  |
| 156 | 6ac5f5897 | llama: read the recurrent conv state by id through ggml_ssm_conv_rs, gathered fallback | carry | a7c618f7c |  |
| 157 | 30afc9758 | cuda : reuse shared matmul activations of up to 16 rows within a graph | carry | ce9113ff7 |  |
| 158 | 1376b053b | cuda : fuse residual add with RMS normalization of up to 16 rows | carry | c9f1580b0 |  |
| 159 | fac1254d2 | cuda : scan the flash-attention mask once per graph evaluation | dead | - | K2: 0 `fattn_kv_max:` lines across 6 served-shape canaries vs 1056-2976 K7 lines in the same logs (M, notes 2026-10-09.md:466-468, results/2026-10-09-s10-*/canary-a1); gates fattn-kv-max-cache.h:30-33 (Q->ne[1] >= 1024 || Q->ne[3] > 1) and :99 (uses < 2) never pass at served ub |
| 160 | 4bf1c2ea3 | cuda : batch top-k over rows in one launch set with a deterministic order | carry (needs-measure) | 6dc6cbef2 | K10: gain measured only vs stock CUB; master #28713 now routes 3-6 rows to radix (M top-k.cu:318-321), so vs master unmeasured [U§7]; drop arm s11-drop-np2-real-j5q |
| 161 | 015a0cbc7 | cuda : shape-gated small-I MMQ tiles for 16 and 24 columns on Blackwell | carry | 7762690c4 |  |
| 162 | 0ab5b19e7 | tests : cover MMQ small-column tile selection | carry | e33d325e4 |  |
| 163 | 3f0814173 | cuda : MMQ stream-k partitions in default-tile units for narrow tiles | carry | 883251c01 |  |
| 164 | c5baada6d | cuda : take SMALL_I MMQ tiles on stream-k launches too | carry | 8dcade3f8 |  |
| 165 | 9ccfb56e0 | tests : cover MMQ small-I tiles on stream-k shapes | carry | 18077ee6e |  |
| 166 | 50af425d0 | llama : skip getter synchronize() when no device work is pending | carry | 7a429f2a6 |  |
| 167 | d5b2af7c9 | meta : stage mirrored host inputs once and upload slowest link first | carry | 36fd6fdd5 |  |
| 168 | a4a3a8c63 | local: K7 paired RMS norm and scale switch, mutants, engagement log | local | c74d6ef03 | local layer on top |
| 169 | 6565c43db | local: K4 SSM_CONV_RS off switch, mutants, engagement log | local | 4a6c50414 | local layer on top |
| 170 | b4efb51a0 | local: K6a shared-activation reuse switch, engagement log, stale-cache mutant | local | 9a97a4fa1 | local layer on top |
| 171 | 31e27ee33 | local: K6b residual add + RMS norm fusion switch, engagement log, numerical mutant | local | 0e3561b1e | local layer on top |
| 172 | d045e5d49 | local: K2 shared KV-max scan off switch and negative controls | dead | - | K2's local switch/mutants; dead with fac1254d2 |
| 173 | 9b2ea143f | local: K10 top-k batching off switch, engagement log, negative-control mutants | local | 783333254 | local layer on top |
| 174 | 7491039b9 | local: K8/L1 MMQ tile A/B switch, auto-drop and stream-k partition mutants, mmq-hash tool | local | 42ff3bdaa | local layer on top |
| 175 | a434dc77b | local: K11a getter sync elision switch, debug check, mutants | local | 097aff7e8 | local layer on top |
| 176 | a8eed261a | local: K11b input staging A/B controls and negative-control mutants | local | 85451a51c | local layer on top |
| 177 | 85bbd7f93 | codex : component tools copied from the p55, K2/K4/K6/K7/K8(+L1)/K9/K10/K11a/K11b trees | scaffolding | - | codex/ only (79 files) |
| 178 | e92e8bced | codex : stack10 drop map and checked drop recipe | scaffolding | - | codex/ only (6 files) |
| 179 | ea38de72d | codex: combined stack10 Mac CPU payload and per-component stage libraries | scaffolding | - | codex/ only (15 files) |
| 180 | 59b923800 | codex: stack10 Mac payload reads the component filter from the OUT_NAME | scaffolding | - | codex/ only (1 files) |
| 181 | 4c8ba0632 | codex: stack10 Mac payload phase and component filter words | scaffolding | - | codex/ only (1 files) |
| 182 | 6fc334fa2 | codex: p55 item templates where gpu_stack9.py globs them | scaffolding | - | codex/ only (4 files) |
| 183 | b992a3cc7 | codex: p55 helpers back at the codex root where gpu_stack9.py reads them | scaffolding | - | codex/ only (6 files) |
| 184 | d43cb54d0 | codex: stack10 Mac payload without gdn4; K6 split into k6a and k6b; COMPONENTS filter | scaffolding | - | codex/ only (6 files) |
| 185 | 671917a74 | codex: stack10 combined CUDA window, gate extraction, stock-MMQ reference, A/B item templates and renderer (not run) | scaffolding | - | codex/ only (22 files) |
| 186 | a3030f921 | codex: stack10 window accepts topic/stack10-* branches | scaffolding | - | codex/ only (1 files) |
| 187 | 6757a0f18 | codex: stack10 window keeps the test-only gdn4 timing commit K8 depends on | scaffolding | - | codex/ only (1 files) |
| 188 | 023961d3e | codex: stack10 item fixtures compare normalized wave-1 fields | scaffolding | - | codex/ only (1 files) |
| 189 | 00a7a590b | codex: stack10 Mac payload runs k4 op stages; full-run evidence at ade74ab90 | scaffolding | - | codex/ only (55 files) |
| 190 | b65d96b0c | codex: stack10 window rejects both gdn4 commits, reads mac-evidence, toml without gated_delta_net.cu | scaffolding | - | codex/ only (4 files) |
| 191 | 8e75755d8 | codex: stack10 stock tree reverts must not run git apply inside the worktrees repo | scaffolding | - | codex/ only (1 files) |
| 192 | 5b54ce48a | codex: stack10 full Mac run evidence at 0128ceadf | scaffolding | - | codex/ only (17 files) |
| 193 | 03a14cedd | codex: stack10 integration report | scaffolding | - | codex/ only (1 files) |
| 194 | a056007d9 | codex: k4 ops gate expects unsupported=0 on CUDA (from s10-k4) | scaffolding | - | codex/ only (1 files) |
| 195 | d9be365e9 | codex: k10_perf parses failing TOP_K lines (from s10-k10) | scaffolding | - | codex/ only (1 files) |
| 196 | 14504eb1a | codex: q8_0 FAIL check matches rows prefixed by the ERR message (from l1-k8) | scaffolding | - | codex/ only (2 files) |
| 197 | ecb1f2524 | llama : keep the real sync in front of state I/O and KV stream copies | carry | 06b866114 |  |
| 198 | 330ab2cb8 | local: K11a mutant 4 (state I/O syncs elidable again) and the caller in the debug-check log | local | fe8fedbd4 | local layer on top |
| 199 | d78ceb718 | codex: stack10 drop map lists the K11a state I/O sync fix (from s10-k11a) | scaffolding | - | codex/ only (2 files) |
| 200 | 353cd186c | codex: k11b glue mutant requires SIGABRT (rc 134) or the fake/staged content check | scaffolding | - | codex/ only (1 files) |
| 201 | d1022e454 | tests : count out-of-range TOP_K output indices as failures instead of reading src through them | carry | 58f4eacc5 | pkg 82 master port (port-a 5f96b0b83), patch-id equal [U§15] |
| 202 | ec0aa3847 | codex: stack10 drop map lists the K10 test_top_k index bounds check | scaffolding | - | codex/ only (2 files) |
| 203 | 013d103bf | codex: k10 controls count only completed runs (from s10-k10) | scaffolding | - | codex/ only (1 files) |
| 204 | 52408e491 | codex: k6 memcheck runs without CUDA graphs (from s10-k6) | scaffolding | - | codex/ only (1 files) |
| 205 | 0e636d9e8 | codex: stack10 k11a Mac stage no longer requires mutant 1 caught (its state-read site now always syncs; from s10-k11a) | scaffolding | - | codex/ only (1 files) |
| 206 | 5f1825dfa | local: K11a debug check queries the backends instead of timing the sync | local | 5ab77cdc9 | local layer on top |
| 207 | 45ff5835e | codex: stack10 k11a gates use the query-based debug check; sync_count adds the mutant-3 control boot (from s10-k11a) | scaffolding | - | codex/ only (7 files) |
| 208 | a51dc1e88 | codex: stack10 drop map lists the K11a query-based debug check (from s10-k11a) | scaffolding | - | codex/ only (2 files) |
| 209 | 572564441 | codex: stack10 full Mac run evidence at 2df562da7 (K11a query-based debug check) | scaffolding | - | codex/ only (16 files) |
| 210 | b47b9849d | codex: stack10 window pins ggml-backend.{h,cpp} (K11a query hook) | scaffolding | - | codex/ only (1 files) |
| 211 | f779174bc | codex: stack10 k9 gate runs PDL-off and stock controls before the verdict; drop map lists the K9 mmq-chain test fix (from s10-k9) | scaffolding | - | codex/ only (3 files) |
| 212 | dbd1a110f | codex: stack10 full Mac run evidence at 6b61cd24a (K9 mmq-chain test fix) | scaffolding | - | codex/ only (13 files) |
| 213 | 40b805adb | codex: full-suite takes the K7/K2 self-test markers from a no-case run | scaffolding | - | codex/ only (2 files) |
| 214 | 387b919a4 | codex: stack10 tail window reuses attempt a1 stages 40-65 after an integrity proof (stages, trees, stamps, hashes, mtimes, must-fail controls) | scaffolding | - | codex/ only (7 files) |
| 215 | 7cddbfaf9 | codex: tail precheck refuses with rc 6 when the reused attempt is unreadable | scaffolding | - | codex/ only (1 files) |
| 216 | cb198fa21 | codex: stack10b purpose | scaffolding | - | codex/ only (1 files) |
| 217 | c5087a10f | ggml : recreate every meta subgraph after the arena is reallocated | carry | ceee21da8 | pkg 86 master port 1cdaba045 (patch-id f9d7c194035b = 86afcf77e = package patch); its test is the in-process graph_compute form [U§14] |
| 218 | 02db31b9b | codex: k11a drive_np2 reports the slot stream errors before the capture error | scaffolding | - | codex/ only (1 files) |
| 219 | a15eb17d5 | codex: stack10b Mac payload drops k9, adds the meta subgraph fix stages (ASAN pre-fix mutant, np2 Meta arms) | scaffolding | - | codex/ only (2 files) |
| 220 | c3d17d5ef | local: GGML_META_SUBGRAPH_MUTANT restores the pre-fix meta subgraph bound (negative control) | local | 24fcf4143 | local layer on top |
| 221 | b0dd3ddba | codex: mfix Mac stages use the GGML_META_SUBGRAPH_MUTANT control and check the crash site | scaffolding | - | codex/ only (2 files) |
| 222 | e9ce68e27 | codex: bw-stack10b window: no K9 gates or K9 stock reverts, np2 joint-replay smoke after the build (pre-fix mutant must die), meta subgraph test-alloc gate | scaffolding | - | codex/ only (9 files) |
| 223 | a907514da | codex: stack10b drop map: K9 dropped, mfix component, new hashes | scaffolding | - | codex/ only (2 files) |
| 224 | fcbc3c4b9 | codex: stack10b engagement table without K9 | scaffolding | - | codex/ only (1 files) |
| 225 | 2ea21e68e | codex: bw-stack10b.toml (rendered by mk_window_toml.py); fixtures and item templates name the stack10b window | scaffolding | - | codex/ only (8 files) |
| 226 | 47cff1bbd | codex: mfix ASAN control accepts any fault in graph_compute_impl (the stale subgraph's first faulting access varies) | scaffolding | - | codex/ only (1 files) |
| 227 | 51b7c9039 | codex: stack10b full Mac run evidence at 47cff1bbd | scaffolding | - | codex/ only (21 files) |
| 228 | 6c28e3e0e | codex: stack10b report | scaffolding | - | codex/ only (1 files) |
| 229 | dd5a11c87 | codex: stack10b window: TMPDIR on /mnt/scratch (nvcc temp files filled /) | scaffolding | - | codex/ only (1 files) |
| 230 | 2d3511bf7 | tests : run the meta subgraph capacity case in-process | carry | ceee21da8 | pkg 86 master port 1cdaba045 (patch-id f9d7c194035b = 86afcf77e = package patch); its test is the in-process graph_compute form [U§14] |
| 231 | 997c8e417 | codex: stack10b gate 54: 300 s timeout per test-alloc run | scaffolding | - | codex/ only (1 files) |
| 232 | d415412bd | codex: stack10b est 7200 s (stack10 a1 stages 7649 s minus K9 1264 s, ccache build, plus freeze/render) | scaffolding | - | codex/ only (1 files) |
| 233 | a588f2a44 | codex: stack10b preflight accepts Mac evidence when only test-alloc.cpp changed (Linux CPU run recorded) | scaffolding | - | codex/ only (5 files) |
