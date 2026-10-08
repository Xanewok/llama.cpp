# Spike chain-m2: MTP draft chain for several sequences (np2)

Branch `topic/chain-m2`, base `topic/chain-np2` 374a465b8 (stack7 48b82b0d1 + da787bb40 single-sequence gate + 374a465b8).
Labels: M measured, D derived, G guess. All file:line refer to the base 374a465b8 unless noted.

## 1. What M1 assumes about one sequence

| Area | Assumption | Where |
|---|---|---|
| Graph predicate | `n_tokens == n_chain && n_outputs == n_chain && token && embd && samplers.size() == 1`: one sequence, and the context may hold only one sampler. An idle second slot's sampler fails the predicate and the decode silently takes the plain path. | `src/models/qwen35.cpp:637-641` |
| Sampler | the graph takes `samplers.begin()`: one seq_id, one sampler, `k_top` of that sampler | `qwen35.cpp:785-788` |
| Mask | asserts `mask->ne[1] == n_tokens && ne[2] == 1 && ne[3] == 1`: one stream, batch order = step order. A per-stream cache (np2 serving, `kv_unified = false`) gives `[n_kv, k, 1, n_stream]`, and `split_equal` orders rows by sequence. | `qwen35.cpp:783` |
| Positions | `view_1d(inp_pos, 1, s)`: row s is step s | `qwen35.cpp:816` |
| K/V destination | `view_1d(self_k_idxs, 1, s)`: row s is step s; with several streams the idxs are stream-major (`llama-kv-cache.cpp:1540`) | `qwen35.cpp:823` |
| Tokens / h | step 0 = row 0 token + h row 0; step s>0 = `t_sampled[s-1]` and the previous step's h | `qwen35.cpp:795-806` |
| h_nextn out | `concat(h_all, h, 1)`: one column per step = batch order only for one sequence | `qwen35.cpp:828` |
| Context gate | `set_mtp_chain` checks arch/vocab/FA/`n_outputs_max_per_seq`, not the predicate: the M1 np2 bug class (chain requested, plain graph run, plain drafts read as chained) | `src/llama-context.cpp:1289-1300` |
| Common | `draft_chain` takes exactly one drafting seq (`n_seq == 1` since da787bb40), batch rows `id_last @ pos0+s` for that seq | `common/speculative.cpp:1867, 1896` |
| Transactional dist | chain dist commits each uniform by `llama_sampler_accept` | `speculative.cpp:1931`, `src/llama-sampler.cpp:1273, 1386-1398, 1448` |
| RNG streams | two per seq: host dist (`smpls[seq]`, step loop, built by `spec_retune`, seed ^ 0x85ebca6b) and chain dist (`mtp_chain_seqs[seq].smpl`, same seed). M1 has no hand-over: a seq that drafts on both paths within a request replays uniforms. At np1 the path never changes mid-request. At np2 it would, whenever the co-slot is greedy or the batch shape changes. | `speculative.cpp:38-72, 1817-1857, 1970` |
| Context params | `n_outputs_max_per_seq = n_max` only when `n_parallel == 1` (transactional dist off at np2) | `speculative.cpp:2869` |
| Server | dparams per generating slot (`pos0 = pos_next`, `id_last`); `result_q` only when `temp > 0` (`use_spec_rejection`); every slot drafts the global k and truncates after drafting (so per-slot k never differs inside the drafter) | `tools/server/server-context.cpp:495, 3236-3310` |
| MTP KV | each step writes its K/V cell during the chain decode; the server trims the draft region after drafting (`seq_rm` from `pos_max+1`); a failed decode rolls back with `seq_rm(pos_min)` | `server-context.cpp` after 3309, `llama-context.cpp` decode error path |

## 2. Gain estimate (decision)

- M: J5q round traces give the D-step hop (period minus sync).
  - 1 row: 150-155 µs.
  - 2 rows on the np2 fixture: 186 µs.
  - np2 real canary: 190 µs.
  - np1 off arm: 145/144/152 µs at 8k/32k/90k.
- M: nsys on stack6 IQ4 gives GPU0 idle per D boundary.
  - np1: 117-127 µs.
  - np2: 160-184 µs.
  - M1 realised 74-79 µs per boundary against nsys ~120 at np1.
- D: M2 at np2 removes about 30-40 µs more per boundary × 4 boundaries than M1 does at np1, on top of the M1 np1 same-binary gain (-0.07/-0.21/-0.21 ms/round).
  - Estimate: **-0.15/-0.29/-0.31 ms/round at 8k/32k/90k**, range -0.15…-0.40.
- This is borderline against the 0.3 ms threshold. I implemented it anyway, for three reasons:
  - the estimate straddles 0.3;
  - the owner rule requires a multi-sequence-capable final form;
  - the M2 refusal turns the M1 predicate mismatch from silent wrong drafts into a failed decode.
- M: a separate host lever is not addressed. The C→D1 gap is np1 329/19 µs vs np2 917/431 µs at 8k/90k (nsys). That points to host work between catch-up and the first draft at np2.
  - Scripts: `codex/m2/bgap.py` and `codex/m2/hop.py`.

## 3. Design (implemented)

### Graph (`src/models/qwen35.cpp`)

- `mtp_chain_layout_of` accepts a batch only when all of these hold:
  - the batch has n_seq whole sequences of k rows each;
  - every row is an output;
  - every row has one seq_id;
  - every sequence has a top-k-first backend sampler.
- Layout:
  - Unified cache (`split_simple`): batch order, which is step-major (row s·n_seq+j).
  - Per-stream cache with n_seq>1 (`split_equal`): seq-major (row j·k+s).
- Per step s, for every sequence:
  - one MTP block over n_seq rows: e_in, h_in, positions `[n_pp][n_seq]` and K/V destination idxs n_seq;
  - mask rows: n_seq rows of one stream, or row s of each stream;
  - one `build_sampling_row` per sequence into ubatch row `row(j,s)`.
- Step s>0 embeds `t_sampled[row(j,s-1)]` per sequence.
- Seq-major support:
  - an own I64 input (`llm_graph_input_mtp_chain_idxs`) reorders the cache's destination cells step-major;
  - the mask is permuted and made contiguous once per graph, because FA asserts a contiguous mask (`ggml.c:5523`; found on the Mac);
  - h_nextn is permuted back to ubatch order.
- With n_seq = 1 the graph is op-for-op the M1 graph.
- `llm_graph_result::n_mtp_chain_seqs`: `process_ubatch` fails the decode (-3, KV rolled back) when the chain is on and the graph did not take it.

### Common (`common/speculative.cpp`)

- The chain drafts every drafting slot. Idle slots are left out and keep their sampler (no `set_sampler` churn).
- Fallback to the step loop for the whole round happens when:
  - any drafting slot lacks `result_q` (greedy);
  - `n_d·k > n_ubatch`;
  - attach or decode fails. A failed decode disables the chain and falls back in the same round.
- RNG hand-over:
  - `rng_chain` per seq, reset in `begin()`;
  - switching to the chain copies host dist → chain dist (`llama_sampler_copy`, after `spec_retune`);
  - switching back copies chain → host.
  - Exact because the host dist and the transactional chain dist draw the same double-precision uniform per sampled token, from committed `rng`.
- Context params: `n_outputs_max = max(n_parallel, n_parallel·n_max)` and `n_outputs_max_per_seq = n_max` for any n_parallel (chain on only).
- Announce: `MTP chain: k=K seqs=N`, once per N. The GPU canary keys on it.

### Owner rule status

- Exact fallback is automatic, the chain is multi-sequence capable, and there is no env var in the upstream commits.
- The test-only hooks (`LLAMA_MTP_CHAIN_DEBUG_SWAP_SEQ`, `..._NO_RNG_SYNC`) live in codex commit 27295d864. That commit also carries M1's existing `REVERSE_U`/`SHIFT_EMBD` style.
- **Not done: default-on.** `--spec-mtp-chain` (env `LLAMA_MTP_CHAIN`) is still opt-in, because both A/B arms need the same flag.
  - Flipping the default plus adding `--no-spec-mtp-chain` is a small separate commit, to make once the GPU A/B is in.

### Commits

| Commit | Subject | Scope |
|---|---|---|
| 59a6f9814 | graph : run the MTP draft chain for several sequences | upstream |
| e641f9939 | graph : give the multi-stream MTP chain a contiguous mask per step | upstream; squash into 59a6f9814 |
| 6abc07a4d | common : draft every probabilistic slot through the MTP chain | upstream |
| eff06d2cb | common : log each MTP chain sequence count once | upstream; squash |
| 04db9819a | common : match the backend-initialised dist sampler in the RNG hand-over | upstream; squash; found on the Mac: backend dist reports `+dist` |
| 27295d864 | codex : M2 np2 chain identity matrix, negative-control hooks and measurement scripts | local |
| ae3e5a132 | codex : M2 build window, GPU item templates, np2 matrix quick mode | local |

## 4. Mac CPU evidence (M)

Commit e641f9939, built with clang. Apple M4 at 4 threads, `-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod` (no i8mm), Qwen3.5-4B fixture. Run via `bin/bw-mac . codex/m2/bw-m2-mac.sh chain-m2`; outputs in `~/repos/qwen38-eval-worktrees/work/build-window-1007/chain-m2-mac-a3/` (`np2/summary.json`, `np2-judge.log`, `np1/report.json`).

Units: one slot of one phase (2 phases × 2 slots) for tokens and draft/accepted/rounds counters, plus every drafted row of `LLAMA_MTP_CHAIN_DUMP` (ids + 10 candidates + q).

**np2 identity: chain off vs on, `judge-np2.py` rc=0**

| Cell | Tokens + counters | Dump ids | max dq | Chain decodes (rows) | Step decodes |
|---|---|---|---|---|---|
| plain k3 s1729 / s2718 | equal | equal (228/228) | 1.2e-7 | 3 and 6 | 0 |
| plain k5 s1729 / s2718 | equal | equal (300, 360) | 1.2e-7 | 5 and 10 | 0 |
| Meta(CPU,CPU) k3 s1729 / s2718 | equal | equal (234, 276) | 1.2e-7 | 3 and 6 | 0 |
| Meta(CPU,CPU) k5 s1729 / s2718 | equal | equal (315, 355) | 1.2e-7 | 5 and 10 | 0 |
| plain k3 unified KV (step-major path) | equal | equal | 1.2e-7 | 3 and 6 | 0 |
| plain k3 greedy slot 1 (slot 0 switches chain ↔ step loop mid-request) | equal | equal (222) | 1.2e-7 | 3 | 126 |
| Meta k3 greedy slot 1 | equal | equal (228) | 6e-8 | 3 | 126 |
| floor: off vs off repeat | equal | bit-equal | 0 | – | – |

- The 2k-row chain decodes prove that both slots drafted in one graph; the k-row ones are single-slot rounds.
- The seq-major path (per-stream KV, the np2 serving layout) covers every non-unified cell. The step-major path is the unified cell.
- dq ≤ 1.2e-7 is the same device-vs-host softmax floor as M1 (np1 report: 1.2e-7).

**Negative controls (all must differ from the off cell): all caught**

| Mutant | Units that differ |
|---|---|
| `SWAP_SEQ` (feed each sequence its neighbour's token), per-stream KV | all 4 |
| `SWAP_SEQ`, unified KV | all 4 |
| `REVERSE_U` | all 4 |
| `SHIFT_EMBD` | all 4 |
| `NO_RNG_SYNC` (greedy co-slot) | slot 0 in both phases, i.e. exactly the slot that switches paths |

**np1 regression (`chain_cpu_identity.py --n-predict 48` + `check-chain.py`)**

PASS (rc=0). There are 9 cases per boot: short/mid/long prompt × (t0 s1729, t1 s1729, t1 s2718). All boots run at np1.

| Cell | Tokens + counters | Dump ids | max dq | Dump rows |
|---|---|---|---|---|
| plain k3 | equal | equal | 1.2e-7 | 486 |
| plain k5 | equal | equal | 1.2e-7 | 665 |
| Meta k3 | equal | equal | 1.2e-7 | 453 |
| Meta k5 | equal | equal | 1.2e-7 | 660 |

- Every chain-on boot logs `MTP chain: k=K seqs=1`.
- Floor (off vs off): bit-equal.
- REVERSE_U and SHIFT_EMBD are caught.
- The n_seq = 1 path therefore still reproduces the step loop exactly, as M1 did.

The two bugs found on the Mac are fixed in the listed commits:
- FA requires a contiguous mask, so a strided per-stream view aborted (`ggml.c:5523`);
- the backend dist is named `+dist`, so the RNG hand-over did not find it.


## 5. Build window and GPU items (checked, NOT submitted)

### Window

- Manifest: `codex/m2/bw-chain-m2.toml`
  - id `bw-chain-m2`, filler, priority 18, `host_sensitive=false`, ccx01-pinned like `bw-np2-getter-ccx01`;
  - `gpuq submit --check`: valid.
- Payload: `codex/m2/bw-chain-m2.sh`.
  1. GCC CPU gate at the frozen SHA: `np2-matrix.sh` quick + `judge-np2.py` quick, 9 runs.
  2. CUDA build with the stack6 RECIPE flags (CCCL 3.4.3, NCCL OFF), DeviceTopK = 25, ggml stamp, M2 refusal string in libllama, `wt freeze chain-m2`.
  3. `make_items.py` renders and checks both items plus their ccx01 copies.
- Preflight refuses:
  - an existing `frozen/chain-m2`;
  - existing experiments;
  - a dirty tree.
- Wall (G) ~80 min, budget 3 h.

### Items rendered by the window

Both are 30 priority, ABBA, real J5q prompts and untraced, and the structure was dry-checked valid.

| Item | Arms | Canary / identity | Expected |
|---|---|---|---|
| `chain-m2-np2-real-j5q` (+ `-ccx01`), np2 serving profile, 8k/32k/90k | `up` = frozen/stack7 with `LLAMA_MTP_CHAIN=0`; `patch` = frozen/chain-m2 with `LLAMA_MTP_CHAIN=1`. Fold is 0 in both, and `GGML_SCHED_INPUT_STAGING=1 LLAMA_ARG_KV_PAD=4096` in both. | Canary (patch, 8k) requires `MTP chain: k=5 seqs=2`, two `n_kv_pad += 4096`, and no `llama_process[chain] returned` / refusal line. Identity is off (arrival-dependent np2). Exactness rests on the CPU matrix. A wrong chain graph would show as a per-arm acceptance drop. | D -0.15/-0.29/-0.31 ms/round |
| `chain-m2-np1-depth-j5q` (+ `-ccx01`), np1 profile, chain=1 fold=1 both arms | stack7 vs chain-m2 | Canary requires `MTP chain: k=5 seqs=1`. **Cross-arm identity** (tokens + draft counters) is the np1 negative control: the n_seq = 1 graph must equal M1. | G neutral |

To submit: the parent submits `codex/m2/bw-chain-m2.toml`. After it passes, it submits the two `-ccx01` drafts from `orchestration/gpuq/items/drafts/`.

## 6. Risks

- **CUDA paths not exercised on the Mac:**
  - `cont` of a permuted F16 mask;
  - `cont` of a strided I32 view (step-0 tokens);
  - FA with `ne[3] = n_stream`, one query row per stream, from a contiguous view;
  - `set_rows` through a view of the new I64 input.
  - The np2 canary proves engagement, but not equality with stack7. A compute-sanitizer pass on one np2 canary boot is cheap if the A/B looks odd.
- **Extra per-round work at np2:** one mask copy (n_kv × 2k F16, ~1.8 MB at 90k), plus per-step concat of n_seq row embeddings. G: a few µs, inside the estimate's range.
- **Gain is borderline (D).** The bigger np2 host gap (C→D1, 0.4-0.9 ms M) is a separate lever. Measuring it next could pay more than M2.
- **Mid-request path switch (chain ↔ step loop) depends on the RNG hand-over.** It is covered by the greedy co-slot cells and caught by the NO_RNG_SYNC mutant on CPU only.
- **Non-consecutive drafting slot ids at np≥3** (e.g. {0, 2}): `split_equal(sequential)` splits them into one ubatch each. Each is a valid one-sequence chain, still exact, with no batching gain. Not relevant at np2.

## 7. Next

1. Submit `bw-chain-m2`.
2. Then submit the np2 A/B and np1 depth items.
3. If np2 is at least 0.25 ms/round better with acceptance unchanged, flip the default-on (opt-out flag) commit and squash the fixups for the upstream package.
