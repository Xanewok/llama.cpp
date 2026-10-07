# SPIKE np2: the two-agent gap after stack7

Branch `topic/np2-kernels` on stack7 48b82b0d1. Labels: M measured, D derived, G guess.

## 1. np2 budget at 8k after stack7 (D)

Sources:
- stack6 IQ4 np2 attribution `analysis/attrib-stack6.md` (M): round 43.71 ms.
- HQ np2 pack `~/repos/attrib-pack/runs/attrib-hq-np2/full-a2/pack/budget.md` (M): round 35.59 ms.
- Ledger `docs/ledger/both-hardware.md:52`: llama − HQ = STEP +4.2, FWD +3.2, KV +1.4, ELEM +0.8, COMM −1.4.
- stack7 np2 gain −1.13 ms/round (D): host trims v2 plus GDN out-proj 2D.

| Rank | Part | llama − HQ after stack7 (D) | What it holds | Lever |
|---|---|---|---|---|
| 1 | STEP | ~3.6 | per-slot sampling and draft steps; AR pool stall 0.51 (M, np1); input upload; ignore_eos host-scan artifact ~0.7 (D) | AR pool depth (drafts exist); M2 multi-slot draft chain: at np2 the chain is refused, so 1 + 1 + k decodes per round vs 3 at np1 |
| 2 | FWD | ~2.7 | MMQ at 12 columns (`mmq23` 18.37 vs `mmvq23x6` 17.10 M) + stream-k fixup 0.43 (M) | L1 shape-gated MMQ small-J, −0.27…−0.45 J5q (D) |
| 3 | KV_GDN | ~0.9 | `k_get_rows_float_vec` 0.55 (M, np2); `gated_delta_net` 1.15 (M) vs HQ 1.04 | **np2 getter (this spike)**, −0.4…−0.55 (D/G) |
| 4 | ELEM | ~0.8 | partly the MMQ q8_1 quantize, 0.40 (M) | follows L1 |
| 5 | KV_ATTN | ~0.5 | draft FA; `mask_to_KV_max` 0.16 (M) | — |
| — | COMM | −1.4 | we lead | — |

### Concrete levers, ranked by np2 ms/round

| Lever | np2 gain | Cost | Status |
|---|---|---|---|
| GDN getter at n_seqs > 1 + joint fused kernel | −0.4…−0.55 (D/G) | done: 3 core commits | built here, GPU items ready |
| L1 MMQ small-J (shape-gated) | −0.27…−0.45 J5q, −0.5 IQ4 (D) | invasive: a 2nd kernel variant per (type, J) through `mul_mat_q` and the template-instance files | design only |
| Fold C for n_seq > 1 | ~−0.2 (D, scaled from np1 −0.23/−0.19/−0.22 M) | ~120–180 core + ~150 test lines (G), 7 files | design only (ranks 3rd) |

L1 derivation, from `wt/mmq-smallj/codex/gpu-results/perf/full-a2/summary.json` (M, n=12):
- Per m=8704 shape, hot/cold: q5_K −2.9/−1.5 µs, q6_K −8.4/−7.0 µs, iq4_xs −3.1/−3.5 µs. lm_head q6_K 124160: −84/−82 µs.
- Every shape with m ≤ 6144 loses; down-proj 5120×8704 loses +27…29 µs cold. So the variant has to be shape-gated.
- J5q estimate: 128 gate/up × (−1.5…−2.9) − 82 ≈ −0.27…−0.45 ms/round (D).

## 2. Findings (file:line)

- **The op and kernel were already per-sequence. Two hazards confined the stack7 getter to n_seqs == 1:**
  1. **Fusion race.** `ggml_cuda_try_gdn_cache_fusion` (`ggml/src/ggml-cuda/ggml-cuda.cu:3034`) lets the GDN kernel write snapshots straight into `cache_s_l*`. With indexed reads and n_seqs > 1:
     - one sequence's block can overwrite a row that another sequence still has to read;
     - this happens with swapped cells (`find_slot`, `src/llama-memory-recurrent.cpp:640-660`) and with the shared zero row `rs_z`.
     - Stack7 therefore refused the fusion. The ungated getter cost +2.3 ms at np2, because unfused snapshot copies run to ~0.9 GB/round.
  2. **Extra states.** When n_rs > n_seqs, `s_copy_tail` moves states of cells that are inside the range but not in the ubatch. That copy is ordered against the in-place reads and the fused writes only by node order.
- **At np2 (`-np 2`) extra states never occur.** The cache has 2 cells, so a 2-sequence ubatch has n_rs = 2. They do occur from 3 sequences up:
  - `find_slot` sets `n = max - min + 1` over the pre-swap tails (`llama-memory-recurrent.cpp:720`);
  - without `kv_unified`, `split_equal(sequential)` joins only consecutive sequence ids (`src/llama-batch.cpp:580`, `llama-memory-hybrid.cpp:89`).
- **Fold C at np2.** It is refused at `common/speculative.cpp:1566-1572` (`n_seq != 1`), and also by:
  - `mtp_fold_init` `get_n_stream()==1` (`src/llama-context.cpp:2351`);
  - the decode gate `n_seqs_unq==1` (`:2044-2049`);
  - a single `h_prev` (`speculative.cpp:1790`, `qwen35.cpp:278-289`).
  
  Only the single `h_prev` is a correctness blocker: seq 1's first row would take seq 0's nextn row. The rest are implementation limits; see §4.

## 3. Design and commits (getter)

The fix is upstreamable. It needs no env var and no new gate: the new conditions are automatic exactness conditions, with exact fallback.

1. `011df3795` **llama : read GDN states by id for ubatches without extra states**
   - The getter now runs whenever `n_rs == n_seqs` (`src/models/delta-net-base.cpp:552`), any n_seqs. Previously it required `n_seqs == 1`.
   - It then copies nothing: a 0-length view `s_copy_none` (`src/llama-graph.h:278`, `src/llama-graph.cpp:3631,3660`).
   - With extra states the ubatch keeps the gather (`s_copy_tail`, unchanged upstream size). That is the exact fallback.
   - The gate is reuse-stable, because `can_reuse` already compares `s_copy` / `s_copy_main` sizes.
2. `858599c30` **cuda : run the sequences of an indexed fused GDN launch in one block per head slice**
   - `gated_delta_net.cu:8` gains the template parameter `NS`. For an indexed fused launch with n_seqs > 1, one block runs every sequence of its (head, column slice). It loads every state before writing any snapshot, so it is race-free for any ids.
   - NS=2 is a specialisation; NS=4 is the general case.
   - The matcher refuses only above `GGML_CUDA_GDN_JOINT_MAX_SEQS` = 4 (`ggml-cuda.cu:3050`, `gated_delta_net.cuh:5`).
   - The arithmetic per sequence is unchanged.
3. Tests:
   - `e2395b5a9` + `3092dee99` + `ccefc0c11`: `test-recurrent-state-rollback --getter-gate` (`tests/test-recurrent-state-rollback.cpp:344`). Scenarios: two slots, swapped cells, joint start from the zero state, and an extra state (seq 2 between seqs 0 and 1). Each phase checks two things:
     - the graph form (by id vs gathered) through an eval-callback probe;
     - logits against single-sequence contexts (nmse ≤ 1e-4).
   - `7656eaf7f`: `test-backend-ops` `GATED_DELTA_NET_INDEXED_CACHE_FUSION` cases for B = 2..5 and a 1024-head race amplifier. Reversed ids make each sequence read the row the other writes.
   - `fecfd797f`: perf cases at the split 48-head verify shape (24 heads, d=128, 6 tokens, 1–2 sequences), gathered vs indexed.
4. Local only, on top: `3a0e07667` `GGML_CUDA_GDN_JOINT_MUTANT`, a test-only negative control that puts each sequence back in its own blocks.
5. Packaging order:
   - squash `3092dee99` and `ccefc0c11` into `e2395b5a9`;
   - put `3a0e07667` last, as the local commit;
   - leave out the `codex:` commits.

## 4. Fold C for several sequences: design (not implemented, ranks 3rd)

- **Today at np2:** each round runs 1 verify, then 1 batched MTP catch-up decode (`speculative.cpp:1712`), then k draft steps, because the chain is refused when more than one sequence drafts (`:1866-1873`).
- **What the fold removes:** the catch-up decode, i.e. its host batch/split, graph launch and embedding round-trip (np1 trace: C phases 338→34, M, notes 2026-10-07:551).
- **Gain estimate:** about the np1 saving, ~−0.2 ms/round (D), because the decode is already shared by both slots. It engages only when both slots verify equally many tokens.
- **Changes:**
  - `mtp_fold_arm` takes `[n_seq][n_embd]` rows indexed by seq_id;
  - `pending_h` is stored contiguously;
  - drop `n_seq != 1` and `n_seqs_unq == 1`;
  - `mtp_fold_prepare` checks contiguity per sequence (`p0_s = seq_pos_max(s) + 1`), and checks that the ubatch row order equals the batch order and matches the draft's own split (multi-stream: ascending ids, equal counts);
  - `build_mtp_fold` uses `get_rows(concat(h_prev_all, h_nextn), idx)`, where a sequence's first row takes `h_prev[s]` and every other row takes nextn row r−1.
- **Exact fallback already exists:** `fold_n_last != n_tokens` sends the batch to the separate catch-up (`speculative.cpp:1673`).
- **Default-on:** replace the `LLAMA_MTP_FOLD_C` env gate with the result of `mtp_fold_init`. Keep any A/B switch as a local commit.
- **Prerequisite for its identity test:** a same-binary fold-vs-no-fold comparison, because np2 cross-boot identity is confounded by the stream collapse.

## 5. Evidence

### Mac CPU (M)

Run with `bin/bw-mac`, payload `codex/np2/cpu-gate.sh`, on the 4B fixture, 4 threads, i8mm off.

- **a1, at `d37b9b5d3`:** rc 21, a test-design error and not a code fault.
  - All two-slot phases read by id with 0 state rows gathered, and logits matched in every scenario. This included swapped cells and the joint zero-state start, with 24 GDN nodes by id per joint ubatch.
  - The "extra state" scenario produced no extra state. Seqs 0 and 2 never share a ubatch without `kv_unified`, so its GATHERED expectation failed.
  - Fixed in `ccefc0c11`: seq 2 is placed between seqs 0 and 1.
- **a2, at `085f744d3`:** @MAC_A2@

### GPU (G until the items run)

| Item | Priority | Expected |
|---|---|---|
| `np2-getter-cuda-checks` | 30 | Indexed fusion passes for 2..4 sequences, with `seqs=2/4 indexed=1 gdn=1 fused=1` and `seqs=5 … fused=0` (refused). The mutant races on the 1024-head case. Joint-kernel perf within ±10 % of two per-sequence blocks (G). |
| `np2-getter-real-j5q` (+`-ccx01`) | 30 | np2 real prompts, ABBA vs frozen/stack7, chain/fold 0 in both arms, 8k/32k/90k: −0.4…−0.55 ms/round (D/G). The canary requires `tokens=6 seqs=2 indexed=N gdn=N fused=N` on both devices. Negative control by record: the stack7 canaries log `seqs=2 indexed=0` (`results/2026-10-07-stack7-np2-real-j5q/canary-a1..a3`, M). |
| `np2-getter-np1-depth-j5q` (+`-ccx01`) | 30 | np1 ABBA, chain/fold 1 in both arms: neutral, with cross-arm tokens and draft counters identical. |

## 6. Window and items (the parent submits them; nothing has been submitted)

- **Manifest:** `~/repos/qwen38-eval-worktrees/work/build-window-1007/bw-np2-getter.toml`. Priority 18, filler, `host_sensitive = false`. `gpuq submit --check`: valid. Source copy: `codex/np2/bw-np2-getter.toml`.
- **Payload:** `codex/np2/bw-np2-getter.sh`. Preflight passes locally.
  1. `cpu-gate.sh` on this host.
  2. CUDA build of `llama-server` and `test-backend-ops` (CUDA 13.2, NCCL OFF, CCCL v3.4.3, 120a), the 25 DeviceTopK check, `wt freeze np2-getter` and `RECIPE.txt`.
  3. `codex/np2/make_items.py`. It renders the three items into `~/repos/qwen38-eval/orchestration/gpuq/items/drafts/` (the two depth items also as `-ccx01.toml`) and runs `gpuq submit --check` on each.
     - Depth items use the stack7 untraced entry `wt/stack7/codex/gpu_ab.sh`, so they compare like for like with the stack7 items.
     - A dry render against frozen/stack7 was contract-valid for both depth items, their ccx01 copies and the CUDA item.
- **rc map:** 0 pass, 2 CPU gate failed, 75 CPU gate skipped for RAM, 8 setup, 9 CUDA, 10 items.

## 7. Risks

- **Compile is unverified.** CUDA has never been compiled; the window is the first compile. This includes the NS template, `float s_shard[NS][rows_per_lane]` register pressure at NS=4, and GCC 13 / nvcc.
- **Joint-kernel perf is unmeasured.** One block now does NS× the serial work per head slice, at the same grid width per sequence and 1/NS the blocks. At 24 heads × 32 slices = 768 blocks for NS=2 the GPU should stay full (G). The CUDA item measures it.
- **The canary regex assumes the per-graph debug line format** seen in the stack7 canaries (`tokens=6 seqs=2 … gdn=1`).
- **np2 identity on GPU is off in the A/B** (arrival-dependent batching). Exactness rests on the CPU identity (getter vs the stack7 gather, plain and Meta) and on test-backend-ops vs CPU.

## 8. Next

1. Run `bw-np2-getter`, then `np2-getter-cuda-checks`. Then the np2 A/B and the np1 A/B, ccx01 copies preferred.
2. If the np2 A/B confirms the gain: package for upstream (the getter + the kernel). The extra-state fallback must be stated in the PR.
3. After that, by size: M2 multi-slot chain (STEP), then L1 small-J (FWD), then fold C for several sequences.
