# Spike: AllReduce as CUDA graph nodes on stack7 (`topic/graph-capture-s7`)

2026-10-08. Labels: M measured, D derived, G guess.

## Verdict first

Expected np1 and np2 gain on stack7 is about zero per round (D), and the per-request capture cost remains
(+30…120 ms per 512-token request, M on stack6). The revisit trigger ("host gaps shrink, so launch overhead becomes the
limiter") does not fit the mechanism. Inside a forward the host is far ahead of the GPUs. The stack7 host trims act
**between** forwards. Recommendation: run the np1 item only if a falsification is wanted (~25 min GPU). Otherwise keep the
lever parked with the new trigger below. The arena fix stays a correctness export either way.

## Findings

1. **The host is ahead inside every verify forward (M, stack6 trace).** `results/2026-10-07-aa-load-j5q/full-a4/0-a/round-trace.jsonl`,
   84 steady np1 rounds, verify phase (`V`): wall 32.65 ms, 128 AR calls, host `ar_wait` (device 0) **28.57 ms**,
   `cuda_launch` 0.57 ms over 127.5 launches per device. `ar_wait` is the AR pool back-pressure
   (`ggml/src/ggml-cuda/allreduce.cu:546-560` `ggml_cuda_ar_acquire_slot`: `cudaEventSynchronize` on the slot the pool
   laps). So the host waits for the GPU, not the reverse, and the 129-graph launch sequence is hidden. This explains the
   stack6 untraced null (−0.10/−0.03/+0.01 ms/round, ledger llama-cpp row 69). It also matches the null of AR pool depth 2→4.
2. **Stack7 does not change that (D).** Its trims (getter, indexed `seq_rm`, bulk verify rows, chain M1, fold C) remove
   host work between forwards: sampling, inputs, KV bookkeeping. That work depends on the previous forward's readback, so
   graph shape cannot overlap it. The verify forward's AR structure and ring depth are unchanged (`allreduce.cu` is untouched
   in stack6→stack7).
3. **Draft steps are the only place where the host is not ahead.** Stack6 `D*` phases: wall 206 µs, 2 ARs, 4.7 launches,
   `ar_wait` 2 µs (M). With chain M1 the five draft steps form one decode of about 11 subgraphs, so node mode saves about
   10 boundaries × a few µs. Estimate −0.0…−0.1 ms/round (G).
4. **The per-request loss is whole-forward capture on the critical path (M/D).** Traced stack6 run
   `wt/graph-capture-fix/codex/gpu-results/depth/full-a1`, per request: the up arm captures 137 subgraphs and submits about
   400 eagerly while the GPU runs (hidden). The node arm captures 9–15 whole forwards: `cuda_capture` 14–28 ms plus
   `cuda_instantiate` 15–24 ms of host time, with the GPU idle (no eager warm-up for AR-node graphs,
   `ggml-cuda.cu` `graph->has_ar_node` branch). That is 30–52 ms/request (D), against +31/+34/+118 ms measured untraced.
   At np2, mixed prefill+decode batches under about 100 rows also qualify for node mode
   (`ggml_backend_meta_ar_node_qualifies_dry`: wire < 1 MiB). That means more one-off shapes, more captures and a larger
   fixed cost (G).
5. **Net per 512-token np1 request:** about 110 rounds × (0…0.1 ms) = 0…11 ms gain, against 30…120 ms capture cost.
   Expected net is **−20…−120 ms/request** (D/G).

## Rebase (what is implemented)

Branch `topic/graph-capture-s7` on stack7 48b82b0d1 (+ the stack7 codex commits ee9cf84c8, ddedffacb). All cherry-picks
applied cleanly. Semantic review: the boundary loop keeps stack7's host-backed Meta view condition (14b23db7c). The CUDA
graph path merges with the stack7 GDN trace code. The round-trace API is unchanged.

| commit | from | what |
|---|---|---|
| 5c6142739 | 5537a7f14 | cuda: AllReduce graph-node path in the internal pipeline |
| bf96cbebf | df61743ce | ggml-meta: insert AR nodes when the backend supports them (`GGML_META_AR_NODE`, default off) |
| 7db3ba4d6 | 346505678 | ggml-meta: create every subgraph of the arena when it is rebuilt (the arena fix) |
| 057a99470, ba730a2ff, 0646f469c, 9e9ae0055 | e54054971, 84acdcc2e, c3ee05e4a, bec62a178 | arena test + assert |
| codex commits | — | `codex/graph-capture/`: CPU gates, fixture (nullptr-terminated device list, ≤ 4 threads, chain env), window payload, item templates |

Code diff vs stack7: `codex/graph-capture/code-vs-stack7.diff` (10 files, +894/−58). Env unset or 0 gives the stack7 plan,
arena sizes and graphs. The same binary serves both arms.

## Mac CPU evidence (M)

Payload `codex/graph-capture/cpu-gates.sh` via `bin/bw-mac`. Base = HEAD with the diff reversed. Every touched file hashes
to its stack7 blob (`stack7-blobs.txt`). Results: `work/build-window-1007/graph-capture-s7-mac-aN/RESULTS.txt`.

@MAC_RESULTS@

## Window and items (not submitted)

- Window manifest `~/repos/qwen38-eval-worktrees/work/build-window-1007/bw-graph-capture-s7.toml`: priority 16, filler,
  host_sensitive false. Payload `codex/graph-capture/bw-graph-capture-s7.sh`. Its preflight requires a passing Mac run of
  the same non-codex tree. Stages: CUDA build (stack7 recipe flags, CCCL 3.4.3), then stamps (DeviceTopK 25, ggml stamp,
  `ggml_backend_comm_allreduce_node` and `meta_ar_node: uid=` strings), then `wt freeze graph-capture-s7` + RECIPE.txt,
  then render items + `gpuq submit --check`. Outputs `work/build-window-1007/graph-capture-s7-aN/items/*-ccx01.gpuq.toml`.
- `graph-capture-s7-np1-j5q` [26]: frozen/graph-capture-s7 in both arms, serving env (chain+fold, staging, kv-pad 4096),
  `GGML_META_AR_NODE` 0 vs 1. 4+4 boots ABBA, 8k/32k/90k, real prompts, untraced, cross-arm token identity.
  Canary on both arms: the off arm must log no `meta_ar_node` (negative control). The node arm must show the k=5 verify graph
  fused (`meta_ar_node: … split=M->K rows=6`, K≠M), at least one site, and never `unavailable`. About 25 min.
- `graph-capture-s7-np2-real-j5q` [26]: same binary, duo env (chain/fold 0, `-ub 128`), two streams, 3+3 boots, identity
  off (np2 CPU byte identity gates exactness). Canary: rows 6 or 12 fused. About 40 min.
- Both are judged on `steady_ms_per_round_D` **and** `fixed_ms_D` (per-request fixed cost) plus `predicted_ms_M` from
  `python3 -m harness report RUN`. The stack6 regression was in fixed/request, not per round.
- Expected outcome: np1 steady Δ 0…−0.1 ms/round (G), fixed/request +30…+120 ms (D from the stack6 M). np2 steady Δ about
  0 ± 0.2 (G), fixed/request at least as large as np1 (G). A per-round gain above 0.3 ms would contradict finding 1 and need
  a trace.

## Upstream form (owner rule: default-on where correct, automatic fallback elsewhere)

This branch is still the spike, not the final form. `GGML_META_AR_NODE` gates the whole feature in `bf96cbebf`. The target
layering is:

1. **upstream commit(s), no env:** the meta backend asks the comm backend for a node descriptor at every qualifying boundary
   (as now). It uses node mode whenever the backend answers and the per-device graph will be captured. Fallback is
   automatic: no backend support (CPU, NCCL, more than 2 GPUs, graphs disabled), a non-qualifying AR (gather, ≥ 1 MiB,
   misaligned) or a non-capturable graph all keep the split plan.
2. **local commit on top:** `GGML_META_AR_NODE=0/1` as the A/B switch, plus the CPU dry-run log and
   `GGML_META_AR_NODE_FORCE_TEST` (test hooks).

What must be true before (1) can be default-on upstream:

- **No per-request regression.** Today the first sighting of every graph shape is captured with the GPU idle: +30…120 ms per
  512-token request (M stack6). The fix must be general: a warm-up analogue in the meta backend that runs the split plan
  on a graph's first use and the node plan only once the same graph (uid) repeats. This mirrors the CUDA backend's own
  2-call warm-up. One-off shapes (prefill tails, first verify of a request, np2 mixed batches) then never pay a capture.
  Not implemented.
- **A fallback instead of an abort for non-capturable graphs.** Today `ggml-cuda.cu` aborts when a graph with AR nodes
  cannot be captured (`MUL_MAT_ID` needing a sync, `GGML_CUDA_DISABLE_GRAPHS` set after planning, unsupported arch).
  Upstream needs the backend to refuse the descriptor for such graphs, or a capturability query at plan time, so the meta
  backend keeps the split plan. Not implemented. (MoE models hit this.)
- **A spin-wait contract upstream will accept.** The node kernel spins on the peer's flag in device memory, bounded by
  `GGML_CUDA_AR_NODE_SPIN_MS` and ending in `__trap()`. Cross-device spin kernels inside graphs need both graphs resident at
  once. Today that holds with two devices launched back to back by one host thread. A reviewer will ask about MPS/MIG,
  preemption and graph-launch ordering. It is also not memcheck-clean by construction.
- **A measured gain at np1 and np2** with identity held. On stack7 the expected per-round gain is about 0 (findings 1–3).
  Without a gain, the feature adds a second AR path for nothing. The upstream case would then rest on setups where the
  host is the limiter inside a forward (slow host, many small forwards). That is not ours, so we cannot show it.

What blocks it now: the first two are missing code (meta warm-up, capturability fallback). The fourth is the deciding one:
on our hardware the lever has no measured gain. Recommendation: do not build the upstream form unless the np1 item shows
≥ 0.2 ms/round steady gain. In that case, implement the meta warm-up first; it should also remove the fixed/request loss.

## Risks

- A capture failure in an AR-node graph aborts the server (`ggml-cuda.cu`: "a graph with AllReduce nodes cannot run
  without CUDA graph capture"). Only `MUL_MAT_ID` blocks capture (dense model: none). Chain M1 sampling ops were already
  captured in split mode. The canary decides.
- The node kernel is not memcheck-clean by construction (spin across devices); owner approved an unsanitized canary on
  stack6.
- np2 shape churn (finding 4) can make np2 clearly slower per request.

## Next

- If the parent wants the falsification: submit the window, then only the np1 item. Skip np2 unless np1 shows a per-round
  gain of 0.2 ms or more.
- New revisit trigger for the lever map: host `ar_wait` per verify forward drops near zero, so the host stops being ahead
  inside a forward (for example a much faster forward or a costlier per-subgraph host path). Also, or instead: a
  capture-cost fix (split plan on the first sighting of a shape, node plan on reuse) **and** a measured per-round gain.
  Neither holds on stack7.
- The arena fix: export as its own package (the base Mac run dies on a signal at stack7, so the bug is latent there, M).
