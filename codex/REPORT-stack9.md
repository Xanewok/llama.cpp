# Stack9

## Conclusion

CPU gates passed. The committed window is ready for the parent to validate and queue; it performs only CUDA compilation, stamps, freeze, and item render/check. No GPU binary, Linux compile/model test, queue submission, or cancellation was run by this worker. GPU correctness, performance, and GCC 13 compilation remain unverified on this integration.

Labels: M measured, D derived, G guess. CPU walls below include loading, verbose logs, and any RAM wait; they are not performance A/B results.

## Commits and origins

Base: `topic/stack8` `5df0dbffc`; comparison binary: `frozen/stack8` source `41d2954a21106c9a65c8dd2a0697e7d33ae4f22f`.

| Commit | Subject | Origin |
|---|---|---|
| `cfce3d732` | `llama : read multi-sequence GDN states by id` | Source/tests from `011df3795`, `858599c30`, `e2395b5a9`, `7656eaf7f`, `3092dee99`, `fecfd797f`, `ccefc0c11`, `96774329b`, `5a855ecc3`; CUDA mutant excluded from this root commit. |
| `c19629b30` | `llama : fold MTP catch-up for several sequences` | `629a725dd` plus M-RoPE fix `e61e6bb31`, merged onto stack8/getter. Default-on, no new environment control. |
| `519069f3e` | `local: add stack9 fold and GDN negative controls` | `538468e53`, sequence-tagged draft dump from `c86c8213a`, CUDA race control from `3a0e07667`. `LLAMA_MTP_FOLD_C=0` disables folding only in this local commit. |
| `0dae1d784` | `tests : integrate stack9 CPU identity gates` | Stack8 window gates; fold driver/matrix including `fe34c623a`, `186e41a52`; adapted getter CPU gate. Preserves the supplied prompt/purpose edits. |
| `925c8dd82` | `codex : prepare stack9 build window and GPU items` | Stack8 renderer/window/runtime guards, current parent parser library, recorded getter artifact. |
| `fd58c11eb` | `codex : match the soak proof to sequential canary requests` | Soak canary sends sequential requests; requires `n_seq=1`. Joint proof stays in np2 depth canary. |
| `68514aae4` | `codex : scope the soak check to fold engagement` | Narrows the new assertion so it accepts the required np2 chain-fallback log. |
| `dc22c96d8` | `codex : keep only the Mac parser dependencies` | Retains only the imported parser dependencies; adds final-source Mac stage selection. |

Final evidence/report commits follow these components. No AI trailers. No changes to other branches.

## Design and conflict resolution

The getter requires `keep`, `offload_kqv`, and `get_n_rs() == n_seqs` (`src/models/delta-net-base.cpp:552`). `s_copy_none` removes every state gather in that case (`src/llama-graph.cpp:3651`, `:3680`). Extra states retain the original gathered path. Indexed fused CUDA launches load every participating sequence's state before snapshot writes, in one block per head/column slice (`ggml/src/ggml-cuda/gated_delta_net.cu:48`). Fusion falls back above the supported joint sequence limit (`ggml/src/ggml-cuda/ggml-cuda.cu:3050`). Per-sequence arithmetic and snapshot-slot mapping are unchanged.

Fold stores one stable contiguous hidden buffer, indexed by sequence id (`common/speculative.cpp:1554`). Each active sequence starts with its own previous hidden row; other tokens select the preceding target row (`src/llama-graph.cpp:536`, `src/models/qwen35.cpp:287`). Token count and active sequence count both constrain graph reuse. Target/draft row order, contiguous sequence blocks, positions, hidden-row bounds, and per-stream equal-count/consecutive-id rules gate folding (`src/llama-context.cpp:2047`, `:2382`). Unsupported layouts keep separate catch-up. Token batches broadcast scalar positions across M-RoPE axes (`src/llama-batch.cpp:829`), so the imported one-axis guard must be removed.

All production patches applied cleanly through three-way merges, including the two independent changes in `src/llama-graph.*`. Stack8's chain fallback and banning-bias fixes remain intact. The only textual conflict was `codex/bw-np2-driver.cpp`: retained `BW_CPU_ONLY`'s thread cap and added `NP2_THREADS` with `min(4, ...)`. The driver also retains stack8's owned-input proxy.

An initially added logging vector was unnecessary: the token limit bounds active sequences, so the imported uint64 mask is safe. It was removed from the component commit. The local chain-dump comment was corrected to its actual format. Final-source build/unit/np2 checks cover these logging-only corrections; inference and sampling operations match the main-suite snapshot.

## Every final source change

- `common/speculative.cpp`: contiguous per-sequence `pending_h`, arm row count, default-on supported folding; local off switch and sequence-tagged draft dump. The chain dump format stays unchanged.
- `src/llama-context.cpp`, `src/llama-context.h`: arm row count, multi-sequence layout/position/fallback checks, per-stream support, one debug engagement marker per active sequence count; local forced-layout control.
- `src/llama-ext.h`: experimental arm API gains `n_seq` and documents the indexed hidden buffer.
- `src/llama-graph.cpp`, `src/llama-graph.h`: empty state-copy view and no-extra-state selection; per-sequence hidden packing/index tensor and reuse checks; local wrong-sequence control.
- `src/models/qwen35.cpp`: hidden-row gather from previous/target rows; removes the old single-sequence fold mutant.
- `src/models/delta-net-base.cpp`: getter selects all ubatches without extra recurrent states.
- `ggml/src/ggml-cuda/gated_delta_net.cu`: joint sequence kernel/dispatch; local racy-layout mutant.
- `ggml/src/ggml-cuda/gated_delta_net.cuh`: joint limit constant.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: indexed cache fusion accepts supported joint sequence counts.
- `tests/test-backend-ops.cpp`: joint/fallback/swapped-state cases, race-amplifying case, served head-split perf shapes.
- `tests/test-recurrent-state-rollback.cpp`: multi-sequence/swap/joint-start/extra-state scenarios, gather probe, identical-batch raw-logit capture/replay, gathered-control validation by actual ubatch shape.
- `codex/bw-np2-driver.cpp`: thread cap/override, k and KV-layout controls, mixed slot workload, model-derived EOG biases for fixed length.

No incidental compile fixes. Shell/Python/build changes are confined to the Mac gate/selection scripts, minimal parser copy, window, renderer, entry/guard, templates, evidence, report and ignore entries. Other inherited `codex/` files are unchanged.

## Mac evidence

Commands, with CUDA hidden and Git lazy fetching disabled:

```sh
BW_MAC_OUTPUT_BASE="$PWD/build-bw-mac-results" ~/repos/qwen38-eval-worktrees/bin/bw-mac "$PWD" codex/bw-stack9-mac.sh stack9
BW_MAC_OUTPUT_BASE="$PWD/build-bw-mac-results" ~/repos/qwen38-eval-worktrees/bin/bw-mac "$PWD" codex/bw-stack9-static-mac.sh stack9-static
BW_MAC_OUTPUT_BASE="$PWD/build-bw-mac-results" ~/repos/qwen38-eval-worktrees/bin/bw-mac "$PWD" codex/bw-stack9-final-mac.sh stack9-final
```

Main model suite: `fa4e41f7ef2fccdef9b2e51fb68eb9d36da70f22`; `build-bw-mac-results/stack9-mac-a1/`. Final source: `dc22c96d87f925c5479d4763ef08a146be03b893`; `build-bw-mac-results/stack9-final-mac-a1/`. Both returned rc 0 (M). Consolidated gate/source provenance: `codex/stack9-evidence/summary.json`; full outputs remain inside this worktree under `/build*`.

CPU configure: CUDA/Metal/BLAS/Accelerate/repack/llamafile off, native off, `armv8.2-a+dotprod`, ccache ON, nice 19, build `-j4`. Model jobs use explicit small contexts and at most four configured threads (D), wait for four GiB free and no known model process, fail after a ten-minute RAM wait or model-runtime cap. Qwen3.5-4B UD-Q4_K_XL MTP fixture; the fold matrix uses q8 target KV, k3/k5, seeds 1729/2718, plain and Meta(CPU,CPU), per-stream and unified KV (D configurations).

| Stage | Main wall s (M) | Final wall s (M) | Result |
|---|---:|---:|---|
| 20 `cpu-build` | 71 | 51 | PASS |
| 21 `unit-tests` | 109 | 97 | PASS |
| 23 `np1-chain-identity` | 3383 | - | PASS |
| 24 `fold-mseq-identity` | 1496 | - | PASS |
| 29 `np2-chain-identity` | 124 | 107 | PASS |
| 31 `recorded-parser-controls` | 0 | 0 | PASS |
| 32 `np2-old-negative-control` | 183 | - | PASS |
| 33 `cand-cpu-identity` | 133 | - | PASS |
| 34 `cand-mutant` | 3 | - | PASS |
| 35 `np2-getter` | 293 | - | PASS |

Chain: four identity pairs (M), each nine prompt/temperature/seed cases and 288 output tokens (M); tokens, counters, draft/candidate ids equal, maximum q delta within the inherited threshold. Repeat is bit-identical; reversed-uniform and shifted-embedding controls caught.

Fold: 24 identity pairs across 52 sequential model processes (M); complete token/counter JSON and sequence-tagged draft dumps byte-identical. Joint engagement and fewer catch-ups required. Per-cell counts: `codex/stack9-evidence/fold-counts.md` and `fold-summary.json`.

Getter: two native raw-logit artifacts, each 123169696 bytes (M), byte-identical for every identical-batch row under both cache fills. Plain and Meta np2 driver outputs/counters equal. Restored build replays the gathered reference exactly. `getter-summary.txt` and `getter-runs.tsv` record success and mutant statuses.

Negative controls (M): old getter gate, extra-state misuse, tail gather and wrong-row mutants each fail with engagement and expected signature; wrong-row driver returns success but differs from control and is caught. Fold wrong-sequence control changes 2 output ids. Forced layout preserves output/dumps and all catch-ups, with positive refusal count and no folds. Mixed-slot off/on preserves output/dumps and the five-row verify, reduces catch-ups from 36 to 4, with 32 folds. Old np2 chain guard fails token/counter comparison after engagement; no-extension candidate mutant fails the full-row-read assertion. CUDA race control was not run here.

Recorded GPU parser replay passed on real llama SSE/np1/np2 records, memory/journal fixtures, and the real np2 getter canary log. Token, overlap, candidate/getter/fallback proof mutants and absent fold markers are rejected. Initial static retry rc 31 (M) was an overly broad soak assertion; corrected static attempt and final replay pass. Historical fixture timings are not stack9 timings. `bash -n`, Python compilation and `python3 -m gpuq.shlint` passed for every authored shell file.

## Window and items

- Window: `codex/bw-stack9.toml`, payload `codex/bw-stack9.sh`. Filler, priority 19, non-preemptible, host-insensitive. Refuses dirty tracked source, changed tested source, or existing `frozen/stack9`. CPU gates are pinned in `codex/stack9-evidence/summary.json`.
- CUDA recipe: 13.2, NCCL OFF, architecture `120`, CCCL 3.4.3, disconnected dependency settings, ccache ON, buildgate `-j4`. Requires full-drain `DeviceTopK` count 25 before freeze. Only the parent runs this window.
- `codex/items/stack9-depth-j5q.toml.in`: priority 30, frozen stack8 vs stack9, ABBA, np1, chain/fold enabled in both, served k5 flags, 8k/32k/90k. `identity = "cross-arm"`: getter/fold changes should preserve tokens and draft counters at np1.
- `codex/items/stack9-np2-real-j5q.toml.in`: priority 30, real session prompts, served duo k5/ub128 flags, plan `s8 s9 s9-nofold s9-nofold s9 s8`, same depths. s9 has no fold environment override; nofold uses the local switch. Canary requires indexed fused getter on both GPUs and `mtp_fold_decode: engaged n_seq=2`, at verbosity 5. Cross-boot identity is off with an explicit arrival-dependent batch-shape reason; deterministic CPU exactness gates are mandatory.
- `codex/items/stack9-duo-soak-j5q.toml.in`: priority 29, frozen stack9 duo flags, fold default-on. Sequential canary proves `n_seq=1`; joint proof belongs to the np2 item.

After freeze, `codex/render_stack9_items.py` snapshots the current parent harness inside this tree and writes/checks `codex/items/rendered/{stack9-depth-j5q,stack9-np2-real-j5q,stack9-duo-soak-j5q}.gpuq.toml`. Every run/canary has the pinned CCX0+1 wrapper and `host_sensitive = false`; one-boot canaries fit the ten-minute budget. Frozen bytes/libraries, exact server PID, scoped kernel faults, steady/overlap gates and untraced timing come from the reused harness. `--cache-ram 0` comes from its default, not hard-coded arm flags. No item pins Mac-result paths.

Real items cannot be rendered/checked before `frozen/stack9` exists. Renderer calls only `gpuq submit --check`. Window structure/budgets are checked by the Mac fixture replay; no host submit/check or CUDA window execution was performed.

## Risks and open items

No GPU speed claim. Parent must run the window, then canaries and A/B/soak. CUDA compilation and integration-specific GPU lifetime/capture behavior remain open; CPU identities do not establish them. This task does not audit or fix the broader graph/meta issues in the background review. Experimental arm callers must pass the new `n_seq` argument.

Architecture caveat: the hard rule specifies `120`, while frozen stack8 and the stack6 recipe use `120a`. The delivered draft retains literal `120`; an optional clarification was unanswered. This compiler-flag difference can confound source-only performance attribution. Resolve it before attributing GPU deltas to these components.

Keep fold component separate if its queued parent A/B is null. The root getter commit remains independently usable; dropping fold also requires removing its local controls and fold-specific validation/item expectations.
