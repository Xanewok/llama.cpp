# Package 67, getter form: work-in-progress evidence (2026-10-06 23:35 CEST)

Status: not ready. np2 decides whether the form ships as is or needs an `n_seqs == 1` gate.

## Shape

The GDN kernel reads the accepted recurrent-state row in place through `ids` on `ggml_gated_delta_net` plus a Mamba-style `build_rs` getter, following the maintainers' precedent for `ggml_ssm_scan(..., ids)` (#29856). The per-layer `get_rows` gather is not built. Backends without ids support reject the op. The old new-op form (`GGML_OP_GATED_DELTA_NET_INDEXED`, the patches in `../`) is superseded and kept only as evidence for that form.

## Sources

- Master form: `~/repos/qwen38-eval-worktrees/wt/gdn-indexed-upstream`, `topic/gdn-getter` on master 7049ff0cb: fa1c8cb6c (ggml ids), 5a91f7b87 (llama getter), 472791d90 (test reserve with an idle shared sequence), then the staging commits 26b2ddca2/48e0e5bc2/3bea534df. The staged patch series and its CPU evidence are in that tree at `codex/upstream/llama.cpp/67-gdn-indexed-rollback/` (`EVIDENCE.md`, `NOTES.md`) and `codex/REPORT-gdn-getter.md`.
- Follow-ups not yet in the staged series: `wt/gdn-getter-v2`, `topic/gdn-getter-v2`: b9b6f3d0c (virtgpu does not claim GDN state ids support), ee3599365 (idle recurrent slots through the getter; `codex/dropextra-mutant.cpp` mutants fail).
- Stack6 port used on the GPU: `topic/gdn-getter-stack6` 743acb55f (c68ec4329, 8d8d2ed1c, 0a473e886 trace, 743acb55f removes the replaced stack6 gathers). Freeze with CCCL 3.4.3: `~/repos/qwen38-eval-worktrees/frozen/gdn-getter-stack6-fc` (1aa34911e).

## Evidence (M measured, D derived)

| Check | Result | Where |
|---|---|---|
| CPU GDN op tests | M 57/57, incl. 13 indexed-equivalence and 3 indexed cache-copy cases; row-0 negative control fails all 13 | worktree `codex/logs/backend-cpu.log`, `negative-row0.*` |
| CPU archs | M 133/133 | `codex/logs/archs-cpu.*` |
| CPU model, 4B rollback/checkpoint np1/np2 | PASS patch and base; split replay FAILS identically on base (pre-existing, not this change) | `codex/logs/model-{patch,base}-np{1,2}.log` |
| GPU sanitizers | M memcheck 41+16, racecheck/synccheck 16/16 clean | ledger row "GDN in-place state read, upstream getter form" |
| GPU np1 depth, J5q, ABBA, CCCL build | M −0.48 / −0.11 / −0.24 ms/round at 8k/32k/90k (D from M intervals); cross-arm token + draft-counter identity on 4 boots; `gdn_gather_skipped` 0 vs 107–128/round | `~/repos/qwen38-eval/results/2026-10-06-gdn-getter-np1-cccl-j5q/full-a1` |
| Same, earlier build without CCCL | +0.10 / +0.20 / +0.14 — build confound (0 vs 25 DeviceTopK strings), not the change | notes 2026-10-06 19:43 |
| Old stack form, for comparison | −0.31 / −0.12 / −0.29 | ledger |

## Open before it is a package

1. np2: `gdn-getter-np2-busy-j5q`, `gdn-getter-np2-idle-j5q` (queued 10-06). The `src[6] && n_seqs > 1` guard turns GDN cache fusion off on every multi-seq ubatch, and fusion also turns off when n_rs > n_seqs (idle slot). Upstream `llama-server` runs several slots by default, so an np2 regression blocks the PR. If np2 regresses: add an `n_seqs == 1` gate and rerun np2.
2. Fold b9b6f3d0c and ee3599365 into the staged series; rebase on current master; CPU suite; fresh independent review; network dupcheck (read-only).
3. Install the staged series here (replace `../000*.patch`, keep the old form under `alt-opcode/`), update `../status`, `../EVIDENCE.md`, UPSTREAM.md row 67. PR text is the owner's.

## Stack7 refresh (2026-10-07, recorded artifacts; not rerun)

The stack7 integration contains the gated getter (`3c97c4af8`: one active GDN sequence only). Source of record: `docs/stack7.md` and `docs/ledger/llama-cpp.md`. The prior multi-sequence getter is not cleared by these results.

M: `results/2026-10-07-stack7-depth-j5q/full-a5`, frozen stack7 `48b82b0d1`, J5q, np1, probabilistic MTP k=5, tensor split, two boots per arm, cross-arm tokens and acceptance identical. D from the recorded M intervals: stack7 minus stack6 is -0.58/-0.62/-0.61 ms/round at 8192/32768/90112 tokens. This includes getter, host trims, out-projection reshape and chain; it does not isolate the getter. Fold-C's additional difference is -0.23/-0.19/-0.22 ms/round (D).

M: `results/2026-10-07-stack7-np2-real-j5q/full-a3`, two boots per arm, real prompts; D: integrated stack7 -1.130/-1.012/-1.277 ms/round at the same depths. Identity was disabled and acceptance differs. The getter uses the explicit-gather fallback on multi-sequence GDN ubatches; the integrated result is not a getter np2 gain.

No patch rewrite, Mac build, new GPU measurement or network duplicate check was performed for this refresh. The existing getter-series open items remain.
