# s9-gdn4: stack9 + #30087 (GDN, 4 columns per warp)

Tree `wt/s9-gdn4`, base `cd776e104` (frozen/stack9). Nothing here was compiled for CUDA or run on a GPU.

## Commits (stack on cd776e104)
- `03c8469f1` tests : time the fused GDN+CPY pair in perf mode. General form, upstreamable by itself (`perf_nodes()` hook in `test-backend-ops.cpp`; a case lists the nodes eval_perf repeats). Also compares the fused-case attention scores (never read before) and adds correctness cases S_v=128, 24 heads, K=6, 1/2/4 indexed seqs (+ swapped seq ids already covered by existing indexed cases) and perf cases at 24-head decode/verify shapes.
- `17ccc7e4f` cuda : process 4 GDN columns per warp. Credits #30087; keeps stack9 NS template, joint layout, `s_shard[NS][rows]`, one block per (head, column slice). Reduction order changes: not bitwise equal, gate is NMSE vs CPU. Not upstreamable as is (stack9 multi-seq kernel is local); the single-sequence part is #30087.
- codex files: window `codex/bw-s9-gdn4.{toml,sh}`, `codex/s9gdn4/*` (Mac payload, parsers, item renderer, `mac-evidence/`), `codex/items/*.toml.in`, reference stack8 drafts in `codex/items/ref-stack8-gdn4/`.

## Done (M)
- Mac CPU gates (bw-mac, run a3, HEAD cc8ee9963): build ok, 82 GDN cases passed (11 fused-indexed S_v=128/6-token rows), perf control: old harness (repeat only `out`) vs new on the 6-token fusion rows, new row 79.5..197.6x larger (e.g. indexed n_seqs=1: 4.94 us -> 900.5 us), untouched rows within 0.33..3x. Evidence `codex/s9gdn4/mac-evidence/summary.json` (source sha256 of kernel+test bound; the window preflight re-checks it).
- `gpuq submit --check codex/bw-s9-gdn4.toml`: valid. `python3 -m gpuq.shlint` rc 0 on `codex/bw-s9-gdn4.sh` and `codex/s9gdn4/bw-s9-gdn4-mac.sh`. `bash codex/bw-s9-gdn4.sh --preflight`: PASS.
- Two timing items rendered against a stand-in `frozen/stack9-gdn4` and `--check`ed valid (`codex/items/s9gdn4-depth-j5q.toml.in`, `s9gdn4-np2-real-j5q.toml.in`; arms frozen/stack9 vs frozen/stack9-gdn4, served `swift15-j5q`/`-duo` flags, `[question]` expected (D): np1 -0.16 ms/round, np2 -0.24..-0.48 ms/round; prefill about -3 % noted in answers). The real ones are rendered by window stage 47 into `codex/items/rendered/`.

## Window `bw-s9-gdn4` (filler, prio 18, non-preemptible, est 4500 s, budget 9000 s)
Stages 40-47: CUDA build of base/cand/mutant-red/mutant-attn/oldh variants; test-backend-ops on both GPU UUIDs; must-fail mutants (reduction over full warp; attention write by lane 0 only) must be caught; `cuobjdump --dump-resource-usage` spill gate (STACK/LOCAL 0, instance set equal to base); ABBA kernel perf base vs cand (non-gating); `wt freeze` as `frozen/stack9-gdn4`; item render + `--check`.
Not submitted. Submit: `gpuq submit codex/bw-s9-gdn4.toml` from the tree. Pins: the files listed in `[pins]` must not be edited; the kernel and test file are not pinned (edited and restored inside the window; the preflight binds them to the Mac hash instead).

## Uncertainty
- The window script is unexecuted beyond `bash -n`, shlint, `--check`, preflight and parser tests on synthetic logs; a first real run may need a fix-up (attempt `retry = never`).
- Expected speedups are derived from stack8 plain-kernel measurements (-32 %/-42 % at 24 heads), not measured on the stack9 joint kernel. 4-head shapes were +9 % (slower) in stack8; the CTA shrink loop is the mitigation, not verified.
- Joint kernel: attention scores of the fused path are now compared, so a first CUDA run could expose a pre-existing stack9 difference rather than a gdn4 regression; the base variant is run through the same tests to tell them apart.
