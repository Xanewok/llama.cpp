# 67 gdn-indexed-rollback: evidence

Labels: M measured, D derived, G guess. Two source revisions, never mixed: **stack form** = stack6 `ac7465885` + `03fe64c15`, `c51652aa1`, `9e999126b` (+ local trace `f5c5bd54f`), GPU-tested head `38c59b368`, frozen build `~/repos/qwen38-eval-worktrees/frozen/gdn-gather-inplace`; **upstream form** = master `7049ff0cb` + 0001..0004 (`topic/gdn-indexed-upstream` 3945edc36), CPU/compile only, no GPU run. GPU numbers belong to the stack form; they are not a promise for the rebased patch.

## GPU A/B (M, stack form, 2026-10-06, gpuq `gdn-gather-inplace-depth-j5q`, 952 GPU s)
- Setup: Swift 1.5 J5q, TP2 on 2x RTX 5060 Ti, np1, MTP k5, `--kv-pad 4096`, random prompts, 512 predicted tokens, ABBA (up, patch, patch, up), 2 boots per arm, depths 8192 / 32768 / 90112. Baseline `up` = frozen stack6 `ac7465885`, `patch` = `38c59b368`. Both arms ran with `GGML_CUDA_DEBUG_SSM_CONV=1`, `GGML_CUDA_DEBUG_GDN_STATE=1`, verbose logging; only the patch arm pays the round-trace gather counter (patch-side overhead, so the delta is conservative, G).
- Raw: `~/repos/qwen38-eval-worktrees/wt/gdn-gather-inplace/codex/gpu-results/depth/full-a1/summary.json` (+ `{0-up,1-patch,2-patch,3-up}/depth-*.json`, `pins.json`); ledger row in `docs/ledger.md` (KV_GDN, 10-06); notes `research/2026-09-28-dual-5060ti/notes/2026-10-06.md` ("11:25").
- Steady ms/round per boot (D, `steady_ms_per_round_D`), sample unit = one boot, n = 2 per arm:

| depth | up boots | patch boots | patch - up (D) |
|---|---|---|---|
| 8192 | 40.752, 40.441 | 40.495, 40.080 | -0.309 ms (-0.76 %) |
| 32768 | 42.037, 41.919 | 42.047, 41.669 | -0.121 ms (-0.29 %) |
| 90112 | 46.522, 46.284 | 46.099, 46.136 | -0.286 ms (-0.62 %) |

- Spread: within-arm boot spread is 0.12-0.42 ms. The 8k and 90k deltas are about one spread; the 32k delta (-0.12) is inside it. Treat 8k/90k as small but consistent, 32k as null (G).
- Identity (M): cross-arm token identity held at all three depths; acceptance identical per depth (0.9148 / 0.3094 / 0.0978 in both arms, M). Decode tok/s: 131.8 -> 133.1, 60.12 -> 60.14, 31.65 -> 32.01 (M).
- Engagement (M, negative control = baseline arm): `gdn_gather_skipped` per round 0 in `up`, 106.8 / 116.6 / 128.2 in `patch` at 8k / 32k / 90k; graph hit/miss per round identical in both arms (hit 6.76 / 6.89 / 6.91, miss 0.16 / 0.13 / 0.16), so no graph-reuse side effect.
- Forecast check: attribution of the gather before the change was 0.30-0.33 ms/round (M, ledger row, from the earlier stack6 attribution); measured -0.31 and -0.29 at 8k and 90k (D). The gather is not removed from the other 40 ms of the round, so this lever cannot exceed ~0.8 %.

## Sanitizers (M, stack form, gpuq `gdn-indexed-memcheck` a2, 2026-10-06 09:05)
Raw: `results/2026-10-06-gdn-indexed-memcheck/a2/verdict.json` (+ `*.tail.txt`; full logs `/mnt/scratch/work/gdn-indexed-memcheck/a2/`). `test-backend-ops -b CUDA0` on the 15 indexed cases plus the old 41 GDN cases as control:
- `mc-old` (plain `GATED_DELTA_NET` 36 + `CACHE_FUSION` 5, memcheck): 41/41, 0 errors, 0 indexed lines.
- `mc-idx` (memcheck): `GATED_DELTA_NET_INDEXED` 4 ok + 8 not supported, `..._INDEXED_CACHE_FUSION` 1 ok + 2 not supported: 5 ok / 10 "not supported" (the n_seqs = 2 cases, the np2 fallback); 0 errors.
- `rc-idx` (racecheck): 5/5, 0 hazards. `sc-idx` (synccheck): 5/5, 0 errors.
- The runs include the CUDA numerical comparison against the CPU reference (rows `ne[3] - 1 - 2*i` select a non-trivial row), which is the only place the CUDA kernel's row selection is checked; `-b CPU` runs compare CPU to CPU and cannot catch a wrong row (see below).

## CPU, upstream form (M, 2026-10-06, `CUDA_VISIBLE_DEVICES=""`, master `7049ff0cb` + 0001..0004)
Build: CPU `-DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON`: `test-backend-ops test-llama-archs test-recurrent-state-rollback` rc=0, 0 warnings. CUDA `-DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a`: `test-backend-ops` + `llama` rc=0, 0 warnings (built before the whitespace-only amend of 0004).
- `test-backend-ops test -b CPU -o GATED_DELTA_NET_INDEXED`: 12/12; `-o GATED_DELTA_NET_INDEXED_CACHE_FUSION`: 3/3; `-o GATED_DELTA_NET`: 36/36; `-o GATED_DELTA_NET_CACHE_FUSION`: 5/5.
- `test-llama-archs -s 42` (all hybrid GDN archs go through the refactored `build_recurrent_attn`, fallback path since `n_rs_seq = 0`): `Summary: all 133 test(s) passed` (Meta rows SKIP on CPU without package 51).
- `test-recurrent-state-rollback -m /mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf -c 4096` (`n_rs_seq = 8`, so the indexed path is engaged on the CPU backend, single seq): rollback + checkpoint replay passes, shared-seq reserve passes. `test_multi_seq_split_replay` FAILS (`max diff 13.4578, nmse 0.018913, first at seq 0 pos 16`) with the identical numbers on unpatched master `7049ff0cb` (build `wt/gdn-indexed-base`): a pre-existing result for this model on this CPU, not a regression (not investigated; relevant to a reviewer who runs this test).
- Stacked on 63 (`topic/gdn-indexed-on63`): `test-recurrent-state-rollback --state-movers -m <same 4B> -ngl 0 -t 1 -tb 1 -c 4096 -b 64 -ub 16 -np 1` and `-np 2`: rc=0, `STATE_MOVERS negative_controls=PASS` both (np2 uses the indexed path with two sequences on CPU).
- Runs used 1-2 pinned cores at nice 19 (gpuq had a `measure` item running; `bin/cpugate` would have waited for it); the build used `bin/buildgate -j4`.

## Negative controls
- Model level (M): forcing the CPU row index to 0 (`state_row = dst->src[6] ? 0 : iv3`, temporary, reverted) makes `test_rollback` fail: `full logits mismatch at position 6, first token 0, nmse 0.0200545`; restored code passes. So the model test engages the indexed read and detects a wrong row.
- Op level: the same mutation leaves `test-backend-ops -b CPU` GATED_DELTA_NET_INDEXED 12/12 and CACHE_FUSION 3/3 passing, because the CPU backend is its own reference. Limitation: the new cases only discriminate a wrong row on another backend (CUDA, see Sanitizers), not in a CPU-only CI job. A CPU-only fail-before test would need a graph that compares the indexed op with an explicit `GET_ROWS` + plain op inside the graph (not written; open point for the owner).
- Stack form, local (Codex report `codex/REPORT-gdn-gather-inplace.md`, M, not rerun): byte-exact comparison against explicit `GET_ROWS` in 192 cases with 192/192 wrong-slot controls detected; real 4B model CPU and Meta(CPU,CPU), np1 and np2: 4/4 baseline-vs-patch transcript comparisons pass, wrong-slot and stale-after-`seq_rm` controls both change the output.
- Engagement control on GPU: baseline arm 0 skipped gathers vs 107-128 in the patch arm (above).

## np2 / multi-sequence fallback
- CUDA `supports_op` is true only for `src[2]->ne[3] == 1` (0002). With two sequences `ggml_backend_dev_supports_op` is false at graph build and `build_recurrent_attn` takes the old `build_rs` path, so np2 output on CUDA is unchanged (D, code; the memcheck run reports the n_seqs = 2 indexed cases as "not supported", M). End-to-end np2 on a GPU with the patch was not run.
- Why not support it: the fused snapshot stores of sequence i can target the row sequence j has yet to read (D). A device-wide read barrier would be needed; the alternative (drop the fusion for n_seqs > 1) adds snapshot copies.
- CPU and Meta(CPU,CPU) support multi-sequence indexed reads (np2 passes, above and in the stack-form report). Meta over CUDA with np1 was exercised in the A/B (TP2).

## Not measured
- Any GPU run of the rebased upstream form (`7049ff0cb` + 0001..0004): kernel, `test-backend-ops -b CUDA0` for the new cases, A/B. The CUDA hunks are textually the tested ones; the model-file hunks differ from the tested form in graph node order (63's prerequisite order is not on master).
- Reserve versus runtime graph shape and `GGML_SCHED_NO_REALLOC` with `n_rs > n_seqs` (NOTES.md open question 2).
- Other backends (Metal, Vulkan, SYCL, WebGPU, OpenVINO, HTP, ET): they report the op as unsupported by default (D from `supports_op` switches; RPC rejects explicitly); no run.
- HIP/MUSA builds of the kernel change.
- Perplexity/KLD: not applicable (same arithmetic, only the read address changes); cross-arm token identity at three depths is the numerical check.

## Risks
- New public op and enum value (RPC patch version bump); a backend whose `supports_op` answers true for unknown ops would run it wrongly (not audited; RPC was handled explicitly, the other backends switch on the op).
- Single-sequence-only CUDA support makes the gain absent at np >= 2 (the serving case for two-agent use).
- Graph node order of the fallback path moved on master (state gather after conv); correctness unaffected (`test-llama-archs` 133/133), but graph splits could differ (not measured).
- Gain is 0.3-0.8 % of a round; a reviewer may weigh it against the op count.

## Getter evidence refresh (2026-10-07)

The stack7 results belong to the getter form, not this superseded opcode series. See `getter/EVIDENCE-wip.md`, section "Stack7 refresh". The integrated np1 result is -0.58/-0.62/-0.61 ms/round (D from recorded M intervals), with token and acceptance identity (M); it cannot be assigned to the getter alone. No production patches were changed in this refresh.
