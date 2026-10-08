# K7 paired q/k RMS norm + scale

Labels: M measured, D derived, G guess. No CUDA code was compiled or run (no nvcc/GPU by rule): the CUDA kernel, launcher and dispatch are unverified until the window.

## Commits (base 0cfc0723e, branch topic/s10-k7)

| Commit | Subject |
|---|---|
| `7ad86f46c` | `cuda : run paired RMS norm and scale chains in one launch` (general) |
| `579617183` | `local: add paired RMS norm and scale off switch, mutants and engagement log` |
| `27f70e2b8` | `local: select the paired RMS norm alias mutant per guard` |
| `b743555cf`, `914b63c44`, `0ba65e053`, `86a7d830c` | codex: Mac payload, parser fix, CUDA window payload, Mac evidence (files under `codex/k7/`; `codex/PURPOSE` rewritten by `wt new`) |

## Design and exactness

Backend-internal fusion, no new ggml op, no concatenation, no GDN change. `ggml_match_rms_norm_scale_pair` (`ggml/src/ggml-rms-norm-scale-pair.h`) matches two independent `RMS_NORM -> SCALE` chains of equal `ne`, separated only by no-op nodes (view/reshape/permute/transpose): the q and k views of a GDN layer. Chain validity reuses `ggml_can_fuse` (single-use, non-output intermediate, same shape), plus F32, `nb0 == 4`, scale bias 0, contiguous scale output. Because chain b is hoisted to chain a's position the matcher also rejects: b's input rooted in a's nodes, and any overlap of out_a with in_b, out_b with in_a, out_a with out_b (`ggml_ssm_conv_overlaps`). Header-only so the CPU test exercises the exact predicate the CUDA backend uses (same precedent as `ggml-ssm-conv.h`).

CUDA: `rms_norm_scale_pair_f32<256|1024>` (`ggml/src/ggml-cuda/norm.cu`) is one grid `(ne1, ne2, 2*ne3)`; `blockIdx.z / ne3` selects the chain. Each block does exactly the per-row code of `rms_norm_f32<bs,false,false,true>`: same block size rule (`ncols < 1024 ? 256 : 1024`), same strided partial sum order, same `block_reduce`, `tmp / ncols`, `rsqrtf(mean + eps)`, `scale_out * (scale * x)`. Per-chain pointers, strides, eps and scale are kernel args, so chains may differ in stride/eps/scale. Hence D: byte-identical to two launches (a duplicated 10-line body, not shared with `rms_norm_f32`, kept in sync by comment and by the byte test). Automatic fallback: any mismatch, `ne2 > 65535`, `ne3 > 32767`, or an active concurrent-stream region (`stream_context().concurrent_events` non-empty) falls through to the existing single-chain fusion. Multi-sequence: `ne3` is the sequence dim and is covered by the test (seqs 1, 2). Dispatch: `ggml_cuda_try_fuse` in `ggml/src/ggml-cuda/ggml-cuda.cu`, before the single `RMS_NORM, SCALE` branch; it returns the node count to skip (the chains plus the no-op nodes between).

## Every source change

- `ggml/src/ggml-rms-norm-scale-pair.h` (new): matcher. Local commits add the alias mutant mask env there.
- `ggml/src/ggml-cuda/norm.cu`, `norm.cuh`: pair kernel, args structs, launcher `ggml_cuda_op_rms_norm_scale_pair_fused`. Local: `GGML_CUDA_RMS_NORM_SCALE_PAIR_MUTANT` (1: chain b scale one ulp high, 2: chain b reads chain a's input).
- `ggml/src/ggml-cuda/ggml-cuda.cu`: include + dispatch. Local: `GGML_CUDA_RMS_NORM_SCALE_PAIR=0` off switch and `rms_norm_scale_pair_fusion:` debug log.
- `tests/test-backend-ops.cpp`: include, `run_rms_norm_scale_pair_identity`, hooked after the alpha check, selected with `-o RMS_NORM_SCALE_PAIR`. No incidental compile fixes.
- Local mutant env `GGML_RMS_NORM_SCALE_PAIR_ALIAS_MUTANT=<mask>` (1: drop out_a/in_b check, 2: out_b/in_a, 4: out/out, 8: dependency).

## Test

`run_rms_norm_scale_pair_identity`: 360 cases (ncols 64/128/256/1000/1536, heads 1/16, tokens 1/3/12, seqs 1/2, six layouts: qwen-shaped shared parent; separate tensors with different eps/scale; different strides; a compute node between the chains (must not pair); k output over q input (must not pair); mismatched shapes (must not pair)), 4 replays each with changing inputs including a zero row, inf, subnormals and overflowing squares. Fused result is compared by bytes against a reference whose RMS node is output-flagged (never fused), with outputs pre-filled with a sentinel (unwritten elements detected) and a one-bit-flip sensitivity control. For layouts that must pair it asserts the matcher pairs, and mutates the graph seven ways (intermediate flagged output a/b, out_a over in_b, out_b over in_a, outputs overlapping, in_b a view of out_a with otherwise disjoint bytes, scale bias != 0) and requires rejection.

## Mac evidence (M; `codex/k7/mac-evidence/`, commit tested `0ba65e053`, source files identical at HEAD)

| Stage | wall s (shared Mac, waits included) | Result |
|---|---:|---|
| 20 CPU build | 122 | PASS |
| 21 `RMS_NORM_SCALE_PAIR` 360 cases + `-o RMS_NORM`, `-o SCALE` | 622 | PASS |
| 22 four per-guard alias mutants | 2 | each fails with its own `matcher accepted: ...` message |
| 23 real Qwen3.5-4B graph (`GGML_SCHED_DEBUG=2`) | 3 | q_conv_rms=384 k_conv_rms=384 adjacent_pairs=384, exactly one no-op node between the chains |

The CPU run proves the matcher, guard negative controls and test harness; it does not exercise any CUDA code (on CPU the fused and reference paths are the same kernels).

## Window (not submitted)

`codex/k7/bw-k7.toml` + `bw-k7.sh`: filler, priority 18, `preemptible = false`, `host_sensitive = false`, same GPU UUIDs and recipe (CUDA 13.2, 120a, NCCL OFF, CCCL 3.4.3, GCC 13, DeviceTopK count 25), builds only `test-backend-ops` under `codex/k7/`'s tree (`build-cuda`), no freeze. On CUDA0 and CUDA1: pair test with graphs and with `GGML_CUDA_DISABLE_GRAPHS=1` (byte pass + engagement log); off switch (pass, no engagement log); mutants 1 and 2 (must fail `byte identity` while engaged); alias mutants 1/2/4/8 (must fail with their own message); `RMS_NORM`, `RMS_NORM_MUL_ADD`, `RMS_NORM_MUL_ROPE`, `SCALE` neighbours. Preflight hashes sources against the Mac evidence.
`gpuq submit --check codex/k7/bw-k7.toml`: `bw-k7: valid` (rc 0). Both `.sh` pass `python3 -m gpuq.shlint`; `bash -n` ok.

## Expected effect and falsifier

D-report cause: q/k norm `rms_norm_f32<256,..>` is 96 launches/round (48 layers x 2), kernel-sum 0.207 ms, exposed 0.122 ms at np2 8k (1.27 us exposed per launch). The pair removes 48 launches and its grid is under one wave: D about -0.03..-0.06 ms/round exposed at np2, depth-flat (8k/90k the same); np1 launches are PDL-overlapped so about 0 (D). Realization is G 0.3-1 (the D-report's own N3 np1 null for similar fusions), so this is below the ~0.1 ms between-boot resolution and should be judged by kernel counts, not round time. Falsifier: np2 trace shows `rms_norm_scale_pair_f32` 48/round and `rms_norm_f32<256,..>` scale-variant 96 -> 0, ELEM partition falls < 0.03 ms. Engagement on the real stack is checked by the debug log `rms_norm_scale_pair_fusion:` (one per layer per captured graph) in the combined-stack canary.

## Open / unverified

- All CUDA code: compile, kernel indexing (`which`/`sample` from `blockIdx.z`), launch via `cudaLaunchKernelEx` with a by-value struct argument, byte identity, CUDA-graph capture.
- Under Meta tensor-parallel the per-device graph order was not inspected; the CPU graph dump shows the unsplit order. If Meta inserts a compute node between the chains, the pair does not engage (safe fallback), visible only through the log.
- The alias-guard rejection is proven at matcher level (deterministic), not by a racy CUDA corruption; the mutant makes the matcher accept, the test's matcher assertion fails.
- `ggml_ssm_conv_overlaps` is reused for the generic range-overlap test despite its name.
