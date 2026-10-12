# zc-pr: zero-copy token table (upstream form, master abee0c847)

Brief: `qwen38-eval/orchestration/briefs/2026-10-12-zc-pr.md`. Design: `work/zc-report/REPORT-zc-getrows.md`. Probe: `work/zc-window/SUMMARY-RESULT.md`.

## Commits (topic/zc-pr)
1. `ggml-cuda : add a host-mapped buffer type that GET_ROWS reads in place`: `CUDA_Host_Mapped` (cudaHostAllocPortable|Mapped,
   used only when the UVA device pointer equals the host pointer, else or under `GGML_CUDA_NO_PINNED` a CPU buffer). It is
   offered via `ggml_backend_dev_get_extra_bufts`. `supports_buft` accepts it on every CUDA device; `supports_op` rejects
   any op that reads it except GET_ROWS src0. Meta: a non-CPU host buft that all simple devices support is MIRRORED
   (`[TAG_META_HOST_MAPPED]`). Test: `tests/test-host-mapped.cpp`.
2. `llama : keep weights that a GPU gathers by rows in host-mapped memory`: `make_cpu_buft_list` puts the mapped buft after
   ACCEL unless `--no-host`. The device's supports_op admits only the token embeddings.
3. `codex:` the Mac gate payload (`codex/zc-pr/mac-gate.sh`, `driver.cpp`, `zc.patch` = commits 1+2 reverse-applied there)
   and this README. Drop it upstream.

Leaf `topic/zc-pr-gather` (wt/zc-pr-gather, the optional commit): GET_ROWS with a host src0 first copies the selected rows
into a pool buffer with int4 loads, all rows in flight, then dequantizes. Measured only by the probe (arm b on the zc build vs
arm b on the gather build = b2).

## Tests
- test-host-mapped, per CUDA device:
  - mapped vs device-buffer GET_ROWS bitwise, every supported type, shapes 1, 13, 512 and 5x2 rows;
  - mul_mat and add on mapped srcs are unsupported;
  - sched: the CPU-buffer control gathers on the CPU, and forcing it onto the device copies the weight (the copy callback
    sees it); mapped gathers on the device with 1 split and no copy; mul_mat on the same weight runs on the CPU, also no copy;
  - `GGML_CUDA_NO_PINNED=1` falls back to the CPU with equal values.
- CPU-only build: SKIP rc 0.
- Mac gate (CPU-only, `bin/bw-mac`): build, test-alloc, model greedy tokens on plain CPU and Meta(CPU,CPU), with and without
  `no_host`, against the same tree with the PR reverted.
- GPU window `work/zc-pr-window` (bw-zc-pr, priority 40):
  - test-host-mapped (normal and NO_PINNED) plus test-backend-ops GET_ROWS on CUDA0/CUDA1, for both builds;
  - probe b vs b2;
  - placement proof boot (J5q duo) on base, zc and zc with NO_PINNED, with GGML_SCHED_DEBUG=2. It checks where token_embd
    lives and its GET_ROWS backend, that no CPU split comes before the first Meta split, that the 16 greedy tokens are equal,
    and records free VRAM per GPU;
  - freeze `frozen/zc-pr`, then render and `--check` the A/B items, and smoke canaries.

## Caveats (decision-relevant)
- Under mmap the table is copied into pinned host RAM: +644 MiB pinned (J5q) per process, 0 VRAM.
- bitnet reads tok_embd in MUL_MAT too. With the PR that MUL_MAT stays on the CPU (supports_op false, so no offload).
  Before, a batch could be offloaded from the host buffer.
- MUSA: the buft is not offered (`cudaHostAlloc` is not mapped in musa.h).
- HIP: compiled, untested.
- The Mac gate cannot exercise the CUDA or Meta host-mapped paths. Only the window does.
