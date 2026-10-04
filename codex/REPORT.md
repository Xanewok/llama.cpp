# Multistream compact KQ masks

Labels: M = measured/observed; D = derived; G = guess. Commit IDs, source lines, seeds, flags, configured depths, thresholds, and workload parameters are identifiers.

## Conclusion and commit

`979c112d95bc1774d503b7d5b508e269f5218213` - `kv-cache : allow compact masks for multiple streams`.

The production change is one condition in `src/llama-graph.cpp:45`: use `n_tokens/n_stream <= 16` without requiring `n_stream == 1`. Any active stream count works. No sampling code, arithmetic kernel, public API, or optimization knob changes.

M: mask bytes, production graph logits, and concurrent CPU server tokens/draft counts match the full-mask control. The CUDA build passed and is frozen. GPU speed and GPU VRAM changes are unmeasured. Existing tests are not entirely green: recurrent multi-sequence replay fails identically on the baseline, and `test-backend-ops` was not run under the explicit prohibition.

## Design and dependencies

`src/llama-kv-cache.cpp:1552` already takes the minimum certified prefix across every query's actual cache stream. Density, sequence membership, causal visibility, and M-RoPE ties are checked. `src/llama-kv-cache.cpp:1640` fills each suffix row from that stream, with `(n_kv - offset)*i`. `src/llama-graph.cpp:85` rechecks the certificate before reusing a compact graph. These functions are unchanged.

A short or invalid stream disables compact masks for the entire ubatch. Independent prefix lengths would require a different graph representation; this patch retains the existing exact rectangular layout and portable `GGML_OP_FILL`/`GGML_OP_CONCAT`, including capability checks and fallback.

M: the fix applies cleanly to master `050439614` after only these mask commits, in order:

- `ce19806d9` - `kv-cache : fill dense sequence masks from position indexes`.
- `77464fdba` - `kv-cache : take the dense-prefix mask path for M-RoPE positions`.
- `24ac966c7` - `kv-cache : build visible mask prefixes on the meta backend`.

The application check used copies under `build-cherry-pick/`. No other branch was changed. The standalone `tests/test-kv-mask.cpp` avoids unrelated stack test dependencies. A separate build of this minimal master combination was not performed.

CUDA's existing concat loop uses each source's dimension-3 stride (`ggml/src/ggml-cuda/concat.cu:148`) and copies F16 bits as integers. No CUDA implementation changes are needed.

## CPU exactness

M PASS: `test-kv-mask`, M 296 cases, both F16 and F32. Every full-mask element is checked against an independent cell oracle; Meta(CPU,CPU) reconstruction is compared byte-for-byte with the full mask. Coverage: unequal depths, reversed stream order, two and three streams, all pairs of accept lengths 0..3, rejected-tail `seq_rm`, holes, sequence-membership and future-position edits, causal/non-causal masks, M-RoPE ties, and per-stream rows 1/4/6/16. The test verifies that the zero-prefix shape stays on Meta. CTest wall time: M 8.74 s.

M PASS: isolated wrong-stream mutant `v_cells.at(0)` exits M 1 with `full mask differs from cell oracle`. Restoring the old production gate exits M 1 with `production multistream compact gate failed`. Production sources/objects are not replaced. See `codex/mask-np2-evidence/mutants.json`.

M PASS: `codex/mask-np2-graph.cpp` runs actual production graphs on Meta(CPU,CPU), with unequal two/three-stream depths, mixed-threshold fallback, per-stream 16-row queries, graph reuse, and cache holes after compact construction. Patch compact callbacks: M 10; baseline: M 0. All M 107,520 output-logit bytes match. Digest: `codex/mask-np2-evidence/graph.json`.

M PASS: original Qwen3.5-4B GGUF, two simultaneous HTTP requests per temperature, `-np 2`, MTP k=3, probabilistic host drafts, q8_0 target KV, f16 draft KV, staging enabled. Both arms return M 48 identical token IDs per request and identical counters:

| Temperature | Slot | Drafted, M | Accepted, M |
|---|---:|---:|---:|
| 0 | 0 | 45 | 32 |
| 0 | 1 | 59 | 27 |
| 1 | 0 | 42 | 32 |
| 1 | 1 | 47 | 31 |

Seeds: 1729/1730. Boot-plus-request wall times: temp 0 up/patch M 14.272/13.515 s; temp 1 up/patch M 12.262/12.637 s. These are identity checks, not performance results. The real-model context is below the threshold; the production graph fixture separately exercises compact masks at long depths.

The control server is renamed `cpu-serving`. `codex/mask_np2_baseline.py` freezes unchanged server/common/ggml bytes and relinks the sole changed production object from `git show 98c695111:src/llama-graph.cpp`. It checks that no other production file differs. An early copy raced a rebuild and was discarded; all reported comparisons use the corrected full-mask library.

M PASS: `test-alloc` (M 16 cases), `test-sampling`, `test-batch-alloc` (M 50 cases, M 327 assertions), `test-llama-archs --input-upload`, `test-llama-archs --tp-sampling -s 42`, `test-reasoning-budget` (M 12 cases), and `test-server-tokens`.

M PASS: `test-sampling-prefilter`, its default M 4,000 draws, including wrong-law control, grammar transitions, budget forcing, seeded trajectories, and rejection verification. The M 35,671-byte seed stream has FNV1a `e97ea9873285a3f9`. A larger 20,000-draw invocation was interrupted; the successful run uses the test's default.

Baseline failure: `tests/test-recurrent-state-rollback.cpp:test_multi_seq_split_replay`. Both patch and `98c695111` exit M 1 with M maximum logit difference 13.4578, M NMSE 0.018913, first mismatch `seq 0 pos 16`; checkpoint rollback passes on both. Its context is below the compact threshold. This failure is unchanged.

Not run: `test-backend-ops`. Hard rule 1 says "Do not run ... test-backend-ops", whereas proof item 2 requests its CPU run. A CPU-only exception was requested but not received; the prohibition was retained.

## CPU fill costs

M medians over M 51 samples of M 20 calls, nice 19, during CPU load from CUDA compilation. F16 host mask fill only, two streams, four rows each; backend FILL/CONCAT and GPU transfers are excluded.

| Depth | Full fill, M us | Compact suffix fill, M us |
|---|---:|---:|
| 8k | 4.101 | 3.902 |
| 32k | 12.458 | 2.139 |
| 90k | 35.086 | 2.435 |

At 8k both calls use the same full-mask path; the difference is timing variation. `codex/mask-np2-evidence/cpu-fill.csv` also records one-row and six-row shapes.

## Bytes and workspace

D bytes per GPU per paired round below assume equal aligned visible depth `d`, `n_kv=d+256`, prefix `d` when above threshold, and no padding boundary crossing. V+C+D masks total 4+4+3=11 rows per stream at k=3, or 6+6+5=17 at the copied experiment's k=5. Bytes = `2 bytes * 2 streams * rows * mask_width`. Aggregate mirrored H2D bytes across both GPUs are twice the table. Unequal depths use the minimum certificate; invalid density/ownership restores full upload.

| Depth | k=3 before, D B | k=3 after, D B | k=5 before, D B | k=5 after, D B |
|---|---:|---:|---:|---:|
| 8k (8192) | 371,712 | 371,712 | 574,464 | 574,464 |
| 32k (32768) | 1,453,056 | 11,264 | 2,245,632 | 17,408 |
| 90k (90112) | 3,976,192 | 11,264 | 6,145,024 | 17,408 |

These are derived aligned examples, not measured GPU transfers. A certificate of 32512 remains full even at `n_kv=32768`; a cached smaller valid prefix can leave a larger suffix.

D mask bytes saved are zero at 8k when both streams are below threshold. G GPU saving is 0.8-2.0 ms per round at 90k, retaining the parent's estimate. No GPU speed is measured. CUDA CONCAT launches one kernel per stream, so byte reduction alone does not predict latency.

KV/model storage is unchanged. Prefix and concat intermediates can increase D logical mask storage by one full mask at concatenation: np2/k5 verify at 90k gives D 2,168,832 B per device before allocator reuse. GPU peak allocation is unknown. M: the synthetic three-stream CPU graph reports Meta compute storage 6.6573 MiB versus initial expected 4.9967 MiB (D difference 1.6606 MiB). The A/B samples minimum free VRAM per GPU.

## CUDA freeze and GPU draft

M PASS: `bash codex/freeze_mask_np2.sh`. CUDA 13.2, NCCL off, arch 120a as in the reference script, CCCL v3.4.3 from the supplied local directory via `FETCHCONTENT_SOURCE_DIR_CCCL`, `FETCHCONTENT_FULLY_DISCONNECTED=ON`, compiler probing forced off, at most `-j8`. No CUDA executable was run.

Freeze: `codex/frozen/mask-np2`, source `979c112d95bc1774d503b7d5b508e269f5218213`, executable `bin/mask-np2-srv`. M PASS: top-level/bin SHA256SUMS. All ELF RUNPATHs use `$ORIGIN`; CUDA libraries additionally use `/usr/local/cuda-13.2/lib64`. Payloads stay local/ignored; provenance/manifests are committed.

`codex/mask-np2.experiment.toml`: original ABBA plan, depths 8192/32768/90112, streams=2, slot_ctx=102400, min_overlap=0.8, k=5, strict cross-arm token/draft identity. Up is the supplied stack3 CCCL 3.4.3 freeze; patch is this freeze; both set `GGML_SCHED_INPUT_STAGING=1`.

`codex/gpu_ab.sh` uses the existing depth harness's frozen checks and owned PID/process-group cleanup, plus current-boot kernel fault checks before/after. Canary: one patch boot at 32k, capped at 600 seconds. Full run: 7200-second cap. G estimate 1100 s per boot x 4 = 4400 s, below D 70% of the cap. The canary validates one boot's native identity; cross-arm equality requires the full ABBA run.

M PASS: `--fixtures`, `--capture`, and `--preflight`. Parsers/gates replay real GPU fixtures and `results/2026-10-04-duo-np2-depth/canary-a2/1-patch`: native tokens/draft counters, final timings, steady intervals, prefill replay, overlap, malformed/missing VRAM rows, bounded fault signatures. Token corruption and absent overlap fail. These recordings are parser evidence, not patch measurements. See `codex/mask-np2-evidence/harness-parsers.json`.

The inherited np2 experiment disabled identity because time-dependent ubatch composition can change even same-build outputs/counters. This draft retains the requested strict gate. Such variation can invalidate an A/B; do not relax the gate to claim speed.

## Defaults and remaining work

Compact masks remain off for non-Meta backends, F32 masks, disabled KQV offload, more than 16 tokens per stream, certified prefix below 32768, SWA, Alibi, or invalid density/visibility. No new environment knob. Input staging and target backend sampling remain off by default; this experiment enables both in both arms. Existing MTP graph/Meta plan reuse defaults are unchanged.

Parent: queue canary, then ABBA; validate GPU identity and VRAM before using timings. CPU backend-ops coverage needs the explicit prohibition lifted. Recurrent replay needs a separate investigation.

## Reproduction

CPU commands: `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`, at most eight threads, `TMPDIR=$PWD/build-tmp`. Configure `cmake -B build-cpu -DGGML_CUDA=OFF` with Release/tests on and ccache/OpenMP fetch/UI downloads off; build with `-j8`.

- `ctest --test-dir build-cpu -R '^test-kv-mask$' --output-on-failure`.
- `python3 codex/mask_np2_baseline.py`.
- `python3 codex/mask_np2_cpu_identity.py`, `LLAMACPP_TEST_MODELFILE=/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf`; localhost sockets must be allowed.
- Compile `codex/mask-np2-graph.cpp` against CPU llama/common/ggml libraries; run with an output path and EXPECT_COMPACT=1. Rerun with `LD_LIBRARY_PATH=$PWD/build-cpu-up/bin`, EXPECT_COMPACT=0; compare files with `cmp`.
- `python3 codex/mask_np2_mutants.py`.
- `bash codex/gpu_ab.sh --fixtures`.
- `bash codex/gpu_ab.sh --preflight --port 18129 --out build-tmp/np2-preflight` (CPU only).
- Parent through gpuq: `bash codex/gpu_ab.sh --canary --out FRESH_DIR`, then `--execute` into a different fresh directory. Neither GPU stage was run here.

Evidence summaries/tails/hashes: `codex/mask-np2-evidence/`. Full logs and binaries: worktree `build*` directories. The following topic commit records tools, report, experiment, and frozen manifests without another production fix.
