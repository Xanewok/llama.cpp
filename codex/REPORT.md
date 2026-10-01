Current scope work: [REPORT-scope.md](REPORT-scope.md). The report below is the supplied historical upstream preparation report.

# CUDA FA q8_0 K/V upstream preparation

Prepared locally on `topic/fattn-mma-quant-kv-rebase`, based on `origin/master` = `05af0d2b1`. GPU validation of this rebase is pending. No PR text or submission was produced.

## Scope and commits

- `22e5c4bcf`: kernel, including the original shared-memory store permutation fix.
- `94cceb8b8`: evaluation and performance cases in the existing `tests/test-backend-ops.cpp`.
- Diff: 6 files, +289/-31 lines (D, Git diff). No Swift sampling, scheduler, meta-backend or conversion prerequisites were carried over.
- `GGML_CUDA_FA_MMA_Q8_0` is absent from committed sources. The original knob branch, `topic/fattn-mma-quant-kv`, remains at `ee6afe4de`.

The replay had no textual conflicts. The test constructor had a fork-only final argument; upstream's existing stream dimensions preserve the two-stream case without it. Cleanup reused `ggml_cuda_is_aligned` and shortened comments. The dequant helper now has the same `CP_ASYNC_AVAILABLE` guard as the raw-load helper; this excludes its CUDA intrinsics from unsupported compilation paths.

## Design and supported cases

`ggml/src/ggml-cuda/fattn.cu:154` gates raw q8_0 dispatch. The existing GQA selection at `fattn.cu:312` must select `ncols2=8`: masked attention, no ALiBi, padded KV length, compatible non-quantized strides, GQA greater than 4. K and V must both be q8_0 with head size 256, query batch 3-8, no logit softcap, aligned base pointers and higher strides, and no selected sparse gather. These are source contracts (D), not new GPU measurements.

`fattn-mma-f16.cuh:502` stages complete raw rows with `cp.async`. `:532` reconstructs signed bytes and multiplies by their f16 scale. K and V share one raw staging buffer; waits and block barriers protect its reuse (`:736`, `:1095`, `:1465`). The lane permutation at `:544` retains the original store-conflict fix. The existing MMA and fixup arithmetic remains in place.

Raw staging adds 8,704 bytes per block (D: 32 rows x 272 bytes). Dynamic shared-memory reservations are 41,792/42,112 bytes for the two instances (D, host formulas at `:2177`), versus 33,792 bytes for their f16 controls. Global f16 scratch remains reserved (`fattn.cu:768`); this change skips conversion work, not scratch allocation.

The exact scalar contract is `half(q) * half(d)`, rounded once, matching `convert.cu:77`. This establishes finite-value dequantization equivalence, not whole-graph token identity. `fattn-common.cuh:1136` queries occupancy and chooses stream-K partitions; resource changes can change floating-point reduction order on another GPU.

## Evidence

M = measured in this session unless explicitly marked prior; D = derived; G = guess. No new GPU throughput measurement was made.

| Check | Result | Evidence |
|---|---|---|
| CPU FA reference | M: 57/57 new cases passed; 11.37 s elapsed | `cpu-fa.log`; optimized CPU vs reference CPU |
| CPU build | M: passed, zero compiler warning lines | `cpu-config.log`, `cpu-build.log`, `cpu-libraries.log` |
| Scalar and packing model | M: 16,252,928 finite-scale/sign-byte products match f16 bits; 256 packed pairs match | `dequant_audit.py`, `dequant-audit.log`; incorrect bias fails |
| CUDA build | M: master comparator and final patch targets compile; zero compiler warning lines | `master-cuda-build.log`, `cuda-build.log`, `cuda-guard-build.log`, `cuda-final-build.log` |
| Other NVIDIA architectures | M: dispatch and both new instances compile for sm_75/80/86/89/90 with `-Werror=all-warnings` | `arch-summary.log`, `arch-commands.jsonl`, `compile-*.log`; compilation only |
| Generated instances | M: both new files match generator output | generation in `build-generated`, `cmp` passed |
| Parent harness | M: shell syntax, frozen hashes and CPU dry run pass | `gpu-dry-run.log`; output, repeat-boot and incomplete-perf negative controls |
| Original fork evidence | M (prior, parent brief): FA 4053/4053; q8_0 kernel 1283 -> 279 us at 98k/batch 4; f16 492 us; served output 11/11 identical | Not rerun here; does not establish performance or identity on this rebase |

Compiler resources below are M from CUDA 13.2.86 ptxas, sm_120a. The requested CMake architecture `120` is normalized to `120a` by upstream CMake.

| K/V type | ncols1 | Registers | Stack bytes | Spill store/load bytes |
|---|---:|---:|---:|---:|
| q8_0 | 4 | 255 | 40 | 44/32 |
| q8_0 | 8 | 255 | 128 | 296/296 |
| f16, master and patched | 4 | 252 | 16 | 0/0 |
| f16, master and patched | 8 | 254 | 56 | 40/40 |

See `ptxas-sm120.txt`, `registers.json` and `compile-master-f16-ncols1_*-sm120.log`. The other active q8_0 targets also use 255 registers (M); spills vary by architecture. These counts are a reviewer risk, not a speed prediction.

## Reproduction and parent queue

Commands used `CUDA_VISIBLE_DEVICES=""` and CPU priority 19; builds used `-j8`. After the initial cache failure, builds set a worktree `TMPDIR` and disabled ccache. The CPU test runner has no `-t` option: a build-local interposer reports hardware concurrency eight, limiting its backend and quantization workers to four; the shipped sources are unchanged by this cap.

CPU configuration used `cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=OFF`, Release and tests enabled. Build target: `test-backend-ops`, `-j8`. Test command:

```sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 LD_PRELOAD="$PWD/build-cpu/cpu-thread-cap.so" nice -n 19 build-cpu/bin/test-backend-ops -b CPU -o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,nh=[24],nr23=\[(6|8),[12]\],kv=(512|1024|4096),nb=[1-8],.*type_K=q8_0' -j 1
```

`prepare_gpu.sh` records the CUDA flags and freezes application executables and DSOs under `frozen/{up,patch}`. Both snapshots are ready. Master production files were checked against every archived Git blob; its only overlay is the identical test matrix. CUDA targets built were `test-backend-ops` and `llama-server`; neither CUDA executable was run.

Parent command, **not executed here**:

```sh
CUDA_VISIBLE_DEVICES="" nice -n 19 bash codex/gpu_check.sh --run
```

Default invocation is CPU-only dry run; `gpu_ab.sh` is an alias. Correctness and interleaved `up patch patch up` performance runs use one GPU, with 48 matched cases per run (D). Served identity uses both GPUs, `patch up patch`, seeds 1-4, fresh/follow-up tool chats, repeated greedy output and a deep prompt: 11 requests per boot (D). It compares finish status, reasoning, content and raw tool arguments, rejects missing work and unstable repeated boots, and writes paired timings to `perf.tsv`. Servers execute as `fa-check-srv`; cleanup stops only the captured child PID. The script checks Xid, occupied memory, port, model size and hashes, with a configured wall budget and bounded request timeouts. Outputs and runtime caches stay inside this worktree.

Upstream lacks the fork's probabilistic-draft flag, so this served check uses upstream MTP behavior. The prior fork's probabilistic identity evidence remains separate.

## Reviewer questions and remaining work

- **Why head size 256 only?** Its raw row is aligned for the existing 16-byte transfers. Smaller common heads need another transfer/layout design; head size 512 uses a different pipeline. Do not broaden without kernel validation.
- **GQA and batch limits?** Dispatch supports GQA greater than 4; added cases cover ratios 6 and 8. Batches 1-2 retain vector kernels on Ada and newer; Ampere batch 2 retains conversion. Wider batches amortize conversion and remain unchanged. Missing explicit new-path boundaries include other GQA ratios, misalignment, unmasked/ALiBi and sparse fallback cases.
- **q4_0?** It requires separate nibble reconstruction and validation. Mixed q8_0/f16 remains on the existing path. No unmeasured quantization generalization was added.
- **Other backends and GPUs?** Turing and older retain fallback. Ampere, Ada and Hopper compile, but correctness, occupancy, short-context performance and speed remain unmeasured there. HIP/MUSA dispatch is excluded; their toolchains were absent, so shared-header builds remain unverified.
- **Before submission:** parent must run the frozen GPU checks, obtain HIP/MUSA build coverage, inspect short-context regressions and spills, and perform human review/disclosure. The parent identifies #29371 as the documented need; current issue and duplicate-PR status were not checked offline. All public text belongs to the user.

## Local Git and preparation limitations

The original `.git` link pointed outside the permitted write tree. This worktree now uses `codex/git`, with the original object store read through an alternate. Original shared refs were not modified. `original-gitdir` preserves the link; `upstream-fa.bundle` contains the final topic ref and requires the master base. Import the bundle to use these commits in another worktree.

Initial Git reads attempted automatic promisor fetches and failed; no download succeeded. Subsequent Git operations disabled lazy fetching and all transports. An initial ccache write failed against the read-only cache; ccache was then disabled. Editing an active preparation script interrupted freezing after successful builds; preparation was restarted, and final source IDs and hashes were verified. None of these failures requires a source change or weakens the recorded CPU/compile checks.
