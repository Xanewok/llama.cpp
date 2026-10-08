# Stack10 candidates

## Conclusion

Package 54, K3 and K5 are implemented. Mac CPU gates passed. K4, K2 and K7 are withheld. The parent-only window builds CUDA, checks the changed operations on both GPUs, freezes, renders the items and calls `gpuq submit --check`; it never submits. CUDA compilation, GPU exactness and performance remain unverified. No GPU executable, Linux compile/model test, queue submission, cancellation, download or network posting was performed by this worker.

Labels: M measured, D derived, G guess. File lines and commit hashes are identifiers. CPU wall times include loading, logging and RAM waits and are not performance A/B results.

## Commits

Base: `cd776e104dbd516c7b74506992d2e8efd7600155`, the source of `frozen/stack9`.

| Commit | Subject | Component |
|---|---|---|
| `a1fff47fc` | `cuda : add grouped one-row q8_0 flash attention` | Package 54 / K1, including multi-stream tests. |
| `758c93a1c` | `llama : retain attention rotations on the cache backend` | K3, allocation fallback and matrix-byte probe. |
| `7f0cb4515` | `cuda : fuse F32 add softplus and broadcast multiply` | K5, NVIDIA guard and same-backend byte test. |
| `efb8115e9` | `local: add stack10 component controls` | Independent off switches, numerical mutants and CUDA engagement logs. |
| `a74a42c77` | `codex : prepare stack10 CPU gates and GPU window` | Mac gates, harness snapshot, window and item templates. |

Evidence/report commits follow these. Only this topic branch was consolidated. Git tree equality preserved the tested final tree; `git diff 239f830af HEAD -- src ggml tests common` is empty. The Mac witness commits retain their original hashes in the evidence, rather than being relabelled as executions of the consolidated history. No attribution trailers were added.

## Design, exactness and conflict resolution

**K1.** Applied the root series from `patches/upstream/llama.cpp/54-grouped-q8-one-row-fa/`. Conflicts in `fattn-mma-f16.cuh`, `fattn.cu` and `generate_cu_files.py` overlap stack9's existing q8 verify implementation. Kept the existing verify selector, instances and scratch allocation; merged the package's narrow-row loader, partial-tile converter, staging-cost gate and one-row instances. The generator emits both instance sets. No model, GPU-name or context blacklist was added. Eligibility and allocation use the same one-row predicate (`ggml/src/ggml-cuda/fattn.cu:182`, `:939`). CPU reference cases passed; prior package evidence is not a GPU result for this integration. The window requires q8 correctness and engagement on each GPU and proves the disabled route has no grouped marker.

Draft KV quantization changes proposal probabilities and can change acceptance and tokens. The flags-only A/B uses one binary and retains rejection verification. Grouped attention can also engage on a target one-row decode with q8 target KV, so the exact np1 item explicitly sets the local `GGML_CUDA_Q8_DECODE=0`. It does not assume that f16 draft KV alone excludes K1.

**K3.** A cache-owned context and buffer retain the same precomputed F32 matrix bytes on the cache backend (`src/llama-kv-cache.cpp:367`). Both builders return persistent leaves (`:1538`, `:1564`); both setters skip those leaves (`:1999`, `:2013`). The buffer is separate from `ctxs_bufs`, so cache clears do not zero it. Uniform buffer ownership, including Meta, selects persistence; mixed layer buffer types, no-allocation contexts and failed allocation retain graph inputs. The Meta allocator previously asserted on a failed child allocation; it now releases the partial allocation and clears tensor bindings before returning failure (`ggml/src/ggml-backend-meta.cpp:1964`). Actual device OOM was not induced; the local allocation-failure control tests the caller fallback.

Raw same-batch logits are byte-identical under persistence, graph inputs and simulated allocation failure, with both cache-fill patterns and the inherited rollback/swap/extra-state scenarios. Plain and Meta(CPU,CPU) np2 tokens/counters also match. CPU uses an FWHT shortcut that ignores matrix bytes, so numerical identity alone cannot validate an upload. The added probe checks the allocated matrix against the Hadamard reference (`tests/test-recurrent-state-rollback.cpp:328`, `:469`); zeroed uploads fail that check. GPU lifetime, replay and matrix consumption still require the queued validation. D: a head width of 256 retains 344064 bytes per cache/backend slice because the precomputed 64, 128 and 256 matrices are all retained; this is not a measured VRAM delta.

**K5.** Only the alpha chain is fused. The matcher accepts contiguous F32 inputs with row-broadcast bias and scale, checks graph consumers and output memory ranges, and preserves the unfused graph for other layouts/types and HIP/MUSA (`ggml/src/ggml-cuda/ggml-cuda.cu:3653`). The kernel uses the existing `op_softplus`, `__fadd_rn` and `__fmul_rn`, preserving the add and multiply rounding boundaries (`ggml/src/ggml-cuda/unary.cu:663`). Beta and GDN reductions are unchanged. The native operation test compares fused and output-flagged unfused branches on the same backend, byte for byte, while changing inputs across repeated evaluations (`tests/test-backend-ops.cpp:12373`). D: 24 graph configurations and 96 evaluations cover broadcast widths, multi-dimensional rows, zero/threshold neighbours, infinities and underflow. The CPU gate validates the test/reference paths; it does not prove CUDA fusion exactness. The window requires the CUDA byte test and an engaged numerical mutant failure on both devices before freeze.

**Withheld components.** These are unresolved implementations, not claims that the optimizations are impossible.

- K4: `build_conv_state` uses the concatenated input for rollback snapshots (`src/models/delta-net-base.cpp:463-520`); `ssm_conv_state_f32` reads snapshots from that input (`ggml/src/ggml-cuda/ssm-conv.cu:250-260`). Merely adding ids leaves snapshot semantics incomplete. Separate inputs need cross-sequence read-before-write protection for overlapping/swapped ids, extra-state ordering, CPU semantics, Meta slicing and backend fallback. That complete vertical slice and its required state tests were not established here.
- K2: the scan's pool allocation is local to a launch (`ggml/src/ggml-cuda/fattn-common.cuh:1163`). Sharing it requires an evaluation boundary, alias-write invalidation and ownership lasting through every retained CUDA graph. A pointer/shape cache without those contracts would reuse stale masks or free replay operands. No capture-safe implementation was established.
- K7: `build_gdn_l2_norm` is RMS norm plus scale (`src/models/models.h:14`), with reduction and scaling in `ggml/src/ggml-cuda/norm.cu:77`. A paired launch needs a multi-output matcher and identical row/reduction mapping for the separate q/k views. Concatenating or folding into GDN does not establish byte identity. No validated paired implementation was established.

## Every source change

- `ggml/src/ggml-cuda/fattn-mma-f16.cuh`: merged q8 row loading, partial conversion, head-pair support, shared sizing and instance declarations; existing verify arithmetic retained.
- `ggml/src/ggml-cuda/fattn.cu`: cost/eligibility predicate, one-row dispatch/allocation, local grouped off switch and engagement marker.
- `ggml/src/ggml-cuda/template-instances/generate_cu_files.py`: retains verify generation and adds decode generation.
- Four new instance files: `fattn-mma-f16-instance-q8_0-ncols1_1-ncols2_{8,16,32}.cu` and `fattn-mma-f16-instance-q8_0-ncols1_4-ncols2_4.cu`.
- `src/llama-kv-cache.h`, `.cpp`: rotation owners/leaves, uniform-buffer allocation, both builder/setter paths, fallback and local off/failure/zero-matrix controls.
- `ggml/src/ggml-backend-meta.cpp`: child-allocation failure cleanup needed by K3 fallback.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: guarded alpha matcher/dispatch, alias checks, local off switch and marker.
- `ggml/src/ggml-cuda/unary.cu`, `.cuh`: rounded alpha kernel/launcher/declaration, NVIDIA-only intrinsic guard, local numerical mutant.
- `tests/test-backend-ops.cpp`: package q8 cases, two-sequence/cache-stream cases, same-backend alpha byte/replay check and hook.
- `tests/test-recurrent-state-rollback.cpp`: matrix-byte probe and quantized-KV engagement requirement. The incidental compile fix reads the matmul hint with `memcpy` from `op_params + 1`; the initially used internal accessor was undeclared.
- `docs/ops.md`: documents the alpha fusion and fallback.

Other edits are confined to `codex/`: supplied prompt/purpose, GPU entry, bounded Mac gate, Mac payloads, minimal copied harness dependencies, copied shlint module, window, templates, renderer, evidence and reports. No other worktree, frozen tree, shared executable or research repository was modified.

## Mac evidence

Primary witness: `239f830af2d6833189fb0e3f98fbf82aa5f2c1a0`, `build-bw-mac-results/stack10-cand-mac-a4/`. Recorded-parser follow-up: `bf2926e72b22812b6801452fc249be9ff41e24df`, `build-bw-mac-results/stack10-cand-static-mac-a2/`. Final payload witness: `594675edd98c8bbcca18a59e09e98c6629476ae3`, `build-bw-mac-results/stack10-cand-static-mac-a3/`. It has the same production-source digest and records the final payload hash in `codex/stack10-evidence/summary.json`.

```sh
CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 BW_MAC_OUTPUT_BASE="$PWD/build-bw-mac-results" ~/repos/qwen38-eval-worktrees/bin/bw-mac "$PWD" codex/bw-stack10-cand-mac.sh stack10-cand
CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 BW_MAC_OUTPUT_BASE="$PWD/build-bw-mac-results" ~/repos/qwen38-eval-worktrees/bin/bw-mac "$PWD" codex/bw-stack10-cand-static-mac.sh stack10-cand-static
```

CPU build: Clang, CUDA/Metal/BLAS/Accelerate/repack/llamafile off, native off, `armv8.2-a+dotprod`, ccache ON, nice 19, `-j4`. Model processes use explicit small contexts and at most four threads; the RAM/busy wait and each process runtime both fail after ten minutes. Qwen3.5-4B UD-Q4_K_XL MTP fixture. No Linux CPU build was run.

| Stage | Wall seconds (M) | Result |
|---|---:|---|
| 20 CPU build | 54 | PASS |
| 21 sampling, allocation, alpha and q8 operation tests | 5 | PASS |
| 22 rotation identity/fallback/negative control | 270 | PASS |
| 31 primary recorded parser controls | 1 | PASS |
| 31 dtype/proof/counter/parser follow-up | 1 | PASS |
| 31 final payload replay and shell lint | 1 | PASS |

M: 18 q8 attention cases passed, including two sequences and a non-unified cache view. M: each raw-logit artifact is 123169696 bytes; comparison is streaming and no artifact over 200 MB was loaded. M: zeroed persistent matrices fail; off/on/allocation-failure raw logits match; all three arms' complete np2 token/counter JSON matches for plain and Meta. Logs and source provenance are in `codex/stack10-evidence/`.

Superseded gates are retained in `superseded-stages.tsv`: the initial zero-upload numerical mutant escaped because FWHT ignores the matrix; the next build exposed the undeclared accessor; the broad shell gate found existing release-script violations; the first dtype replay used an unavailable remote log path. Corrected gates passed. Full-repository shlint findings remain in unchanged `scripts/{build-profile,check-release-apiabi,check-requirements,make-release-checks,make-release-desc}.sh`. All `codex/` shell payloads pass the copied `python3 -m gpuq.shlint` from `~/repos/gpuq`; local `bash -n` and Python compilation also pass.

## Window and expected outcomes

Window: `codex/bw-stack10-cand.toml`; payload: `codex/bw-stack10-cand.sh`. Filler, priority 19, host-insensitive, non-preemptible. It checks production-source and final-payload hashes against the Mac evidence, requires the parent parser bytes to match the tested snapshot, refuses a dirty tree/existing freeze, compiles under buildgate `-j4`, checks the GCC 13 compiler stamp and full-drain `DeviceTopK` count, runs changed CUDA cases and controls on both devices, then freezes as `frozen/stack10-cand`. Recipe: CUDA 13.2, architecture `120a`, NCCL OFF, offline CCCL 3.4.3, ccache ON. UUIDs stay hidden during build; only the explicit operation stage exposes them inside this gpuq item. G: 3000-second estimate; D: 9000-second cap. The parent must queue the window; it was not executed here.

Renderer: `codex/render_stack10_items.py`. It creates/checks `codex/items/rendered/*.gpuq.toml` after the freeze exists. These rendered paths cannot be checked against frozen bytes before parent execution. Every run/canary receives the pinned `bash .../work/aa-load/ccx01.sh` prefix and `host_sensitive = false`; the reused harness supplies `--cache-ram 0`, joint np2 start, exact-PID stop, frozen/library checks, scoped journal faults and boot budgets. Canaries use one boot and a ten-minute cap, real prompts, verbosity 5 and component/draft-cache log proofs. Timing is untraced.

| Template | Priority | Arms/outcome and falsifier |
|---|---:|---|
| `codex/items/s10c-np2-real-j5q.toml.in` | 27 | Stack9 vs stack10 plus independent no-rotation/no-alpha/no-grouped arms, two interleaved boots each, served duo k5/ub128, 8k/32k/90k. Exact arms must have matched per-stream decode counters or the result fails as insufficient data. G: a small depth-flat K3/K5 gain; a null beyond between-boot spread falsifies usefulness. No additive gain is claimed. |
| `codex/items/s10c-q8dkv-np2-real-j5q.toml.in` | 26 | One binary, f16/q8_0 draft KV flags only, ABBA, same real depths. Reports acceptance and D tokens/round as well as time. G: q8 helps at depth only if acceptance and request time compensate for quantization. A tokens/round drop above G: 3 percent, or no net request-time gain, falsifies adoption. |
| `codex/items/s10c-depth-j5q.toml.in` | 26 | Stack9 vs stack10 np1 ABBA, f16 draft KV and grouped q8 explicitly disabled in the candidate. Cross-arm tokens/counters must match. Any mismatch falsifies this integration's identity claim. |

The harness retains untrimmed counter-derived periods and exclusion reasons, with no median-based trim. Boot spread is the uncertainty unit. K1's grouped off control and K5's numerical control are queued, not observed GPU evidence. K3's extra resident bytes, true device-allocation failure, CUDA capture/replay, GPU engagement/performance and all withheld components remain open.
