# K9 fix: private read-only review

Not established safe. Q1 UNKNOWN, Q2 UNKNOWN, Q3 OK at source level, Q4 defect in proof coverage. Keep the upstream PDL restrict policy for producer-written inputs until ordering and non-coherent visibility are established separately.

Reviewed `117c87e14`, `153c8f868`, upstream `9e58d4d69` (#24030), and the window gates at HEAD `a5fcdd386`. Locations below refer to this worktree. Observed facts, inferences and unresolved claims are distinguished; M labels disassembly counts. No GPU execution, builds, model runs, network, commits or changes outside this file.

## Q1: Does the zero offset prevent every load before the wait?

**UNKNOWN end-to-end; the source dependency is present.** `ggml/src/ggml-cuda/common.cuh:147` exposes an opaque output register to nvcc/NVVM, but exposes the literal `mov.u32 ..., 0` to ptxas. Inference: the front end cannot ordinarily eliminate the address dependency through the opaque asm output, including loads from inlined helpers; the `memory` clobber alone is not a proof for loads treated as invariant.

The zero is not opaque to ptxas. Constant propagation or moving the register initialization can erase the artificial dependency; `volatile` is not a promise that the PTX statements remain an indivisible scheduling unit in SASS. Whether ptxas then treats `griddepcontrol.wait` as a sufficient barrier to `ld.global.nc` hoisting is unresolved here, so the unconditional claim at `common.cuh:142` is stronger than the evidence.

Concrete risk, not an observed failure: fold the offset to zero, schedule the first `y` load before the wait, and read quantizer output while the quantizer still writes it. LICM of an invariant load has the same consequence if its address no longer depends on the asm output.

Observed: no original-pointer escape or pre-offset derived pointer exists in these kernel bodies. `mul_mat_q_process_tile` receives updated pointers at `mmq.cuh:1077`, `:1171`, `:1255`; its `load_tiles`/`write_back` calls at `:917`, `:952`, `:954` forward those values. A helper's own `__restrict__` annotation does not independently recover an un-offset argument.

What decides it: inspect the exact fix cubins for the deployment toolchain and every supported specialization, checking that the wait dominates each dependent global access on normal, non-mutant paths. A passing cubin establishes ordering for that build; a general compiler guarantee requires documented compiler treatment of this idiom. The permitted stock/old-K9 objects do not contain the zero-offset fix, and this tree has no corresponding built object.

## Q2: Is post-wait LDG.CONSTANT guaranteed coherent?

**UNKNOWN; no basis here to certify that guarantee.** Under the question's whole-kernel-lifetime read-only requirement, producer-written inputs do not meet the precondition merely because their first source-level read follows the wait. `mul_mat_q` reads freshly quantized `y` at `mmq.cuh:924`; `quantize_mmq_q8_1` reads producer-generated `x` at `quantize.cu:501`, and potentially generated `ids` at `:490`/`:538`. Static weights are a different case: unchanged weight storage can satisfy the lifetime condition.

Observed local CUDA contract: `/usr/local/cuda-13.2/include/cuda_device_runtime_api.h:457` says grid dependency synchronization waits for direct dependencies to complete, and `:466` implements it as `griddepcontrol.wait` with a memory clobber. The trigger explicitly provides no visibility guarantee (`:444`). This header does not specify non-coherent L1/texture-cache invalidation at the wait; a compiler clobber cannot provide hardware cache coherence.

Concrete conditional failure: a line containing old `y` is populated in the non-coherent cache before the producer's final store, survives the dependency wait, and a post-wait `.nc` load hits it. An earlier alias load, prefetch, or hoisted load could populate such a line; whether launch/wait invalidation rules prevent that execution on this architecture is unknown, not a demonstrated stale read.

A producer sharing the SM and writing before the wait completes is legal PDL overlap, not by itself a failing scenario. A correctly executed wait cannot return while its direct primary grid is still executing; writes after it returns would require another unordered writer or a missing dependency. Residency overlap alone therefore does not prove a cache bug.

What decides it: an authoritative PTX/CUDA guarantee explicitly covering `ld.global.nc`/the read-only cache across `griddepcontrol.wait`, including pre-existing lines and prefetch. No local PTX ISA/Programming Guide copy was located; online specifications were not checked because network access is forbidden. A cache-stress test can refute safety but cannot establish a universal memory-model guarantee.

**#24030 does not distinguish the causes.** Observed diff: it removes kernel-parameter `__restrict__` and adds `GGML_CUDA_RESTRICT`, empty for PDL on Hopper and newer; it adds neither a cache operation nor an ordering test. That change can affect both invariant-load optimization and selection of the non-coherent read-only path. The commit describes incompatibility/races, not a demonstrated hoisting-only root cause; `common.cuh:143` should not be treated as upstream evidence of that diagnosis.

## Q3: Pointer coverage before first access

**OK at source level with `pdl_mutant=0`.** Every pointer parameter is reassigned before any dereference or helper forwarding. This is separate from Q1's machine-code ordering and Q2's visibility question.

| Kernel | Offsets | First access/forwarding sites |
|---|---|---|
| `mul_mat_q` | `mmq.cuh:981`: `x`, `y`, `ids_dst`, `expert_bounds`; `:982`: `dst`, `tmp_fixup`, `y_scale` | `expert_bounds` at `:1034`/`:1122`/`:1211`; `ids_dst` at `:1057`/`:1151`; tile calls at `:1077`/`:1171`/`:1255` forward all other pointers. Tile `x` load dispatch is `:917`, `y` load is `:924`, destination/fixup/scale write-back dispatch is `:952`/`:954` (scale reads in write-back at `:467`/`:517`). |
| `mul_mat_q_stream_k_fixup` | `mmq.cuh:1269`: `ids_dst`, `expert_bounds`, `dst`, `tmp_last_tile` | `tmp_last_tile` at `:1321`; ordinary `dst` read/write at `:1365`; MoE `expert_bounds` at `:1371`, `ids_dst` at `:1376`, `dst` at `:1397`. |
| `quantize_mmq_q8_1` | `quantize.cu:471`: `x`, `ids`, `vy` | `ids` at `:490` or scatter `:538`; `x4` derived at `:494` and read at `:501`; `y` derived from `vy` at `:495` and written through `yqs4` at `:547`. |

The earlier calls to `ggml_cuda_pdl_lc()` receive no pointer argument. Branch-specific derived pointers, including `y_scale_tile`, are formed after the offsets. The local mutant bits intentionally bypass the main/fixup waits at `mmq.cuh:975`/`:1268`; they do not change this source-coverage result for the normal arm.

## Q4: Do the window gates catch hoisting or stale reads?

**Defect if treated as a complete ordering/coherence proof; useful bounded checks otherwise.** `codex/k9fix_sass.py:57`-`:62` rejects a recognized global opcode listed before the first `ACQBULK`. It does not establish control-flow dominance: `parse()` discards predicates, operands and branch destinations at `:36`, so a branch around the wait or a false-predicated wait can pass; the normal path needs analysis separately from the intentional mutant paths.

The opcode classification is also partial: `GLOBAL` at `k9fix_sass.py:21` omits `UTMALDG`, although `STRICT` at `:20` recognizes it. The window selects particular MMQ types, tile widths and `fallback=false` prefixes (`bw-k9-fix.sh:130`-`:153`), not every specialization or the NVFP4 scale path. These are coverage limits, not evidence that those paths currently fail.

Observed parser check against real recorded disassembly: `SP/dry/base-mmq-iq4_xs.sass` and `SP/dry/k9old-mmq-iq4_xs.sass`, kernel prefix `_Z9mul_mat_qIL9ggml_type23ELi16ELb0EL9ggml_prec30EEv`. Old K9 is rejected: read-only LDG counts are stock 73 versus old K9 0 (M). In-memory mutations of that recorded stock opcode sequence show: adding an entry wait passes; moving an `LDG` before it fails; adding a `BRA.U` that can jump past it still passes because its target is discarded. No fixture files were changed.

`bw-k9-fix.sh:208`/`:210` compare printed output hashes, with replay mismatch detection at `tests/test-backend-ops.cpp:12959`-`:12965`. They catch output-changing hoisted or stale reads that occur in those executions, including deterministic wrong output versus stock; a matching run does not prove coherence for other cache states. SASS placement cannot detect a stale cache hit after a correctly placed wait.

The slow-producer controls at `bw-k9-fix.sh:221`/`:226`/`:229` test whether eliminating the main/fixup wait exposes a race. They do not create an isolated negative control for stale post-wait `.nc` reads or for quantizer wait removal. The chain tests ordinary MUL_MAT (`test-backend-ops.cpp:12916`/`:12917`); the separate MUL_MAT_ID suites do not extend that hash/mutant identity proof to generated MoE metadata.

## Evidence and limits

Read the supplied background reports, the full reviewed diffs, pointer forwarding helpers, `mmq-chain`, window script and SASS parser. Review applied scope, security, general, backend and design checks; the material issue is the unsupported correctness guarantee, not a style finding.

Commands: `CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 nice -n 19 git show REV -- ggml`; filtered `nice -n 19 /usr/local/cuda-13.2/bin/cuobjdump -sass -fun EXACT_MANGLED_NAME OBJECT`; single-process `nice -n 19 python3` with `PYTHONDONTWRITEBYTECODE=1` to load parser definitions via `ast`, parse recorded SASS and check in-memory mutations.

Filtered stock/old-K9 object inspection confirmed stock `LDG.E.CONSTANT` and old-K9 `LDG.E`; old-K9's wait is at SASS `0x0080` and its first global load at `0x1d00`. Objects: `/mnt/scratch/s10-k9/base/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-iq4_xs.cu.o` and the same relative path under `/home/xanewok/repos/qwen38-eval-worktrees/wt/s10-k9/build-cuda`. This is not evidence of the fix's generated code or GPU speed.

`SP` is `/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad`. Required next evidence is the exact fix SASS and an explicit non-coherent-cache PDL contract; a passing window alone leaves Q2 open.
