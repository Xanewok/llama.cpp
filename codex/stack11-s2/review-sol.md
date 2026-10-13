| file:line | issue | evidence | severity | fix |
|---|---|---|---|---|
| `tests/test-host-mapped.cpp:112` | A later allocation fallback erases an earlier test failure. | `test_bitwise()` accumulates failures in `ok`, but returns unconditional `true` if a subsequent allocation falls back. This defect also exists in package 90's `master-abee0c8` form. | P2, test correctness | Return `ok` from the fallback branch. Preserve the distinction between skipped coverage and successful comparisons. |
| `tests/test-llama-archs.cpp:695` | The reset test does not exercise simultaneous np2 sequences or the device chain. | It uses only `seq_id = 1`, `n_max = 1`, and `backend_sampling = false`. `n_seq_max = 2` verifies indexing, not two active sequences. | P3, coverage | Add a fresh-plus-established batch and verify that the established carry survives; verify fold resumption afterward. |
| `tests/test-alloc.cpp:1492`, `tests/test-host-mapped.cpp:309` | Real two-GPU Meta integration remains unverified on this stack. | The Meta regression uses fake host memory over CPU backends; the CUDA test exercises each device separately. `MAC.md` explicitly excludes CUDA and real-device Meta. | Validation gap | In the authorized GPU window, verify placement and row bytes through Meta, including fallback, at np1/np2 for both profiles. |

**`713d8b9de`: OK.**

The carry reset preserves the package's production semantics, adapted correctly to flat `[n_seq][n_embd]` storage. It resets only sequences present in the catch-up batch whose **own draft memory** is empty. An idle empty slot does not affect another sequence. Non-empty continuations retain their carry, including subsequent prompt chunks.

The fold guard is needed with the reset located in `process()`: a successful fold writes draft K/V during target decoding and bypasses that catch-up branch. Without the guard, an armed carry from the previous request reaches the new request before it can be reset. The recorded `p87-nofold` and `p87-nospec` failures independently support both requirements.

`p0 == 0` correctly detects empty draft memory: `seq_pos_max()` returns `-1` for an absent sequence (`src/llama-kv-cells.h:386`). This tests memory state, not the incoming token position. An empty-memory restart at a positive position also falls back because the existing position-continuity check fails. An image chunk continuing at position zero retains its carry when memory is non-empty.

The refusal is **batch-wide**. If any participating sequence is fresh, all sequences use catch-up; `common/speculative.cpp:1620` zeros only the fresh sequences. The header's wording does not explicitly describe this batch-wide consequence, but the implementation is coherent.

G: an otherwise fold-eligible short fresh prompt can therefore cost an extra C decode. At np2, a fresh slot mixed with an established slot can also remove the established slot's fold for that batch. This is a request-boundary cost, potentially recurring with slot churn; it is not a permanent loss of folding. Large batches or incompatible layouts already fell back before this commit.

G: ordinary steady verification retains folding because draft rollback preserves the committed prefix (`tools/server/server-context.cpp:3536`). The np1 device chain is unchanged and receives the updated carry. At np2, `draft_chain()` already requires `n_seq == 1`; folding C remains independently available.

**`a2f920c53`: OK with fixes.** The required fix is the test's unconditional-success return; I found no production correctness defect in the reviewed paths.

The stack11-specific branch at `ggml/src/ggml-backend-meta.cpp:1298` is necessary and correct. It resolves foreign sources before `ggml_backend_meta_find_split_state()` casts their buffer context to a Meta context. It preserves source traversal order and avoids inserting foreign buffers into the Meta cache. The recorded `p90-nodriver` SIGSEGV supports this failure mechanism.

The two-argument resolver marks an ordinary foreign host weight `MIRRORED`, while preserving the existing host-view treatment. Meta accepts the mapped type only when every sub-device accepts it. When creating device tensors, it replaces Meta-backed sources but leaves the foreign table pointer intact (`ggml/src/ggml-backend-meta.cpp:1511`). Thus both GPUs read the same complete table; token rows are not partitioned or numerically reduced.

G: with successful mapping, token `GET_ROWS` is bit-identical to the device-resident path at np1 and np2 for J5q and IQ4_XS. `ggml_cuda_op_get_rows()` passes `src0->data` directly to the same type-dispatched kernel (`ggml/src/ggml-cuda/getrows.cu:457`); memory residence introduces no alternate arithmetic. Package logs record zero differing IQ4_XS elements on both GPUs, but those measurements do not certify this stack's integration.

The CUDA allocation uses portable mapped memory, checks pointer equality on every mapping-capable device, and restores the previous device. Allocation or mapping failure returns an actual CPU buffer, so the scheduler sees the fallback type. No reviewed path hands that ordinary CPU buffer directly to a CUDA kernel.

G: lifetime and alignment are sound under normal model/context ownership. The model retains the buffer; loading copies the original bytes synchronously; destruction uses `cudaFreeHost`. CPU tensor alignment satisfies the reviewed `GET_ROWS` loads, and those kernels need no CUDA allocation padding.

`GGML_CUDA_NO_PINNED` safely restores CPU gathering and loses the zero-copy speed benefit. The fallback test verifies F16 values and placement, not quantized row identity. G: finite IQ4_XS rows should also match because CPU and CUDA use the same scale reconstruction and multiplication sequence; this is not a universal CPU/CUDA exactness guarantee for every quantization type.

The loader retains package 90's restrictions: automatic token embeddings only, GPU first layer, no `--no-host`, no bitnet, and no duplicated output tensor. CPU overrides remain on the original CPU list.

This was a read-only source and recorded-evidence review. No binaries, tests, or GPU tools were executed. `G` labels conclusions inferred from code rather than measured on this stack.