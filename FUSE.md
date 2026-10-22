# Decode AllReduce residual RMSNorm fusion

Base: `6ef25f572c544b705d3fe7a37c02978bc2d1aa56` (package 101 fork). This is a local engine experiment, with no new ggml operation or default-path environment switch.

## Node chains established from source

`src/models/qwen35.cpp:llama_model_qwen35::graph::graph` and `src/models/qwen3next.cpp:llama_model_qwen3next::graph::graph` build the following chains. `src/llama-graph.cpp:llm_graph_context::build_norm` emits separate `RMS_NORM` and `MUL` nodes for RMS normalization with a weight.

| Boundary | Chain after the partial projection and AllReduce |
| --- | --- |
| Recurrent attention output (`ssm_out`) or full attention output (`wo`) | `ADD(cur, inpSA)` (`attn_residual`) -> `RMS_NORM` -> `MUL(attn_post_norm.weight)` -> FFN projections |
| Dense FFN output (`ffn_down`) | `ADD(cur, ffn_residual)` (`post_ffn`) -> optional control vector -> next layer's `RMS_NORM` -> `MUL(attn_norm.weight)` -> attention projections |
| MoE FFN output | Expert weighting/reduction and the shared branch can stay partial until their sum; then `ADD(cur, ffn_residual)` (`post_moe`) -> optional control vector -> next layer's `RMS_NORM` -> `MUL(attn_norm.weight)` |
| Last FFN output | Residual `ADD` -> optional control vector -> `RMS_NORM` -> `MUL(output_norm.weight)` (`h_nextn`, `result_norm`) -> optional output-row crop -> LM head |

`ggml/src/ggml-backend-meta.cpp:ggml_backend_meta_graph_compute_impl` constructs per-rank graphs ending at `PARTIAL` nodes. Its delayed-reduction logic can carry a partial through expert weights/views/sums and combine independent partial branches. Execution submits those subgraphs to `ggml_backend_comm_graph_compute`, or computes a subgraph followed by a separate collective on fallback. AllReduce itself is not a ggml graph node.

The last-layer attention crop can put `GET_ROWS` between a projection and the residual add. An active control vector can put an extra `ADD` before the next normalization. These chains do not match. The LM head does not have a following residual normalization. The table is a source-derived pattern, not a runtime graph trace or an assertion that every collective matches.

## Existing fusion and selected pattern

`ggml/src/ggml-cuda/ggml-cuda.cu:ggml_cuda_can_fuse` / `ggml_cuda_try_fuse` already select `RMS_NORM + MUL`, optionally with a following bias `ADD`, and the Q/K `RMS_NORM + MUL + ROPE` variants. The bias add is after normalization; it is not the residual add. This base has an `ADD_RMS_NORM` test but no CUDA residual-add fusion dispatch.

`ggml_cuda_match_ar_norm` matches the next subgraph's prefix `ADD + RMS_NORM [+ MUL]` against the preceding AllReduce tensor. It uses `ggml_can_fuse_subgraph` with the residual add and final result as materialized outputs. The add can have later consumers. An observed or multiply-used norm remains materialized by selecting the two-node prefix and leaving the multiply in the graph.

The matcher accepts contiguous, same-shaped F32 tensors, widths 256 through 8192 divisible by eight, and 1 through 16 rows. A fused weight must be a contiguous F32 vector of the row width. It rejects shifted overlaps, output/weight overlaps, broadcast residuals, and aliasing between the AllReduce input and residual. Exact row-aligned in-place output aliases are safe because a whole row is loaded before the epilogue writes it.

`ggml_backend_cuda_comm_graph_compute` selects matching prefixes on both ranks, with equal row geometry and no concurrent stream regions, only while recording the existing internal P2P collective graph. Warmup, disabled graphs, host staging, copy-engine collectives, HIP/MUSA, large batches, unmatched prefixes, and asymmetric ranks keep their original dispatch. `GGML_CUDA_DISABLE_FUSION` also disables this match, using the existing backend switch.

`ggml/src/ggml-cuda/allreduce.cu:ggml_cuda_ar_kernel<..., true>` implements the epilogue. All eight existing epoch/arrival slots advance, including blocks without rows. Each block stages and consumes complete rows `blockIdx.x + k*8`; this changes ownership without changing the epoch count. Both ranks select the same ownership before any work is captured. Rows assigned to one block need no inter-block reduction or new scratch allocation. Dynamic shared memory holds one residual row and the block-reduction scratch (at most 32896 bytes).

The kernel still writes the reduced tensor and the residual-add tensor to honor existing consumers and allocation lifetimes. It computes the norm and optional weight multiply from shared memory. The capture evaluator skips the matched prefix without modifying the cgraph, so property snapshots still include every skipped node, its sources and epsilon. Activation quantization for subsequent matmuls and the Q8 cache are unchanged.

## Passes and arithmetic

For a matched weighted chain, the base launches AllReduce, ADD, and the already-fused RMSNorm/MUL kernel. The patch launches one kernel: two standalone passes and two launches removed per matched collective per device. For an unweighted chain the same count applies. The AllReduce and residual output stores remain; the ADD's read of the reduced tensor and the norm's read of the residual-add tensor are removed. No saving is claimed for unmatched collectives or activation quantization.

Wire rounding, inactive-shard zeroing, and the F32 two-rank sum remain unchanged. The kernel computes `(wire(local) + wire(peer)) + residual` in F32 before accumulating squares. The F32 square-sum order changes: 256 threads accumulate wire-vector groups and use `block_reduce`, rather than the standalone RMSNorm kernel's vector/block configuration. Compiler FMA choices can also change the square-sum rounding. Results are not promised bitwise equal. The numerical gate is normalized MSE <= `1e-7`, the default `test-backend-ops` tolerance; model-level KLD is required before judging the optimization acceptable.

## Verification and queued work

`tests/test-ar-p2p.cpp:norm_cases` adds 15 split-versus-collective cases, with fractional inputs and a positive/negative fusion-log check. Shapes include 5120 x {1,6,8,12,16}, width 256, and 8192 x 16. Cases cover optional weights, swapped ADD inputs, in-place ADD, an observed norm, live residual outputs, changed inputs, epsilon recapture, and an inactive rank. Negative cases cover 17 rows, width 260, transposed and broadcast residuals, asymmetric prefixes, and equal-element-count/different-row-shape ranks. Both the reduced and added tensors and the final output are checked; observed norms are checked separately.

The standard graph test includes these cases with BF16 wire. `test-ar-p2p --norm-only` permits a separate native-F32-wire run without the package 101 multi-chunk suite's BF16 requirement. New test calls use exported APIs or existing static inline op-parameter accessors; graph views copy the existing internal struct without calling non-exported helpers.

Host syntax checks passed with `g++ -std=c++17 -Wall -Wextra -Werror -fsyntax-only` for the test with and without `GGML_TEST_CUDA_GRAPH`, and for the extracted host matcher with the real ggml headers and a stub for the existing environment helper. No CUDA compiler, built executable, test, or GPU tool was run. CUDA compilation/linking, runtime tolerances, exact matching on Swift's allocated graphs, peer replay behavior, and performance remain unverified. In particular, using fewer active communication blocks at very small row counts may lose PCIe throughput; only the queued A/B can establish the tradeoff.

`item/ar-norm-fuse.{toml,sh}` drafts the gated CUDA 13.2 build, package 101 gates, both wire-mode norm gates, and freeze. `item/arnorm-depth-iq4.toml` and `item/arnorm-np2-real-iq4.toml` draft the frozen package 101 versus fused A/B. Their questions require KLD because cross-arm token identity is not a numerical correctness gate for this change. These are queue drafts only; no queue, freeze, or benchmark was executed.
