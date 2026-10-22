# Package 101 reduction port

Upstream base: package 94 at `1ffc1dcf3`, above master `ff5a23376`. Fork base: `726f57ffe`, which already contains 94. Original 101 tips were `4f752eb85` and `5733c813f`.

The reduction removes both old optional procedures, the CUSTOM encoding, merged arena, timeout trap and standalone graph test. Its sole new optional procedure is `ggml_backend_comm_graph_compute`. The fork must retain owner-local comm lifetime, plan-local split graphs, prepare-only behavior, exact gather fallback and all existing input-staging procedures and environment aliases.

Both host syntax and whitespace checks pass with exit 0 and no diagnostics. The meta conflict was resolved against the fork base: comm lookup stays on `owner`, graph lists use plan-local `backend_ctx`, `prepare_only` returns before collective compute, and any gather boundary retains exact split execution. CUDA, headers, tests and CMake merged without source conflicts; the existing input-staging procedures and P2P alias remain. No tests, built executables, nvcc, GPU tools or builds are run. See `VALIDATION-101.md` for the exact syntax command and limits.
