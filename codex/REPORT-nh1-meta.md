# NH1 Meta planner correction

Labels: M measured, D derived, G guess. Worktree `nh1-fold-c`, branch `topic/nh1-fold-c`, starting commit `cd4a9d49a`.

## Cause and design

The fold ends with draft K/V writes at `src/models/qwen35.cpp:340`. `llama_kv_cache::cpy_v` (`src/llama-kv-cache.cpp:1396`, FA path) returns `ggml_set_rows`, which aliases the persistent cache through `view_src` (`ggml/src/ggml.c:4027`). The cache has `op == GGML_OP_NONE`.

`Meta(CPU,CPU)` buffers report as host buffers because all constituent buffer types are host buffers (`ggml/src/ggml-backend-meta.cpp:346`). Before this fix, the subgraph-boundary loop at `:2463` skipped every node whose `view_src` was a host-buffer leaf. It therefore skipped the final folded `SET_ROWS` before the end-of-graph boundary check at `:2471`. The last subgraph was never closed, causing `i_start == cgraph->n_nodes` to fail. The nodes were still mapped to per-device tensors; this filter affects planning, not their inclusion in execution subgraphs. This explains why ordinary graphs with cache stores followed by attention/head nodes did not abort.

The node-mapping loop at `:2289` already excludes Meta buffers from its host-view workaround. Apply the same exclusion to the boundary loop and the delayed-AllReduce branch scan at `:2423`. Both inconsistent filters already exist in base `68d99c67d` (`ggml/src/ggml-backend-meta.cpp:2216`, `:2255`). This is a general backend correction: CPU Meta views, copies and row writes must participate in split-state analysis. A dummy terminal fold node would hide the assertion while preserving the inconsistent filter. No fold graph, sampling, scheduler lifetime or CUDA implementation changes are needed. The backend correction and regression test are committed separately as `376e5c8d9`.

## CPU verification

M: the new terminal-view regression aborts with exit 134 against the original library at `ggml-backend-meta.cpp:2500`, matching the parent's folded-graph failure. M: all 40 allocator tests pass after the fix, including the six new Meta cases. Logs: `codex/nh1-meta-results/test-alloc-original.log`, `test-alloc-fixed.log`. The CPU server target and `fold-cpu` were rebuilt using the exact CPU recipe from `codex/bw-nh1.sh`, without executing that payload's CUDA/freeze/item stages; build log: `codex/nh1-meta-results/build-cpu.log`.

NH1 model identity is pending a CPU window. The initial GPU measurements finished and the CPU rebuild/tests ran, but further GPU measurements (`host-trims-v2-np2-iq4-same`, followed by `host-trims-v2-np2-iq4-stack6`) were queued before model verification could start. A separate queued CPU model fixture also occupied the one-model slot during the earlier gap. Builds and tests passed through `cpugate`; every compile additionally passed through `buildgate`, with `nice -n 19`, at most eight CPU threads, and `-j4` for CMake builds.

Executed build recipe: `cpugate buildgate cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF`, then `cpugate buildgate cmake --build build-cpu --target llama-server -j4`, then the `g++` fixture command from `codex/bw-nh1.sh:50` through both gates. Gate executables: `/home/xanewok/repos/qwen38-eval-worktrees/bin/{cpugate,buildgate}`. Allocator test: `cpugate build-cpu/bin/test-alloc`.

With the build-window recipe disabling CMake tests, the allocator executable was compiled directly: `cpugate buildgate g++ -std=c++17 -O2 -pthread -DGGML_SHARED -Iggml/include -Iggml/src tests/test-alloc.cpp -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lggml -lggml-cpu -lggml-base -o build-cpu/bin/test-alloc`.

Next required verification (not yet executed): `CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 codex/fold_cpu_identity.py codex/nh1-meta-results/cpu`, which runs every model boot through `cpugate`, with explicit `-c 4096`. It must pass all plain/Meta k3/k5 identity and engagement checks, the repeat baseline and the incorrect-row mutant. Afterwards run `cpugate python3 codex/test_nh1_meta_engagement.py codex/nh1-meta-results/cpu/meta-k3-off` and the corresponding k5 control. No serving claim is made until these tests pass.

`tests/test-alloc.cpp:1125` adds terminal `VIEW`, `CPY` and `SET_ROWS` graphs on mirrored and dimension-split CPU Meta buffers. Writes are checked byte-for-byte, including untouched cache rows, on direct computation and reuse of an explicit Meta graph plan.

`codex/test_nh1_meta_engagement.py` replays real Meta fold-off artifacts as an on arm with an initialization canary but no folds. M: replay of the parent's `nh1-a1/cpu/meta-k3-off` artifact passes identity but is rejected for zero folds over 282 draft rounds and unchanged C counts (330). Evidence: `codex/nh1-meta-results/control-meta-disengagement-recorded.json`. This tests the silent-disengagement failure independently of the incorrect-hidden-row mutant in `codex/fold_cpu_identity.py`.

## GPU risks and remaining work

GPU execution and speed are unverified. CUDA Meta device buffers report as non-host buffers, so these additional checks do not change their planner branches (source inference). The parent still needs the CUDA build and the existing frozen depth experiment after reviewing these changes.

The fold adds live consumers and draft cache writes to the target graph. CUDA fusion/capture, per-device hidden-state equality, draft-cache visibility across the target/draft backends, and AllReduce placement require GPU cross-arm identity and engagement checks. CPU Meta exercises split-state derivation and generic AllReduce, but does not establish CUDA graph behavior or internal CUDA AllReduce numerics. The fold remains restricted to its existing single-sequence, single-ubatch eligibility contract.

No GPU binary, whole build-window payload, freeze, gpuq submission or network operation was run. Project file edits stay in this worktree.
