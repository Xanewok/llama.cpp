# GDN recurrent-state getter

The getter form is implemented on upstream master and stack6. CPU op equivalence and the row-0 negative control pass. CUDA builds pass; no GPU executable was run. GPU correctness, sanitizer findings and latency remain unknown until the parent runs the prepared items.

## Source and API

Upstream branch `topic/gdn-getter`, base `7049ff0cb`, source head `472791d90`:

- `fa1c8cb6c`: optional state ids, CPU/CUDA readers, backend guards and op tests.
- `5a91f7b87`: GDN layers use the recurrent-state getter; shared-sequence reserve enables rollback.
- `472791d90`: decode one copied child while the other is idle to require extra-state preservation.

`ggml_gated_delta_net(..., state, ids, K)` adds a nullable contiguous I32 vector with one id per output sequence (`ggml/include/ggml.h:2653`, `ggml/src/ggml.c:6375`). A NULL input preserves the prior contiguous input contract. With ids, the input pool can have more rows than the batch; CPU (`ggml/src/ggml-cpu/ops.cpp:11212`) and CUDA (`ggml/src/ggml-cuda/gated_delta_net.cu:46`) select the row and retain output sequence order. This extends the existing op like `ggml_ssm_scan`, without a new opcode or RPC version change. The C signature changes, so callers and libraries must be rebuilt together.

`src/models/delta-net-base.cpp:548` selects the getter only for rollback mode, KV offload enabled, and support on the cache device. It calls ordinary `build_rs` (`src/llama-graph.cpp:3587`), preserving its zeroing, tail gather and relocation. There is no `n_rs == n_seqs` condition. Rejected indexed inputs use the existing gather-then-plain graph with NULL ids. Non-rollback mode retains its prior arithmetic path.

The indexed reader supplies exactly the floats that GET_ROWS would copy; GDN arithmetic, snapshots and sampling are unchanged. This establishes equivalence of the state-selection transformation. End-to-end GPU token identity is still a gate in the A/B item, not a measured result here.

CUDA waits for the PDL dependency before reading ids. Indexed multi-sequence snapshot fusion is disabled (`ggml/src/ggml-cuda/ggml-cuda.cu:2828`): a sequence can otherwise overwrite a row another sequence still reads. Separate output and cache-copy nodes preserve ordering. Existing first-real-node matching also prevents fusion when an extra-row tail gather intervenes. This may cost multi-sequence performance; no speed claim is made.

## Backend rejection

Each listed `supports_op` returns false for `GATED_DELTA_NET` with `src[6]`:

| Backend | Location |
|---|---|
| Metal | `ggml/src/ggml-metal/ggml-metal-device.m:1539` |
| OpenCL | `ggml/src/ggml-opencl/ggml-opencl.cpp:8838` |
| HTP/Hexagon | `ggml/src/ggml-hexagon/ggml-hexagon.cpp:8302` |
| SYCL | `ggml/src/ggml-sycl/ggml-sycl.cpp:6457` |
| Vulkan | `ggml/src/ggml-vulkan/ggml-vulkan.cpp:15325` |
| WebGPU | `ggml/src/ggml-webgpu/ggml-webgpu.cpp:4381` |
| ET | `ggml/src/ggml-et/ggml-et.cpp:866` |
| OpenVINO | `ggml/src/ggml-openvino/ggml-openvino.cpp:1744` |
| RPC | `ggml/src/ggml-rpc/ggml-rpc.cpp:2221` |

Their GDN graph uses GET_ROWS followed by the plain op, or the scheduler's existing CPU fallback if that backend cannot execute plain GDN. Prior backend snapshot limitations remain. CPU accepts indexed input; Meta delegates support to its component devices and requires mirrored ids. HIP/MUSA share the CUDA source but were not built. Spacemit's GDN entries are commented-out placeholders, not separate implementations. The backend occurrence audit is `rg -l GGML_OP_GATED_DELTA_NET ggml/src`.

## CPU evidence

All executions used empty `CUDA_VISIBLE_DEVICES`, `nice -n 19` through `cpugate`, and at most eight CPU threads. The reproduction scripts serialize model runs with `cpu-model.lock` and pass explicit bounded contexts. Counts and timings below are M (measured); they are validation wall times, not performance comparisons.

- M57/57 CPU GDN cases pass: M36 plain, M13 indexed equivalence, M5 plain cache-copy cases, M3 indexed cache-copy cases. `codex/logs/backend-cpu.log`. The renamed executable `gdn-backend-cpu` ran `test -b CPU -o 'GATED_DELTA_NET.*' -j 1`, with the hardware-concurrency shim capped at eight.
- M13/13 indexed cases fail with the row-0 mutant; the negative-control script passes. M1.131 s wall time, `codex/logs/negative-row0.{log,json}`. `codex/row0-mutant.cpp` changes only indexed reads; it is compiled with `g++ -std=c++17 -O2 -fPIC -shared -Iggml/include codex/row0-mutant.cpp -ldl -o build-cpu/bin/row0-mutant.so` through buildgate; the gather reference retains its ids. The tests explicitly compare indexed-minus-gather output against zero even for CPU versus itself (`tests/test-backend-ops.cpp:4795`), excluding unwritten snapshot slots. Cases include high/permuted rows, repeated ids, larger pools, KDA and non-contiguous q/k/v.
- M133/133 architecture tests pass in M17.043 s, `codex/logs/archs-cpu.{log,json}`. The CPU-only test source was copied with its `n_ctx = 0` replaced by explicit `256` (the fixture's original trained context), avoiding an unbounded model initialization. CPU Meta skips remain as on the base.
- Qwen3.5-4B fixture, patch np1/np2: rollback/checkpoint restore and shared-sequence reserve pass with cache fill 0x00. The idle-child reserve case uses `n_rs_seq = 8` and `seq_cp` under NO_REALLOC (`tests/test-recurrent-state-rollback.cpp:501,529`). M24.710/M26.088 s, `codex/logs/model-patch-np{1,2}.{log,json}`.
- The model executable exits 1 because split replay fails: Mmax absolute difference 13.4578, MNMSE 0.018913, first at seq 0 pos 16. The same executable with base `7049ff0cb` libraries reproduces the same mismatch at np1/np2, while rollback/checkpoint/shared reserve pass. Base wall M19.770/M19.395 s, `codex/logs/model-base-np{1,2}.{log,json}`. This is observed base behavior, not evidence of a getter regression; it remains unresolved. The test exits after this fill, so cache fill 0x3e is not reached in these model runs.

Model command: `test-recurrent-state-rollback -m /mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf -ngl 0 -t 8 -tb 8 -c 4096 -b 64 -ub 16 -np {1,2}`. `codex/run_model_checks.sh` records the exact library paths. The shared subtest uses explicit context 512.

Stack6 `--state-movers`, np1/np2, passes with its built-in negative controls: M8/M16 trial-sequence results, M8.713/M13.084 s wall time, `codex/logs/stack-state-movers-np{1,2}.{log,json}`. Command: `build-stack-cpu/bin/test-recurrent-state-rollback --state-movers` plus the same model/context/thread/batch flags above. NO_REALLOC is ON.

The renamed CPU server `gdn-reserve-server` completes MTP with np2 and explicit context 4096 under NO_REALLOC, logs `n_rs_seq = 3` and `copying state to child`, serves both completions, and exits cleanly when its exact Popen process is stopped. M11.533 s validation wall time; `codex/logs/reserve-server.{log,json}` and `codex/logs/reserve-server/{command,request,response}.json`, `server.log`. `codex/reserve_cpu.py` records the request before sending it and checks for allocation/assertion failures. Its optional extra-row probe recorded M0 events; the separate idle-child test supplies the extra-row coverage, not this server run.

## Builds and stack port

Build commands used `codex/wait_build.sh` then `buildgate`, waiting when a running item id contained `depth` or `perf`. Compilation used `-j4`, shared ccache and empty CUDA visibility.

CPU: `cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DLLAMA_BUILD_TESTS=ON -DGGML_SCHED_NO_REALLOC=ON`; tools/server enabled for the renamed reserve server. Targets `test-backend-ops`, `test-recurrent-state-rollback`, `test-llama-archs`, `llama-server` build. Logs are `codex/logs/build-cpu*.log` and `build-reserve*.log`.

CUDA: `cmake -B build-cuda -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_CCACHE=ON -DLLAMA_BUILD_TESTS=ON`; `buildgate cmake --build build-cuda -j4 --target test-backend-ops llama` passes (`codex/logs/build-cuda.log`). No CUDA executable was run.

Stack branch `topic/gdn-getter-stack6`, base `ac7465885`, source head `743acb55f`: `c68ec4329` API/kernel port, `8d8d2ed1c` getter port, `0a473e886` local engagement trace, `743acb55f` removes the three stack-specific obsolete gathers. Its inherited state-mover test is unchanged; the upstream-only idle-child test is not part of the stack port. The stack diff retains convolution fusion and all other stack behavior; adds the same getter/backend/test changes plus a CUDA log and round counter. Ddiff size: 28 files, 251 insertions, 74 deletions.

`build-stack-cuda` builds `test-backend-ops` and server with the same CUDA flags; server is copied to `round-trace` before freezing. `build-stack-cpu` builds rollback/state-movers with CUDA OFF and NO_REALLOC ON. Logs: `codex/logs/build-stack-{cuda,cpu}.log`.

Freeze: `QWEN38_WT_ROOT=$PWD/codex wt freeze $PWD gdn-getter-stack6 --build $PWD/build-stack-cuda`. `codex/frozen/gdn-getter-stack6/SOURCE` pins `743acb55f9873ea6bea15057458fd5249bdf83ea`; the frozen ggml-base contains `743acb55f\0`. `SHA256SUMS` and local RUNPATH are checked by preflight. Recipe: `codex/frozen/gdn-getter-stack6/RECIPE.txt`.

## Prepared GPU items

- `codex/memcheck/item.toml`, driver `codex/memcheck/memcheck.py`: memcheck, racecheck, synccheck; sanitizer output goes to its own `--log-file`, target output separately; `GPUQ_ACTIVE="1"`; strict coverage, engagement, fault and budget gates. Full coverage is D41 existing and D16 indexed cases per relevant stage. Canary has one plain and three indexed cases per tool, D210 s stage caps inside a D450 s budget. No intended-fault stage.
- `codex/depth-item.toml`, experiment `codex/depth-experiment.toml`, wrapper `codex/gpu_ab.sh`: frozen stack6 versus getter, J5q TP2 np1 k5, ABBA, depths 8192/32768/90112, cross-arm native token/counter identity, steady-round gates and per-GPU indexed log engagement. Full budget D3450 s. Canary is one patch boot at 8192, D600 s budget; it also compares native identity with the recorded stack6 canary baseline.

The depth item is rendered with `python3 -m harness.mkitem` from the existing harness. Its command wrapper adds offline git settings and the one-boot identity control; pins, budget and gates are retained. The kernel-only sanitizer item follows the prior sanitizer manifest because mkitem supports server experiments, not kernel suites. `harness/*.py` was not edited.

Recorded-artifact parser checks PASS (`codex/logs/parser-check.log`). Fixtures include unmodified real sanitizer logs from `gdn-indexed-memcheck/a2`, real native SSE streams and round traces from the prior GDN depth A/B, and the shared gpuq fixture directory. Controls reject missing engagement, changed token identity, failed cases, inadequate coverage, sanitizer findings and fault lines; the parser also recognizes all M13 CPU row-0 failures. These are format fixtures, not getter GPU results.

CPU-only frozen/input preflights PASS (`codex/logs/items-check.log`, `codex/gpu-results/depth/preflight-plan/`). Initial manifest validation caught unbounded signature regexes; corrected word-bounded signatures pass. Both final `gpuq submit --check` commands PASS: `codex/logs/memcheck-contract.log`, `codex/logs/depth-contract.log`. Commands and regeneration are in `codex/check_gpu_items.sh`. No item was submitted, and no canary or full GPU run was executed.

## Package and limits

Staged package: `codex/upstream/llama.cpp/67-gdn-indexed-rollback/`. Primary series recreates the `472791d90` tree exactly when applied to `7049ff0cb` (`codex/logs/{apply,source}-tree.txt`). The previous opcode package and its evidence remain under `alt-opcode/`, with a one-line explanation. `apply.txt`, `EVIDENCE.md`, `NOTES.md`, `status` describe the getter primary form. The parent must install this replacement into the canonical package 67; external package files were read-only here.

Open: getter-form GPU correctness and speed, CUDA sanitizers, rejecting-platform builds/runtime, HIP/MUSA, and acceptance of the ggml signature change. Older opcode GPU numbers are not getter measurements. No sampling change, PR text, network call, GPU run, push or queue submission was performed.
