# Package 101 source validation

No binaries, tests, GPU tools, nvcc or builds were run. All commands used `CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1`, nice 19 and one compiler process.

Upstream form, exit 0 with no diagnostics:

    nice -n 19 g++ -std=c++17 -Wall -Wextra -fsyntax-only -Iggml/include -Iggml/src ggml/src/ggml-backend-meta.cpp tests/test-ar-p2p.cpp tests/test-ar-p2p-graph.cpp
    nice -n 19 git diff --check

`ggml-cuda.cu` and `allreduce.cu` need nvcc; host syntax checks do not validate them.

## Tests added, not executed

- `test-ar-p2p-graph`: compares merged vs split meta execution bitwise for both residual and row-split output, including an inactive rank shard. Eight rounds per phase change inputs while stable rounds assert no new merged capture. Direct copy-engine AllReduces use the same comm pipeline between graph calls and compare both ranks.
- Phase sequence: five AllReduces with a forced unsupported boundary (two merged segments); one AllReduce plus 40 scales (more graph nodes, fewer split and merged subgraphs, forces arena replacement); five AllReduces again (reuse retained slots); copy-engine-sized meta graph (must stay split). The third phase changes a SCALE parameter and graph UID, requires a split stabilization call without capture, then requires recapture and stable replay.
- `test-ar-p2p-graph-disabled`: same source with `GGML_CUDA_DISABLE_GRAPHS=1`. Support query returns skip 77. A graph-disabled build also skips.
- `test-ar-p2p-graph-timeout`: isolated `--timeout` child warms both ranks, captures and launches rank 0 alone, then synchronizes. CMake requires nonzero termination with `peer missed AllReduce epoch`; external timeout and unrelated crashes fail. Unsupported configurations skip. Only the captured launch passes a nonzero timeout; host-driven launches pass zero.

## Limits

No runtime identity, progress, performance, GPU graph count, CUDA compiler acceptance or timeout diagnostic is established here. The epoch deadline bounds an executing block's poll, not driver submission or scheduling. The CUDA fatal trap cannot recover the comm context. Stability gating avoids transient captures but cannot eliminate cold capture for every new graph variant. The optional comm protocol and backend-owned CUSTOM node representation need maintainer discussion before any submission.
