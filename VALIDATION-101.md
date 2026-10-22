# Package 101 source validation

Environment for every check: `CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1`, nice 19, one compiler process. No built binaries, tests, GPU tools, nvcc or builds are executed. The only compiler invocation is syntax-only, as requested by the brief.

Host syntax command (master and fork):

    nice -n 19 g++ -std=c++17 -Wall -Wextra -fsyntax-only -Iggml/include -Iggml/src ggml/src/ggml-backend-meta.cpp tests/test-ar-p2p.cpp
    nice -n 19 git diff --check

Master: exit 0, no diagnostics. Fork status is recorded in `NOTES-101-port.md` after porting.

`test-ar-p2p.cpp:graph_cases` compares split and collective results bitwise for residuals and a projection split across both ranks. Changed inputs exercise repeated execution; an inactive layer tests zero contribution. Five reductions, one reduction plus 40 scales, then five reductions exercise arena replacement and reuse. A changed scale parameter and UID must decline optimization once before resuming. The spy asserts that stable calls keep using collective execution once available; it does not count captures or prove a driver-level cache hit. Null context and incomplete rank lists must decline on the host. The existing direct AllReduce cases retain type, size, scalar-tail and inactive-rank coverage.

`test-ar-p2p-graph` now invokes the same existing binary with `GGML_CUDA_ALLREDUCE=internal;GGML_CUDA_P2P=1`. No standalone graph binary, timeout child, CMake script, or deliberate device fault remains. An unavailable graph procedure or unsupported peer capture prints a subtest skip while ordinary reductions still run.

Unverified: nvcc acceptance, simultaneous independent rank-stream captures, bitwise identity, progress, cache reuse and speed. Prior GPU results belong to the old CUSTOM-node design. Dropped tests no longer establish copy-engine interleaving or device-side peer timeout; late peer loss is unbounded. No evidence supports attributing the fixed per-request cost to capture/instantiate, so this reduction adds no workaround for it.
