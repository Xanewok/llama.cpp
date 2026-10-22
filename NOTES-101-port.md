# Package 101 fork port

Target: `726f57ffecedcdbfdbbaf8892fcdc8321779fd56`; package 94 is already in this fork. Ported the upstream design and implementation; did not reapply 94.

- Design commit: `NOTES-94-port.md` modify/delete conflict. Retained the master port notes as reference; they do not describe an extra 94 application to the fork.
- `ggml-backend-meta.cpp`: arena sizing conflict resolved with plan-local high-water marks and owner-local `n_reduce_steps`. All new comm calls use `owner`, since the fork separates comm ownership from graph plans. Retained `prepare_only` return before readiness observation, so plan creation does not consume the cold execution. Gather boundaries fail graph qualification and retain the fork's exact reduction path; the exactness check reads the selected layout's boundary.
- `ggml-cuda.cu`: registry insertion conflict resolved by retaining both new optional comm procedures and all existing input-staging procedures with their original guards.
- `allreduce.cu`, headers, and CMake auto-merged. Retained the fork's `GGML_CUDA_AR_P2P=1` alias and all copy/BF16 policies. No new local gate was added; A/B uses the two branch binaries under the existing P2P settings.

Fork host syntax check and whitespace check both returned 0:

    nice -n 19 g++ -std=c++17 -Wall -Wextra -fsyntax-only -Iggml/include -Iggml/src ggml/src/ggml-backend-meta.cpp tests/test-ar-p2p.cpp tests/test-ar-p2p-graph.cpp
    nice -n 19 git diff --check

Environment: `CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1`. No nvcc, builds, tests, binaries or GPU tools were run.
