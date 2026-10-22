# Package 94 port to ff5a23376

- `0001-WIP-94.patch`: `tests/CMakeLists.txt` context changed from `llama_build(test-backend-ops.cpp)` to `llama_build(test-backend-ops.cpp LINK ggml)`. Inserted the original test registration after that target. CUDA source and test source applied unchanged.
- `0002-WIP-94.patch`: no conflicts.
