#pragma once

// DO NOT MERGE. Diagnostic capacity; acquire_slot must still wait before reuse.
static constexpr int GGML_CUDA_AR_POOL_DEFAULT = 2;
static constexpr int GGML_CUDA_AR_POOL_MAX = 16;

// Zero rejects a malformed or out-of-range override.
static inline int ggml_cuda_ar_pool_size(const char * value) {
    if (value == nullptr) {
        return GGML_CUDA_AR_POOL_DEFAULT;
    }
    int size = 0;
    for (const char * c = value; *c; ++c) {
        if (*c < '0' || *c > '9') {
            return 0;
        }
        size = size * 10 + (*c - '0');
        if (size > GGML_CUDA_AR_POOL_MAX) {
            return 0;
        }
    }
    return size >= GGML_CUDA_AR_POOL_DEFAULT ? size : 0;
}
