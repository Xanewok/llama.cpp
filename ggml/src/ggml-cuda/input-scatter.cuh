#pragma once

#include "common.cuh"

// Entries per launch; a longer list takes several launches over the same device region.
#define GGML_CUDA_INPUT_SCATTER_MAX 32
#define GGML_CUDA_INPUT_SCATTER_THREADS 256

// Entry e copies size[e] bytes from src + offs[e] to dst[e]. Destinations must not overlap: entries run in no order.
// Passed by value, so the table lives in kernel parameter space and needs no upload of its own.
struct ggml_cuda_input_scatter_args {
    const char * src;
    int          n;
    char *       dst [GGML_CUDA_INPUT_SCATTER_MAX];
    size_t       offs[GGML_CUDA_INPUT_SCATTER_MAX];
    size_t       size[GGML_CUDA_INPUT_SCATTER_MAX];
};

static inline int ggml_cuda_input_scatter_blocks(size_t max_size) {
    const size_t per_block = 16*GGML_CUDA_INPUT_SCATTER_THREADS;
    const size_t blocks = (max_size + per_block - 1) / per_block;
    return (int) (blocks < 1 ? 1 : blocks > 1024 ? 1024 : blocks);
}

// The work of thread tid out of nthreads. Byte-exact for any alignment; 16- or 4-byte words where dst, src and size allow.
static __host__ __device__ __forceinline__ void ggml_cuda_input_scatter_thread(
        const ggml_cuda_input_scatter_args & a, size_t tid, size_t nthreads) {
#pragma unroll
    for (int e = 0; e < GGML_CUDA_INPUT_SCATTER_MAX; ++e) {
        if (e < a.n) {
            char       * dst  = a.dst[e];
            const char * src  = a.src + a.offs[e];
            const size_t size = a.size[e];
            const size_t mis  = (size_t) (uintptr_t) dst | (size_t) (uintptr_t) src | size;
            if (mis % 16 == 0) {
                for (size_t i = tid; i < size/16; i += nthreads) {
                    ((int4 *) dst)[i] = ((const int4 *) src)[i];
                }
            } else if (mis % 4 == 0) {
                for (size_t i = tid; i < size/4; i += nthreads) {
                    ((int *) dst)[i] = ((const int *) src)[i];
                }
            } else {
                for (size_t i = tid; i < size; i += nthreads) {
                    dst[i] = src[i];
                }
            }
        }
    }
}

// max_size: the largest size[e], which sizes the grid.
void ggml_cuda_input_scatter(const ggml_cuda_input_scatter_args & args, size_t max_size, cudaStream_t stream);
