#include "input-scatter.cuh"

static __global__ void k_input_scatter(const ggml_cuda_input_scatter_args args) {
    ggml_cuda_input_scatter_thread(args, (size_t) blockIdx.x*blockDim.x + threadIdx.x, (size_t) gridDim.x*blockDim.x);
}

void ggml_cuda_input_scatter(const ggml_cuda_input_scatter_args & args, size_t max_size, cudaStream_t stream) {
    k_input_scatter<<<ggml_cuda_input_scatter_blocks(max_size), GGML_CUDA_INPUT_SCATTER_THREADS, 0, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
}
