#pragma once

#include "common.cuh"

bool ggml_cuda_syncfree_supported(const ggml_tensor * op);
void ggml_cuda_syncfree(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
cudaError_t ggml_cuda_syncfree_stream_synchronize(cudaStream_t stream);
uint64_t ggml_cuda_syncfree_stream_count();
