#pragma once

#include "common.cuh"

#define MMVQ_MMA_MAX_BATCH_SIZE 16 // Max. number of src1 columns for the tensor-core MMVQ kernel.

// Meant for the widths at which MMVQ is not used: below its cut it already reads src0 at close to the memory bandwidth.
bool ggml_cuda_should_use_mmvq_mma(const ggml_tensor * src0, int64_t ne11, int cc);

void ggml_cuda_mul_mat_vec_q_mma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
