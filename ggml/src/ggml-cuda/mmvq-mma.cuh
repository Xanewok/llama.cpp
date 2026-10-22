#pragma once

#include "common.cuh"

bool ggml_cuda_should_use_mmvq_mma(const ggml_tensor * src0, int64_t ne11, int cc);

void ggml_cuda_mul_mat_vec_q_mma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
