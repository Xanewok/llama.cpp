#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

struct ggml_ssm_conv_state_fusion;
void ggml_cuda_op_ssm_conv_state(ggml_backend_cuda_context & ctx, const ggml_ssm_conv_state_fusion & match);
