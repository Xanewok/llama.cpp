#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

struct ggml_ssm_conv_state_fusion;
void ggml_cuda_op_ssm_conv_state(ggml_backend_cuda_context & ctx, const ggml_ssm_conv_state_fusion & match);

// Most sequences and tokens one ssm_conv_rs launch handles: a thread keeps the state of every sequence in registers
// until all are read, and a longer batch keeps the gathered path.
#define GGML_CUDA_SSM_CONV_RS_MAX_SEQS   8
#define GGML_CUDA_SSM_CONV_RS_MAX_TOKENS 32

bool ggml_cuda_ssm_conv_rs_supported(const ggml_tensor * op);
void ggml_cuda_op_ssm_conv_rs(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = nullptr);
