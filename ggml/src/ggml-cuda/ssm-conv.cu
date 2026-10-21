#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"
#include "ggml-ssm-conv.h"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 7:  launch_kernel(std::integral_constant<int, 7 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 7, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}


template <bool apply_silu, int d_conv>
static __global__ void ssm_conv_state_f32(const char * input, const char * weights,
        float * out, char * cache, int64_t channels, int64_t n_tokens, int64_t slots,
        size_t x1, size_t x2, size_t w1,
        size_t cache_seq_stride, size_t cache_slot_stride) {
    ggml_cuda_pdl_lc();
    const int64_t channel = blockIdx.y * blockDim.x + threadIdx.x;
    const int64_t seq = blockIdx.x;
    float x[d_conv];
    float w[d_conv];
    ggml_cuda_pdl_sync();
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        w[j] = *(const float *) (weights + channel * w1 + j * sizeof(float));
        x[j] = *(const float *) (input + channel * x1 + seq * x2 + j * sizeof(float));
    }
    for (int64_t i = 0; i < n_tokens; ++i) {
        if (i > 0) {
            x[(i - 1) % d_conv] = *(const float *) (input + channel * x1 + seq * x2 + (i + d_conv - 1) * sizeof(float));
        }
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            sum += x[(i + j) % d_conv] * w[j];
        }
        sum += 0.0f;
        out[(seq * n_tokens + i) * channels + channel] = apply_silu ? ggml_cuda_op_silu_single(sum) : sum;
    }
    // Older slots clamp to the input state when this ubatch has fewer tokens than slots.
    for (int64_t slot = 0; slot < slots; ++slot) {
        const int64_t start = n_tokens > slot ? n_tokens - slot : 0;
#pragma unroll
        for (int j = 0; j < d_conv - 1; ++j) {
            const float value = *(const float *) (input + channel * x1 + seq * x2 + (start + j) * sizeof(float));
            *(float *) (cache + slot * cache_slot_stride + seq * cache_seq_stride +
                (channel * (d_conv - 1) + j) * sizeof(float)) = value;
        }
    }
}

void ggml_cuda_op_ssm_conv_state(ggml_backend_cuda_context & ctx, const ggml_ssm_conv_state_fusion & match) {
    const ggml_tensor * concat = match.conv->src[0];
    const ggml_tensor * weights = match.conv->src[1];
    const dim3 blocks(concat->ne[2], concat->ne[1] / 128);
    const auto params = ggml_cuda_kernel_launch_params(blocks, 128, 0, ctx.stream());
    auto launch = [&](auto nc, auto silu) {
        ggml_cuda_kernel_launch(ssm_conv_state_f32<decltype(silu)::value, decltype(nc)::value>, params,
            (const char *) concat->data, (const char *) weights->data,
            (float *) match.out->data, (char *) match.cache->data, concat->ne[1], match.out->ne[1], match.slots,
            concat->nb[1], concat->nb[2], weights->nb[1],
            match.cache->nb[1], match.slot_stride);
    };
    auto launch_nc = [&](auto nc) {
        if (match.out != match.conv) {
            launch(nc, std::true_type{});
        } else {
            launch(nc, std::false_type{});
        }
    };
    switch (weights->ne[0]) {
        case 3:  launch_nc(std::integral_constant<int, 3>{}); break;
        case 4:  launch_nc(std::integral_constant<int, 4>{}); break;
        case 5:  launch_nc(std::integral_constant<int, 5>{}); break;
        case 9:  launch_nc(std::integral_constant<int, 9>{}); break;
        case 15: launch_nc(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("unsupported fused conv size");
    }
}

// One thread owns a channel for every sequence: it reads that channel's state of all sequences before it writes any
// snapshot, and channels never share data, so a snapshot row that another sequence reads is safe without a barrier.
// The arithmetic is that of ssm_conv_state_f32.
template <bool apply_silu, int d_conv>
static __global__ void ssm_conv_rs_f32(const char * x, const char * weights, const char * states, const int32_t * ids,
        char * snap, float * out, int64_t channels, int64_t n_t, int64_t n_s, int64_t n_slots,
        size_t x0, size_t x1, size_t x2, size_t w1, size_t states_row, size_t snap_row, size_t snap_slot) {
    ggml_cuda_pdl_lc();
    const int64_t channel = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const bool live = channel < channels;
    float hist[GGML_CUDA_SSM_CONV_RS_MAX_SEQS][d_conv - 1];
    float w[d_conv];
    ggml_cuda_pdl_sync();
    if (!live) {
        return;
    }
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        w[j] = *(const float *) (weights + channel * w1 + j * sizeof(float));
    }
#pragma unroll
    for (int s = 0; s < GGML_CUDA_SSM_CONV_RS_MAX_SEQS; ++s) {
        if (s < n_s) {
            const char * row = states + (int64_t) ids[s] * states_row + channel * (d_conv - 1) * sizeof(float);
#pragma unroll
            for (int j = 0; j < d_conv - 1; ++j) {
                hist[s][j] = *(const float *) (row + j * sizeof(float));
            }
        }
    }
#pragma unroll
    for (int s = 0; s < GGML_CUDA_SSM_CONV_RS_MAX_SEQS; ++s) {
        if (s >= n_s) {
            continue;
        }
        char * snap_s = snap + (int64_t) s * snap_row + channel * (d_conv - 1) * sizeof(float);
        float win[d_conv];
#pragma unroll
        for (int j = 0; j < d_conv - 1; ++j) {
            win[j] = hist[s][j];
        }
        for (int64_t i = 0; i < n_t; ++i) {
            win[d_conv - 1] = *(const float *) (x + channel * x1 + i * x0 + s * x2);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < d_conv; ++j) {
                sum += win[j] * w[j];
            }
            sum += 0.0f;
            out[(s * n_t + i) * channels + channel] = apply_silu ? ggml_cuda_op_silu_single(sum) : sum;
            // Slot k starts at token max(0, n_t - k); slots older than the batch hold the input state.
            for (int64_t k = 1; k < n_slots; ++k) {
                if ((n_t > k ? n_t - k : 0) == i) {
#pragma unroll
                    for (int j = 0; j < d_conv - 1; ++j) {
                        *(float *) (snap_s + k * snap_slot + j * sizeof(float)) = win[j];
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < d_conv - 1; ++j) {
                win[j] = win[j + 1];
            }
        }
#pragma unroll
        for (int j = 0; j < d_conv - 1; ++j) {
            *(float *) (snap_s + j * sizeof(float)) = win[j];
        }
    }
}

bool ggml_cuda_ssm_conv_rs_supported(const ggml_tensor * op) {
    const ggml_tensor * x = op->src[0];
    const ggml_tensor * w = op->src[1];
    switch (w->ne[0]) {
        case 3: case 4: case 5: case 9: case 15: break;
        default: return false;
    }
    return op->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 &&
        op->src[2]->type == GGML_TYPE_F32 && op->src[4]->type == GGML_TYPE_F32 && w->nb[0] == sizeof(float) &&
        x->ne[0] <= GGML_CUDA_SSM_CONV_RS_MAX_TOKENS && x->ne[2] <= GGML_CUDA_SSM_CONV_RS_MAX_SEQS;
}

void ggml_cuda_op_ssm_conv_rs(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst) {
    const ggml_tensor * x      = dst->src[0];
    const ggml_tensor * w      = dst->src[1];
    const ggml_tensor * states = dst->src[2];
    const ggml_tensor * ids    = dst->src[3];
    const ggml_tensor * snap   = dst->src[4];
    const ggml_tensor * out    = silu_dst ? silu_dst : dst;

    GGML_ASSERT(ggml_cuda_ssm_conv_rs_supported(dst));
    GGML_ASSERT(ggml_is_contiguous(out) && out->type == GGML_TYPE_F32);

    const int64_t channels = w->ne[1];
    const dim3 blocks((channels + 127) / 128);
    const auto params = ggml_cuda_kernel_launch_params(blocks, 128, 0, ctx.stream());
    auto launch = [&](auto nc, auto silu) {
        ggml_cuda_kernel_launch(ssm_conv_rs_f32<decltype(silu)::value, decltype(nc)::value>, params,
            (const char *) x->data, (const char *) w->data, (const char *) states->data, (const int32_t *) ids->data,
            (char *) snap->data, (float *) out->data, channels, x->ne[0], x->ne[2], snap->ne[2],
            x->nb[0], x->nb[1], x->nb[2], w->nb[1], states->nb[1], snap->nb[1], snap->nb[2]);
    };
    auto launch_nc = [&](auto nc) {
        if (silu_dst) {
            launch(nc, std::true_type{});
        } else {
            launch(nc, std::false_type{});
        }
    };
    switch (w->ne[0]) {
        case 3:  launch_nc(std::integral_constant<int, 3>{}); break;
        case 4:  launch_nc(std::integral_constant<int, 4>{}); break;
        case 5:  launch_nc(std::integral_constant<int, 5>{}); break;
        case 9:  launch_nc(std::integral_constant<int, 9>{}); break;
        case 15: launch_nc(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("unsupported conv size");
    }
}
