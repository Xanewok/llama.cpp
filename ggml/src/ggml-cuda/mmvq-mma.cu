#include "mmvq-mma.cuh"
#include "mma.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"

// MMVQ for a few src1 columns on int8 tensor cores.
//
// A warp owns 16 rows and every MMVQ_MMA_NWARPS-th src0 block of them; the warps of a CUDA block add up their partial sums
// at the end, as in MMVQ. Each lane loads the src0 bytes of its own mma fragments from VRAM and decodes them in registers:
// src0 is read once, with no shared-memory tile and no barrier in the k loop, so more columns only add mma work.
//
// The A fragment of mma m16n8k16/k32 holds, for lane (g, t) = (lane/4, lane%4), rows g and g+8 at ints t (and 4+t for k32)
// of the k chunk. A chunk must stay inside one src0 sub-block (one scale per row), and src1 is MMVQ's q8_1 (one scale per 32
// values): q6_K uses k16 chunks (16-value sub-blocks), iq4_xs k32 chunks (32-value sub-blocks).

using namespace ggml_cuda_mma;

#define MMVQ_MMA_NWARPS 4
#define MMVQ_MMA_ROWS   16

// The src0 data of one block for rows g and g+8, as lane (g, t) uses it.
template <ggml_type type> struct mmvq_mma_block;

template <> struct mmvq_mma_block<GGML_TYPE_Q6_K> {
    typedef block_q6_K block_t;

    int  ql[2][8]; // [row][4*h + s4]: ql int 16*h + 4*s4 + t; low nibbles of sub-block 8*h + s4, high nibbles of 8*h + 4 + s4
    int  qh[2][4]; // [row][2*h + p]:  qh int  8*h + 4*p  + t; high bits of sub-blocks 8*h + p + {0, 2, 4, 6}
    int  sc[2][4]; // [row][i]: scales 4*i .. 4*i + 3
    half d[2];

    // 210-byte blocks are only 2-byte aligned.
    __device__ __forceinline__ void load(const block_t * x0, const block_t * x1, const int t) {
        const block_t * x[2] = {x0, x1};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                ql[r][i] = get_int_b2(x[r]->ql, 4*i + t);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                qh[r][i] = get_int_b2(x[r]->qh, 4*i + t);
                sc[r][i] = get_int_b2(x[r]->scales, i);
            }
            d[r] = x[r]->d;
        }
    }
};

template <> struct mmvq_mma_block<GGML_TYPE_IQ4_XS> {
    typedef block_iq4_xs block_t;

    int qs[2][8]; // [row][ib]: qs int 4*ib + t; low nibbles are values 4*t .. 4*t + 3 of sub-block ib, high nibbles 16 + 4*t ..
    int dh[2];    // [row]: d | scales_h << 16
    int sl[2];    // [row]: scales_l

    // Host contract (ggml_cuda_should_use_mmvq_mma): blocks are 4-byte aligned.
    __device__ __forceinline__ void load(const block_t * x0, const block_t * x1, const int t) {
        const block_t * x[2] = {x0, x1};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int ib = 0; ib < QK_K/32; ++ib) {
                qs[r][ib] = get_int_b4(x[r]->qs, 4*ib + t);
            }
            dh[r] = get_int_b4(x[r], 0);
            sl[r] = get_int_b4(x[r]->scales_l, 0);
        }
    }
};

// acc[nt][l]: rows g (l < 2) and g+8 (l >= 2), columns 8*nt + 2*t + l%2.
template <int ntiles>
static __device__ __forceinline__ void mmvq_mma_dot(
        const mmvq_mma_block<GGML_TYPE_Q6_K> & b, const block_q8_1 * __restrict__ y, const int stride_col_y, const int ncols,
        float (&acc)[ntiles][4]) {
    const int g = threadIdx.x / 4;
    const int t = threadIdx.x % 4;

    const float d[2] = {__half2float(b.d[0]), __half2float(b.d[1])};

#pragma unroll
    for (int sb = 0; sb < QK_K/16; ++sb) {
        const int h     = sb / 8;
        const int nib   = (sb / 4) % 2;
        const int s4    = sb % 4;
        const int shift = 2*((sb % 8) / 2);

        tile<16, 4, int> A;
        float ds[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int vl = (b.ql[r][4*h + s4] >> (4*nib)) & 0x0F0F0F0F;
            const int vh = ((b.qh[r][2*h + sb % 2] >> shift) << 4) & 0x30303030;
            A.x[r] = __vsub4(vl | vh, 0x20202020);
            ds[r]  = d[r] * (int8_t) (b.sc[r][sb / 4] >> (8*(sb % 4)));
        }

#pragma unroll
        for (int nt = 0; nt < ntiles; ++nt) {
            const int jb = 8*nt + g;
            tile<8, 4, int> B;
            B.x[0] = jb < ncols ? get_int_b4(y[jb*stride_col_y + sb/2].qs, 4*(sb % 2) + t) : 0;

            tile<16, 8, int> C;
            mma(C, A, B);

            const int   jc  = 8*nt + 2*t;
            const float dy0 = jc + 0 < ncols ? __low2float(y[(jc + 0)*stride_col_y + sb/2].ds) : 0.0f;
            const float dy1 = jc + 1 < ncols ? __low2float(y[(jc + 1)*stride_col_y + sb/2].ds) : 0.0f;
            acc[nt][0] += ds[0]*dy0 * C.x[0];
            acc[nt][1] += ds[0]*dy1 * C.x[1];
            acc[nt][2] += ds[1]*dy0 * C.x[2];
            acc[nt][3] += ds[1]*dy1 * C.x[3];
        }
    }
}

template <int ntiles>
static __device__ __forceinline__ void mmvq_mma_dot(
        const mmvq_mma_block<GGML_TYPE_IQ4_XS> & b, const block_q8_1 * __restrict__ y, const int stride_col_y, const int ncols,
        float (&acc)[ntiles][4]) {
    const int g = threadIdx.x / 4;
    const int t = threadIdx.x % 4;

    const float d[2] = {
        __half2float(__ushort_as_half((unsigned short) (b.dh[0] & 0xFFFF))),
        __half2float(__ushort_as_half((unsigned short) (b.dh[1] & 0xFFFF))),
    };

#pragma unroll
    for (int ib = 0; ib < QK_K/32; ++ib) {
        tile<16, 8, int> A;
        float dl[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int2 v = get_int_from_table_16(b.qs[r][ib], kvalues_iq4nl);
            A.x[r + 0] = v.x;
            A.x[r + 2] = v.y;
            const int ls = ((b.sl[r] >> (4*ib)) & 0x0F) | (((b.dh[r] >> (16 + 2*ib)) & 0x03) << 4);
            dl[r] = d[r] * (ls - 32);
        }

#pragma unroll
        for (int nt = 0; nt < ntiles; ++nt) {
            const int jb = 8*nt + g;
            tile<8, 8, int> B;
            if (jb < ncols) {
                const block_q8_1 * yb = y + jb*stride_col_y + ib;
                B.x[0] = get_int_b4(yb->qs, t);
                B.x[1] = get_int_b4(yb->qs, 4 + t);
            }

            tile<16, 8, int> C;
            mma(C, A, B);

            const int   jc  = 8*nt + 2*t;
            const float dy0 = jc + 0 < ncols ? __low2float(y[(jc + 0)*stride_col_y + ib].ds) : 0.0f;
            const float dy1 = jc + 1 < ncols ? __low2float(y[(jc + 1)*stride_col_y + ib].ds) : 0.0f;
            acc[nt][0] += dl[0]*dy0 * C.x[0];
            acc[nt][1] += dl[0]*dy1 * C.x[1];
            acc[nt][2] += dl[1]*dy0 * C.x[2];
            acc[nt][3] += dl[1]*dy1 * C.x[3];
        }
    }
}

template <ggml_type type, int ntiles>
__launch_bounds__(MMVQ_MMA_NWARPS*WARP_SIZE, 4)
static __global__ void mul_mat_vec_q_mma(
        const char * __restrict__ vx, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
        const int blocks_per_row, const int nrows, const int ncols,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst) {
#ifdef TURING_MMA_AVAILABLE
    typedef typename mmvq_mma_block<type>::block_t block_t;

    const int g = threadIdx.x / 4;
    const int t = threadIdx.x % 4;

    const int row0        = MMVQ_MMA_ROWS*blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int sample_dst  = blockIdx.z;

    const block_t * x = (const block_t *) vx
        + int64_t(sample_dst / sample_ratio)*stride_sample_x + int64_t(channel_dst / channel_ratio)*stride_channel_x;
    const block_t * x0 = x + int64_t(min(row0 + g,     nrows - 1))*stride_row_x;
    const block_t * x1 = x + int64_t(min(row0 + 8 + g, nrows - 1))*stride_row_x;
    y   += int64_t(sample_dst)*stride_sample_y   + int64_t(channel_dst)*stride_channel_y;
    dst += int64_t(sample_dst)*stride_sample_dst + int64_t(channel_dst)*stride_channel_dst;

    float acc[ntiles][4] = {{0.0f}};

    if (int(threadIdx.y) < blocks_per_row) {
        mmvq_mma_block<type> cur;
        cur.load(x0 + threadIdx.y, x1 + threadIdx.y, t);
        for (int kb = threadIdx.y; kb < blocks_per_row; kb += MMVQ_MMA_NWARPS) {
            // The next block is in flight while this one is multiplied; the last iteration reloads its own block.
            const int kb_next = kb + MMVQ_MMA_NWARPS < blocks_per_row ? kb + MMVQ_MMA_NWARPS : kb;
            mmvq_mma_block<type> next;
            next.load(x0 + kb_next, x1 + kb_next, t);
            mmvq_mma_dot<ntiles>(cur, y + kb*(QK_K/QK8_1), stride_col_y, ncols, acc);
            cur = next;
        }
    }

    __shared__ float acc_shared[MMVQ_MMA_NWARPS - 1][ntiles][4][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int nt = 0; nt < ntiles; ++nt) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                acc_shared[threadIdx.y - 1][nt][l][threadIdx.x] = acc[nt][l];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int nt = 0; nt < ntiles; ++nt) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
#pragma unroll
            for (int w = 0; w < MMVQ_MMA_NWARPS - 1; ++w) {
                acc[nt][l] += acc_shared[w][nt][l][threadIdx.x];
            }
            const int row = row0 + 8*(l / 2) + g;
            const int col = 8*nt + 2*t + l % 2;
            if (row < nrows && col < ncols) {
                dst[int64_t(col)*stride_col_dst + row] = acc[nt][l];
            }
        }
    }
#else
    GGML_UNUSED_VARS(vx, y, dst, blocks_per_row, nrows, ncols, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}

template <ggml_type type>
static void mul_mat_vec_q_mma_launch(
        const char * x, const block_q8_1 * y, float * dst,
        const int blocks_per_row, const int nrows, const int ncols,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_dst, const int channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_dst,  const int sample_ratio,  const int stride_sample_x,  const int stride_sample_y,  const int stride_sample_dst,
        cudaStream_t stream) {
    const dim3 block_nums((nrows + MMVQ_MMA_ROWS - 1) / MMVQ_MMA_ROWS, nchannels_dst, nsamples_dst);
    const dim3 block_dims(WARP_SIZE, MMVQ_MMA_NWARPS, 1);

    if (ncols <= 8) {
        mul_mat_vec_q_mma<type, 1><<<block_nums, block_dims, 0, stream>>>
            (x, y, dst, blocks_per_row, nrows, ncols, stride_row_x, stride_col_y, stride_col_dst,
             channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
             sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    } else {
        mul_mat_vec_q_mma<type, 2><<<block_nums, block_dims, 0, stream>>>
            (x, y, dst, blocks_per_row, nrows, ncols, stride_row_x, stride_col_y, stride_col_dst,
             channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
             sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    }
}

bool ggml_cuda_should_use_mmvq_mma(const ggml_tensor * src0, const int64_t ne11, const int cc) {
    if (!turing_mma_available(cc) || ne11 > MMVQ_MMA_MAX_BATCH_SIZE) {
        return false;
    }
    switch (src0->type) {
        case GGML_TYPE_Q6_K:
            break;
        case GGML_TYPE_IQ4_XS:
            if ((uintptr_t) src0->data % 4 != 0 || src0->nb[1] % 4 != 0 || src0->nb[2] % 4 != 0 || src0->nb[3] % 4 != 0) {
                return false;
            }
            break;
        default:
            return false;
    }
    // local study knob: GGML_CUDA_MMVQ_MMA=0 keeps MMQ
    static const bool mma_off = getenv("GGML_CUDA_MMVQ_MMA") && atoi(getenv("GGML_CUDA_MMVQ_MMA")) == 0;
    return !mma_off;
}

void ggml_cuda_mul_mat_vec_q_mma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(nb00 == ts_src0);
    GGML_ASSERT(nb10 == ts_src1);
    GGML_ASSERT(nb0  == ts_dst);
    GGML_ASSERT(ggml_blck_size(src0->type) == QK_K && ne00 % QK_K == 0);
    GGML_ASSERT(ne11 <= MMVQ_MMA_MAX_BATCH_SIZE);
    GGML_ASSERT(ne12 % ne02 == 0 && ne13 % ne03 == 0);

    cudaStream_t stream = ctx.stream();

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = nb11 / ts_src1;
        const int64_t s12 = nb12 / ts_src1;
        const int64_t s13 = nb13 / ts_src1;
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
            ne10_padded, ne11, ne12, ne13, stream);
    }

    const int64_t s01 = nb01 / ts_src0;
    const int64_t s02 = nb02 / ts_src0;
    const int64_t s03 = nb03 / ts_src0;
    const int64_t s1  = nb1  / ts_dst;
    const int64_t s2  = nb2  / ts_dst;
    const int64_t s3  = nb3  / ts_dst;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    static const bool mma_debug = getenv("GGML_CUDA_DEBUG_MMVQ_MMA") != nullptr;
    if (mma_debug) {
        fprintf(stderr, "mmvq_mma: device=%d type=%s columns=%lld ntiles=%d rows=%lld k=%lld channels=%lld samples=%lld\n",
                ctx.device, ggml_type_name(src0->type), (long long) ne11, ne11 <= 8 ? 1 : 2, (long long) ne01, (long long) ne00,
                (long long) ne12, (long long) ne13);
    }

    const char       * x = (const char *) src0->data;
    const block_q8_1 * y = (const block_q8_1 *) src1_q8_1.get();
    float            * d = (float *) dst->data;

    switch (src0->type) {
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_mma_launch<GGML_TYPE_Q6_K>(x, y, d, ne00/QK_K, ne01, ne11, s01, s11, s1,
                ne12, ne12/ne02, s02, s12, s2, ne13, ne13/ne03, s03, s13, s3, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_mma_launch<GGML_TYPE_IQ4_XS>(x, y, d, ne00/QK_K, ne01, ne11, s01, s11, s1,
                ne12, ne12/ne02, s02, s12, s2, ne13, ne13/ne03, s03, s13, s3, stream);
            break;
        default:
            GGML_ABORT("unsupported type %s", ggml_type_name(src0->type));
    }
}
