#include "syncfree.cuh"
#include "../ggml-syncfree.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

static std::atomic<uint64_t> syncfree_stream_count{0};

cudaError_t ggml_cuda_syncfree_stream_synchronize(cudaStream_t stream) {
    static const bool enabled = [] { const char * p = std::getenv("LLAMA_SYNCFREE"); return p && std::strcmp(p, "1") == 0; }();
    if (enabled) {
        syncfree_stream_count.fetch_add(1, std::memory_order_relaxed);
    }
    return cudaStreamSynchronize(stream);
}

uint64_t ggml_cuda_syncfree_stream_count() {
    return syncfree_stream_count.load(std::memory_order_relaxed);
}

static __global__ void syncfree_accept_kernel(
        const float * p, const int32_t * ids, const int32_t * sampled,
        const float * packet, int32_t * out, int k, int n_pos, int nc) {
    const int s = blockIdx.x;
    ggml_syncfree_accept(p + s*nc*(k + 1), ids + s*nc*(k + 1), sampled + s*(k + 1),
        packet + s*(2 + 23*k), out + s*(4 + k + n_pos*k), k, n_pos, nc);
}

static __global__ void syncfree_mask_kernel(
        const float * mask, const int32_t * positions, const int32_t * accept,
        const float * packet, float * out, int nkv, int rows, int k, int accept_stride) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < nkv*rows) {
        const int s = i/(nkv*k);
        out[i] = ggml_syncfree_mask_value(mask[i], positions[s*nkv + i % nkv],
            accept + s*accept_stride, packet + s*(2 + 23*k), k);
    }
}

bool ggml_cuda_syncfree_supported(const ggml_tensor * op) {
    if (op->op != GGML_OP_CUSTOM || !op->src[0] || !op->src[1] || !op->src[2] || !op->src[3] ||
            op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_I32 ||
            op->src[2]->type != GGML_TYPE_I32 || op->src[3]->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1]) ||
            !ggml_is_contiguous(op->src[2]) || !ggml_is_contiguous(op->src[3])) {
        return false;
    }
    const int64_t k = (op->src[3]->ne[0] - 2)/23;
    const int64_t ns = op->src[3]->ne[1];
    if (k < 2 || k > 16 || ns < 1 || ns > 1024 || op->src[3]->ne[0] != 2 + 23*k ||
            op->src[3]->ne[2] != 1 || op->src[3]->ne[3] != 1) {
        return false;
    }
    if (std::strcmp(op->name, "syncfree_accept") == 0) {
        const int64_t positions = op->ne[0] - 4 - k;
        return op->type == GGML_TYPE_I32 && op->ne[1] == ns && op->ne[2] == 1 && op->ne[3] == 1 &&
            op->src[2]->ne[0] == k + 1 && op->src[2]->ne[1] == ns && op->src[2]->ne[2] == 1 && op->src[2]->ne[3] == 1 &&
            op->src[0]->ne[0] >= 1 && op->src[0]->ne[0] <= 40 && op->src[0]->ne[1] == k + 1 &&
            op->src[0]->ne[2] == ns && op->src[0]->ne[3] == 1 &&
            ggml_are_same_shape(op->src[0], op->src[1]) &&
            (positions == k || positions == 4*k);
    }
    if (std::strcmp(op->name, "syncfree_mask") == 0) {
        return op->type == GGML_TYPE_F32 && ggml_are_same_shape(op, op->src[0]) &&
            op->src[1]->ne[0] == op->ne[0] && op->src[1]->ne[1] == ns &&
            op->src[1]->ne[2] == 1 && op->src[1]->ne[3] == 1 &&
            (op->src[2]->ne[0] == 4 + 2*k || op->src[2]->ne[0] == 4 + 5*k) &&
            op->src[2]->ne[1] == ns && op->src[2]->ne[2] == 1 && op->src[2]->ne[3] == 1 &&
            op->ne[0] >= 1 && op->ne[0] <= (1 << 20) && op->ne[2] == 1 &&
            ((op->ne[3] == 1 && op->ne[1] == k*ns) || (op->ne[3] == ns && op->ne[1] == k)) &&
            ggml_nelements(op) <= INT32_MAX - 255;
    }
    return false;
}

void ggml_cuda_syncfree(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_syncfree_supported(dst));
    if (std::strcmp(dst->name, "syncfree_accept") == 0) {
        const int k = dst->src[2]->ne[0] - 1;
        const int n_pos = (dst->ne[0] - 4 - k)/k;
        syncfree_accept_kernel<<<dst->ne[1], 1, 0, ctx.stream()>>>(
            (const float *) dst->src[0]->data, (const int32_t *) dst->src[1]->data,
            (const int32_t *) dst->src[2]->data, (const float *) dst->src[3]->data,
            (int32_t *) dst->data, k, n_pos, dst->src[0]->ne[0]);
    } else {
        const int n = ggml_nelements(dst);
        const int k = (dst->src[3]->ne[0] - 2)/23;
        syncfree_mask_kernel<<<(n + 255)/256, 256, 0, ctx.stream()>>>(
            (const float *) dst->src[0]->data, (const int32_t *) dst->src[1]->data,
            (const int32_t *) dst->src[2]->data, (const float *) dst->src[3]->data, (float *) dst->data,
            dst->ne[0], n/dst->ne[0], k, dst->src[2]->ne[0]);
    }
}
