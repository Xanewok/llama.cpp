// local: fable-bcd probe. See fable-probe.h.
#include "common.cuh"
#include "fable-probe.h"
#include "mmvq.cuh"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstdint>
#include <vector>

static __device__ __forceinline__ unsigned long long fable_globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

static __global__ void fable_flush_kernel(float4 * buf, size_t n4) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = buf[i];
        v.x += 1.0f;
        buf[i] = v;
    }
}

// 8 blocks like ggml_cuda_ar_kernel: the threads other than 0 issue the prefetch, thread 0 spins for the wait.
static __global__ void fable_spin_kernel(const char * pf, size_t pf_bytes, int mode, unsigned long long spin_ns, unsigned * sink) {
    const size_t tid  = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nthr = (size_t) gridDim.x * blockDim.x;
    unsigned acc = 0;
    if (mode == 1) {
        for (size_t off = tid * 128; off < pf_bytes; off += nthr * 128) {
            ggml_cuda_fable_prefetch_l2(pf + off);
        }
    } else if (mode == 2) {
        for (size_t off = tid * 128; off < pf_bytes; off += nthr * 128 * 8) {
            unsigned v[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const size_t o = off + (size_t) u * nthr * 128;
                unsigned t = 0;
                if (o < pf_bytes) {
                    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(t) : "l"(pf + o));
                }
                v[u] = t;
            }
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                acc ^= v[u];
            }
        }
    } else if (mode == 3) {
#if __CUDA_ARCH__ >= 900
        for (size_t off = tid * 65536; off < pf_bytes; off += nthr * 65536) {
            const unsigned n = (unsigned) min((size_t) 65536, pf_bytes - off);
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(pf + off), "r"(n) : "memory");
        }
#endif // __CUDA_ARCH__ >= 900
    }
    if (spin_ns > 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        const unsigned long long t0 = fable_globaltimer();
        while (fable_globaltimer() - t0 < spin_ns) {
        }
    }
    if (acc == 0x9e3779b9u) {
        *sink = acc;
    }
}

static __global__ void fable_fill_kernel(char * dst, size_t n, const char * pat, size_t patn) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = pat[i % patn];
    }
}

static ggml_backend_cuda_context * fable_ctx(ggml_backend_t be) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) be->context;
    ggml_cuda_set_device(ctx->device);
    return ctx;
}

static size_t fable_l2_bytes(ggml_backend_t be) {
    ggml_backend_cuda_context * ctx = fable_ctx(be);
    int v = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrL2CacheSize, ctx->device));
    return (size_t) v;
}

static int fable_nsm(ggml_backend_t be) {
    ggml_backend_cuda_context * ctx = fable_ctx(be);
    return ggml_cuda_info().devices[ctx->device].nsm;
}

static int fable_route_mmvq(ggml_backend_t be, int type, int n) {
    ggml_backend_cuda_context * ctx = fable_ctx(be);
    const int cc = ggml_cuda_info().devices[ctx->device].cc;
    return ggml_cuda_should_use_mmvq((ggml_type) type, cc, n) ? 1 : 0;
}

static void * fable_alloc(ggml_backend_t be, size_t n) {
    fable_ctx(be);
    void * p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n));
    CUDA_CHECK(cudaMemset(p, 0, n));
    return p;
}

static void fable_free(ggml_backend_t be, void * p) {
    fable_ctx(be);
    CUDA_CHECK(cudaFree(p));
}

static void fable_fill_pattern(ggml_backend_t be, void * dst, size_t n, const void * host_pat, size_t patn) {
    ggml_backend_cuda_context * ctx = fable_ctx(be);
    void * d_pat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pat, patn));
    CUDA_CHECK(cudaMemcpy(d_pat, host_pat, patn, cudaMemcpyHostToDevice));
    fable_fill_kernel<<<1024, 256, 0, ctx->stream()>>>((char *) dst, n, (const char *) d_pat, patn);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream()));
    CUDA_CHECK(cudaFree(d_pat));
}

static void fable_run(ggml_backend_t be, const fable_arm * arm, int reps, fable_result * out) {
    ggml_backend_cuda_context * ctx = fable_ctx(be);
    cudaStream_t s = ctx->stream();
    out->rc = 0;

    // Calls 1-2 run direct, call 3 captures the CUDA graph under this arm's flag, later calls replay it.
    g_fable_probe_pf = arm->pf_flag;
    for (int i = 0; i < 4; ++i) {
        if (ggml_backend_graph_compute_async(be, arm->gf) != GGML_STATUS_SUCCESS) {
            out->rc = 1;
            return;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(s));

    const int nsm = ggml_cuda_info().devices[ctx->device].nsm;
    std::vector<cudaEvent_t> e0(reps);
    std::vector<cudaEvent_t> e1(reps);
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventCreate(&e0[i]));
        CUDA_CHECK(cudaEventCreate(&e1[i]));
    }
    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0, s));
    const int64_t h0 = ggml_time_us();
    for (int r = 0; r < reps; ++r) {
        if (arm->flush) {
            fable_flush_kernel<<<nsm * 4, 256, 0, s>>>((float4 *) arm->flush_buf, arm->flush_bytes / sizeof(float4));
            CUDA_CHECK(cudaGetLastError());
        }
        if (arm->spin_ns > 0 || arm->mode > 0) {
            fable_spin_kernel<<<8, 256, 0, s>>>((const char *) arm->pf_ptr, arm->pf_bytes, arm->mode, arm->spin_ns, arm->sink);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(e0[r], s));
        if (ggml_backend_graph_compute_async(be, arm->gf) != GGML_STATUS_SUCCESS) {
            out->rc = 2;
        }
        CUDA_CHECK(cudaEventRecord(e1[r], s));
    }
    const int64_t h1 = ggml_time_us();
    CUDA_CHECK(cudaEventRecord(t1, s));
    CUDA_CHECK(cudaEventSynchronize(t1));

    std::vector<float> us(reps);
    for (int r = 0; r < reps; ++r) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0[r], e1[r]));
        us[r] = ms * 1000.0f;
    }
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, t0, t1));

    std::sort(us.begin(), us.end());
    const int cut = reps / 10;
    double sum = 0.0;
    for (int r = cut; r < reps - cut; ++r) {
        sum += us[r];
    }
    out->trim_us   = sum / (double) (reps - 2 * cut);
    out->median_us = us[reps / 2];
    out->total_us  = (double) total_ms * 1000.0 / reps;
    out->host_us   = (double) (h1 - h0) / reps;

    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventDestroy(e0[i]));
        CUDA_CHECK(cudaEventDestroy(e1[i]));
    }
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    g_fable_probe_pf = 0;
}

extern "C" const struct fable_api * fable_probe_api(void) {
    static const struct fable_api api = {
        fable_l2_bytes, fable_nsm, fable_route_mmvq, fable_alloc, fable_free, fable_fill_pattern, fable_run,
    };
    return &api;
}
