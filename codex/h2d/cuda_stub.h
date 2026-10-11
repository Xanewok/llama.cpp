// Host stand-in for the CUDA runtime (copied from wt/s10-k11b, extended with the input scatter) and backend context, just enough to compile and run the input staging code of
// ggml-cuda.cu on a machine without CUDA. Streams are queues: an operation runs only when the stream is progressed,
// an event completes when its stream has run past it, and cudaMemcpyAsync reads its source when it runs, so a host
// that reuses a staging slot too early corrupts the copy, as on a GPU.
#pragma once

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-round-trace.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#define CUDART_VERSION 13000
#define GGML_CUDA_MAX_DEVICES 16

enum cudaError_t { cudaSuccess = 0, cudaErrorNotReady = 600 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
enum { cudaHostAllocDefault = 0, cudaHostAllocPortable = 1 };
enum { cudaEventDisableTiming = 2, cudaStreamNonBlocking = 1 };
enum cudaMemcpySrcAccessOrder { cudaMemcpySrcAccessOrderStream };
struct cudaMemcpyAttributes { cudaMemcpySrcAccessOrder srcAccessOrder; };

struct CUstream_st {
    int device = 0;
    std::deque<std::function<void()>> queue;
    uint64_t enqueued = 0;
    uint64_t done = 0;
};
struct CUevent_st {
    CUstream_st * stream = nullptr;
    uint64_t seq = 0;
    bool recorded = false;
};
typedef CUstream_st * cudaStream_t;
typedef CUevent_st * cudaEvent_t;

struct sim_state {
    int current_device = 0;
    double link_ms[8] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    bool malloc_fails = false;
    int portable_allocs = 0;
    int default_allocs = 0;
    int event_syncs = 0;
    int slot_waits = 0;
    int memcpy_calls = 0;
    int batch_calls = 0;
    int scatter_launches = 0;
    std::vector<CUstream_st *> streams;
};
inline sim_state & sim() { static sim_state s; return s; }

// One queued operation runs per call; false if the stream is idle.
inline bool sim_step(CUstream_st * s) {
    if (s->queue.empty()) {
        return false;
    }
    auto op = std::move(s->queue.front());
    s->queue.pop_front();
    op();
    ++s->done;
    return true;
}
inline void sim_drain(CUstream_st * s, uint64_t until = UINT64_MAX) {
    while (s->done < until && sim_step(s)) {
    }
}
inline void sim_enqueue(CUstream_st * s, std::function<void()> op) {
    s->queue.push_back(std::move(op));
    ++s->enqueued;
}

inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t * s, unsigned) {
    *s = new CUstream_st;
    (*s)->device = sim().current_device;
    sim().streams.push_back(*s);
    return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t s) {
    sim_drain(s);
    sim().streams.erase(std::find(sim().streams.begin(), sim().streams.end(), s));
    delete s;
    return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t s) { sim_drain(s); return cudaSuccess; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t * e, unsigned) { *e = new CUevent_st; return cudaSuccess; }
inline cudaError_t cudaEventCreate(cudaEvent_t * e) { *e = new CUevent_st; return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t e) { delete e; return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s) {
    e->stream = s;
    e->seq = s->enqueued;
    e->recorded = true;
    return cudaSuccess;
}
inline cudaError_t cudaEventQuery(cudaEvent_t e) {
    return !e->recorded || e->stream->done >= e->seq ? cudaSuccess : cudaErrorNotReady;
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t e) {
    ++sim().event_syncs;
    if (e->recorded) {
        sim_drain(e->stream, e->seq);
    }
    return cudaSuccess;
}
inline cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t, cudaEvent_t end) {
    *ms = (float) sim().link_ms[end->stream->device];
    return cudaSuccess;
}
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) { return cudaSuccess; }
inline cudaError_t cudaHostAlloc(void ** p, size_t n, unsigned flags) {
    ++(flags & cudaHostAllocPortable ? sim().portable_allocs : sim().default_allocs);
    *p = malloc(n);
    return *p ? cudaSuccess : (cudaError_t) 2;
}
inline cudaError_t cudaFreeHost(void * p) { free(p); return cudaSuccess; }
inline cudaError_t cudaMalloc(void ** p, size_t n) {
    *p = sim().malloc_fails ? nullptr : malloc(n);
    return *p ? cudaSuccess : (cudaError_t) 2;
}
inline cudaError_t cudaFree(void * p) { free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t n, cudaMemcpyKind, cudaStream_t s) {
    ++sim().memcpy_calls;
    sim_enqueue(s, [=] { memcpy(dst, src, n); });
    return cudaSuccess;
}
inline cudaError_t cudaMemcpyBatchAsync(void * const * dsts, const void * const * srcs, size_t * sizes, size_t count, cudaMemcpyAttributes *, size_t * attrs_index, size_t, cudaStream_t s) {
    *attrs_index = 0;
    ++sim().batch_calls;
    std::vector<void *> d(dsts, dsts + count);
    std::vector<const void *> c(srcs, srcs + count);
    std::vector<size_t> n(sizes, sizes + count);
    sim_enqueue(s, [=] {
        for (size_t i = 0; i < count; ++i) {
            memcpy(d[i], c[i], n[i]);
        }
    });
    return cudaSuccess;
}

#define CUDA_CHECK(x)                                                                  \
    do {                                                                               \
        const cudaError_t e_ = (x);                                                    \
        if (e_ != cudaSuccess) {                                                       \
            fprintf(stderr, "sim: CUDA error %d at %s:%d: %s\n", (int) e_, __FILE__, __LINE__, #x); \
            abort();                                                                   \
        }                                                                              \
    } while (0)

inline void ggml_cuda_set_device(int device) { sim().current_device = device; }

struct ggml_backend_cuda_context {
    int device = 0;
    cudaStream_t compute = nullptr;
    struct input_slot {
        void * data = nullptr;
        size_t capacity = 0;
        cudaEvent_t uploaded = nullptr;
        bool pending = false;
    };
    input_slot input_slots[2];
    size_t input_next = 0;
    cudaStream_t input_stream = nullptr;
    cudaEvent_t input_compute_done = nullptr;
    std::vector<void *> input_dsts;
    std::vector<const void *> input_srcs;
    std::vector<size_t> input_sizes;
    void * input_scatter_buf = nullptr;
    size_t input_scatter_capacity = 0;
    cudaStream_t stream() {
        if (!compute) {
            ggml_cuda_set_device(device);
            cudaStreamCreateWithFlags(&compute, 0);
        }
        return compute;
    }
};

inline const char * sim_backend_name(ggml_backend_t) { return "sim-cuda"; }
inline bool ggml_backend_is_cuda(ggml_backend_t backend) { return backend && backend->iface.get_name == sim_backend_name; }
inline ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static ggml_backend_buffer_type types[8];
    return &types[device];
}

// The scatter kernel runs on the host when its stream reaches it: every thread of the real grid, in thread order, so
// entries interleave (overlapping destinations come out in no list order, as on a GPU).
#define __host__
#define __device__
#define __forceinline__ inline
struct int4 { int x, y, z, w; };
#include SCATTER_SNIPPET
inline void ggml_cuda_input_scatter(const ggml_cuda_input_scatter_args & args, size_t max_size, cudaStream_t s) {
    ++sim().scatter_launches;
    const size_t nthreads = (size_t) ggml_cuda_input_scatter_blocks(max_size) * GGML_CUDA_INPUT_SCATTER_THREADS;
    sim_enqueue(s, [args, nthreads] {
        for (size_t t = 0; t < nthreads; ++t) {
            ggml_cuda_input_scatter_thread(args, t, nthreads);
        }
    });
}
