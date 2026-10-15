#pragma once

#include "common.cuh"
#include "ggml-round-trace.h"

// DO NOT MERGE: events stay outside graph capture.
struct ggml_cuda_trace_batch;
struct ggml_cuda_trace_scope {
    ggml_cuda_trace_batch * batch = nullptr;
    size_t index = 0;
    cudaStream_t stream;

    ggml_cuda_trace_scope(const char * kind, int device, cudaStream_t stream, bool enabled = true) : stream(stream) {
        if (enabled && ggml_round_trace_gpu_enabled()) { begin(kind, device); }
    }
    ~ggml_cuda_trace_scope() { if (batch) { end(); } }
    void begin(const char * kind, int device);
    void end();
    ggml_cuda_trace_scope(const ggml_cuda_trace_scope &) = delete;
    ggml_cuda_trace_scope & operator=(const ggml_cuda_trace_scope &) = delete;
};

inline void ggml_cuda_trace_wait(int device, cudaStream_t stream, cudaEvent_t event, unsigned flags = 0) {
    ggml_cuda_trace_scope gpu_wait("wait", device, stream);
    CUDA_CHECK(cudaStreamWaitEvent(stream, event, flags));
}
