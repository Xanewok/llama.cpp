#pragma once

#include "ggml.h"
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

// DO NOT MERGE: single submission-thread attribution, enabled once at process startup.
inline bool ggml_round_trace_enabled() {
    static const bool enabled = [] { const char * p = std::getenv("LLAMA_ROUND_TRACE"); return p && *p; }();
    return enabled;
}

GGML_API int64_t ggml_round_trace_now();
GGML_API void ggml_round_trace_begin(const char * phase, int n_tokens);
GGML_API void ggml_round_trace_end(int result);
// Event, count and scope kinds must use immutable static storage; the writer can outlive the producer round.
GGML_API void ggml_round_trace_event(const char * kind, int64_t begin, int64_t end,
                                   int device = -1, uint64_t count = 1, uint64_t bytes = 0);
GGML_API void ggml_round_trace_flush();

struct ggml_round_trace_scope {
    const char * kind;
    int device;
    uint64_t count;
    uint64_t bytes;
    int64_t begin;

    explicit ggml_round_trace_scope(const char * kind, int device = -1, uint64_t count = 1, uint64_t bytes = 0)
        : kind(kind), device(device), count(count), bytes(bytes), begin(ggml_round_trace_enabled() ? ggml_round_trace_now() : -1) {}
    ~ggml_round_trace_scope() {
        if (begin >= 0) { ggml_round_trace_event(kind, begin, ggml_round_trace_now(), device, count, bytes); }
    }
};

inline void ggml_round_trace_count(const char * kind, int device = -1) {
    if (ggml_round_trace_enabled()) {
        const auto now = ggml_round_trace_now();
        ggml_round_trace_event(kind, now, now, device);
    }
}

// DO NOT MERGE: additive schema-1 gpu array. Times use CLOCK_MONOTONIC nanoseconds.
// kind distinguishes compute, allreduce, transfer and wait; wait is not GPU busy.
// stream identifies a stream within its device; phase indexes phases.
// error_ns covers the reference bracket and approximate event timer precision, assuming equal host/GPU clock rates.
inline bool ggml_round_trace_gpu_enabled() {
    if (!ggml_round_trace_enabled()) { return false; }
    static const bool enabled = [] { const char * p = std::getenv("LLAMA_ROUND_TRACE_GPU"); return p && p[0] == '1' && p[1] == '\0'; }();
    return enabled;
}

struct ggml_round_trace_gpu_interval {
    const char * kind;
    size_t phase;
    int device;
    uint64_t stream;
    int64_t begin_ns;
    int64_t end_ns;
    int64_t error_ns;
};

inline int64_t ggml_round_trace_gpu_map(int64_t lo, int64_t hi, float before_reference_ms) {
    return lo + (hi - lo) / 2 - std::llround(double(before_reference_ms) * 1e6);
}

inline int64_t ggml_round_trace_gpu_error(int64_t lo, int64_t hi, float before_reference_ms) {
    return (hi - lo + 1) / 2 + 500 + int64_t(std::ceil(std::abs(double(before_reference_ms)) * 1e6 * std::numeric_limits<float>::epsilon()));
}

struct ggml_round_trace_gpu_source {
    virtual ~ggml_round_trace_gpu_source() = default;
    // Called only by the writer, after the producer has submitted the complete round.
    virtual void read(std::vector<ggml_round_trace_gpu_interval> & out) = 0;
};

GGML_API uint64_t ggml_round_trace_gpu_round(size_t & phase);
GGML_API void ggml_round_trace_gpu_defer(std::shared_ptr<ggml_round_trace_gpu_source> source);
