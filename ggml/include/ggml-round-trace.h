#pragma once

#include "ggml.h"
#include <cstdlib>
#include <cstdint>

// DO NOT MERGE: single submission-thread attribution, enabled once at process startup.
inline bool ggml_round_trace_enabled() {
    static const bool enabled = [] { const char * p = std::getenv("LLAMA_ROUND_TRACE"); return p && *p; }();
    return enabled;
}

GGML_API int64_t ggml_round_trace_now();
GGML_API void ggml_round_trace_begin(const char * phase, int n_tokens);
GGML_API void ggml_round_trace_end(int result);
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
