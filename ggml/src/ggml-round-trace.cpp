#include "ggml-round-trace.h"

#include <cstdio>
#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>

#ifdef GGML_ROUND_TRACE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

struct trace_phase {
    std::string name;
    int n_tokens;
    int result = -999;
    int64_t begin;
    int64_t end = 0;
};

struct trace_event {
    const char * kind;
    size_t phase;
    int device;
    uint64_t count;
    uint64_t bytes;
    int64_t begin;
    int64_t end;
};

struct trace_state {
    uint64_t round = 0;
    std::vector<trace_phase> phases;
    std::vector<trace_event> events;
    bool decoding = false;
};

static thread_local trace_state state;

int64_t ggml_round_trace_now() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

static void trace_flush(int64_t next_begin) {
    if (state.phases.empty()) { return; }
    // Keep writes from separate contexts intact. File I/O stays outside decode ranges.
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    FILE * out = std::fopen(std::getenv("LLAMA_ROUND_TRACE"), "a");
    if (!out) {
        std::perror("LLAMA_ROUND_TRACE");
        std::abort();
    }
    std::fprintf(out, "{\"schema\":1,\"pid\":%ld,\"tid\":%ld,\"round\":%llu,\"next_begin_ns\":%lld,\"phases\":[",
        long(getpid()), long(syscall(SYS_gettid)), (unsigned long long) state.round, (long long) next_begin);
    for (size_t i = 0; i < state.phases.size(); ++i) {
        const auto & p = state.phases[i];
        std::fprintf(out, "%s{\"phase\":\"%s\",\"n_tokens\":%d,\"result\":%d,\"begin_ns\":%lld,\"end_ns\":%lld,\"counters\":[",
            i ? "," : "", p.name.c_str(), p.n_tokens, p.result, (long long) p.begin, (long long) p.end);
        std::map<std::pair<std::string, int>, std::vector<const trace_event *>> groups;
        for (const auto & e : state.events) {
            if (e.phase == i) { groups[{e.kind, e.device}].push_back(&e); }
        }
        bool comma = false;
        for (auto & group : groups) {
            auto & events = group.second;
            std::sort(events.begin(), events.end(), [](const trace_event * a, const trace_event * b) { return a->begin < b->begin; });
            uint64_t count = 0, bytes = 0;
            int64_t ns = 0, end = 0;
            for (const auto * e : events) {
                count += e->count;
                bytes += e->bytes;
                ns += std::max<int64_t>(0, e->end - std::max(end, e->begin));
                end = std::max(end, e->end);
            }
            std::fprintf(out, "%s{\"kind\":\"%s\",\"device\":%d,\"count\":%llu,\"bytes\":%llu,\"host_us\":%.3f}",
                comma ? "," : "", group.first.first.c_str(), group.first.second,
                (unsigned long long) count, (unsigned long long) bytes, ns / 1000.0);
            comma = true;
        }
        std::fprintf(out, "]}");
    }
    std::fprintf(out, "],\"events\":[");
    for (size_t i = 0; i < state.events.size(); ++i) {
        const auto & e = state.events[i];
        std::fprintf(out, "%s{\"kind\":\"%s\",\"phase\":%zu,\"device\":%d,\"count\":%llu,\"bytes\":%llu,\"begin_ns\":%lld,\"end_ns\":%lld}",
            i ? "," : "", e.kind, e.phase, e.device, (unsigned long long) e.count, (unsigned long long) e.bytes,
            (long long) e.begin, (long long) e.end);
    }
    std::fprintf(out, "]}\n");
    if (std::fclose(out) != 0) { std::perror("LLAMA_ROUND_TRACE"); std::abort(); }
    state.phases.clear();
    state.events.clear();
}

void ggml_round_trace_begin(const char * phase, int n_tokens) {
    if (!ggml_round_trace_enabled()) { return; }
    if (phase[0] == 'V') {
        trace_flush(ggml_round_trace_now());
        ++state.round;
    }
    if (!state.round) { return; }
    state.phases.push_back({phase, n_tokens, -999, ggml_round_trace_now(), 0});
    state.decoding = true;
#ifdef GGML_ROUND_TRACE_NVTX
    const std::string label = "round_trace:" + std::to_string(syscall(SYS_gettid)) + ":" +
        std::to_string(state.round) + ":" + phase + ":" + std::to_string(state.phases.size() - 1);
    nvtxRangePushA(label.c_str());
#endif
}

void ggml_round_trace_end(int result) {
    if (!ggml_round_trace_enabled() || !state.decoding) { return; }
    state.phases.back().end = ggml_round_trace_now();
    state.phases.back().result = result;
    state.decoding = false;
#ifdef GGML_ROUND_TRACE_NVTX
    nvtxRangePop();
#endif
}

void ggml_round_trace_event(const char * kind, int64_t begin, int64_t end, int device, uint64_t count, uint64_t bytes) {
    if (state.phases.empty()) { return; }
    state.events.push_back({kind, state.phases.size() - 1, device, count, bytes, begin, end});
}

void ggml_round_trace_flush() {
    if (ggml_round_trace_enabled()) { trace_flush(0); }
}
