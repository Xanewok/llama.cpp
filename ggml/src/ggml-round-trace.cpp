#include "ggml-round-trace.h"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
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
    std::string kind;
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

    ~trace_state();
};

static thread_local trace_state state;

int64_t ggml_round_trace_now() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

struct trace_record {
    uint64_t round;
    long pid;
    long tid;
    int64_t next_begin;
    std::vector<trace_phase> phases;
    std::vector<trace_event> events;
};

static FILE * trace_open() {
    FILE * out = std::fopen(std::getenv("LLAMA_ROUND_TRACE"), "a");
    if (!out) {
        std::perror("LLAMA_ROUND_TRACE");
        std::abort();
    }
    return out;
}

static void trace_write(FILE * out, const trace_record & record) {
    std::fprintf(out, "{\"schema\":1,\"pid\":%ld,\"tid\":%ld,\"round\":%llu,\"next_begin_ns\":%lld,\"phases\":[",
        record.pid, record.tid, (unsigned long long) record.round, (long long) record.next_begin);
    for (size_t i = 0; i < record.phases.size(); ++i) {
        const auto & p = record.phases[i];
        std::fprintf(out, "%s{\"phase\":\"%s\",\"n_tokens\":%d,\"result\":%d,\"begin_ns\":%lld,\"end_ns\":%lld,\"counters\":[",
            i ? "," : "", p.name.c_str(), p.n_tokens, p.result, (long long) p.begin, (long long) p.end);
        std::map<std::pair<std::string, int>, std::vector<const trace_event *>> groups;
        for (const auto & e : record.events) {
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
    for (size_t i = 0; i < record.events.size(); ++i) {
        const auto & e = record.events[i];
        std::fprintf(out, "%s{\"kind\":\"%s\",\"phase\":%zu,\"device\":%d,\"count\":%llu,\"bytes\":%llu,\"begin_ns\":%lld,\"end_ns\":%lld}",
            i ? "," : "", e.kind.c_str(), e.phase, e.device, (unsigned long long) e.count, (unsigned long long) e.bytes,
            (long long) e.begin, (long long) e.end);
    }
    std::fprintf(out, "]}\n");
    if (std::ferror(out)) { std::perror("LLAMA_ROUND_TRACE"); std::abort(); }
}

struct trace_writer {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<trace_record> queue;
    std::thread worker;
    bool stopping = false;
    const bool sync = [] { const char * p = std::getenv("GGML_ROUND_TRACE_SYNC"); return p && std::string(p) == "1"; }();

    ~trace_writer() { finish(); }

    void run() {
        FILE * out = trace_open();
        std::vector<char> buffer(1024 * 1024);
        if (std::setvbuf(out, buffer.data(), _IOFBF, buffer.size()) != 0) {
            std::perror("LLAMA_ROUND_TRACE setvbuf");
            std::abort();
        }
        while (true) {
            trace_record record;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [this] { return stopping || !queue.empty(); });
                if (queue.empty()) { break; }
                record = std::move(queue.front());
                queue.pop_front();
            }
            changed.notify_all();
            trace_write(out, record);
            // Readers can inspect each complete round before the server stops.
            if (std::fflush(out) != 0) { std::perror("LLAMA_ROUND_TRACE"); std::abort(); }
        }
        if (std::fclose(out) != 0) { std::perror("LLAMA_ROUND_TRACE"); std::abort(); }
    }

    void submit(trace_record record) {
        std::unique_lock<std::mutex> lock(mutex);
        // At most 64 pending rounds, plus the round owned by the writer. Backpressure must not drop records.
        changed.wait(lock, [this] { return !stopping && queue.size() < 64; });
        if (sync) {
            FILE * out = trace_open();
            trace_write(out, record);
            if (std::fclose(out) != 0) { std::perror("LLAMA_ROUND_TRACE"); std::abort(); }
            return;
        }
        queue.push_back(std::move(record));
        if (!worker.joinable()) { worker = std::thread([this] { run(); }); }
        lock.unlock();
        changed.notify_all();
    }

    void finish() {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [this] { return !stopping; });
        if (!worker.joinable()) { return; }
        stopping = true;
        lock.unlock();
        changed.notify_all();
        worker.join();
        lock.lock();
        stopping = false;
        lock.unlock();
        changed.notify_all();
    }
};

static trace_writer & trace_get_writer() {
    static trace_writer writer;
    return writer;
}

static void trace_flush(int64_t next_begin) {
    if (state.phases.empty()) { return; }
    trace_record record {state.round, long(getpid()), long(syscall(SYS_gettid)), next_begin, {}, {}};
    record.phases = std::move(state.phases);
    record.events = std::move(state.events);
    trace_get_writer().submit(std::move(record));
}

trace_state::~trace_state() {
    // Thread-local tails must be queued before the process-wide writer is destroyed.
    if (ggml_round_trace_enabled()) { trace_flush(0); }
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
    if (ggml_round_trace_enabled()) {
        trace_flush(0);
        trace_get_writer().finish();
    }
}
