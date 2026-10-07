#include "ggml-round-trace.h"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>

static bool deterministic = true;
static thread_local long test_tid = 200;
static thread_local int64_t test_ns = 1000000000;

static int64_t wall_now() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

// Only trace timestamps and producer IDs are replaced. Worker scheduling and file I/O stay real.
static int test_clock_gettime(clockid_t clock, timespec * ts) {
    if (!deterministic) { return clock_gettime(clock, ts); }
    test_ns += 17;
    ts->tv_sec = test_ns / 1000000000;
    ts->tv_nsec = test_ns % 1000000000;
    return 0;
}
static pid_t test_getpid() { return 100; }
static long test_syscall(long call, ...) {
    if (call != SYS_gettid) { std::abort(); }
    return test_tid;
}

#define clock_gettime test_clock_gettime
#define getpid test_getpid
#define syscall test_syscall
#ifndef GGML_ROUND_TRACE_TEST_SOURCE
#define GGML_ROUND_TRACE_TEST_SOURCE "../ggml/src/ggml-round-trace.cpp"
#endif
#include GGML_ROUND_TRACE_TEST_SOURCE
#undef clock_gettime
#undef getpid
#undef syscall

static std::string contents() {
    const char * path = std::getenv("LLAMA_ROUND_TRACE");
    if (!path) { return {}; }
    std::ifstream in(path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

static void round_events(int round) {
    ggml_round_trace_begin("V", 4);
    const auto base = ggml_round_trace_now();
    // Deliberately unsorted, overlapping intervals; device and kind both affect grouping.
    ggml_round_trace_event("zeta", base + 50, base + 100, 1, 2, 77);
    ggml_round_trace_event("alpha", base + 30, base + 90, 0, 3, 88);
    ggml_round_trace_event("alpha", base + 10, base + 60, 0, 5, 99);
    ggml_round_trace_event("alpha", base + 20, base + 40, 1, 7, 111);
    ggml_round_trace_event("alpha", base + 10, base + 60, 0, 11, 222);
    ggml_round_trace_count("empty");
    ggml_round_trace_end(round);
    ggml_round_trace_end(-123);
    ggml_round_trace_begin("C", 1);
    { ggml_round_trace_scope scope("submit", -1, 13, 333); }
    ggml_round_trace_end(0);
    for (int i = 1; i <= 3; ++i) {
        const std::string phase = "D" + std::to_string(i);
        ggml_round_trace_begin(phase.c_str(), 1);
        ggml_round_trace_event("input", base + i, base + i + 100, i % 2, i, 400 + i);
        ggml_round_trace_end(i);
    }
}

static int visible(size_t expected) {
    const auto data = contents();
    const auto n = size_t(std::count(data.begin(), data.end(), '\n'));
    if (n == expected) { return 0; }
    std::fprintf(stderr, "flush visibility: expected %zu records, got %zu\n", expected, n);
    return 15;
}

static int emit() {
    ggml_round_trace_event("outside", 1, 2);
    ggml_round_trace_begin("C", 1);
    for (int r = 0; r < 5; ++r) { round_events(r); }
    ggml_round_trace_flush();
    if (visible(5)) { return 15; }
    // A flush ends this writer session, but does not disable future rounds.
    for (int r = 5; r < 8; ++r) { round_events(r); }
    ggml_round_trace_flush();
    ggml_round_trace_flush();
    if (visible(8)) { return 15; }
    // Separate TLS contexts, serialized submissions: compare order and producer IDs exactly.
    for (int t = 0; t < 3; ++t) {
        std::thread producer([t] {
            test_tid = 201 + t;
            test_ns = 2000000000 + t * 1000000;
            for (int r = 0; r < 3; ++r) { round_events(r); }
            ggml_round_trace_flush();
        });
        producer.join();
    }
    return visible(17);
}

static int ownership() {
    ggml_round_trace_begin("V", 4);
    std::string kind(96, 'a');
    const auto now = ggml_round_trace_now();
    ggml_round_trace_event(kind.c_str(), now, now + 1);
    kind.assign(96, 'b');
    ggml_round_trace_flush();
    if (contents().find(std::string(96, 'a')) == std::string::npos) {
        std::fprintf(stderr, "kind ownership failed\n");
        return 16;
    }
    return 0;
}

static int stress() {
    std::vector<std::thread> producers;
    for (int t = 0; t < 4; ++t) {
        producers.emplace_back([t] {
            test_tid = 300 + t;
            test_ns += t * 1000000;
            for (int r = 0; r < 192; ++r) { round_events(r); }
            // Exercise concurrent finish calls and producer backpressure.
            ggml_round_trace_flush();
        });
    }
    for (auto & t : producers) { t.join(); }
    ggml_round_trace_flush();
    return visible(768);
}

static int timing() {
    deterministic = false;
    std::vector<int64_t> elapsed;
    for (int r = 0; r < 16; ++r) {
        ggml_round_trace_begin("V", 4);
        for (int i = 0; i < 1800; ++i) {
            ggml_round_trace_event("input", 1000000000000000 + i, 1000000000000001 + i, i % 2, 1, 4096);
        }
        ggml_round_trace_end(0);
        const auto start = wall_now();
        ggml_round_trace_begin("V", 4);
        elapsed.push_back(wall_now() - start);
        ggml_round_trace_end(0);
        // Drain outside the timed span so the test measures enqueue rather than a growing backlog.
        ggml_round_trace_flush();
    }
    std::sort(elapsed.begin(), elapsed.end());
    std::printf("{\"begin_v_median_ns_M\":%lld,\"rounds\":16,\"events_per_record\":1800}\n", (long long) elapsed[elapsed.size() / 2]);
    return 0;
}

int main(int argc, char ** argv) {
    const std::string mode = argc == 2 ? argv[1] : "";
    if (mode == "--emit") { return emit(); }
    if (mode == "--ownership") { return ownership(); }
    if (mode == "--stress") { return stress(); }
    if (mode == "--timing") { return timing(); }
    if (mode == "--exit-tail") {
        round_events(0);
        std::thread producer([] { test_tid = 201; round_events(0); });
        producer.join();
        return 0;
    }
    if (mode == "--disabled") {
        round_events(0);
        ggml_round_trace_flush();
        if (!state.phases.empty() || !state.events.empty()) { return 17; }
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.compare(0, 8, "Threads:") == 0 && std::stoi(line.substr(8)) != 1) { return 17; }
        }
        return 0;
    }
    std::fprintf(stderr, "expected --emit, --ownership, --stress, --timing, --exit-tail or --disabled\n");
    return 18;
}
