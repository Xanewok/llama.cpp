#include "ggml-round-trace.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <sstream>
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
    static const char kind[] = "static_event_kind_longer_than_sso";
    std::string phase = "V" + std::string(96, 'a');
#ifdef GGML_ROUND_TRACE_TEST_HOOKS
    auto & writer = trace_get_writer();
    {
        std::lock_guard<std::mutex> lock(writer.mutex);
        writer.test_pause = !writer.sync;
    }
#endif
    ggml_round_trace_begin(phase.c_str(), 7);
    const auto now = ggml_round_trace_now();
    ggml_round_trace_event(kind, now, now + 1, 2, 23, 456);
    ggml_round_trace_end(91);
    const auto original = phase;
    ggml_round_trace_begin("V", 4);
    phase.assign(97, 'b');
    state.phases.back().name = "mutated_phase";
    state.phases.back().n_tokens = 42;
    ggml_round_trace_event("mutated_event", now + 10, now + 20, 3, 99, 999);
    ggml_round_trace_end(-7);
#ifdef GGML_ROUND_TRACE_TEST_HOOKS
    {
        std::lock_guard<std::mutex> lock(writer.mutex);
        writer.test_pause = false;
    }
    writer.changed.notify_all();
#endif
    ggml_round_trace_flush();
    const auto data = contents();
    const auto first_end = data.find('\n');
    const auto first = data.substr(0, first_end);
    if (first.find("\"phase\":\"" + original + "\",\"n_tokens\":7,\"result\":91") == std::string::npos ||
        first.find("\"kind\":\"" + std::string(kind) + "\",\"phase\":0,\"device\":2,\"count\":23,\"bytes\":456") == std::string::npos ||
        first.find("mutated_") != std::string::npos) {
        std::fprintf(stderr, "record ownership failed after producer state mutation\n");
        return 16;
    }
    return 0;
}

static int stress() {
#ifdef GGML_ROUND_TRACE_TEST_HOOKS
    auto & writer = trace_get_writer();
    {
        std::lock_guard<std::mutex> lock(writer.mutex);
        writer.test_pause = !writer.sync;
    }
#endif
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
    bool reached = true;
#ifdef GGML_ROUND_TRACE_TEST_HOOKS
    if (!writer.sync) {
        std::unique_lock<std::mutex> lock(writer.mutex);
        reached = writer.changed.wait_for(lock, std::chrono::seconds(30), [&writer] { return writer.test_full_queue_waits > 0; });
        writer.test_pause = false;
        lock.unlock();
        writer.changed.notify_all();
    }
#endif
    for (auto & t : producers) { t.join(); }
    ggml_round_trace_flush();
    if (!reached) {
        std::fprintf(stderr, "stress did not reach 64 pending rounds and block a producer\n");
        return 20;
    }
    if (visible(768)) { return 15; }
    uint64_t next[4] = {1, 1, 1, 1};
    std::istringstream lines(contents());
    std::string line;
    while (std::getline(lines, line)) {
        const auto tid = std::stol(line.substr(line.find("\"tid\":") + 6));
        const auto round = std::stoull(line.substr(line.find("\"round\":") + 8));
        if (tid < 300 || tid >= 304 || round != next[tid - 300]++) {
            std::fprintf(stderr, "stress producer order failed: tid=%ld round=%llu\n", tid, round);
            return 19;
        }
    }
    for (auto n : next) {
        if (n != 193) { return 19; }
    }
#ifdef GGML_ROUND_TRACE_TEST_HOOKS
    std::printf("{\"records_D\":768,\"full_queue_waits_M\":%llu}\n", (unsigned long long) writer.test_full_queue_waits);
#endif
    return 0;
}

static int timing() {
    deterministic = false;
    std::vector<int64_t> elapsed;
    ggml_round_trace_begin("V", 4);
    ggml_round_trace_end(0);
    ggml_round_trace_begin("V", 4);
    for (int r = 0; r < 16; ++r) {
        for (int i = 0; i < 1800; ++i) {
            ggml_round_trace_event("input", 1000000000000000 + i, 1000000000000001 + i, i % 2, 1, 4096);
        }
        ggml_round_trace_end(0);
        const auto start = wall_now();
        ggml_round_trace_begin("V", 4);
        elapsed.push_back(wall_now() - start);
        ggml_round_trace_end(0);
    }
    // The warm worker stays alive throughout the loop; these records fit below the queue bound.
    ggml_round_trace_flush();
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
