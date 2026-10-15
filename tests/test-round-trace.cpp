#include "ggml-round-trace.h"

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <unistd.h>

static void check(bool ok) {
    if (!ok) { throw std::runtime_error("round trace check failed"); }
}

struct source : ggml_round_trace_gpu_source {
    std::thread::id producer = std::this_thread::get_id();
    std::atomic<int> & reads;
    std::atomic<int> & frees;

    source(std::atomic<int> & reads, std::atomic<int> & frees) : reads(reads), frees(frees) {}
    ~source() { ++frees; }
    void read(std::vector<ggml_round_trace_gpu_interval> & out) override {
        check(std::this_thread::get_id() != producer);
        ++reads;
        out.push_back({"compute", 0, 0, 17, 100, 200, 501});
        out.push_back({"wait", 1, 1, 18, 200, 300, 502});
    }
};

int main() {
    check(ggml_round_trace_gpu_map(10000000, 10002000, 2.0f) == 8001000);
    check(ggml_round_trace_gpu_error(10000000, 10002000, 2.0f) == 1501);
    check(ggml_round_trace_gpu_map(100, 102, 0.000002f) == 99);
    check(ggml_round_trace_gpu_map(100, 102, -0.000002f) == 103);
    char path[] = "/tmp/test-round-trace-XXXXXX";
    int fd = mkstemp(path);
    check(fd >= 0);
    close(fd);
    setenv("LLAMA_ROUND_TRACE", path, 1);
    setenv("LLAMA_ROUND_TRACE_GPU", "1", 1);
    setenv("GGML_ROUND_TRACE_SYNC", "1", 1);
    check(ggml_round_trace_gpu_enabled());
    std::atomic<int> reads{0}, frees{0};
    size_t phase = 99;
    check(ggml_round_trace_gpu_round(phase) == 0);
    ggml_round_trace_begin("V", 6);
    check(ggml_round_trace_gpu_round(phase) == 1 && phase == 0);
    ggml_round_trace_gpu_defer(std::make_shared<source>(reads, frees));
    ggml_round_trace_end(0);
    ggml_round_trace_begin("D", 1);
    check(ggml_round_trace_gpu_round(phase) == 1 && phase == 1);
    ggml_round_trace_end(0);
    ggml_round_trace_begin("V", 6);
    ggml_round_trace_end(0);
    ggml_round_trace_flush();
    check(reads == 1 && frees == 1);
    FILE * f = std::fopen(path, "r");
    check(f != nullptr);
    char line[4096];
    check(std::fgets(line, sizeof(line), f) != nullptr);
    std::string first(line);
    check(first.find("\"gpu\":[{\"kind\":\"compute\",\"phase\":0,\"device\":0,\"stream\":17,\"begin_ns\":100,\"end_ns\":200,\"error_ns\":501}") != std::string::npos);
    check(first.find("\"kind\":\"wait\",\"phase\":1,\"device\":1") != std::string::npos);
    check(std::fgets(line, sizeof(line), f) != nullptr);
    check(std::string(line).find("\"gpu\"") == std::string::npos);
    check(std::fgets(line, sizeof(line), f) == nullptr);
    std::fclose(f);
    unlink(path);
}
