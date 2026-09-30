// DO NOT MERGE: see ggml-host-prof.h
#include "ggml-host-prof.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {
struct prof_state {
    bool enabled = false;
    std::mutex mtx;
    std::map<std::string, std::vector<int64_t>> samples; // ns
    std::map<std::string, int64_t> counts;
    prof_state() {
        const char * e = getenv("GGML_HOST_PROF");
        enabled = e && atoi(e) != 0;
    }
    void dump() {
        std::lock_guard<std::mutex> lk(mtx);
        const char * path = getenv("GGML_HOST_PROF_OUT");
        FILE * f = path ? fopen(path, "w") : stderr;
        if (!f) f = stderr;
        fprintf(f, "%-48s %8s %10s %10s %10s %10s\n", "name", "n", "mean_us", "p50_us", "p10_us", "p90_us");
        for (auto & kv : samples) {
            auto v = kv.second;
            if (v.empty()) continue;
            std::sort(v.begin(), v.end());
            double sum = 0; for (auto x : v) sum += x;
            auto pct = [&](double p) { return v[std::min(v.size() - 1, (size_t) (p * (v.size() - 1) + 0.5))] / 1e3; };
            fprintf(f, "%-48s %8zu %10.2f %10.2f %10.2f %10.2f\n", kv.first.c_str(), v.size(), sum / v.size() / 1e3, pct(0.5), pct(0.1), pct(0.9));
        }
        for (auto & kv : counts) {
            fprintf(f, "%-48s %8lld (count)\n", kv.first.c_str(), (long long) kv.second);
        }
        if (f != stderr) fclose(f);
    }
};
prof_state & st() {
    static prof_state * s = [] {
        auto * p = new prof_state();
        if (p->enabled) atexit([] { st().dump(); });
        return p;
    }();
    return *s;
}
int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}

int64_t ggml_host_prof_now(void) { return st().enabled ? now_ns() : 0; }

void ggml_host_prof_add(const char * name, int64_t t_start) {
    if (t_start == 0) return;
    const int64_t d = now_ns() - t_start;
    auto & s = st();
    std::lock_guard<std::mutex> lk(s.mtx);
    s.samples[name].push_back(d);
}

void ggml_host_prof_count(const char * name) {
    auto & s = st();
    if (!s.enabled) return;
    std::lock_guard<std::mutex> lk(s.mtx);
    s.counts[name]++;
}

void ggml_host_prof_reset(void) {
    auto & s = st();
    std::lock_guard<std::mutex> lk(s.mtx);
    s.samples.clear();
    s.counts.clear();
}
