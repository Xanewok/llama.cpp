// Probe (not a unit test): CUDA GET_ROWS on an IQ4_XS [5120, 248320] token table, comparing
//   a: table in the device buffer (VRAM)
//   b: table in a CUDA_Host pinned buffer, read in place by the kernel over PCIe (zero-copy, one table for all GPUs)
//   d: table in plain malloc memory registered with ggml_backend_cuda_register_host_buffer, read in place
//   m: table in a read-only file mmap (the llama.cpp model layout), registered, read in place
//   c: today's path: CPU GET_ROWS (dequant to F32) into a pinned buffer + H2D of the F32 rows
// Outputs of b/d/m must be bitwise equal to a; c is reported, not gated (CPU vs CUDA dequant).
// The kernel receives the host pointer as is: a correct output from b/d/m proves an in-place read (nothing is copied).
// d/m can fault on a platform without host-pointer access to registered memory: run them in their own process.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef ZC_NVTX
#include <nvtx3/nvToolsExt.h>
#define ZC_RANGE_PUSH(s) nvtxRangePushA(s)
#define ZC_RANGE_POP()   nvtxRangePop()
#else
#define ZC_RANGE_PUSH(s) ((void) (s))
#define ZC_RANGE_POP()   ((void) 0)
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

static const int64_t   NE0  = 5120;
static const int64_t   NE1  = 248320;
static const ggml_type TYPE = GGML_TYPE_IQ4_XS;

using clk = std::chrono::steady_clock;

static double us_since(clk::time_point t0) {
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

struct stats { double med, p10, p90; };

static stats summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    auto q = [&](double f) { return v[std::min(v.size() - 1, (size_t) (f * (v.size() - 1) + 0.5))]; };
    return { q(0.5), q(0.1), q(0.9) };
}

static thread_local uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng() {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return rng_state;
}

// random IQ4_XS blocks with a finite, non-tiny fp16 scale: random scale bits could be NaN/Inf and hide a mismatch
static void fill_table(uint8_t * p, size_t size) {
    for (size_t i = 0; i + 8 <= size; i += 8) {
        const uint64_t r = rng();
        memcpy(p + i, &r, 8);
    }
    const size_t bs = ggml_type_size(TYPE);
    for (size_t off = 0; off + bs <= size; off += bs) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.001f + (float) (rng() % 1000) * 5e-5f);
        memcpy(p + off, &d, sizeof(d));
    }
}

static std::vector<int32_t> random_ids(size_t n) {
    std::vector<int32_t> ids(n);
    for (auto & id : ids) {
        id = (int32_t) (rng() % NE1);
    }
    return ids;
}

struct opts {
    std::vector<std::string> devs  = { "CUDA0", "CUDA1" };
    std::string              arms  = "abc";
    std::vector<int>         rows  = { 1, 2, 6, 12, 64, 512 };
    int                      iters = 300;
    int                      reps  = 20;
    int                      threads = 6;
    bool                     contend = false;
    bool                     pipelined = true;
    std::string              mmap_file;
};

static std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> out;
    size_t a = 0;
    while (a <= s.size()) {
        size_t b = s.find(',', a);
        if (b == std::string::npos) b = s.size();
        if (b > a) out.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return out;
}

// a table placement: a tensor whose data the CUDA kernel reads
struct table {
    char                  arm;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         t   = nullptr;
};

static table make_table(char arm, ggml_backend_buffer_type_t buft, void * ptr, size_t size) {
    table tb;
    tb.arm = arm;
    ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, true };
    tb.ctx = ggml_init(ip);
    tb.t   = ggml_new_tensor_2d(tb.ctx, TYPE, NE0, NE1);
    ggml_set_name(tb.t, "token_embd");
    if (buft) {
        tb.buf = ggml_backend_alloc_ctx_tensors_from_buft(tb.ctx, buft);
    } else {
        GGML_ASSERT(size >= ggml_nbytes(tb.t));
        tb.buf = ggml_backend_cpu_buffer_from_ptr(ptr, size);
        GGML_ASSERT(ggml_backend_tensor_alloc(tb.buf, tb.t, ptr) == GGML_STATUS_SUCCESS);
    }
    GGML_ASSERT(tb.buf);
    return tb;
}

// GET_ROWS graphs on one device: g1 = one node (sync mode), gk = k independent nodes (pipelined mode)
struct rows_graph {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         ids = nullptr;
    std::vector<ggml_tensor *> outs;
    ggml_cgraph *         g1 = nullptr;
    ggml_cgraph *         gk = nullptr;
};

static rows_graph make_graph(ggml_tensor * tab, int rows, int k, ggml_backend_buffer_type_t buft) {
    rows_graph rg;
    ggml_init_params ip = { ggml_tensor_overhead() * (4 + 3 * (size_t) k) + 2 * ggml_graph_overhead_custom(2 * k + 8, false), nullptr, true };
    rg.ctx = ggml_init(ip);
    rg.ids = ggml_new_tensor_1d(rg.ctx, GGML_TYPE_I32, (int64_t) rows * k);
    for (int i = 0; i < k; i++) {
        ggml_tensor * v = ggml_view_1d(rg.ctx, rg.ids, rows, (size_t) i * rows * sizeof(int32_t));
        rg.outs.push_back(ggml_get_rows(rg.ctx, tab, v));
    }
    rg.g1 = ggml_new_graph_custom(rg.ctx, 2 * k + 8, false);
    ggml_build_forward_expand(rg.g1, rg.outs[0]);
    rg.gk = ggml_new_graph_custom(rg.ctx, 2 * k + 8, false);
    for (auto * o : rg.outs) {
        ggml_build_forward_expand(rg.gk, o);
    }
    // the table is allocated elsewhere; only ids, views and outputs land here
    rg.buf = ggml_backend_alloc_ctx_tensors_from_buft(rg.ctx, buft);
    GGML_ASSERT(rg.buf);
    return rg;
}

static void free_graph(rows_graph & rg) {
    ggml_backend_buffer_free(rg.buf);
    ggml_free(rg.ctx);
}

static int pipelined_k(int rows) { return rows <= 64 ? 32 : 8; }

static void print_stats(const char * dev, char arm, int rows, const char * mode, const char * cond, const std::vector<double> & v) {
    const stats s = summarize(v);
    printf("zc: dev=%s arm=%c rows=%d mode=%s cond=%s n=%zu med_us=%.2f p10_us=%.2f p90_us=%.2f\n",
        dev, arm, rows, mode, cond, v.size(), s.med, s.p10, s.p90);
    fflush(stdout);
}

// sync mode: refresh ids (untimed), then time compute + synchronize of one GET_ROWS
static std::vector<double> run_sync(ggml_backend_t be, rows_graph & rg, int rows, int iters) {
    std::vector<double> v;
    v.reserve(iters);
    for (int it = -10; it < iters; it++) {
        const auto ids = random_ids(rows);
        ggml_backend_tensor_set(rg.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
        const auto t0 = clk::now();
        GGML_ASSERT(ggml_backend_graph_compute(be, rg.g1) == GGML_STATUS_SUCCESS);
        const double us = us_since(t0);
        if (it >= 0) v.push_back(us);
    }
    return v;
}

// pipelined mode: k GET_ROWS with distinct ids in one graph; per-call = wall / k (launches overlap the kernels)
static std::vector<double> run_pipelined(ggml_backend_t be, rows_graph & rg, int rows, int k, int reps) {
    std::vector<double> v;
    for (int r = -2; r < reps; r++) {
        const auto ids = random_ids((size_t) rows * k);
        ggml_backend_tensor_set(rg.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
        const auto t0 = clk::now();
        GGML_ASSERT(ggml_backend_graph_compute(be, rg.gk) == GGML_STATUS_SUCCESS);
        const double us = us_since(t0) / k;
        if (r >= 0) v.push_back(us);
    }
    return v;
}

struct cpu_path {
    ggml_context *        ctx  = nullptr;
    ggml_backend_buffer_t buf  = nullptr;
    ggml_tensor *         ids  = nullptr;
    ggml_tensor *         out  = nullptr;
    ggml_cgraph *         g    = nullptr;
};

// today's path: ids and F32 rows in the pinned host buffer (llama-context gives the CPU backend the GPU host buft)
static cpu_path make_cpu_path(ggml_tensor * tab_host, int rows, ggml_backend_buffer_type_t host_buft) {
    cpu_path cp;
    ggml_init_params ip = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true };
    cp.ctx = ggml_init(ip);
    cp.ids = ggml_new_tensor_1d(cp.ctx, GGML_TYPE_I32, rows);
    cp.out = ggml_get_rows(cp.ctx, tab_host, cp.ids);
    cp.g   = ggml_new_graph(cp.ctx);
    ggml_build_forward_expand(cp.g, cp.out);
    cp.buf = ggml_backend_alloc_ctx_tensors_from_buft(cp.ctx, host_buft);
    GGML_ASSERT(cp.buf);
    return cp;
}

static std::vector<double> run_cpu_path(ggml_backend_t cpu, ggml_backend_t be, cpu_path & cp, ggml_tensor * dev_out, int rows, int iters) {
    std::vector<double> v;
    v.reserve(iters);
    for (int it = -10; it < iters; it++) {
        const auto ids = random_ids(rows);
        ggml_backend_tensor_set(cp.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
        const auto t0 = clk::now();
        GGML_ASSERT(ggml_backend_graph_compute(cpu, cp.g) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_set_async(be, dev_out, cp.out->data, 0, ggml_nbytes(cp.out));
        ggml_backend_synchronize(be);
        const double us = us_since(t0);
        if (it >= 0) v.push_back(us);
    }
    return v;
}

// background H2D stream on another GPU: PCIe / host-memory contention
struct contender {
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> bytes{0};
    double seconds = 0;
    std::thread th;
    void start(ggml_backend_dev_t dev) {
        th = std::thread([this, dev]() {
            const size_t n = 64ull << 20;
            ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, true };
            ggml_context * cs = ggml_init(ip);
            ggml_context * cd = ggml_init(ip);
            ggml_tensor * src = ggml_new_tensor_1d(cs, GGML_TYPE_I8, n);
            ggml_tensor * dst = ggml_new_tensor_1d(cd, GGML_TYPE_I8, n);
            ggml_backend_buffer_t bs = ggml_backend_alloc_ctx_tensors_from_buft(cs, ggml_backend_dev_host_buffer_type(dev));
            ggml_backend_buffer_t bd = ggml_backend_alloc_ctx_tensors_from_buft(cd, ggml_backend_dev_buffer_type(dev));
            memset(src->data, 1, n);
            const auto t0 = clk::now();
            while (!stop.load()) {
                ggml_backend_tensor_set(dst, src->data, 0, n);
                bytes += n;
            }
            seconds = us_since(t0) * 1e-6;
            ggml_backend_buffer_free(bs);
            ggml_backend_buffer_free(bd);
            ggml_free(cs);
            ggml_free(cd);
        });
    }
    double finish() {
        stop = true;
        th.join();
        return bytes.load() / seconds / 1e9;
    }
};

static size_t count_diff(const std::vector<float> & x, const std::vector<float> & y) {
    GGML_ASSERT(x.size() == y.size());
    size_t n = 0;
    for (size_t i = 0; i < x.size(); i++) {
        n += memcmp(&x[i], &y[i], sizeof(float)) != 0;
    }
    return n;
}

int main(int argc, char ** argv) {
    opts o;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { GGML_ASSERT(i + 1 < argc); return argv[++i]; };
        if      (a == "--devs")         o.devs = split(next());
        else if (a == "--arms")         o.arms = next();
        else if (a == "--rows")         { o.rows.clear(); for (auto & s : split(next())) o.rows.push_back(std::stoi(s)); }
        else if (a == "--iters")        o.iters = std::stoi(next());
        else if (a == "--reps")         o.reps = std::stoi(next());
        else if (a == "--threads")      o.threads = std::stoi(next());
        else if (a == "--contend")      o.contend = true;
        else if (a == "--no-pipelined") o.pipelined = false;
        else if (a == "--mmap-file")    o.mmap_file = next();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    // plain launches: the probe measures the kernel, not CUDA graph replay
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
    ggml_backend_load_all();

    auto has = [&](char c) { return o.arms.find(c) != std::string::npos; };

    std::vector<ggml_backend_dev_t> devs;
    std::vector<ggml_backend_t>     bes;
    for (auto & name : o.devs) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
        if (!dev) { printf("zc: FAIL no device %s\n", name.c_str()); return 1; }
        devs.push_back(dev);
        bes.push_back(ggml_backend_dev_init(dev, nullptr));
        printf("zc-dev: %s %s\n", name.c_str(), ggml_backend_dev_description(dev));
    }
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(devs[0]);
    if (!host_buft) {
        // --devs CPU: a CPU-only smoke of the harness (Mac gate), no zero-copy meaning
        host_buft = ggml_backend_cpu_buffer_type();
    }

    const size_t nbytes = ggml_row_size(TYPE, NE0) * NE1;
    const size_t asize  = (nbytes + 4095) & ~(size_t) 4095;
    uint8_t * src = (uint8_t *) aligned_alloc(4096, asize);
    GGML_ASSERT(src);
    fill_table(src, nbytes);
    printf("zc-table: type=%s ne=[%" PRId64 ", %" PRId64 "] bytes=%zu row_bytes=%zu\n", ggml_type_name(TYPE), NE0, NE1, nbytes, ggml_row_size(TYPE, NE0));

    // shared placements
    std::vector<table> host_tables;
    if (has('b')) {
        table tb = make_table('b', host_buft, nullptr, 0);
        // CUDA_Host falls back to a plain CPU buffer when pinning fails: then b would not be zero-copy
        if (strcmp(ggml_backend_buffer_name(tb.buf), "CUDA_Host") != 0) {
            printf("zc: FAIL arm b buffer is %s, not CUDA_Host\n", ggml_backend_buffer_name(tb.buf));
            return 1;
        }
        ggml_backend_tensor_set(tb.t, src, 0, nbytes);
        host_tables.push_back(tb);
    }
    typedef bool (*reg_fn_t)(void *, size_t);
    reg_fn_t reg_fn = nullptr;
    if (has('d') || has('m')) {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
        GGML_ASSERT(reg);
        reg_fn = (reg_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_register_host_buffer");
        GGML_ASSERT(reg_fn);
        setenv("GGML_CUDA_REGISTER_HOST", "1", 1);
    }
    if (has('d')) {
        // a copy, so that arm c (which reads src) is not served from registered memory
        uint8_t * p = (uint8_t *) aligned_alloc(4096, asize);
        GGML_ASSERT(p);
        memcpy(p, src, nbytes);
        if (!reg_fn(p, asize)) {
            printf("zc: FAIL arm d register failed\n");
            return 1;
        }
        host_tables.push_back(make_table('d', nullptr, p, asize));
    }
    if (has('m')) {
        GGML_ASSERT(!o.mmap_file.empty());
        int fd = open(o.mmap_file.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        GGML_ASSERT(fd >= 0);
        GGML_ASSERT(write(fd, src, nbytes) == (ssize_t) nbytes);
        fsync(fd);
        close(fd);
        fd = open(o.mmap_file.c_str(), O_RDONLY);
        GGML_ASSERT(fd >= 0);
        // same flags as llama_mmap
        void * p = mmap(nullptr, nbytes, PROT_READ, MAP_SHARED, fd, 0);
        GGML_ASSERT(p != MAP_FAILED);
        close(fd);
        if (!reg_fn(p, nbytes)) {
            printf("zc: FAIL arm m register failed\n");
            return 1;
        }
        host_tables.push_back(make_table('m', nullptr, p, nbytes));
    }
    table cpu_table = make_table('c', nullptr, src, asize);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, o.threads);

    bool ok = true;
    for (size_t di = 0; di < devs.size(); di++) {
        const char * dn = o.devs[di].c_str();
        ggml_backend_t be = bes[di];
        ggml_backend_buffer_type_t dev_buft = ggml_backend_dev_buffer_type(devs[di]);

        std::vector<table> tabs;
        table dev_table = make_table('a', dev_buft, nullptr, 0);
        ggml_backend_tensor_set(dev_table.t, src, 0, nbytes);
        tabs.push_back(dev_table);
        for (auto & h : host_tables) tabs.push_back(h);

        for (int rows : o.rows) {
            // equality on a fixed id set incl. the first and last row
            auto check_ids = random_ids(rows);
            check_ids[0] = 0;
            if (rows > 1) check_ids[rows - 1] = (int32_t) (NE1 - 1);
            std::map<char, std::vector<float>> outs;

            for (auto & tb : tabs) {
                const int k = pipelined_k(rows);
                rows_graph rg = make_graph(tb.t, rows, k, dev_buft);

                ggml_backend_tensor_set(rg.ids, check_ids.data(), 0, rows * sizeof(int32_t));
                GGML_ASSERT(ggml_backend_graph_compute(be, rg.g1) == GGML_STATUS_SUCCESS);
                outs[tb.arm].resize((size_t) rows * NE0);
                ggml_backend_tensor_get(rg.outs[0], outs[tb.arm].data(), 0, ggml_nbytes(rg.outs[0]));

                char label[128];
                snprintf(label, sizeof(label), "dev=%s arm=%c rows=%d mode=sync", dn, tb.arm, rows);
                ZC_RANGE_PUSH(label);
                print_stats(dn, tb.arm, rows, "sync", "idle", run_sync(be, rg, rows, o.iters));
                ZC_RANGE_POP();
                if (o.pipelined) {
                    snprintf(label, sizeof(label), "dev=%s arm=%c rows=%d mode=pipelined", dn, tb.arm, rows);
                    ZC_RANGE_PUSH(label);
                    print_stats(dn, tb.arm, rows, "pipelined", "idle", run_pipelined(be, rg, rows, k, o.reps));
                    ZC_RANGE_POP();
                }
                if (o.contend && tb.arm == 'b' && devs.size() > 1) {
                    contender c;
                    c.start(devs[(di + 1) % devs.size()]);
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    snprintf(label, sizeof(label), "dev=%s arm=b rows=%d mode=sync cond=h2d-other", dn, rows);
                    ZC_RANGE_PUSH(label);
                    auto v = run_sync(be, rg, rows, o.iters);
                    ZC_RANGE_POP();
                    const double gbs = c.finish();
                    print_stats(dn, 'b', rows, "sync", "h2d-other", v);
                    printf("zc-contend: dev=%s rows=%d other_h2d_GBps=%.2f\n", dn, rows, gbs);
                }
                free_graph(rg);
            }

            if (has('c')) {
                rows_graph rg = make_graph(dev_table.t, rows, 1, dev_buft);
                cpu_path cp = make_cpu_path(cpu_table.t, rows, host_buft);
                ggml_backend_tensor_set(cp.ids, check_ids.data(), 0, rows * sizeof(int32_t));
                GGML_ASSERT(ggml_backend_graph_compute(cpu, cp.g) == GGML_STATUS_SUCCESS);
                outs['c'].assign((const float *) cp.out->data, (const float *) cp.out->data + (size_t) rows * NE0);
                char label[128];
                snprintf(label, sizeof(label), "dev=%s arm=c rows=%d mode=sync", dn, rows);
                ZC_RANGE_PUSH(label);
                print_stats(dn, 'c', rows, "sync", "idle", run_cpu_path(cpu, be, cp, rg.outs[0], rows, o.iters));
                ZC_RANGE_POP();
                ggml_backend_buffer_free(cp.buf);
                ggml_free(cp.ctx);
                free_graph(rg);
            }

            for (auto & kv : outs) {
                if (kv.first == 'a') continue;
                const size_t nd = count_diff(outs['a'], kv.second);
                const bool gated = kv.first != 'c';
                printf("zc-eq: dev=%s rows=%d a_vs_%c differing=%zu/%zu%s\n", dn, rows, kv.first, nd, kv.second.size(), gated ? "" : " (info)");
                if (gated && nd != 0) ok = false;
            }
            fflush(stdout);
        }
        ggml_backend_buffer_free(dev_table.buf);
        ggml_free(dev_table.ctx);
    }

    // both GPUs reading the one pinned table at once (the -sm tensor shape)
    if (o.contend && has('b') && devs.size() > 1) {
        table & hb = host_tables[0];
        for (int rows : o.rows) {
            std::vector<rows_graph> rgs;
            for (size_t di = 0; di < devs.size(); di++) rgs.push_back(make_graph(hb.t, rows, 1, ggml_backend_dev_buffer_type(devs[di])));
            std::vector<std::vector<double>> res(devs.size());
            std::vector<std::thread> ths;
            for (size_t di = 0; di < devs.size(); di++) {
                ths.emplace_back([&, di]() { res[di] = run_sync(bes[di], rgs[di], rows, o.iters); });
            }
            for (auto & t : ths) t.join();
            for (size_t di = 0; di < devs.size(); di++) {
                print_stats(o.devs[di].c_str(), 'b', rows, "sync", "both-gpus", res[di]);
                free_graph(rgs[di]);
            }
        }
    }

    if (!o.mmap_file.empty()) unlink(o.mmap_file.c_str());
    printf("zc-getrows: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
