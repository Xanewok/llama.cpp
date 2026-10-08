// Kernel time of q8_0 K/V flash attention at prefill shapes, in-kernel dequantization against convert + f16, through the public
// ggml API. The dispatch knob GGML_CUDA_FA_Q8_QTILES (local commit) is switched in-process: a graph is allocated and computed
// under one value, so the buffer of the graph holds exactly the scratch that setting needs.
//   fa-q8-bench DEVICE        prints one "bench ..." line per shape
// ik = cap 1000000 (in-kernel wherever the dispatch can), cv = cap 0 (convert + f16). Order per shape: cv ik ik cv; each arm
// is a fresh graph with 3 warm-up calls and 20 timed calls; the arm time is the median, the shape time the mean of its two arms.
// us per call covers the whole ggml_backend_graph_compute call (launch, conversion kernels, FA kernel, synchronize).
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::atomic<int> g_mq{0};

static void log_cb(enum ggml_log_level, const char * text, void *) {
    if (strstr(text, "fattn_q8_mq:")) {
        ++g_mq;
    }
}

struct rng {
    uint64_t s;
    explicit rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
    float uniform(float lo, float hi) { return lo + (hi - lo)*float(next() >> 40)/float(1 << 24); }
};

struct shape {
    int64_t dk, gqa, seqs, kv, nb;
};

// Fills a contiguous q8_0 (or f32) tensor by repeating one quantized block of 4096 rows: timing does not depend on the values.
static void fill_tiled(ggml_tensor * t, rng & r, float lo, float hi) {
    const int64_t row_len = t->ne[0], rows = ggml_nelements(t)/row_len, chunk_rows = std::min<int64_t>(rows, 4096);
    std::vector<float> f(chunk_rows*row_len);
    for (auto & x : f) {
        x = r.uniform(lo, hi);
    }
    std::vector<uint8_t> q;
    if (t->type == GGML_TYPE_F32) {
        q.resize(f.size()*sizeof(float));
        memcpy(q.data(), f.data(), q.size());
    } else {
        q.resize(chunk_rows*ggml_row_size(t->type, row_len));
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, chunk_rows, row_len, nullptr);
    }
    const size_t total = ggml_nbytes(t);
    for (size_t off = 0; off < total; off += q.size()) {
        ggml_backend_tensor_set(t, q.data(), off, std::min(q.size(), total - off));
    }
}

static uint64_t fnv64(const uint8_t * p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ p[i])*0x100000001b3ull;
    }
    return h;
}

struct result {
    double us;
    int engaged;
    double buffer_mib;
    uint64_t digest;
};

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct inputs {
    ggml_context * ctx;
    ggml_backend_buffer_t buf;
    ggml_tensor *q, *k, *v, *m, *s;
};

// K/V are views of a cache twice as long (one sequence) or of two streams (two sequences), permuted like the KV cache.
static ggml_tensor * kv_view(ggml_context * ctx, ggml_tensor * base, const shape & sh) {
    ggml_tensor * view = sh.seqs == 2 ?
        ggml_view_4d(ctx, base, base->ne[0], base->ne[1], base->ne[2]/2, base->ne[3], base->nb[1], base->nb[2], base->nb[3], 0) :
        ggml_view_4d(ctx, base, base->ne[0], base->ne[1]/2, base->ne[2], base->ne[3], base->nb[1], base->nb[2], base->nb[3], 0);
    return ggml_permute(ctx, view, 0, 2, 1, 3);
}

static inputs make_inputs(ggml_backend_t backend, const shape & sh, int64_t nh) {
    inputs in;
    in.ctx = ggml_init({ ggml_tensor_overhead()*16, nullptr, true });
    in.q = ggml_new_tensor_4d(in.ctx, GGML_TYPE_F32, sh.dk, sh.nb, nh*sh.gqa, sh.seqs);
    // logical K: (dk, kv, nh, seqs); stored as (dk, nh, kv, seqs) and permuted (0,2,1,3)
    if (sh.seqs == 2) {
        in.k = ggml_new_tensor_4d(in.ctx, GGML_TYPE_Q8_0, sh.dk, nh, 2*sh.kv, sh.seqs);
        in.v = ggml_new_tensor_4d(in.ctx, GGML_TYPE_Q8_0, sh.dk, nh, 2*sh.kv, sh.seqs);
    } else {
        in.k = ggml_new_tensor_4d(in.ctx, GGML_TYPE_Q8_0, sh.dk, 2*nh, sh.kv, sh.seqs);
        in.v = ggml_new_tensor_4d(in.ctx, GGML_TYPE_Q8_0, sh.dk, 2*nh, sh.kv, sh.seqs);
    }
    in.m = ggml_new_tensor_4d(in.ctx, GGML_TYPE_F16, sh.kv, sh.nb, 1, sh.seqs);
    in.s = ggml_new_tensor_1d(in.ctx, GGML_TYPE_F32, nh*sh.gqa);
    in.buf = ggml_backend_alloc_ctx_tensors(in.ctx, backend);
    if (!in.buf) {
        return in;
    }
    rng r(sh.dk*1000003 + sh.kv + sh.nb);
    fill_tiled(in.q, r, -1.0f, 1.0f);
    fill_tiled(in.k, r, -1.0f, 1.0f);
    fill_tiled(in.v, r, -1.0f, 1.0f);
    fill_tiled(in.s, r, -2.0f, 2.0f);
    std::vector<ggml_fp16_t> mask(sh.kv*sh.nb);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int64_t i = 0; i < sh.nb; ++i) {
        for (int64_t j = 0; j < sh.kv; ++j) {
            mask[i*sh.kv + j] = j < sh.kv - sh.nb + 1 + i ? zero : ninf;
        }
    }
    for (int64_t s3 = 0; s3 < sh.seqs; ++s3) {
        ggml_backend_tensor_set(in.m, mask.data(), s3*mask.size()*sizeof(ggml_fp16_t), mask.size()*sizeof(ggml_fp16_t));
    }
    return in;
}

static bool run_arm(ggml_backend_t backend, const inputs & in, const shape & sh, const char * qtiles, result & res) {
    setenv("GGML_CUDA_FA_Q8_QTILES", qtiles, 1);
    ggml_context * ctx = ggml_init({ ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true });
    ggml_tensor * k = kv_view(ctx, in.k, sh);
    ggml_tensor * v = kv_view(ctx, in.v, sh);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, in.q, k, v, in.m, 1.0f/sqrtf((float) sh.dk), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(out, in.s);
    ggml_flash_attn_ext_set_n_kv_max(out, 0);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    if (!ggml_backend_supports_op(backend, out)) {
        ggml_free(ctx);
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    g_mq = 0;
    bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    res.engaged = g_mq;
    for (int i = 0; i < 2 && ok; ++i) {
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }
    std::vector<double> t(20);
    for (auto & x : t) {
        const double t0 = now_us();
        ok = ok && ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        x = now_us() - t0;
    }
    std::sort(t.begin(), t.end());
    res.us = 0.5*(t[9] + t[10]);
    res.buffer_mib = ggml_backend_buffer_get_size(buf)/1048576.0;
    std::vector<uint8_t> bytes(ggml_nbytes(out));
    ggml_backend_tensor_get(out, bytes.data(), 0, bytes.size());
    res.digest = fnv64(bytes.data(), bytes.size());
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s DEVICE\n", argv[0]);
        return 2;
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(argv[1]);
    if (!dev) {
        fprintf(stderr, "no device %s\n", argv[1]);
        return 2;
    }
    ggml_log_set(log_cb, nullptr);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    const int64_t nh = 2;
    std::vector<shape> shapes;
    for (int64_t seqs : {1, 2}) {
        for (int64_t kv : {8192, 32768, 90112}) {
            for (int64_t nb : {16, 32, 64, 128, 256, 512}) {
                shapes.push_back({256, 6, seqs, kv, nb});
            }
        }
    }
    for (int64_t gqa : {4, 8}) {
        for (int64_t kv : {8192, 32768}) {
            for (int64_t nb : {64, 128, 512}) {
                shapes.push_back({128, gqa, 1, kv, nb});
            }
        }
    }
    for (int64_t gqa : {4, 8}) {
        for (int64_t nb : {64, 128, 512}) {
            shapes.push_back({256, gqa, 1, 32768, nb});
        }
    }
    for (int64_t nb : {64, 128}) {
        shapes.push_back({128, 2, 1, 8192, nb});
        shapes.push_back({256, 2, 1, 32768, nb});
    }
    shapes.push_back({256, 1, 1, 32768, 64});
    int bad = 0;
    for (const auto & sh : shapes) {
        inputs in = make_inputs(backend, sh, nh);
        if (!in.buf) {
            fprintf(stderr, "alloc failed dk=%lld kv=%lld seqs=%lld\n", (long long) sh.dk, (long long) sh.kv, (long long) sh.seqs);
            ggml_free(in.ctx);
            ++bad;
            continue;
        }
        // tiles as the dispatch counts them: ceil(rows / wide columns) once the batch exceeds the narrow tile
        const int ncols2 = sh.gqa > 4 ? 8 : sh.gqa > 2 ? 4 : sh.gqa > 1 ? 2 : 1;
        const int64_t narrow = 32/ncols2, wide = 64/ncols2;
        const int64_t tiles = sh.nb <= narrow ? 1 : (sh.nb + wide - 1)/wide;
        const char * order[4] = { "0", "1000000", "1000000", "0" };
        result res[4] = {};
        bool ok = true;
        for (int a = 0; a < 4; ++a) {
            ok = ok && run_arm(backend, in, sh, order[a], res[a]);
        }
        if (!ok) {
            fprintf(stderr, "run failed dk=%lld gqa=%lld seqs=%lld kv=%lld nb=%lld\n", (long long) sh.dk, (long long) sh.gqa,
                (long long) sh.seqs, (long long) sh.kv, (long long) sh.nb);
            ++bad;
        } else {
            const double cv = 0.5*(res[0].us + res[3].us), ik = 0.5*(res[1].us + res[2].us);
            printf("bench dev=%s dk=%lld gqa=%lld seqs=%lld kv=%lld nb=%lld tiles=%lld ik_us=%.1f cv_us=%.1f ratio=%.4f "
                   "ik_reps=%.1f,%.1f cv_reps=%.1f,%.1f engaged_ik=%d,%d engaged_cv=%d,%d buf_ik_mib=%.1f buf_cv_mib=%.1f same=%d\n",
                argv[1], (long long) sh.dk, (long long) sh.gqa, (long long) sh.seqs, (long long) sh.kv, (long long) sh.nb,
                (long long) tiles, ik, cv, ik/cv, res[1].us, res[2].us, res[0].us, res[3].us, res[1].engaged, res[2].engaged,
                res[0].engaged, res[3].engaged, res[1].buffer_mib, res[0].buffer_mib,
                (int) (res[1].digest == res[0].digest && res[2].digest == res[0].digest && res[3].digest == res[0].digest));
            fflush(stdout);
        }
        ggml_backend_buffer_free(in.buf);
        ggml_free(in.ctx);
    }
    unsetenv("GGML_CUDA_FA_Q8_QTILES");
    ggml_backend_free(backend);
    return bad ? 1 : 0;
}
