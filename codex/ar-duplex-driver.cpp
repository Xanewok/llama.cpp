// AllReduce microbenchmark and byte-identity driver for the chunked kernel path, through the public backend API only
// (ggml_backend_comm_init / ggml_backend_comm_allreduce_tensor), so the same source runs against any build.
//   ar-duplex-driver identity [small]    per-case output hashes of both GPUs + CPU reference check (rc 1 on a mismatch)
//   ar-duplex-driver perf ARM [calls]    per-call us at 1/6/12/18/64 rows and 0.5 MB: median, p25, p75 (batch 1 and 8)
//   ar-duplex-driver selftest            CPU only: the reference checker must flag a flipped bit
// rc: 0 pass | 1 result mismatch | 2 setup | 3 allreduce did not take the internal path
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint64_t rng_state;
static uint32_t rnd() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; return (uint32_t) (rng_state >> 11); }
static float urand() { return (rnd() & 0xFFFFFF) / (float) 0x1000000; }   // [0,1)

enum { K_NORMAL, K_TINY, K_DENORM, K_HUGE, K_INF, K_NAN, K_ZERO, K_CANCEL, K_COUNT };

struct input {
    std::vector<float>   v;      // exactly representable in the tensor type
    std::vector<uint8_t> bytes;
    std::vector<uint8_t> skip;   // elements the reference check ignores (denormal rows: the device flushes with -use_fast_math)
};

static float to_type(ggml_type t, float x) {
    switch (t) {
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
        default:             return x;
    }
}

// Same x for both ranks on K_CANCEL (the ranks hold x and -x); everything else is rank-specific.
static input make_input(ggml_type t, int64_t ne, uint64_t seed, int rank) {
    input in;
    in.v.resize(ne);
    in.skip.assign(ne, 0);
    const int64_t row_len = 5120;
    const float big   = t == GGML_TYPE_F16 ? 6.0e4f : 3.0e38f;
    const float denrm = t == GGML_TYPE_F16 ? 3.0e-6f : 1.0e-40f;
    for (int64_t e = 0; e < ne; ++e) {
        const int64_t row = e / row_len;
        const int kind = (int) ((row + seed + rank * 3) % K_COUNT);
        rng_state = (seed + 1) * 0x9e3779b97f4a7c15ull ^ ((uint64_t) e * 0xbf58476d1ce4e5b9ull) ^ (kind == K_CANCEL ? 0 : (uint64_t) (rank + 1) << 40);
        for (int i = 0; i < 4; ++i) rnd();
        float x;
        switch (kind) {
            case K_NORMAL: x = (urand() - 0.5f) * 8.0f; break;
            case K_TINY:   x = (urand() - 0.5f) * 2.0e-3f; break;
            case K_DENORM: x = (urand() - 0.5f) * 2.0f * denrm; in.skip[e] = 1; break;
            case K_HUGE:   x = (urand() < 0.5f ? -1.0f : 1.0f) * big * (0.5f + 0.5f * urand()); break;
            case K_INF:    x = e % 7 == 0 ? ((rank == 1 && e % 14 == 0) ? -INFINITY : INFINITY) : (urand() - 0.5f) * 8.0f; break;
            case K_NAN:    x = e % 11 == 0 ? NAN : (urand() - 0.5f) * 8.0f; break;
            case K_ZERO:   x = (rnd() & 1) ? 0.0f : -0.0f; break;
            default:       x = (urand() - 0.5f) * 200.0f; if (rank == 1) x = -x; break;   // K_CANCEL
        }
        in.v[e] = to_type(t, x);
        if (kind == K_DENORM) {   // a row is denormal on one rank only: the whole element pair is skipped
            in.skip[e] = 1;
        }
    }
    const size_t ts = ggml_type_size(t);
    in.bytes.resize(ne * ts);
    for (int64_t e = 0; e < ne; ++e) {
        switch (t) {
            case GGML_TYPE_F16:  { ggml_fp16_t h = ggml_fp32_to_fp16(in.v[e]); memcpy(&in.bytes[e * ts], &h, ts); break; }
            case GGML_TYPE_BF16: { ggml_bf16_t h = ggml_fp32_to_bf16(in.v[e]); memcpy(&in.bytes[e * ts], &h, ts); break; }
            default:             memcpy(&in.bytes[e * ts], &in.v[e], ts);
        }
    }
    return in;
}

// Does the device reduce this tensor through the BF16 wire? Mirrors ggml_cuda_ar_allreduce.
static bool wire_is_bf16(ggml_type t, int64_t ne) {
    const char * s = getenv("GGML_CUDA_AR_BF16_THRESHOLD");
    const unsigned long long thr = (s && s[0]) ? strtoull(s, nullptr, 10) : 1;
    return t == GGML_TYPE_F32 && thr > 0 && (unsigned long long) ne * ggml_type_size(t) >= thr;
}

static float wire(ggml_type t, bool bf16_wire, float x) { return bf16_wire ? ggml_bf16_to_fp32(ggml_fp32_to_bf16(x)) : to_type(t, x); }

// Expected bytes of one rank's output: dst(wire(own) + wire(peer)).
static std::vector<uint8_t> expected(ggml_type t, int64_t ne, bool bf16_wire, const std::vector<float> & own, const std::vector<float> & peer) {
    const size_t ts = ggml_type_size(t);
    std::vector<uint8_t> out(ne * ts);
    for (int64_t e = 0; e < ne; ++e) {
        const float s = wire(t, bf16_wire, own[e]) + wire(t, bf16_wire, peer[e]);
        switch (t) {
            case GGML_TYPE_F16:  { ggml_fp16_t h = ggml_fp32_to_fp16(s); memcpy(&out[e * ts], &h, ts); break; }
            case GGML_TYPE_BF16: { ggml_bf16_t h = ggml_fp32_to_bf16(s); memcpy(&out[e * ts], &h, ts); break; }
            default:             memcpy(&out[e * ts], &s, ts);
        }
    }
    return out;
}

static float load_f(ggml_type t, const uint8_t * p) {
    switch (t) {
        case GGML_TYPE_F16:  { ggml_fp16_t h; memcpy(&h, p, 2); return ggml_fp16_to_fp32(h); }
        case GGML_TYPE_BF16: { ggml_bf16_t h; memcpy(&h, p, 2); return ggml_bf16_to_fp32(h); }
        default:             { float f; memcpy(&f, p, 4); return f; }
    }
}

// Bitwise equal, any NaN equal to any NaN (payloads are not specified), skipped elements ignored. Returns the mismatch count.
static int64_t compare(ggml_type t, int64_t ne, const std::vector<uint8_t> & got, const std::vector<uint8_t> & want,
                       const std::vector<uint8_t> & skip0, const std::vector<uint8_t> & skip1) {
    const size_t ts = ggml_type_size(t);
    int64_t bad = 0;
    for (int64_t e = 0; e < ne; ++e) {
        if (skip0[e] || skip1[e]) continue;
        if (memcmp(&got[e * ts], &want[e * ts], ts) == 0) continue;
        if (std::isnan(load_f(t, &got[e * ts])) && std::isnan(load_f(t, &want[e * ts]))) continue;
        ++bad;
    }
    return bad;
}

static uint64_t fnv(uint64_t h, const std::vector<uint8_t> & b) {
    for (uint8_t c : b) { h ^= c; h *= 0x100000001b3ull; }
    return h;
}

struct pair_ctx {
    ggml_backend_t be[2] = {};
    void * comm = nullptr;
    ggml_backend_comm_allreduce_tensor_t fn = nullptr;
};

static bool setup(pair_ctx & p) {
    for (int g = 0; g < 2; ++g) {
        p.be[g] = ggml_backend_cuda_init(g);
        if (!p.be[g]) { fprintf(stderr, "ar-duplex: no CUDA device %d\n", g); return false; }
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(p.be[0]));
    auto init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    p.fn = (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");
    if (!init || !p.fn) { fprintf(stderr, "ar-duplex: backend has no comm API\n"); return false; }
    p.comm = init(p.be, 2);
    return p.comm != nullptr;
}

struct cell {
    ggml_context * ctx[2] = {};
    ggml_backend_buffer_t buf[2] = {};
    ggml_tensor * t[2] = {};
    ~cell() {
        for (int g = 0; g < 2; ++g) { if (buf[g]) ggml_backend_buffer_free(buf[g]); if (ctx[g]) ggml_free(ctx[g]); }
    }
};

static bool make_cell(const pair_ctx & p, cell & c, ggml_type t, int64_t ne, bool active1) {
    for (int g = 0; g < 2; ++g) {
        ggml_init_params ip = { ggml_tensor_overhead() * 4 + 1024, nullptr, true };
        c.ctx[g] = ggml_init(ip);
        c.t[g] = ggml_new_tensor_1d(c.ctx[g], t, ne);
        c.buf[g] = ggml_backend_alloc_ctx_tensors(c.ctx[g], p.be[g]);
        if (!c.buf[g]) return false;
        if (g == 0 || active1) c.t[g]->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }
    return true;
}

static bool allreduce(const pair_ctx & p, cell & c) {
    ggml_tensor * ts[2] = { c.t[0], c.t[1] };
    return p.fn(p.comm, ts);
}

struct id_case { const char * name; ggml_type t; int64_t ne; bool active1; int reps; };

static int identity(pair_ctx & p, bool small) {
    const int64_t R = 5120;
    std::vector<id_case> cases;
    for (ggml_type t : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16 }) {
        const char * n = t == GGML_TYPE_F32 ? "f32" : t == GGML_TYPE_F16 ? "f16" : "bf16";
        // sizes: tail-only, one vector, tails (ne % 8 == 4 only for F32: its input is 16 B aligned), decode rows, 0.5 MB
        std::vector<int64_t> sizes = { 8, 16, 24, 5120 + 8, R, R * 2, R * 6, R * 12, R * 18, R * 64, 262144, 262144 + 8 };
        if (t == GGML_TYPE_F32) { sizes.push_back(4); sizes.push_back(12); sizes.push_back(5124); sizes.push_back(R * 12 + 4); sizes.push_back(262144 + 4); }
        if (small) { sizes = { 8, 5120 + 8, R * 6 }; if (t == GGML_TYPE_F32) { sizes.push_back(4); sizes.push_back(5124); } }
        // above 1 MB of wire the copy engine takes over unless GGML_CUDA_AR_COPY_THRESHOLD=0 (the window runs both)
        if (!small && getenv("GGML_CUDA_AR_COPY_THRESHOLD")) { sizes.push_back(524288 + 8); sizes.push_back(3 * 524288 + 4); sizes.push_back(2 * 524288 * 2 + 12); }
        for (int64_t ne : sizes) {
            if (t != GGML_TYPE_F32 && ne % 8) continue;
            char buf[64]; snprintf(buf, sizeof buf, "%s-%lld", n, (long long) ne);
            cases.push_back({ strdup(buf), t, ne, true, small ? 2 : 3 });
        }
        cases.push_back({ strdup((std::string(n) + "-inactive1-30720").c_str()), t, 30720, false, 2 });
    }
    int rc = 0;
    for (const id_case & c : cases) {
        cell cl;
        if (!make_cell(p, cl, c.t, c.ne, c.active1)) { fprintf(stderr, "ar-duplex: alloc failed %s\n", c.name); return 2; }
        const bool bf16_wire = wire_is_bf16(c.t, c.ne);
        uint64_t h[2] = { 0xcbf29ce484222325ull, 0xcbf29ce484222325ull };
        int64_t bad = 0;
        for (int rep = 0; rep < c.reps; ++rep) {
            const uint64_t seed = 17 * rep + c.ne;
            input in[2] = { make_input(c.t, c.ne, seed, 0), make_input(c.t, c.ne, seed, 1) };
            std::vector<float> contrib1 = in[1].v;
            if (!c.active1) {
                std::fill(contrib1.begin(), contrib1.end(), 0.0f);   // an inactive shard contributes zeros whatever its buffer holds
            }
            for (int g = 0; g < 2; ++g) ggml_backend_tensor_set(cl.t[g], in[g].bytes.data(), 0, in[g].bytes.size());
            if (!allreduce(p, cl)) { fprintf(stderr, "ar-duplex: allreduce returned false (%s)\n", c.name); return 3; }
            std::vector<uint8_t> out[2] = { std::vector<uint8_t>(in[0].bytes.size()), std::vector<uint8_t>(in[0].bytes.size()) };
            for (int g = 0; g < 2; ++g) { ggml_backend_synchronize(p.be[g]); ggml_backend_tensor_get(cl.t[g], out[g].data(), 0, out[g].size()); h[g] = fnv(h[g], out[g]); }
            const std::vector<uint8_t> w0 = expected(c.t, c.ne, bf16_wire, in[0].v, contrib1);
            const std::vector<uint8_t> w1 = expected(c.t, c.ne, bf16_wire, contrib1, in[0].v);
            std::vector<uint8_t> none(c.ne, 0);
            bad += compare(c.t, c.ne, out[0], w0, in[0].skip, in[1].skip);
            bad += compare(c.t, c.ne, out[1], w1, in[0].skip, in[1].skip);
        }
        printf("id %-24s h0=%016llx h1=%016llx ref=%s\n", c.name, (unsigned long long) h[0], (unsigned long long) h[1],
               bad ? "BAD" : "ok");
        if (bad) { fprintf(stderr, "ar-duplex: FAIL %s: %lld mismatching elements vs the CPU reference\n", c.name, (long long) bad); rc = 1; }
    }
    fprintf(stderr, "ar-duplex: identity %s\n", rc ? "FAIL" : "PASS");
    return rc;
}

static double pct(std::vector<double> & v, double q) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t) (q * (v.size() - 1) + 0.5))];
}

static int perf(pair_ctx & p, const char * arm, int calls) {
    struct pc { const char * label; int64_t ne; };
    const int64_t R = 5120;
    const pc cells[] = { { "rows1", R }, { "rows6", R * 6 }, { "rows12", R * 12 }, { "rows18", R * 18 }, { "rows64", R * 64 }, { "mb0.5", 262144 } };
    for (const pc & c : cells) {
        cell cl;
        if (!make_cell(p, cl, GGML_TYPE_F32, c.ne, true)) return 2;
        input in[2] = { make_input(GGML_TYPE_F32, c.ne, 1, 0), make_input(GGML_TYPE_F32, c.ne, 1, 1) };
        for (int batch : { 1, 8 }) {
            std::vector<double> us;
            const int samples = std::max(12, calls / batch);
            for (int s = -10; s < samples; ++s) {   // 10 warm-up samples
                for (int g = 0; g < 2; ++g) ggml_backend_tensor_set(cl.t[g], in[g].bytes.data(), 0, in[g].bytes.size());
                for (int g = 0; g < 2; ++g) ggml_backend_synchronize(p.be[g]);
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < batch; ++i) {
                    if (!allreduce(p, cl)) { fprintf(stderr, "ar-duplex: allreduce returned false (%s)\n", c.label); return 3; }
                }
                for (int g = 0; g < 2; ++g) ggml_backend_synchronize(p.be[g]);
                const auto t1 = std::chrono::steady_clock::now();
                if (s >= 0) us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / batch);
            }
            printf("perf arm=%s cell=%s ne=%lld batch=%d calls=%d med_us=%.2f p25_us=%.2f p75_us=%.2f\n", arm, c.label, (long long) c.ne, batch,
                   (int) us.size() * batch, pct(us, 0.5), pct(us, 0.25), pct(us, 0.75));
        }
    }
    return 0;
}

// CPU only. The reference must accept its own output and flag a single flipped bit, per type and wire.
static int selftest() {
    int rc = 0;
    for (ggml_type t : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16 }) {
        const int64_t ne = 5120 * 8 + (t == GGML_TYPE_F32 ? 4 : 0);
        input a = make_input(t, ne, 5, 0), b = make_input(t, ne, 5, 1);
        for (int bf16 = 0; bf16 < (t == GGML_TYPE_F32 ? 2 : 1); ++bf16) {
            std::vector<uint8_t> w = expected(t, ne, bf16, a.v, b.v);
            const int64_t same = compare(t, ne, w, w, a.skip, b.skip);
            // flip a bit of an element that is not skipped and not NaN
            int64_t e = 0;
            const size_t ts = ggml_type_size(t);
            while (e < ne && (a.skip[e] || b.skip[e] || std::isnan(load_f(t, &w[e * ts])))) ++e;
            std::vector<uint8_t> w2 = w;
            w2[e * ts] ^= 1;
            const int64_t flagged = compare(t, ne, w2, w, a.skip, b.skip);
            int64_t kinds = 0;
            for (int64_t i = 0; i < ne; ++i) kinds += std::isnan(load_f(t, &w[i * ts])) + std::isinf(load_f(t, &w[i * ts]));
            printf("selftest type=%s bf16_wire=%d self_mismatch=%lld flipped_bit_flagged=%lld nan_or_inf_outputs=%lld\n", ggml_type_name(t), bf16,
                   (long long) same, (long long) flagged, (long long) kinds);
            if (same != 0 || flagged != 1 || kinds == 0) rc = 1;
        }
    }
    printf("ar-duplex: selftest %s\n", rc ? "FAIL" : "PASS");
    return rc;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: ar-duplex-driver identity | perf ARM [samples] | selftest\n"); return 2; }
    const std::string mode = argv[1];
    if (mode == "selftest") return selftest();
    pair_ctx p;
    if (!setup(p)) return 2;
    if (mode == "identity") return identity(p, argc > 2 && !strcmp(argv[2], "small"));
    if (mode == "perf") return perf(p, argc > 2 ? argv[2] : "arm", argc > 3 ? atoi(argv[3]) : 240);
    return 2;
}
