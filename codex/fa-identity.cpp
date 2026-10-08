// Deterministic q8_0 flash-attention output digests through the public ggml API only.
// The binary links no library path of its own: run it once per library set (LD_LIBRARY_PATH) and diff the lines.
//   fa-identity DEVICE GROUP     e.g. fa-identity CUDA0 verify
// GROUP: decode (one query row, grouped one-row kernels), verify (3-8 queries, head size 256), other (f16 K/V, q8_0 with 2 or
// 9+ queries: paths this fix must not touch). Prints "GROUP CASE FNV64 OUT_BYTES"; the digest covers the raw f32 output bytes.
#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct rng {
    uint64_t s;
    explicit rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
    float uniform(float lo, float hi) { return lo + (hi - lo)*float(next() >> 40)/float(1 << 24); }
};

struct fa_case {
    const char * group;
    int64_t hsk, hsv, nh;
    std::array<int64_t, 2> nr23;
    int64_t kv, nb;
    bool sinks;
    ggml_type type;
    std::array<int32_t, 4> permute;
    bool v_is_view_of_k, stream_view;
};

struct kv_tensor {
    ggml_tensor * view;
    ggml_tensor * base;
};

static std::string name_of(const fa_case & c) {
    char b[256];
    snprintf(b, sizeof b, "hsk=%lld,hsv=%lld,nh=%lld,nr=%lldx%lld,kv=%lld,nb=%lld,sinks=%d,type=%s,perm=%d%d%d%d,vk=%d,stream=%d",
        (long long) c.hsk, (long long) c.hsv, (long long) c.nh, (long long) c.nr23[0], (long long) c.nr23[1], (long long) c.kv,
        (long long) c.nb, (int) c.sinks, ggml_type_name(c.type), c.permute[0], c.permute[1], c.permute[2], c.permute[3],
        (int) c.v_is_view_of_k, (int) c.stream_view);
    return b;
}

static void fill(ggml_tensor * t, rng & r, float lo, float hi) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> f(n);
    for (auto & x : f) {
        x = r.uniform(lo, hi);
    }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(n);
        ggml_fp32_to_fp16_row(f.data(), h.data(), n);
        ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
    } else {
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, n/t->ne[0], t->ne[0], nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    }
}

static uint64_t fnv64(const uint8_t * p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ p[i])*0x100000001b3ull;
    }
    return h;
}

// Shapes follow test_flash_attn_ext in tests/test-backend-ops.cpp: K/V are views of a cache twice as long (or twice as many
// streams), laid out like the KV cache through the permute.
static int run(ggml_backend_t backend, const fa_case & c, uint64_t seed, bool & ok) {
    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    rng r(seed);

    auto permuted = [&](ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, bool is_view) {
        int64_t ne[4] = { ne0, ne1, ne2, ne3 }, np[4];
        for (int i = 0; i < 4; ++i) {
            np[c.permute[i]] = ne[i];
        }
        kv_tensor out;
        if (is_view && c.stream_view) {
            out.base = ggml_new_tensor_4d(ctx, type, np[0], np[1], 2*np[2], np[3]);
            out.view = ggml_view_4d(ctx, out.base, np[0], np[1], np[2], np[3], out.base->nb[1], out.base->nb[2], out.base->nb[3], 0);
        } else if (is_view) {
            out.base = ggml_new_tensor_4d(ctx, type, np[0], 2*np[1], np[2], np[3]);
            out.view = ggml_view_4d(ctx, out.base, np[0], np[1], np[2], np[3], out.base->nb[1], out.base->nb[2], out.base->nb[3], 0);
        } else {
            out.base = out.view = ggml_new_tensor_4d(ctx, type, np[0], np[1], np[2], np[3]);
        }
        if (c.permute != std::array<int32_t, 4>{0, 1, 2, 3}) {
            out.view = ggml_permute(ctx, out.view, c.permute[0], c.permute[1], c.permute[2], c.permute[3]);
        }
        return out;
    };

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.hsk, c.nb, c.nh*c.nr23[0], c.nr23[1]);
    kv_tensor k = permuted(c.type, c.hsk, c.kv, c.nh, c.nr23[1], true);
    kv_tensor v;
    if (c.v_is_view_of_k) {
        v.view = ggml_view_4d(ctx, k.view, c.hsv, c.kv, c.nh, c.nr23[1], k.view->nb[1], k.view->nb[2], k.view->nb[3], 0);
        v.base = nullptr;
    } else {
        v = permuted(c.type, c.hsv, c.kv, c.nh, c.nr23[1], true);
    }
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.kv, c.nb, 1, c.nr23[1]);
    ggml_tensor * s = c.sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, q->ne[2]) : nullptr;

    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k.view, v.view, m, 1.0f/sqrtf((float) c.hsk), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(out, s);
    ggml_flash_attn_ext_set_n_kv_max(out, 0);
    ggml_prec_set_acc(out, GGML_PREC_F32);

    if (!ggml_backend_supports_op(backend, out)) {
        printf("%s %s UNSUPPORTED\n", c.group, name_of(c).c_str());
        ggml_free(ctx);
        return 0;
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "alloc failed: %s\n", name_of(c).c_str());
        ggml_free(ctx);
        ok = false;
        return 1;
    }
    fill(q, r, -1.0f, 1.0f);
    fill(k.base, r, -1.0f, 1.0f);
    if (v.base) {
        fill(v.base, r, -1.0f, 1.0f);
    }
    if (s) {
        fill(s, r, -2.0f, 2.0f);
    }
    {
        // each row sees the first kv - nb + 1 + i positions, like a causal verify batch; a few random holes before that
        std::vector<ggml_fp16_t> mask(ggml_nelements(m));
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int64_t s3 = 0; s3 < m->ne[3]; ++s3) {
            for (int64_t i = 0; i < c.nb; ++i) {
                for (int64_t j = 0; j < c.kv; ++j) {
                    const bool visible = j < c.kv - c.nb + 1 + i && (j < 64 || r.next() % 16 != 0);
                    mask[((s3*c.nb) + i)*c.kv + j] = visible ? zero : ninf;
                }
            }
        }
        ggml_backend_tensor_set(m, mask.data(), 0, mask.size()*sizeof(ggml_fp16_t));
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed: %s\n", name_of(c).c_str());
        ok = false;
    }
    std::vector<uint8_t> bytes(ggml_nbytes(out));
    ggml_backend_tensor_get(out, bytes.data(), 0, bytes.size());
    size_t nonfinite = 0;
    for (size_t i = 0; i + 4 <= bytes.size(); i += 4) {
        float f;
        memcpy(&f, bytes.data() + i, 4);
        nonfinite += !std::isfinite(f);
    }
    printf("%s %s %016llx %zu nonfinite=%zu\n", c.group, name_of(c).c_str(), (unsigned long long) fnv64(bytes.data(), bytes.size()),
        bytes.size(), nonfinite);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return 0;
}

static std::vector<fa_case> cases(const std::string & group) {
    std::vector<fa_case> list;
    const std::array<int32_t, 4> kvc = {0, 2, 1, 3}, plain = {0, 1, 2, 3};
    const ggml_type q8 = GGML_TYPE_Q8_0;
    if (group == "decode") {
        for (int hs : {64, 96, 128, 256}) {
            for (int gqa : {2, 4, 6, 8}) {
                list.push_back({"decode", hs, hs, 2, {gqa, 1}, 2048, 1, gqa == 2 || gqa == 6, q8, kvc, false, false});
            }
        }
        // Qwen3.8-like (hs 256, 6 query heads per KV head), sequences, short and long caches
        for (int64_t kv : {256, 2048, 8192, 32768}) {
            for (int64_t seqs : {1, 2}) {
                list.push_back({"decode", 256, 256, 4, {6, seqs}, kv, 1, false, q8, kvc, false, false});
                list.push_back({"decode", 256, 256, 4, {6, seqs}, kv, 1, true, q8, kvc, false, seqs == 2});
            }
        }
        for (const auto & sh : {std::array<int, 3>{192, 128, 8}, {192, 128, 16}, {320, 256, 32}, {512, 512, 2}, {512, 512, 6}}) {
            list.push_back({"decode", sh[0], sh[1], 2, {sh[2], 1}, 2048, 1, false, q8, plain, false, false});
        }
        for (int gqa : {16, 20}) {
            list.push_back({"decode", 576, 512, 1, {gqa, 1}, 2048, 1, false, q8, plain, true, false});
        }
    } else if (group == "verify") {
        for (int64_t kv : {256, 2048, 8192, 32768}) {
            for (int64_t nb : {3, 4, 5, 6, 7, 8}) {
                for (int64_t seqs : {1, 2}) {
                    list.push_back({"verify", 256, 256, 4, {6, seqs}, kv, nb, false, q8, kvc, false, seqs == 2});
                    list.push_back({"verify", 256, 256, 4, {6, seqs}, kv, nb, true, q8, kvc, false, false});
                }
            }
        }
        for (int64_t nb : {3, 4, 5, 8}) {
            list.push_back({"verify", 256, 256, 2, {8, 1}, 2048, nb, false, q8, kvc, false, false});
            list.push_back({"verify", 256, 256, 2, {16, 1}, 2048, nb, true, q8, kvc, false, false});
        }
    } else if (group == "other") {
        for (int64_t nb : {2, 9, 16, 64}) { // nb=1 is the grouped decode path: group decode
            list.push_back({"other", 256, 256, 4, {6, 1}, 2048, nb, false, q8, kvc, false, false});
        }
        for (int64_t nb : {1, 4, 8, 64}) {
            list.push_back({"other", 256, 256, 4, {6, 1}, 2048, nb, false, GGML_TYPE_F16, kvc, false, false});
            list.push_back({"other", 128, 128, 4, {6, 1}, 2048, nb, true, GGML_TYPE_F16, kvc, false, false});
        }
        for (int64_t nb : {4, 8}) {
            list.push_back({"other", 128, 128, 4, {6, 1}, 2048, nb, false, q8, kvc, false, false});
        }
    }
    return list;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s DEVICE decode|verify|other\n", argv[0]);
        return 2;
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(argv[1]);
    if (!dev) {
        fprintf(stderr, "no device %s\n", argv[1]);
        return 2;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    const auto list = cases(argv[2]);
    if (list.empty()) {
        fprintf(stderr, "unknown group %s\n", argv[2]);
        return 2;
    }
    bool ok = true;
    uint64_t seed = 1;
    for (const auto & c : list) {
        run(backend, c, seed++, ok);
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
