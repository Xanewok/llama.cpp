// Windowed vs one-pass q8_0 flash attention through the public ggml API only.
//   fa-window DEVICE save DIR   writes the raw f32 outputs, one file per case
//   fa-window DEVICE cmp DIR    prints "CASE nmse=.. maxabs=.. bitwise=0|1" against the files in DIR
// Run "save" with GGML_CUDA_FA_KV_WINDOW=0 (one-pass conversion), then "cmp" with the default window, a small window and each mutant.
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
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

static const char * g_dir;
static bool g_save;
static int g_index;

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
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    size_t nonfinite = 0;
    for (float f : res) {
        nonfinite += !std::isfinite(f);
    }
    char file[1024];
    snprintf(file, sizeof file, "%s/%03d.bin", g_dir, g_index);
    if (g_save) {
        FILE * f = fopen(file, "wb");
        if (!f || fwrite(res.data(), 4, res.size(), f) != res.size() || fclose(f)) {
            fprintf(stderr, "write failed: %s\n", file);
            ok = false;
        }
        printf("%03d %s saved nonfinite=%zu\n", g_index, name_of(c).c_str(), nonfinite);
    } else {
        std::vector<float> ref(res.size());
        FILE * f = fopen(file, "rb");
        if (!f || fread(ref.data(), 4, ref.size(), f) != ref.size()) {
            fprintf(stderr, "read failed: %s\n", file);
            ok = false;
            ref.assign(ref.size(), 0.0f);
        }
        if (f) {
            fclose(f);
        }
        double num = 0, den = 0, maxabs = 0;
        for (size_t i = 0; i < res.size(); ++i) {
            const double d = (double) res[i] - ref[i];
            num += d*d;
            den += (double) ref[i]*ref[i];
            maxabs = std::max(maxabs, std::fabs(d));
        }
        printf("%03d %s nmse=%.3e maxabs=%.3e bitwise=%d nonfinite=%zu\n", g_index, name_of(c).c_str(), den > 0 ? num/den : -1.0, maxabs,
            (int) (memcmp(res.data(), ref.data(), res.size()*4) == 0), nonfinite);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return 0;
}

// Windowed shapes: 1 window (8192), a short second window (8448), several windows, one and two sequences, sinks, GQA.
static std::vector<fa_case> cases() {
    const std::array<int32_t, 4> kvc = {0, 2, 1, 3};
    const ggml_type q8 = GGML_TYPE_Q8_0;
    std::vector<fa_case> list;
    list.push_back({"w", 256, 256, 4, {6, 1}, 8192, 512, false, q8, kvc, false, false});
    list.push_back({"w", 256, 256, 4, {6, 1}, 8448, 512, false, q8, kvc, false, false});
    list.push_back({"w", 256, 256, 4, {6, 1}, 20480, 128, true, q8, kvc, false, false});
    list.push_back({"w", 256, 256, 4, {6, 1}, 32768, 512, false, q8, kvc, false, false});
    list.push_back({"w", 256, 256, 4, {6, 2}, 32768, 64, false, q8, kvc, false, true});
    list.push_back({"w", 256, 256, 4, {6, 1}, 90112, 128, false, q8, kvc, false, false});
    list.push_back({"w", 256, 256, 4, {6, 2}, 90112, 64, true, q8, kvc, false, true});
    for (int gqa : {1, 4, 8}) {
        list.push_back({"w", 128, 128, 4, {gqa, 1}, 32768, 128, false, q8, kvc, false, false});
    }
    return list;
}

int main(int argc, char ** argv) {
    if (argc != 4 || (strcmp(argv[2], "save") && strcmp(argv[2], "cmp"))) {
        fprintf(stderr, "usage: %s DEVICE save|cmp DIR\n", argv[0]);
        return 2;
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(argv[1]);
    if (!dev) {
        fprintf(stderr, "no device %s\n", argv[1]);
        return 2;
    }
    g_save = !strcmp(argv[2], "save");
    g_dir = argv[3];
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    bool ok = true;
    uint64_t seed = 1;
    for (const auto & c : cases()) {
        run(backend, c, seed++, ok);
        ++g_index;
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
