// Exactness of token-embedding rows dequantized on a device (llm_graph_input_tok_rows): GET_ROWS over raw rows gathered
// on the host, on every backend, against the CPU GET_ROWS of the whole table, bit for bit.
// IQ4_XS values are exact products (fp16 d: 11 bits, ls - 32: 6, grid: 7; 24 <= float's 24), so any association or FMA
// gives the same bits; the test asserts that, and that a wrong row, swapped nibbles or an f16 intermediate are caught.
// env TOK_ROWS_REQUIRE_GPU=N: fail with fewer than N GPU devices. argv: types that must be bitwise on every backend.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); n_fail++; } } while (0)

// out = get_rows(src, idx) on backend, src given as raw bytes of type [ne0, nrows]
static std::vector<float> get_rows(ggml_backend_t backend, ggml_type type, int64_t ne0, int64_t nrows,
                                   const void * src_data, const std::vector<int32_t> & idx) {
    ggml_init_params ip = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * src = ggml_new_tensor_2d(ctx, type, ne0, nrows);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) idx.size());
    ggml_tensor * out = ggml_get_rows(ctx, src, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    std::vector<float> res;
    if (!ggml_backend_supports_op(backend, out)) {
        ggml_free(ctx);
        return res;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(src, src_data, 0, ggml_nbytes(src));
    ggml_backend_tensor_set(ids, idx.data(), 0, ggml_nbytes(ids));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    res.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

static size_t n_bit_diff(const std::vector<float> & a, const std::vector<float> & b, double * maxabs) {
    GGML_ASSERT(a.size() == b.size());
    size_t n = 0;
    *maxabs = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            n++;
            *maxabs = std::fmax(*maxabs, std::fabs((double) a[i] - b[i]));
        }
    }
    return n;
}

static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

enum iq4_variant { SAME_ORDER, REASSOCIATED, NIBBLES_SWAPPED, F16_INTERMEDIATE };

// IQ4_XS super-block: half d, u16 scales_h, u8 scales_l[4], u8 qs[128] (136 bytes, 256 values)
static void dequant_iq4_xs(const uint8_t * row, int64_t ne0, float * y, iq4_variant v) {
    for (int64_t ib = 0; ib < ne0/256; ++ib) {
        const uint8_t * b = row + ib*136;
        ggml_fp16_t dh;
        uint16_t scales_h;
        memcpy(&dh, b, 2);
        memcpy(&scales_h, b + 2, 2);
        const uint8_t * scales_l = b + 4;
        const uint8_t * qs = b + 8;
        const float d = ggml_fp16_to_fp32(dh);
        for (int j = 0; j < 8; ++j) {
            const int ls = ((scales_l[j/2] >> 4*(j%2)) & 0xf) | (((scales_h >> 2*j) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int k = 0; k < 16; ++k) {
                int lo = kvalues_iq4nl[qs[16*j + k] & 0xf];
                int hi = kvalues_iq4nl[qs[16*j + k] >> 4];
                if (v == NIBBLES_SWAPPED) {
                    std::swap(lo, hi);
                }
                float * o = y + 256*ib + 32*j + k;
                o[ 0] = v == REASSOCIATED ? d * (float) ((ls - 32)*lo) : dl * lo;
                o[16] = v == REASSOCIATED ? d * (float) ((ls - 32)*hi) : dl * hi;
                if (v == F16_INTERMEDIATE) {
                    o[ 0] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(o[ 0]));
                    o[16] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(o[16]));
                }
            }
        }
    }
}

int main(int argc, char ** argv) {
    std::set<std::string> required;
    for (int i = 1; i < argc; ++i) {
        required.insert(argv[i]);
    }

    std::vector<ggml_backend_t> backends = { ggml_backend_cpu_init() };
    int n_gpu = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto t = ggml_backend_dev_type(dev);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            backends.push_back(ggml_backend_dev_init(dev, nullptr));
            n_gpu++;
        }
    }
    const char * req_gpu = getenv("TOK_ROWS_REQUIRE_GPU");
    CHECK(!req_gpu || n_gpu >= atoi(req_gpu), "%d GPU devices, TOK_ROWS_REQUIRE_GPU=%s", n_gpu, req_gpu);
    printf("tok_rows: backends=%zu gpus=%d\n", backends.size(), n_gpu);

    const int64_t ne0 = 5120, n_vocab = 300;
    std::mt19937 rng(1729);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> table_f32(ne0*n_vocab);
    for (auto & v : table_f32) {
        v = nd(rng);
    }
    // 46 rows per round at np2 (D): duplicates and both table ends included
    std::vector<int32_t> tokens = { 0, (int32_t) n_vocab - 1, 7, 7 };
    std::uniform_int_distribution<int32_t> ud(0, (int32_t) n_vocab - 1);
    while (tokens.size() < 46) {
        tokens.push_back(ud(rng));
    }
    const int32_t n = (int32_t) tokens.size();
    std::vector<int32_t> iota(n);
    for (int32_t i = 0; i < n; ++i) {
        iota[i] = i;
    }

    const ggml_type types[] = {
        GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_MXFP4,
    };

    std::set<std::string> seen;
    for (ggml_type type : types) {
        const std::string tname = ggml_type_name(type);
        seen.insert(tname);
        if (ggml_quantize_requires_imatrix(type)) {
            printf("tok_rows: %s skipped (needs an imatrix)\n", tname.c_str());
            continue;
        }
        const size_t rs = ggml_row_size(type, ne0);
        std::vector<uint8_t> table(rs*n_vocab);
        ggml_quantize_chunk(type, table_f32.data(), table.data(), 0, n_vocab, ne0, nullptr);

        // reference: the CPU path before the change, GET_ROWS on the whole table
        const std::vector<float> ref = get_rows(backends[0], type, ne0, n_vocab, table.data(), tokens);
        CHECK(!ref.empty(), "CPU GET_ROWS unsupported for %s", tname.c_str());
        if (ref.empty()) {
            continue;
        }

        std::vector<uint8_t> rows(rs*n), rows_mut(rs*n);
        for (int32_t i = 0; i < n; ++i) {
            memcpy(rows.data() + i*rs, table.data() + tokens[i]*rs, rs);
        }
        rows_mut = rows;
        memcpy(rows_mut.data(), table.data() + ((tokens[0] + 1) % n_vocab)*rs, rs);

        for (ggml_backend_t backend : backends) {
            const char * bname = ggml_backend_name(backend);
            const std::vector<float> got = get_rows(backend, type, ne0, n, rows.data(), iota);
            if (got.empty()) {
                printf("tok_rows: %s %s unsupported (the CPU path stays)\n", bname, tname.c_str());
                CHECK(!required.count(tname), "%s %s unsupported but required", bname, tname.c_str());
                continue;
            }
            double maxabs;
            const size_t nd_ = n_bit_diff(ref, got, &maxabs);
            printf("tok_rows: %s %s bitwise=%d differing=%zu/%zu maxabs=%.3g\n", bname, tname.c_str(), nd_ == 0, nd_,
                   ref.size(), maxabs);
            const bool cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
            CHECK(nd_ == 0 || !(cpu || required.count(tname)), "%s %s not bitwise", bname, tname.c_str());

            const std::vector<float> mut = get_rows(backend, type, ne0, n, rows_mut.data(), iota);
            const size_t nm = n_bit_diff(ref, mut, &maxabs);
            CHECK(nm > 0, "row mutant escaped on %s %s", bname, tname.c_str());
            if (nm > 0) {
                printf("tok_rows: %s %s row mutant caught differing=%zu\n", bname, tname.c_str(), nm);
            }
        }

        if (type == GGML_TYPE_IQ4_XS) {
            const char * names[] = { "same-order", "reassociated", "nibbles-swapped", "f16-intermediate" };
            for (int v = SAME_ORDER; v <= F16_INTERMEDIATE; ++v) {
                std::vector<float> y(ref.size());
                for (int32_t i = 0; i < n; ++i) {
                    dequant_iq4_xs(rows.data() + i*rs, ne0, y.data() + i*ne0, (iq4_variant) v);
                }
                double maxabs;
                const size_t nv = n_bit_diff(ref, y, &maxabs);
                printf("tok_rows: iq4_xs %s differing=%zu/%zu maxabs=%.3g\n", names[v], nv, ref.size(), maxabs);
                if (v <= REASSOCIATED) {
                    CHECK(nv == 0, "iq4_xs %s dequant differs from the CPU", names[v]);
                } else {
                    CHECK(nv > 0, "iq4_xs %s mutant escaped the bitwise comparison", names[v]);
                }
            }
        }
    }
    for (const auto & r : required) {
        CHECK(seen.count(r), "required type %s not tested", r.c_str());
    }

    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }
    printf("test-tok-rows: %s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}
