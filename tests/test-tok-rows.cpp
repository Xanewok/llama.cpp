// llm_graph_input_tok_rows: GET_ROWS over the raw rows gathered on the host (indices 0..n-1) must give, on every
// backend, exactly the bits of GET_ROWS over the whole table with the token ids.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); n_fail++; } } while (0)

// empty if the backend does not support the op
static std::vector<float> get_rows(ggml_backend_t backend, ggml_type type, int64_t ne0, int64_t nrows,
                                   const void * table, const std::vector<int32_t> & idx) {
    ggml_init_params ip = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * src = ggml_new_tensor_2d(ctx, type, ne0, nrows);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) idx.size());
    ggml_tensor * out = ggml_get_rows(ctx, src, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    std::vector<float> res;
    if (ggml_backend_supports_op(backend, out)) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(src, table, 0, ggml_nbytes(src));
        ggml_backend_tensor_set(ids, idx.data(), 0, ggml_nbytes(ids));
        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        res.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    return res;
}

static size_t n_bit_diff(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        n += memcmp(&a[i], &b[i], sizeof(float)) != 0;
    }
    return n;
}

int main() {
    ggml_backend_load_all();

    std::vector<ggml_backend_t> backends;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        backends.push_back(ggml_backend_dev_init(ggml_backend_dev_get(i), nullptr));
    }

    const int64_t ne0 = 2048, n_vocab = 300;
    std::mt19937 rng(1729);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> table_f32(ne0*n_vocab);
    for (auto & v : table_f32) {
        v = nd(rng);
    }

    // both table ends and a repeated id
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

    int n_cpu_types = 0;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const ggml_type type = (ggml_type) t;
        const ggml_type_traits * traits = ggml_get_type_traits(type);
        const bool float_or_quant = type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || ggml_is_quantized(type);
        if (!float_or_quant || (!traits->to_float && type != GGML_TYPE_F32) || ne0 % ggml_blck_size(type) != 0 || ggml_quantize_requires_imatrix(type)) {
            continue;
        }

        const size_t rs = ggml_row_size(type, ne0);
        std::vector<uint8_t> table(rs*n_vocab);
        ggml_quantize_chunk(type, table_f32.data(), table.data(), 0, n_vocab, ne0, nullptr);

        std::vector<uint8_t> rows(rs*n);
        for (int32_t i = 0; i < n; ++i) {
            memcpy(rows.data() + i*rs, table.data() + tokens[i]*rs, rs);
        }
        // the first row taken from the next token's row
        std::vector<uint8_t> rows_wrong = rows;
        memcpy(rows_wrong.data(), table.data() + ((tokens[0] + 1) % n_vocab)*rs, rs);

        for (ggml_backend_t backend : backends) {
            const char * bname = ggml_backend_name(backend);
            const std::vector<float> ref = get_rows(backend, type, ne0, n_vocab, table.data(), tokens);
            if (ref.empty()) {
                printf("tok_rows: %s %s GET_ROWS unsupported\n", bname, traits->type_name);
                continue;
            }
            const bool cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
            n_cpu_types += cpu;

            const size_t nd_ = n_bit_diff(ref, get_rows(backend, type, ne0, n, rows.data(), iota));
            printf("tok_rows: %s %s differing=%zu/%zu\n", bname, traits->type_name, nd_, ref.size());
            CHECK(nd_ == 0, "%s %s: gathered rows differ from the whole-table lookup", bname, traits->type_name);

            const size_t nw = n_bit_diff(ref, get_rows(backend, type, ne0, n, rows_wrong.data(), iota));
            CHECK(nw > 0, "%s %s: wrong-row mutant not detected", bname, traits->type_name);
        }
    }
    CHECK(n_cpu_types >= 10, "only %d types tested on the CPU backend", n_cpu_types);

    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }
    printf("test-tok-rows: %s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}
