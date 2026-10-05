#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(bool ok, const char * msg) { if (!ok) { throw std::runtime_error(msg); } }

static void run_case(int S, int B, int T, int K, bool kda, bool duplicate) {
    auto * backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 1);
    ggml_init_params params = {16 * 1024 * 1024, nullptr, true};
    auto * ctx = ggml_init(params);
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, 2, T, B);
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, 2, T, B);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, 4, T, B);
    auto * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kda ? S : 1, 4, T, B);
    auto * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 4, T, B);
    auto * cache = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, 4, 3 * B + 1);
    auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, B);
    auto * wrong = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, B);
    auto * rows = ggml_get_rows(ctx, ggml_reshape_2d(ctx, cache, S * S * 4, cache->ne[3]), ids);
    auto * state = ggml_reshape_4d(ctx, rows, S, S, 4, B);
    auto * ref = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, K);
    auto * indexed = ggml_gated_delta_net_indexed(ctx, q, k, v, g, beta, cache, ids, K);
    auto * bad = ggml_gated_delta_net_indexed(ctx, q, k, v, g, beta, cache, wrong, K);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ref);
    ggml_build_forward_expand(graph, indexed);
    ggml_build_forward_expand(graph, bad);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer, "allocation failed");
    int leaf = 1;
    for (auto * t : {q, k, v, g, beta, cache}) {
        std::vector<float> values(ggml_nelements(t));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = t == g ? -0.1f : t == beta ? 0.5f : std::sin(float(i + leaf * 31) * 0.13f) * 0.1f;
        }
        ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
        ++leaf;
    }
    std::vector<int32_t> indices(B), wrong_indices(B, 0);
    for (int i = 0; i < B; ++i) { indices[i] = duplicate ? 3 * B : 3 * B - 2 * i; }
    ggml_backend_tensor_set(ids, indices.data(), 0, B * sizeof(int32_t));
    ggml_backend_tensor_set(wrong, wrong_indices.data(), 0, B * sizeof(int32_t));
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute failed");
    const size_t elems = S * 4 * T * B + S * S * 4 * B * std::min(T, K);
    std::vector<float> a(elems), b(elems), c(elems);
    ggml_backend_tensor_get(ref, a.data(), 0, elems * sizeof(float));
    ggml_backend_tensor_get(indexed, b.data(), 0, elems * sizeof(float));
    ggml_backend_tensor_get(bad, c.data(), 0, elems * sizeof(float));
    check(std::memcmp(a.data(), b.data(), elems * sizeof(float)) == 0, "indexed state differs from GET_ROWS");
    check(std::memcmp(a.data(), c.data(), elems * sizeof(float)) != 0, "wrong-slot negative control escaped");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

int main() {
    try {
        int cases = 0;
        for (int S : {16, 32, 64, 128}) {
            for (int B : {1, 2}) {
                for (bool kda : {false, true}) {
                    for (bool duplicate : {false, true}) {
                        for (int T : {1, 6, 8}) {
                            for (int K : {1, 6}) {
                                run_case(S, B, T, K, kda, duplicate);
                                ++cases;
                            }
                        }
                    }
                }
            }
        }
        std::printf("M: %d exact GET_ROWS comparisons and %d wrong-slot controls passed\n", cases, cases);
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
