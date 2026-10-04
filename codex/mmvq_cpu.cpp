#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    const bool negative = argc == 2 && std::strcmp(argv[1], "--negative-control") == 0;
    const auto begin = std::chrono::steady_clock::now();
    int cases = 0;
    for (ggml_type type : {GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0}) {
        for (int n : {1, 2, 4, 6, 8}) {
            const int k = 5120, m = 37;
            ggml_context * ctx = ggml_init({32*1024*1024, nullptr, false});
            ggml_tensor * a = ggml_new_tensor_2d(ctx, type, k, m);
            ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
            std::vector<float> input(k*m);
            for (int i = 0; i < k*m; ++i) {
                input[i] = std::sin(float(i)*0.13f);
            }
            ggml_quantize_chunk(type, input.data(), a->data, 0, m, k, nullptr);
            for (int i = 0; i < k*n; ++i) {
                static_cast<float *>(b->data)[i] = std::cos(float(i)*0.17f);
            }
            ggml_tensor * dst = ggml_mul_mat(ctx, a, b);
            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, dst);
            ggml_cplan plan = ggml_graph_plan(graph, 8, nullptr);
            std::vector<uint8_t> work(plan.work_size);
            plan.work_data = work.data();
            if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) {
                return 2;
            }
            const float * values = static_cast<float *>(dst->data);
            const ggml_type rhs_type = ggml_get_type_traits_cpu(type)->vec_dot_type;
            std::vector<uint8_t> rhs_quant(ggml_row_size(rhs_type, k)*n);
            std::vector<float> rhs(k*n), lhs(k*m), expected(m*n);
            ggml_get_type_traits(type)->to_float(a->data, lhs.data(), k*m);
            for (int col = 0; col < n; ++col) {
                const size_t offset = ggml_row_size(rhs_type, k)*col;
                ggml_get_type_traits_cpu(rhs_type)->from_float(static_cast<float *>(b->data) + col*k, rhs_quant.data() + offset, k);
                if (rhs_type == GGML_TYPE_Q8_K) {
                    dequantize_row_q8_K(reinterpret_cast<const block_q8_K *>(rhs_quant.data() + offset), rhs.data() + col*k, k);
                } else {
                    ggml_get_type_traits(rhs_type)->to_float(rhs_quant.data() + offset, rhs.data() + col*k, k);
                }
                for (int row = 0; row < m; ++row) {
                    double sum = 0;
                    for (int i = 0; i < k; ++i) {
                        sum += double(lhs[row*k + i])*rhs[col*k + i];
                    }
                    expected[col*m + row] = sum;
                }
            }
            if (negative) {
                expected[0] += 1e6f;
            }
            double error = 0, norm = 0;
            for (int i = 0; i < m*n; ++i) {
                const double delta = values[i] - expected[i];
                error += delta*delta;
                norm += double(expected[i])*expected[i];
            }
            const double nmse = error/std::fmax(norm, 1e-30);
            std::printf("%s ncols=%d nmse_D=%.12g\n", ggml_type_name(type), n, nmse);
            ggml_free(ctx);
            if (!(nmse < 1e-7)) {
                return 1;
            }
            ++cases;
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    std::printf("cases_M=%d wall_ms_M=%.3f\n", cases, ms);
    return 0;
}
