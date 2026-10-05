#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double error = 0, norm = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        error += double(a[i] - b[i]) * (a[i] - b[i]);
        norm += double(b[i]) * b[i];
    }
    return error / norm;
}

static std::vector<float> split_attention(const std::vector<float> & q, const std::vector<float> & k,
        const std::vector<float> & v, const std::vector<float> & mask, int kv, int tile, int splits, bool broken) {
    const int dim = 256, heads = 12, gqa = 6;
    const int tiles = (kv + tile - 1) / tile;
    std::vector<float> out(heads * dim);
    for (int h = 0; h < heads; ++h) {
        float maximum = -INFINITY, sum = 0;
        std::vector<float> accum(dim, 0);
        for (int split = 0; split < splits; ++split) {
            if (broken && split == splits - 1) {
                continue;
            }
            float part_max = -INFINITY, part_sum = 0;
            std::vector<float> part(dim, 0);
            const int begin = tiles * split / splits;
            const int end = tiles * (split + 1) / splits;
            for (int t = begin; t < end; ++t) {
                for (int row = t * tile; row < std::min(kv, (t + 1) * tile); ++row) {
                    if (!std::isfinite(mask[row])) {
                        continue;
                    }
                    const size_t offset = size_t(h / gqa * kv + row) * dim;
                    float score = 0;
                    for (int d = 0; d < dim; ++d) {
                        score += q[h * dim + d] * k[offset + d];
                    }
                    score = score / 16 + mask[row];
                    const float next = std::max(part_max, score);
                    const float old_scale = std::exp(part_max - next);
                    const float weight = std::exp(score - next);
                    for (int d = 0; d < dim; ++d) {
                        part[d] = part[d] * old_scale + weight * v[offset + d];
                    }
                    part_sum = part_sum * old_scale + weight;
                    part_max = next;
                }
            }
            if (part_sum == 0) {
                continue;
            }
            const float next = std::max(maximum, part_max);
            const float old_scale = std::exp(maximum - next);
            const float part_scale = std::exp(part_max - next);
            for (int d = 0; d < dim; ++d) {
                accum[d] = accum[d] * old_scale + part[d] * part_scale;
            }
            sum = sum * old_scale + part_sum * part_scale;
            maximum = next;
        }
        for (int d = 0; d < dim; ++d) {
            out[h * dim + d] = accum[d] / sum;
        }
    }
    return out;
}

static std::vector<float> upload(ggml_tensor * t, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1, 1);
    std::vector<float> values(ggml_nelements(t));
    for (float & value : values) {
        value = dist(rng);
    }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
    } else {
        std::vector<char> raw(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, values.data(), raw.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
        ggml_backend_tensor_set(t, raw.data(), 0, raw.size());
        ggml_get_type_traits(t->type)->to_float(raw.data(), values.data(), values.size());
    }
    return values;
}

static bool check(int kv, ggml_type type) {
    const auto start = std::chrono::steady_clock::now();
    ggml_context * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 1, 12, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, type, 256, kv, 2, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, type, 256, kv, 2, 1);
    ggml_tensor * v_float = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, kv, 2, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, 1, 1, 1);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / 16, 0, 0);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 1);
    ggml_backend_cpu_set_use_ref(cpu, true);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    std::mt19937 rng(1234);
    auto q_values = upload(q, rng);
    auto k_values = upload(k, rng);
    auto v_values = upload(v, rng);
    ggml_backend_tensor_set(v_float, v_values.data(), 0, ggml_nbytes(v_float));
    std::vector<float> mask(kv, 0);
    std::vector<ggml_fp16_t> mask_raw(kv);
    for (int row = 0; row < kv; ++row) {
        mask[row] = row % 17 == 0 ? -INFINITY : float(row % 3) * -0.125f;
        mask_raw[row] = ggml_fp32_to_fp16(mask[row]);
    }
    ggml_backend_tensor_set(m, mask_raw.data(), 0, ggml_nbytes(m));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    const bool computed = ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> reference(12 * 256);
    ggml_backend_tensor_get(out, reference.data(), 0, ggml_nbytes(out));
    ggml_tensor * oracle = ggml_flash_attn_ext(ctx, q, k, v_float, m, 1.0f / 16, 0, 0);
    ggml_prec_set_acc(oracle, GGML_PREC_F32);
    ggml_backend_buffer_t oracle_buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    ggml_cgraph * oracle_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(oracle_graph, oracle);
    const bool oracle_computed = ggml_backend_graph_compute(cpu, oracle_graph) == GGML_STATUS_SUCCESS;
    std::vector<float> exact(12 * 256);
    ggml_backend_tensor_get(oracle, exact.data(), 0, ggml_nbytes(oracle));
    ggml_tensor * q_pad = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 4, 12, 1);
    ggml_tensor * m_pad = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, 4, 1, 1);
    ggml_tensor * padded = ggml_flash_attn_ext(ctx, q_pad, k, v_float, m_pad, 1.0f / 16, 0, 0);
    ggml_prec_set_acc(padded, GGML_PREC_F32);
    ggml_backend_buffer_t padded_buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    std::vector<float> padded_q(12 * 4 * 256, 0.25f);
    for (int h = 0; h < 12; ++h) {
        std::copy_n(q_values.data() + h * 256, 256, padded_q.data() + h * 4 * 256);
    }
    ggml_backend_tensor_set(q_pad, padded_q.data(), 0, ggml_nbytes(q_pad));
    for (int row = 0; row < 4; ++row) {
        ggml_backend_tensor_set(m_pad, mask_raw.data(), row * kv * sizeof(ggml_fp16_t), kv * sizeof(ggml_fp16_t));
    }
    ggml_cgraph * padded_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(padded_graph, padded);
    const bool padded_computed = ggml_backend_graph_compute(cpu, padded_graph) == GGML_STATUS_SUCCESS;
    std::vector<float> padded_first(12 * 256);
    ggml_backend_tensor_get(padded, padded_first.data(), 0, padded_first.size() * sizeof(float));
    const bool padding_identical = padded_first == exact;
    const ggml_type q_type = ggml_get_type_traits_cpu(type)->vec_dot_type;
    std::vector<char> q_raw(q_values.size() / ggml_blck_size(q_type) * ggml_type_size(q_type));
    ggml_get_type_traits_cpu(q_type)->from_float(q_values.data(), q_raw.data(), q_values.size());
    ggml_get_type_traits(q_type)->to_float(q_raw.data(), q_values.data(), q_values.size());
    const auto before = split_attention(q_values, k_values, v_values, mask, kv, 64, 18, false);
    const auto after = split_attention(q_values, k_values, v_values, mask, kv, 32, 54, false);
    const auto mutant = split_attention(q_values, k_values, v_values, mask, kv, 32, 54, true);
    const double e_before = nmse(before, exact), e_after = nmse(after, exact), e_mutant = nmse(mutant, exact);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printf("M: type=%s kv=%d CPU seconds=%.3f nmse64=%.9g nmse32=%.9g dropped_split_nmse=%.9g native_reference_nmse=%.9g padding_identical=%d\n", ggml_type_name(type), kv, seconds, e_before, e_after, e_mutant, nmse(reference, exact), padding_identical);
    ggml_backend_buffer_free(padded_buffer);
    ggml_backend_buffer_free(oracle_buffer);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return computed && oracle_computed && padded_computed && padding_identical && e_before < 5e-4 && e_after < 5e-4 && e_mutant > 5e-4;
}

int main() {
    bool ok = true;
    for (int kv : {8192, 32768, 90112, 513}) {
        for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
            ok = check(kv, type) && ok;
        }
    }
    return ok ? 0 : 1;
}
