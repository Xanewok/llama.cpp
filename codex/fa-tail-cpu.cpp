#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) {
        fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

int main(int argc, char **) {
    check(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    const bool mutant = argc > 1;
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 8);
    int cases = 0;
    double max_error = 0;
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        for (int nq : {1, 4, 6, 8}) {
            for (int tail : {0, 1, 255, 256, 257, 4095}) {
                const int nk = 4352, live = nk - tail, dim = 256, nh = 4, gqa = 6;
                ggml_init_params params = {16*1024*1024, nullptr, true};
                ggml_context * ctx = ggml_init(params);
                ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, nq, nh*gqa, 1);
                ggml_tensor * k = ggml_new_tensor_4d(ctx, type, dim, nk, nh, 1);
                ggml_tensor * v = ggml_new_tensor_4d(ctx, type, dim, nk, nh, 1);
                ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nk, nq, 1, 1);
                int cut = mutant ? ((live + 255)/256 - 1)*256 : nk;
                ggml_tensor * kp = ggml_view_4d(ctx, k, dim, cut, nh, 1, k->nb[1], k->nb[2], k->nb[3], 0);
                ggml_tensor * vp = ggml_view_4d(ctx, v, dim, cut, nh, 1, v->nb[1], v->nb[2], v->nb[3], 0);
                ggml_tensor * mp = ggml_view_4d(ctx, m, cut, nq, 1, 1, m->nb[1], m->nb[2], m->nb[3], 0);
                mp = ggml_cont(ctx, mp);
                ggml_tensor * out = ggml_flash_attn_ext(ctx, q, kp, vp, mp, 1.0f/16, 0, 0);
                ggml_tensor * kr = ggml_view_4d(ctx, k, dim, live, nh, 1, k->nb[1], k->nb[2], k->nb[3], 0);
                ggml_tensor * vr = ggml_view_4d(ctx, v, dim, live, nh, 1, v->nb[1], v->nb[2], v->nb[3], 0);
                ggml_tensor * mr = ggml_view_4d(ctx, m, live, nq, 1, 1, m->nb[1], m->nb[2], m->nb[3], 0);
                mr = ggml_cont(ctx, mr);
                ggml_tensor * ref = ggml_flash_attn_ext(ctx, q, kr, vr, mr, 1.0f/16, 0, 0);
                ggml_cgraph * graph = ggml_new_graph_custom(ctx, 256, false);
                ggml_build_forward_expand(graph, out);
                ggml_build_forward_expand(graph, ref);
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
                check(buf != nullptr, "allocation failed");
                ggml_backend_buffer_clear(buf, 0);
                std::vector<float> vals(ggml_nelements(v), 0);
                const int tile_start = ((live - 1)/256)*256;
                for (int h = 0; h < nh; ++h) {
                    for (int i = tile_start; i < live; ++i) {
                        std::fill_n(vals.begin() + (h*nk + i)*dim, dim, 100.0f);
                    }
                }
                std::vector<unsigned char> quant(ggml_nbytes(v));
                ggml_quantize_chunk(type, vals.data(), quant.data(), 0, nk*nh, dim, nullptr);
                ggml_backend_tensor_set(v, quant.data(), 0, quant.size());
                std::vector<ggml_fp16_t> mask(nk*nq, ggml_fp32_to_fp16(-INFINITY));
                for (int i = 0; i < nq; ++i) {
                    std::fill_n(mask.begin() + i*nk, live, ggml_fp32_to_fp16(0));
                }
                ggml_backend_tensor_set(m, mask.data(), 0, ggml_nbytes(m));
                check(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS, "compute failed");
                std::vector<float> a(ggml_nelements(out)), b(a.size());
                ggml_backend_tensor_get(out, a.data(), 0, ggml_nbytes(out));
                ggml_backend_tensor_get(ref, b.data(), 0, ggml_nbytes(ref));
                for (size_t i = 0; i < a.size(); ++i) {
                    check(std::isfinite(a[i]) && std::isfinite(b[i]), "non-finite output");
                    max_error = std::max(max_error, double(std::abs(a[i] - b[i])));
                }
                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
                ++cases;
            }
        }
    }
    ggml_backend_free(cpu);
    printf("M: cases=%d max_abs_error=%.9g mutant=%d\n", cases, max_error, mutant);
    return max_error <= 1e-5 ? 0 : 1;
}
