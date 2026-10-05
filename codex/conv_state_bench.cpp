// The galloc-model mode preserves projection, state gather, concat, snapshots, conv, and SiLU order.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-ssm-conv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct layer { ggml_tensor *ids, *state, *tok, *concat, *conv, *silu, *w, *cache, *first; };

static bool overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t pa = (uintptr_t) a->data, pb = (uintptr_t) b->data;
    return pa <= pb ? pb - pa < ggml_nbytes(a) : pa - pb < ggml_nbytes(b);
}

int main(int argc, char ** argv) {
    std::string dev = "CPU";
    int L = 48, C = 5120, T = 6, K = 6, iters = 300, layout_only = 0, old_contract = 0;
    std::string only;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--backend") dev = argv[i + 1];
        else if (a == "--layers") L = atoi(argv[i + 1]);
        else if (a == "--channels") C = atoi(argv[i + 1]);
        else if (a == "--tokens") T = atoi(argv[i + 1]);
        else if (a == "--K") K = atoi(argv[i + 1]);
        else if (a == "--iters") iters = atoi(argv[i + 1]);
        else if (a == "--check-layout") layout_only = atoi(argv[i + 1]);
        else if (a == "--old-contract") old_contract = atoi(argv[i + 1]);
        else if (a == "--mode") only = argv[i + 1];
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (L < 1 || L > 48 || C < 1 || C > 5120 || T < 1 || T > 32 || K < 1 || K > 6 || iters < 1) {
        return 2;
    }
    const int H = 3; // conv kernel 4
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name(dev.c_str(), nullptr);
    if (!backend) { fprintf(stderr, "no backend %s\n", dev.c_str()); return 2; }
    printf("backend %s layers %d channels %d tokens %d K %d iters %d\n", ggml_backend_name(backend), L, C, T, K, iters);

    // weights and caches persist across modes (cache = conv_states_all: K planes of one [H*C] row)
    // SSRC/XSRC: persistent sources gathered by get_rows, as build_rs and the qkv projection feed the concat in the model
    ggml_init_params wp = { ggml_tensor_overhead() * (4 * L + 4), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    std::vector<ggml_tensor *> W(L), CACHE(L), SSRC(L), XSRC(L);
    for (int l = 0; l < L; ++l) {
        W[l] = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, H + 1, C);
        CACHE[l] = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, (int64_t) H * C, K);
        SSRC[l] = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, (int64_t) H * C, 1);
        XSRC[l] = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, C, T);
    }
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    auto fill = [&](ggml_tensor * t) {
        std::vector<float> v(ggml_nelements(t));
        for (auto & f : v) f = U(rng);
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    };
    for (int l = 0; l < L; ++l) { fill(W[l]); fill(SSRC[l]); fill(XSRC[l]); }
    std::vector<int32_t> ids_x(T), ids_s(1, 0);
    for (int t = 0; t < T; ++t) ids_x[t] = t;

    std::vector<std::vector<float>> ref_out, ref_cache;
    const char * modes[] = { "sep-unfused", "sep-fused", "galloc-model", "galloc-pinned" };
    for (const char * mode : modes) {
        if (!only.empty() && only != mode) continue;
        fprintf(stderr, "BENCH_MODE %s\n", mode);
        const bool sep = strncmp(mode, "sep", 3) == 0, unfused = strcmp(mode, "sep-unfused") == 0;
        const bool pinned = strcmp(mode, "galloc-pinned") == 0; // concat inputs outlive the conv: the alloc-dep fix
        ggml_init_params p = { ggml_tensor_overhead() * (size_t) L * (16 + 3 * K) + ggml_graph_overhead_custom(8192, false), nullptr, true };
        ggml_context * ctx = ggml_init(p);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
        std::vector<layer> ls(L);
        for (int l = 0; l < L; ++l) {
            layer & y = ls[l];
            y.w = W[l]; y.cache = CACHE[l];
            y.ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T); ggml_set_input(y.ids);
            // tokens first, then the state row: the model's order (qkv projection, then build_rs inside build_conv_state)
            y.tok = ggml_get_rows(ctx, XSRC[l], y.ids);
            ggml_build_forward_expand(gf, y.tok);
            ggml_tensor * rows = ggml_get_rows(ctx, SSRC[l], ggml_view_1d(ctx, y.ids, 1, 0));
            if (pinned) { ggml_set_output(y.tok); ggml_set_output(rows); }
            y.state = ggml_reshape_3d(ctx, rows, H, C, 1);
            y.concat = ggml_concat(ctx, y.state, ggml_transpose(ctx, y.tok), 0);

            y.first = nullptr;
            for (int t = 1; t <= K; ++t) {
                const int64_t s_idx = std::max<int64_t>(0, (int64_t) T - K + t);
                const int64_t slot = K - t;
                ggml_tensor * src = ggml_view_3d(ctx, y.concat, H, C, 1, y.concat->nb[1], y.concat->nb[2], s_idx * sizeof(float));
                ggml_tensor * dst = ggml_view_2d(ctx, y.cache, (int64_t) H * C, 1, y.cache->nb[1], slot * y.cache->nb[1]);
                auto * copy = ggml_cpy(ctx, src, dst);
                if (!y.first) y.first = copy;
                ggml_build_forward_expand(gf, copy);
            }
            y.conv = ggml_ssm_conv(ctx, y.concat, y.w);
            ggml_format_name(y.conv, "conv_output_raw-%d", l);
            if (unfused) ggml_set_output(y.conv);
            y.silu = ggml_silu(ctx, y.conv);
            ggml_set_output(y.silu);
            ggml_build_forward_expand(gf, y.silu);
        }
        ggml_backend_buffer_t buf = nullptr;
        ggml_gallocr_t ga = nullptr;
        if (sep) {
            buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        } else {
            ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(ga, gf)) { fprintf(stderr, "gallocr failed\n"); return 2; }
        }
        int n_ov_tok = 0, n_ov_state = 0;
        for (auto & y : ls) {
            n_ov_tok += overlap(y.silu, y.tok);
            n_ov_state += overlap(y.silu, y.state);
        }
        int eligible = 0;
        for (const auto & y : ls) {
            int start = 0;
            while (start < gf->n_nodes && gf->nodes[start] != y.first) ++start;
            ggml_ssm_conv_state_fusion match;
            const char * reason = nullptr;
            bool matched = start < gf->n_nodes && ggml_match_ssm_conv_state_fusion(gf, start, match, &reason);
            if (old_contract && matched && overlap(y.silu, y.tok)) {
                matched = false;
                reason = "old output-vs-dead-input contract";
            }
            eligible += matched;
            if (matched && (match.slots != K || match.out != y.silu)) return 8;
            if (!matched && !unfused) {
                fprintf(stderr, "engagement rejected: %s\n", reason ? reason : "missing copy");
                return 8;
            }
            if (matched && unfused) return 8;
        }
        printf("LAYOUT mode=%s eligible=%d/%d silu_overlaps_tokens=%d/%d\n", mode, eligible, L, n_ov_tok, L);
        if (!sep && !pinned && T == 6 && n_ov_tok != L) return 8;
        if (layout_only) {
            if (ga) ggml_gallocr_free(ga);
            if (buf) ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            continue;
        }
        for (int l = 0; l < L; ++l) ggml_backend_tensor_set(ls[l].ids, ids_x.data(), 0, ggml_nbytes(ls[l].ids));
        for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(backend, gf); // CUDA graph warmup needs 2 stable calls
        // correctness vs the unfused reference: outputs and every snapshot plane
        std::vector<std::vector<float>> out(L), cache(L);
        for (int l = 0; l < L; ++l) {
            out[l].resize(ggml_nelements(ls[l].silu)); cache[l].resize(ggml_nelements(CACHE[l]));
            ggml_backend_tensor_get(ls[l].silu, out[l].data(), 0, ggml_nbytes(ls[l].silu));
            ggml_backend_tensor_get(CACHE[l], cache[l].data(), 0, ggml_nbytes(CACHE[l]));
        }
        double max_out = 0, max_cache = 0;
        if (unfused) { ref_out = out; ref_cache = cache; }
        else if (!ref_out.empty()) {
            for (int l = 0; l < L; ++l) {
                for (size_t i = 0; i < out[l].size(); ++i) max_out = std::max(max_out, (double) std::fabs(out[l][i] - ref_out[l][i]));
                for (size_t i = 0; i < cache[l].size(); ++i) max_cache = std::max(max_cache, (double) std::fabs(cache[l][i] - ref_cache[l][i]));
            }
        }
        if (!ref_out.empty() && !unfused) {
            for (int l = 0; l < L; ++l) {
                if (memcmp(out[l].data(), ref_out[l].data(), out[l].size() * sizeof(float)) ||
                    memcmp(cache[l].data(), ref_cache[l].data(), cache[l].size() * sizeof(float))) {
                    fprintf(stderr, "bitwise output or cache mismatch\n");
                    return 8;
                }
            }
        }
        std::vector<double> sync_us, async_us;
        for (int rep = 0; rep < 5; ++rep) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(backend, gf);
            auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) ggml_backend_graph_compute_async(backend, gf);
            ggml_backend_synchronize(backend);
            auto t2 = std::chrono::steady_clock::now();
            sync_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / iters);
            async_us.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count() / iters);
        }
        std::sort(sync_us.begin(), sync_us.end()); std::sort(async_us.begin(), async_us.end());
        printf("RESULT mode=%s us_per_graph_sync_med=%.2f us_per_graph_async_med=%.2f us_per_layer_async=%.3f "
               "silu_overlaps_tokens=%d/%d silu_overlaps_state=%d/%d max_abs_out=%.3g max_abs_cache=%.3g\n",
            mode, sync_us[2], async_us[2], async_us[2] / L, n_ov_tok, L, n_ov_state, L, max_out, max_cache);
        fflush(stdout);
        if (ga) ggml_gallocr_free(ga);
        if (buf) ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(backend);
    return 0;
}
