#pragma once

#include "sampling.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

static void test_sampling_bias(
        llama_model * model,
        const std::function<llama_context_ptr(llama_sampler *)> & make_ctx,
        const std::function<void(llama_context *)> & decode,
        const std::vector<std::vector<float>> & raw,
        const std::vector<int32_t> & rows) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int32_t k = 8;
    const auto & first = raw.front();
    const llama_token best = std::max_element(first.begin(), first.end()) - first.begin();
    const llama_token low = std::min_element(first.begin(), first.end()) - first.begin();
    const float lift = first[best] - first[low] + 0.125f;
    const auto other_begin = low < n_vocab/2 ? first.begin() + n_vocab/2 : first.begin();
    const auto other_end = low < n_vocab/2 ? first.end() : first.begin() + n_vocab/2;
    const llama_token low_other = std::min_element(other_begin, other_end) - first.begin();
    const float lift_other = first[best] - first[low_other] - 0.125f;

    for (const bool finite : { false, true }) {
        common_params_sampling params;
        params.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_MIN_P, COMMON_SAMPLER_TYPE_TEMPERATURE };
        params.top_k = k;
        params.top_p = 0.95f;
        params.min_p = 0.05f;
        params.temp = 1.0f;
        params.backend_sampling = true;
        params.logit_bias = finite ? std::vector<llama_logit_bias>{{ low, lift }, { low_other, lift_other }, { best, -0.375f }}
                                  : std::vector<llama_logit_bias>{{ best, -INFINITY }};
        common_sampler_ptr backend(common_sampler_init(model, params));
        llama_context_ptr ctx = make_ctx(common_sampler_get_backend(backend.get()));
        decode(ctx.get());

        for (size_t r = 0; r < rows.size(); ++r) {
            const int32_t idx = rows[r];
            const float * logits = llama_get_sampled_logits_ith(ctx.get(), idx);
            const llama_token * ids = llama_get_sampled_candidates_ith(ctx.get(), idx);
            const uint32_t count = llama_get_sampled_candidates_count_ith(ctx.get(), idx);
            GGML_ASSERT(logits && ids && count == k + 1);
            auto expected = raw[r];
            for (const auto & lb : params.logit_bias) {
                expected[lb.token] += lb.bias;
            }
            auto sorted = expected;
            std::sort(sorted.begin(), sorted.end(), std::greater<float>());
            for (uint32_t i = 0; i < count; ++i) {
                GGML_ASSERT(ids[i] >= 0 && ids[i] < n_vocab);
                GGML_ASSERT(logits[i] == expected[ids[i]]);
                GGML_ASSERT(logits[i] >= sorted[k]);
            }
            if (r == 0) {
                GGML_ASSERT(finite ? std::find(ids, ids + count, low) != ids + count
                                   : std::find(ids, ids + count, best) == ids + count);
                GGML_ASSERT(first[low] < sorted[k]);
                if (finite) {
                    GGML_ASSERT(std::find(ids, ids + count, low_other) != ids + count);
                }
            }
            for (uint32_t seed = 0; seed < 300; ++seed) {
                params.seed = seed;
                params.backend_sampling = false;
                common_sampler_ptr host(common_sampler_init(model, params));
                params.backend_sampling = true;
                common_sampler_ptr candidate(common_sampler_init(model, params));
                common_sampler_row row;
                row.n_vocab = n_vocab;
                row.logits = [&]() { return raw[r].data(); };
                const llama_token want = common_sampler_sample_row(host.get(), row);
                const llama_token got = common_sampler_sample(candidate.get(), ctx.get(), idx);
                GGML_ASSERT(got == want);
                const auto * a = common_sampler_get_candidates(host.get(), false);
                const auto * b = common_sampler_get_candidates(candidate.get(), false);
                GGML_ASSERT(a->size == b->size);
                for (size_t i = 0; i < a->size; ++i) {
                    GGML_ASSERT(a->data[i].id == b->data[i].id);
                    GGML_ASSERT(a->data[i].logit == b->data[i].logit);
                    GGML_ASSERT(a->data[i].p == b->data[i].p);
                }
            }
            GGML_ASSERT(std::memcmp(llama_get_logits_ith(ctx.get(), idx), raw[r].data(), n_vocab*sizeof(float)) == 0);
        }
        printf("leading bias %s: %zu rows x 300 seeds, tokens/logits/probabilities identical\n", finite ? "finite" : "-INF", rows.size());

        // An invalid certificate must fetch raw logits and reapply the original bias exactly once.
        for (int32_t mode = 0; mode < 5; ++mode) {
            const bool grammar_first = mode == 2;
            params.grammar = mode == 2 || mode == 3 ? common_grammar{COMMON_GRAMMAR_TYPE_USER, "root ::= \"a\""} : common_grammar{};
            params.reasoning_budget_start = mode == 4 ? llama_tokens{1} : llama_tokens{};
            params.reasoning_budget_end = mode == 4 ? std::vector<llama_tokens>{{2}} : std::vector<llama_tokens>{};
            params.reasoning_budget_forced = mode == 4 ? llama_tokens{2} : llama_tokens{};
            params.reasoning_budget_tokens = mode == 4 ? 1 : -1;
            params.seed = 17;
            params.backend_sampling = false;
            common_sampler_ptr host(common_sampler_init(model, params));
            params.backend_sampling = true;
            common_sampler_ptr candidate(common_sampler_init(model, params));
            if (mode == 4) {
                common_sampler_accept(host.get(), 1, true);
                common_sampler_accept(candidate.get(), 1, true);
                GGML_ASSERT(common_sampler_reasoning_budget_force(host.get()));
                GGML_ASSERT(common_sampler_reasoning_budget_force(candidate.get()));
            }
            std::vector<float> tied(k + 1, mode == 1 ? -INFINITY : 1.0f);
            std::vector<llama_token> ids(k + 1);
            for (int32_t i = 0; i <= k; ++i) {
                ids[i] = i;
            }
            bool fetched = false;
            common_sampler_row row;
            row.n_vocab = n_vocab;
            row.logits = [&]() { fetched = true; return first.data(); };
            const auto want = common_sampler_sample_row(host.get(), row, grammar_first);
            fetched = false;
            row.cand_logits = mode >= 3 ? llama_get_sampled_logits_ith(ctx.get(), rows.front()) : tied.data();
            row.cand_ids = mode >= 3 ? llama_get_sampled_candidates_ith(ctx.get(), rows.front()) : ids.data();
            row.n_cand = k + 1;
            GGML_ASSERT(common_sampler_sample_row(candidate.get(), row, grammar_first) == want);
            GGML_ASSERT(fetched);
        }
        printf("leading bias %s: tie, nonfinite, grammar-first, grammar-retry, budget fallbacks identical\n", finite ? "finite" : "-INF");
    }

    common_params_sampling duplicate;
    duplicate.backend_sampling = true;
    duplicate.top_k = k;
    duplicate.logit_bias = {{ low, lift }, { low, 0.5f }};
    common_sampler_ptr fallback(common_sampler_init(model, duplicate));
    GGML_ASSERT(common_sampler_get_backend(fallback.get()) == nullptr);
}
