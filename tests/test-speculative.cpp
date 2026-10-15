// Model-free contract tests of common_speculative: the draft prompt is built only for drafters that read it.

#include "common.h"
#include "speculative.h"
#include "../ggml/src/ggml-syncfree.h"

#include <cmath>
#include <random>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(cond)                                                           \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

static llama_tokens periodic_prompt(size_t n, size_t period) {
    llama_tokens res(n);
    for (size_t i = 0; i < n; ++i) {
        res[i] = 100 + (llama_token) (i % period);
    }
    return res;
}

// mimics tools/server: pass the prompt only if the drafters read it
static llama_tokens draft_once(common_speculative * spec, const llama_tokens & prompt, llama_token id_last) {
    llama_tokens result;
    llama_tokens spec_prompt;

    const bool uses_prompt = common_speculative_uses_prompt(spec);
    if (uses_prompt) {
        spec_prompt = prompt;
    }

    common_speculative_begin(spec, 0, prompt);

    common_speculative_get_draft_params(spec, 0) = {
        /* .drafting = */ true,
        /* .n_max    = */ 8,
        /* .pos0     = */ (llama_pos) prompt.size(),
        /* .id_last  = */ id_last,
        /* .prompt   = */ uses_prompt ? &spec_prompt : nullptr,
        /* .result   = */ &result,
    };

    common_speculative_draft(spec);

    return result;
}

static void test_ngram(common_speculative_type type, bool expect_draft) {
    common_params_speculative params;
    params.types = { type };
    params.ngram_simple.size_n = 4;
    params.ngram_simple.size_m = 6;
    params.ngram_map_k.size_n  = 4;
    params.ngram_map_k.size_m  = 6;

    common_speculative * spec = common_speculative_init(params, 1);
    CHECK(spec != nullptr);

    // every n-gram drafter reads the prompt: leaving it null would crash
    CHECK(common_speculative_uses_prompt(spec));

    const size_t period = 20;
    const llama_tokens prompt = periodic_prompt(200, period);
    const llama_token id_last = 100 + (llama_token) (prompt.size() % period);

    const llama_tokens draft = draft_once(spec, prompt, id_last);

    if (expect_draft) {
        CHECK(!draft.empty());
        for (size_t i = 0; i < draft.size(); ++i) {
            CHECK(draft[i] == 100 + (llama_token) ((prompt.size() + 1 + i) % period));
        }
    }

    printf("%s: uses_prompt, draft size %zu: OK\n", common_speculative_type_to_str(type).c_str(), draft.size());

    common_speculative_free(spec);
}

static void test_syncfree_greedy() {
    constexpr int k = 5;
    constexpr int np = 4;
    float packet[2 + 23*k] = {};
    packet[0] = 8192;
    packet[1] = 1;
    int32_t sampled[k + 1];
    for (int i = 0; i < k; ++i) {
        packet[2 + i] = 100 + i;
    }
    for (int prefix = 0; prefix <= k; ++prefix) {
        for (int i = 0; i <= k; ++i) {
            sampled[i] = i < prefix ? 100 + i : 200 + i;
        }
        int32_t out[4 + k + np*k];
        ggml_syncfree_accept(nullptr, nullptr, sampled, packet, out, k, np, 1);
        CHECK(out[0] == prefix);
        CHECK(out[1] == 200 + prefix);
        CHECK(out[2] == 8192 + prefix + 1);
        for (int i = 0; i <= prefix; ++i) {
            CHECK(out[3 + i] == (i < prefix ? 100 + i : 200 + prefix));
        }
        for (int axis = 0; axis < np; ++axis) {
            for (int step = 0; step < k; ++step) {
                CHECK(out[4 + k + axis*k + step] == out[2] + step);
            }
        }
    }
    printf("syncfree: greedy prefixes 0..5 and four position axes: OK\n");
}

static void test_syncfree_rejection() {
    constexpr int draws = 100000;
    std::mt19937 rng(1729);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    const float p[] = {0.2f, 0.3f, 0.5f, 0.2f, 0.3f, 0.5f};
    const int32_t ids[] = {0, 1, 2, 0, 1, 2};
    float packet[25] = {};
    packet[0] = 8192;
    for (int i = 0; i < 10; ++i) {
        packet[3 + i] = -1;
    }
    packet[3] = 0;
    packet[4] = 1;
    packet[13] = 0.75f;
    packet[14] = 0.25f;
    int counts[3] = {};
    int accepted = 0;
    for (int i = 0; i < draws; ++i) {
        packet[2] = uniform(rng) < 0.75f ? 0 : 1;
        packet[23] = uniform(rng);
        packet[24] = uniform(rng);
        int32_t sampled[2];
        for (auto & token : sampled) {
            const float u = uniform(rng);
            token = u < 0.2f ? 0 : (u < 0.5f ? 1 : 2);
        }
        int32_t out[9];
        ggml_syncfree_accept(p, ids, sampled, packet, out, 1, 4, 3);
        CHECK(out[0] >= 0 && out[0] <= 1);
        CHECK(out[3] >= 0 && out[3] < 3);
        ++counts[out[3]];
        accepted += out[0];
    }
    CHECK(std::abs((double) accepted/draws - 0.45) < 0.01);
    for (int i = 0; i < 3; ++i) {
        CHECK(std::abs((double) counts[i]/draws - p[i]) < 0.01);
    }
    printf("syncfree: rejection %d draws, acceptance %.5f, output %.5f %.5f %.5f: OK\n",
        draws, (double) accepted/draws, (double) counts[0]/draws, (double) counts[1]/draws, (double) counts[2]/draws);
}

int main() {
    test_syncfree_greedy();
    test_syncfree_rejection();
    CHECK(!common_speculative_uses_prompt(nullptr));

    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, true);
    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,  true);
    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, false);
    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_MOD,    false);
    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,  false);

    return 0;
}
