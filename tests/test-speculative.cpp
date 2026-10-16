// Model-free contract tests of common_speculative: the draft prompt is built only for drafters that read it.

#include "common.h"
#include "speculative.h"
#include "../ggml/src/ggml-syncfree.h"
#include "../src/llama-ext.h"

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
    constexpr int ns_max = 2;
    float packet[ns_max*(2 + 23*k)] = {};
    for (int seq = 0; seq < ns_max; ++seq) {
        auto * data = packet + seq*(2 + 23*k);
        data[0] = 8192 + seq*4096;
        data[1] = 1;
        for (int i = 0; i < k; ++i) {
            data[2 + i] = 100 + seq*1000 + i;
        }
    }
    for (int ns : {1, 2}) {
        for (int np : {1, 4}) {
            for (int p0 = 0; p0 <= k; ++p0) {
                for (int p1 = 0; p1 <= (ns == 1 ? 0 : k); ++p1) {
                    const int prefixes[] = {p0, p1};
                    int32_t sampled[ns_max*(k + 1)];
                    for (int seq = 0; seq < ns; ++seq) {
                        for (int i = 0; i <= k; ++i) {
                            sampled[seq*(k + 1) + i] = (i < prefixes[seq] ? 100 : 200) + seq*1000 + i;
                        }
                    }
                    std::vector<int32_t> out(ns*(4 + k + np*k));
                    ggml_syncfree_accept_batch(nullptr, nullptr, sampled, packet, out.data(), k, np, 1, ns);
                    for (int seq = 0; seq < ns; ++seq) {
                        const int prefix = prefixes[seq];
                        const auto * row = out.data() + seq*(4 + k + np*k);
                        const int base = 8192 + seq*4096;
                        CHECK(row[0] == prefix);
                        CHECK(row[1] == 200 + seq*1000 + prefix);
                        CHECK(row[2] == base + prefix + 1);
                        for (int i = 0; i <= prefix; ++i) {
                            CHECK(row[3 + i] == (i < prefix ? 100 + seq*1000 + i : row[1]));
                        }
                        for (int axis = 0; axis < np; ++axis) {
                            for (int step = 0; step < k; ++step) {
                                CHECK(row[4 + k + axis*k + step] == row[2] + step);
                            }
                        }
                        for (int pos = base - 1; pos <= base + 2*k + 1; ++pos) {
                            const bool rejected = pos >= base + prefix + 1 && pos <= base + k;
                            CHECK(ggml_syncfree_mask_value(0.0f, pos, row, packet + seq*(2 + 23*k), k) == (rejected ? -INFINITY : 0.0f));
                            CHECK(ggml_syncfree_mask_value(-INFINITY, pos, row, packet + seq*(2 + 23*k), k) == -INFINITY);
                        }
                    }
                }
            }
        }
    }
    printf("syncfree: np1/np2 greedy prefix pairs, 1/4 position axes and per-sequence masks: OK\n");
}

static void test_syncfree_rejection() {
    constexpr int draws = 100000;
    constexpr int ns = 2;
    std::mt19937 rng(1729);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    const float p[] = {0.2f, 0.3f, 0.5f, 0.2f, 0.3f, 0.5f, 0.6f, 0.1f, 0.3f, 0.6f, 0.1f, 0.3f};
    const int32_t ids[] = {0, 1, 2, 0, 1, 2, 10, 11, 12, 10, 11, 12};
    const float q[ns][2] = {{0.75f, 0.25f}, {0.1f, 0.9f}};
    float packet[ns*25] = {};
    for (int seq = 0; seq < ns; ++seq) {
        auto * data = packet + seq*25;
        data[0] = 8192 + seq*4096;
        for (int i = 0; i < 10; ++i) {
            data[3 + i] = -1;
        }
        data[3] = seq*10;
        data[4] = seq*10 + 1;
        data[13] = q[seq][0];
        data[14] = q[seq][1];
    }
    int counts[ns][3] = {};
    int accepted[ns] = {};
    for (int i = 0; i < draws; ++i) {
        int32_t sampled[ns*2];
        for (int seq = 0; seq < ns; ++seq) {
            auto * data = packet + seq*25;
            data[2] = seq*10 + (uniform(rng) < q[seq][0] ? 0 : 1);
            data[23] = uniform(rng);
            data[24] = uniform(rng);
            for (int j = 0; j < 2; ++j) {
                const float u = uniform(rng);
                sampled[seq*2 + j] = seq*10 + (u < p[seq*6] ? 0 : (u < p[seq*6] + p[seq*6 + 1] ? 1 : 2));
            }
        }
        int32_t out[ns*9];
        ggml_syncfree_accept_batch(p, ids, sampled, packet, out, 1, 4, 3, ns);
        for (int seq = 0; seq < ns; ++seq) {
            const auto * row = out + seq*9;
            CHECK(row[0] >= 0 && row[0] <= 1);
            CHECK(row[3] >= seq*10 && row[3] < seq*10 + 3);
            CHECK(row[2] == (int) packet[seq*25] + row[0] + 1);
            ++counts[seq][row[3] - seq*10];
            accepted[seq] += row[0];
        }
    }
    const float expected[] = {0.45f, 0.2f};
    for (int seq = 0; seq < ns; ++seq) {
        CHECK(std::abs((double) accepted[seq]/draws - expected[seq]) < 0.01);
        for (int i = 0; i < 3; ++i) {
            CHECK(std::abs((double) counts[seq][i]/draws - p[seq*6 + i]) < 0.01);
        }
        printf("syncfree: np2 seq=%d rejection %d draws, acceptance %.5f, output %.5f %.5f %.5f: OK\n",
            seq, draws, (double) accepted[seq]/draws, (double) counts[seq][0]/draws, (double) counts[seq][1]/draws, (double) counts[seq][2]/draws);
    }
}

static void test_syncfree_output_limits() {
    for (int n_seq : {1, 2}) {
        for (bool chain : {false, true}) {
            common_params params;
            params.n_parallel = n_seq;
            params.speculative.draft.mtp_chain = chain;
            params.speculative.draft.backend_sampling = true;
            params.speculative.draft.n_max = 5;
            const auto draft = common_base_params_to_speculative(params);
            const int per_seq = chain && (n_seq == 1 || llama_syncfree_enabled()) ? 5 : 1;
            CHECK(draft.n_outputs_max_per_seq == per_seq);
            CHECK(draft.n_outputs_max == n_seq*per_seq);
        }
    }
    printf("syncfree: per-sequence sampler output limits, gate=%d: OK\n", llama_syncfree_enabled());
}

int main() {
    test_syncfree_greedy();
    test_syncfree_rejection();
    test_syncfree_output_limits();
    CHECK(!common_speculative_uses_prompt(nullptr));

    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, true);
    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,  true);
    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, false);
    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_MOD,    false);
    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,  false);

    return 0;
}
