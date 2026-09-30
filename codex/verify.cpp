#include "common.h"
#include "sampling.h"
#include "llama-cpp.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

struct test_row {
    std::vector<float> full;
    std::vector<float> values;
    std::vector<llama_token> ids;
    int reads = 0;

    void select(int k) {
        ids.resize(full.size());
        std::iota(ids.begin(), ids.end(), 0);
        std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int a, int b) { return full[a] > full[b]; });
        ids.resize(k);
        values.clear();
        for (const auto id : ids) {
            values.push_back(full[id]);
        }
    }

    common_sampler_row row(bool candidates) {
        common_sampler_row r;
        r.n_vocab = full.size();
        r.logits = [&]() { ++reads; return full.data(); };
        if (candidates) {
            r.cand_ids = ids.data();
            r.cand_logits = values.data();
            r.n_cand = ids.size();
        }
        return r;
    }
};

static common_params_sampling params(uint32_t seed, bool backend) {
    common_params_sampling p;
    p.seed = seed;
    p.backend_sampling = backend;
    p.top_k = 20;
    p.top_p = 0.95f;
    p.min_p = 0.05f;
    p.temp = 1.0f;
    p.no_perf = true;
    return p;
}

static bool equal_candidates(common_sampler * a, common_sampler * b) {
    auto * ca = common_sampler_get_candidates(a, true);
    auto * cb = common_sampler_get_candidates(b, true);
    if (ca->size != cb->size || ca->selected != cb->selected) {
        return false;
    }
    for (size_t i = 0; i < ca->size; ++i) {
        if (ca->data[i].id != cb->data[i].id || ca->data[i].logit != cb->data[i].logit || ca->data[i].p != cb->data[i].p) {
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    try {
        check(argc == 2 || (argc == 3 && std::strcmp(argv[2], "--decode") == 0), "usage: verify MODEL [--decode]");
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = argc == 2;
        mp.n_gpu_layers = 0;
        llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
        check(model != nullptr, "model load failed");
        const auto * vocab = llama_model_get_vocab(model.get());
        auto token = [&](const std::string & s) {
            auto t = common_tokenize(vocab, s, false, false);
            check(t.size() == 1, "test text must be one token");
            return t[0];
        };
        const llama_token a = token("a"), b = token("b"), c = token("c");

        test_row row;
        row.full.resize(llama_vocab_n_tokens(vocab));
        for (size_t i = 0; i < row.full.size(); ++i) {
            row.full[i] = -20.0f + float(i)*0.000001f;
        }
        for (int i = 0; i < 32; ++i) {
            row.full[i] = 4.0f - 0.025f*i;
        }
        row.select(21);
        if (std::getenv("VERIFY_NEGATIVE_CONTROL")) {
            row.values[0] += 8.0f;
        }

        int negative_differences = 0;
        for (uint32_t seed = 0; seed < 256; ++seed) {
            auto ph = params(seed, false), pc = params(seed, true);
            common_sampler_ptr h(common_sampler_init(model.get(), ph)), fast(common_sampler_init(model.get(), pc));
            row.reads = 0;
            const auto got = common_sampler_sample_row(fast.get(), row.row(true));
            check(row.reads == 0, "candidate first draw read full logits");
            const auto want = common_sampler_sample_row(h.get(), row.row(false));
            check(got == want && equal_candidates(h.get(), fast.get()), "seed or distribution mismatch");
            common_sampler_ptr broken(common_sampler_init(model.get(), pc));
            auto mutated = row;
            mutated.values[0] += 8.0f;
            negative_differences += common_sampler_sample_row(broken.get(), mutated.row(true)) != want;
        }
        check(negative_differences > 0, "negative control did not detect changed logits");
        std::printf("M: 256 seed identities and exact candidate probabilities; negative control differs in %d seeds\n", negative_differences);

        auto law_p = params(42, false), draw_p = params(42, true);
        common_sampler_ptr law(common_sampler_init(model.get(), law_p)), draws(common_sampler_init(model.get(), draw_p));
        common_sampler_sample_row(law.get(), row.row(false));
        const auto * distribution = common_sampler_get_candidates(law.get(), true);
        const std::vector<llama_token_data> reference_law(distribution->data, distribution->data + distribution->size);
        auto chi = [&](bool broken) {
            auto test = row;
            if (broken) {
                test.values[0] += 8.0f;
            }
            std::vector<int> counts(row.full.size());
            const int n = 10000;
            for (int i = 0; i < n; ++i) {
                ++counts[common_sampler_sample_row(draws.get(), test.row(true))];
            }
            double chi2 = 0.0;
            for (const auto & e : reference_law) {
                const double expected = n*e.p;
                const double delta = counts[e.id] - expected;
                chi2 += delta*delta/expected;
                counts[e.id] = 0;
            }
            check(std::accumulate(counts.begin(), counts.end(), 0) == 0, "candidate draw escaped host support");
            const double df = reference_law.size() - 1;
            const double z = (std::cbrt(chi2/df) - (1.0 - 2.0/(9.0*df)))/std::sqrt(2.0/(9.0*df));
            const double p = 0.5*std::erfc(z/std::sqrt(2.0));
            std::printf("M: %s chi-square=%.3f df=%.0f p=%.6g, %d draws (Wilson-Hilferty p approximation)\n", broken ? "negative control" : "candidates", chi2, df, p, n);
            return p;
        };
        check(chi(false) > 1e-6, "candidate distribution failed host-law chi-square");
        check(chi(true) < 1e-6, "negative control passed host-law chi-square");

        // q=p exercises all four rows; an off-support q exercises the residual draw.
        for (bool residual : {false, true}) {
            for (uint32_t seed = 0; seed < 128; ++seed) {
                auto ph = params(seed, false), pc = params(seed, true);
                common_sampler_ptr h(common_sampler_init(model.get(), ph)), fast(common_sampler_init(model.get(), pc));
                common_sampler_ptr proposal(common_sampler_init(model.get(), ph));
                common_sampler_sample_row(proposal.get(), row.row(false));
                auto * dist = common_sampler_get_candidates(proposal.get(), true);
                std::vector<llama_token_data> q(dist->data, dist->data + dist->size);
                llama_tokens draft(3, q[0].id);
                if (residual) {
                    q = {{a, 0.0f, 1.0f}};
                    draft.assign(3, a);
                }
                std::vector<std::vector<llama_token_data>> qs(3, q);
                auto want = common_sampler_sample_and_accept_n_rejection_rows(h.get(), [&](size_t) { return row.row(false); }, draft, qs);
                row.reads = 0;
                auto got = common_sampler_sample_and_accept_n_rejection_rows(fast.get(), [&](size_t) { return row.row(true); }, draft, qs);
                check(got == want && row.reads == 0, "rejection verify identity failed");
                check(got.size() == (residual ? 1 : 4), "verify did not exercise intended branch");
            }
        }
        std::puts("M: 128 all-accepted four-row rounds and 128 residual rounds are seed-identical");

        test_row small;
        small.full.resize(128);
        for (int i = 0; i < 128; ++i) {
            small.full[i] = -20.0f + i*0.001f;
        }
        check(a < 128 && b < 128 && c < 128, "ASCII token ids outside test row");
        small.full[b] = 10.0f;
        small.full[a] = -30.0f;
        small.select(21);
        check(std::find(small.ids.begin(), small.ids.end(), a) == small.ids.end(), "valid grammar token must be outside candidates");
        for (uint32_t seed = 0; seed < 32; ++seed) {
            auto ph = params(seed, false), pc = params(seed, true);
            ph.grammar = pc.grammar = { COMMON_GRAMMAR_TYPE_USER, "root ::= \"a\"*" };
            common_sampler_ptr h(common_sampler_init(model.get(), ph)), fast(common_sampler_init(model.get(), pc));
            const auto want = common_sampler_sample_row(h.get(), small.row(false));
            small.reads = 0;
            const auto got = common_sampler_sample_row(fast.get(), small.row(true));
            check(got == a && got == want && small.reads == 1 && equal_candidates(h.get(), fast.get()), "grammar rejection did not fetch the valid tail");
        }
        std::puts("M: 32 grammar rejection identities, with valid token outside top-K and one full-row read");

        auto valid = small;
        valid.full[a] = 12.0f;
        valid.select(21);
        for (uint32_t seed = 0; seed < 32; ++seed) {
            auto ph = params(seed, false), pc = params(seed, true);
            ph.grammar = pc.grammar = { COMMON_GRAMMAR_TYPE_USER, "root ::= \"a\"*" };
            common_sampler_ptr h(common_sampler_init(model.get(), ph)), fast(common_sampler_init(model.get(), pc));
            llama_tokens draft(3, a);
            std::vector<std::vector<llama_token_data>> qs(3, {{a, 0.0f, 1.0f}});
            auto want = common_sampler_sample_and_accept_n_rejection_rows(h.get(), [&](size_t) { return valid.row(false); }, draft, qs);
            valid.reads = 0;
            auto got = common_sampler_sample_and_accept_n_rejection_rows(fast.get(), [&](size_t) { return valid.row(true); }, draft, qs);
            check(got == want && got == llama_tokens(4, a), "grammar verify mixture differs");
        }
        std::puts("M: 32 grammar-active four-row verify identities");

        for (bool grammar_first : {false, true}) {
            auto ph = params(1, false), pc = params(1, true);
            ph.grammar = pc.grammar = { COMMON_GRAMMAR_TYPE_USER, "root ::= \"c\" \"a\"*" };
            ph.grammar_lazy = pc.grammar_lazy = true;
            ph.grammar_triggers = pc.grammar_triggers = {{ COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN, "", c }};
            common_sampler_ptr h(common_sampler_init(model.get(), ph)), fast(common_sampler_init(model.get(), pc));
            small.reads = 0;
            check(common_sampler_sample_row(h.get(), small.row(false)) == common_sampler_sample_row(fast.get(), small.row(true)), "lazy draw differs");
            common_sampler_accept(h.get(), c, true);
            common_sampler_accept(fast.get(), c, true);
            auto want = common_sampler_sample_row(h.get(), small.row(false), grammar_first);
            small.reads = 0;
            auto got = common_sampler_sample_row(fast.get(), small.row(true), grammar_first);
            check(got == a && got == want && small.reads == 1, "lazy trigger or grammar-first fallback differs");
        }
        std::puts("M: lazy trigger and grammar-first fallback identities");

        auto ph = params(9, false), pc = params(9, true);
        ph.reasoning_budget_tokens = pc.reasoning_budget_tokens = 0;
        ph.reasoning_budget_start = pc.reasoning_budget_start = {c};
        ph.reasoning_budget_end = pc.reasoning_budget_end = {{a}};
        ph.reasoning_budget_forced = pc.reasoning_budget_forced = {a};
        common_sampler_ptr h(common_sampler_init(model.get(), ph)), fast(common_sampler_init(model.get(), pc));
        common_sampler_accept(h.get(), c, true);
        common_sampler_accept(fast.get(), c, true);
        auto want = common_sampler_sample_row(h.get(), small.row(false));
        small.reads = 0;
        auto got = common_sampler_sample_row(fast.get(), small.row(true));
        check(got == a && got == want && small.reads == 1 && equal_candidates(h.get(), fast.get()), "FORCING missed token outside candidates");
        std::puts("M: reasoning budget FORCING identity and full-row fallback");

        small.full[b] = small.full[c] = 10.0f;
        small.select(21);
        auto pt = params(1, true), pr = params(1, false);
        common_sampler_ptr tie(common_sampler_init(model.get(), pt)), tie_ref(common_sampler_init(model.get(), pr));
        want = common_sampler_sample_row(tie_ref.get(), small.row(false));
        small.reads = 0;
        got = common_sampler_sample_row(tie.get(), small.row(true));
        check(got == want && small.reads == 1 && equal_candidates(tie_ref.get(), tie.get()), "top-K tie did not use full-row ordering");
        std::puts("M: top-K tie fallback identity");

        for (bool bias : {false, true}) {
            auto p = params(1, true);
            if (bias) {
                p.logit_bias = {{b, 1.0f}};
            } else {
                p.penalty_repeat = 1.1f;
            }
            p.grammar = {COMMON_GRAMMAR_TYPE_USER, "root ::= \"a\"*"};
            common_sampler_ptr ineligible(common_sampler_init(model.get(), p));
            check(common_sampler_get_backend(ineligible.get()) == nullptr, "logit-changing prefix offloaded under grammar");
        }

        auto clone_p = params(1, true);
        common_sampler_ptr original(common_sampler_init(model.get(), clone_p));
        common_sampler_sample_row(original.get(), row.row(true));
        common_sampler_ptr cloned(common_sampler_clone(original.get()));
        common_sampler_ptr copied(common_sampler_init(model.get(), clone_p));
        common_sampler_copy(original.get(), copied.get());
        want = common_sampler_sample_row(original.get(), row.row(true));
        check(common_sampler_sample_row(cloned.get(), row.row(true)) == want && common_sampler_sample_row(copied.get(), row.row(true)) == want, "candidate clone or copy changed RNG");
        check(common_sampler_get_backend(original.get()) != common_sampler_get_backend(cloned.get()), "cloned backend candidate chain shares ownership");
        std::puts("M: ineligible prefixes fall back; clone and copy preserve host RNG");

        for (bool backend : {false, true}) {
            auto p = params(42, backend);
            common_sampler_ptr bench(common_sampler_init(model.get(), p));
            common_sampler_sample_row(bench.get(), row.row(backend));
            const int rounds = 2000;
            const int64_t start = ggml_time_us();
            for (int i = 0; i < rounds; ++i) {
                for (int r = 0; r < 4; ++r) {
                    common_sampler_sample_row(bench.get(), row.row(backend));
                }
            }
            std::printf("M: %s host four-row chain %.3f us/round, %d rounds, vocab=%zu (decode and candidate compute excluded)\n",
                    backend ? "candidates" : "full", double(ggml_time_us() - start)/rounds, rounds, row.full.size());
        }

        auto proposal_p = params(42, false);
        common_sampler_ptr proposal(common_sampler_init(model.get(), proposal_p));
        common_sampler_sample_row(proposal.get(), row.row(false));
        const auto * d = common_sampler_get_candidates(proposal.get(), true);
        const std::vector<llama_token_data> q(d->data, d->data + d->size);
        const std::vector<std::vector<llama_token_data>> qs(3, q);
        const llama_tokens draft(3, q[0].id);
        for (bool backend : {false, true}) {
            auto p = params(42, backend);
            common_sampler_ptr bench(common_sampler_init(model.get(), p));
            const int rounds = 2000;
            const int64_t start = ggml_time_us();
            for (int i = 0; i < rounds; ++i) {
                const auto got = common_sampler_sample_and_accept_n_rejection_rows(bench.get(), [&](size_t) { return row.row(backend); }, draft, qs);
                check(got.size() == 4, "timing round rejected q=p");
            }
            std::printf("M: %s all-accepted rejection verify %.3f us/round, %d rounds (decode and candidate compute excluded)\n",
                    backend ? "candidates" : "full", double(ggml_time_us() - start)/rounds, rounds);
        }
        if (argc == 3) {
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = 128;
            cp.n_batch = 16;
            cp.n_ubatch = 16;
            cp.n_threads = cp.n_threads_batch = 8;
            cp.n_outputs_max_per_seq = 4;
            std::vector<std::vector<float>> reference(4);
            auto decode = [&](llama_context * ctx) {
                llama_batch batch = llama_batch_init(4, 0, 1);
                for (int i = 0; i < 4; ++i) {
                    common_batch_add(batch, i % 2 ? a : b, i, {0}, true);
                }
                const int rc = llama_decode(ctx, batch);
                llama_batch_free(batch);
                check(rc == 0, "model four-row decode failed");
            };
            {
                llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
                check(ctx != nullptr, "host context failed");
                decode(ctx.get());
                for (int i = 0; i < 4; ++i) {
                    const auto * logits = llama_get_logits_ith(ctx.get(), i);
                    reference[i].assign(logits, logits + row.full.size());
                }
            }
            auto p = params(1, true);
            common_sampler_ptr installed(common_sampler_init(model.get(), p));
            llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
            check(ctx != nullptr && llama_set_sampler(ctx.get(), 0, common_sampler_get_backend(installed.get())), "candidate context failed");
            decode(ctx.get());
            for (int i = 0; i < 4; ++i) {
                for (uint32_t seed = 0; seed < 64; ++seed) {
                    auto host_p = params(seed, false), candidate_p = params(seed, true);
                    common_sampler_ptr host(common_sampler_init(model.get(), host_p)), candidate(common_sampler_init(model.get(), candidate_p));
                    common_sampler_row full;
                    full.n_vocab = row.full.size();
                    full.logits = [&]() { return reference[i].data(); };
                    const auto want = common_sampler_sample_row(host.get(), full);
                    const auto got = common_sampler_sample(candidate.get(), ctx.get(), i);
                    check(got == want && equal_candidates(host.get(), candidate.get()), "real model seed or probability mismatch");
                }
                check(std::memcmp(llama_get_logits_ith(ctx.get(), i), reference[i].data(), row.full.size()*sizeof(float)) == 0, "real model full logits differ");
            }
            std::puts("M: Qwen3.5-4B CPU decode, four rows x 64 seeds and full logits bit-identical");
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
