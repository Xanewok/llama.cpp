// Law check for the probabilistic-draft verify (common_sampler_sample_and_accept_n_rejection_rows):
// with d ~ q, the first emitted token must follow the target chain's law p exactly, and the accept rate
// must be sum_x min(p(x), q(x)). The negative control hands the verifier a q that d was not drawn from.
#include "common.h"
#include "sampling.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
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
        r.logits = [this]() { return full.data(); };
        if (candidates) {
            r.cand_ids = ids.data();
            r.cand_logits = values.data();
            r.n_cand = ids.size();
        }
        return r;
    }
};

// the serving chain: --temp 1 --top-k 20 --top-p 0.95 --min-p 0.05
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

// upper tail of chi-square (Wilson-Hilferty)
static double chi2_p(double chi2, double df) {
    const double z = (std::cbrt(chi2/df) - (1.0 - 2.0/(9.0*df)))/std::sqrt(2.0/(9.0*df));
    return 0.5*std::erfc(z/std::sqrt(2.0));
}

int main(int argc, char ** argv) {
    try {
        check(argc == 2 || argc == 4, "usage: rejection-law MODEL [SEED N]");
        const uint32_t seed = argc == 4 ? std::atoi(argv[2]) : 42;
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
        check(model != nullptr, "model load failed");
        const auto * vocab = llama_model_get_vocab(model.get());

        test_row row;
        row.full.assign(llama_vocab_n_tokens(vocab), -20.0f);
        for (int i = 0; i < 32; ++i) {
            row.full[i] = 4.0f - 0.12f*i;
        }
        row.select(21);

        // draft law: overlaps p, skews it, and puts mass on tokens the target chain cuts
        std::vector<llama_token_data> q;
        {
            float sum = 0.0f;
            for (int i = 0; i < 40; ++i) {
                const float w = std::exp(-0.03f*float((i - 8)*(i - 8)));
                q.push_back({ (llama_token) i, 0.0f, w });
                sum += w;
            }
            for (auto & e : q) {
                e.p /= sum;
            }
        }
        std::vector<llama_token_data> q_wrong = q;
        for (auto & e : q_wrong) {
            e.p = 1.0f/q_wrong.size();
        }

        const int n = argc == 4 ? std::atoi(argv[3]) : 60000;
        int failures = 0;
        for (const bool candidates : { false, true }) {
            auto law_params = params(7, candidates);
            common_sampler_ptr law(common_sampler_init(model.get(), law_params));
            common_sampler_sample_row(law.get(), row.row(candidates));
            const auto * dist = common_sampler_get_candidates(law.get(), true);
            std::map<llama_token, double> p;
            for (size_t k = 0; k < dist->size; ++k) {
                if (dist->data[k].p > 0.0f) {
                    p[dist->data[k].id] = dist->data[k].p;
                }
            }
            double accept_theory = 0.0;
            for (const auto & e : q) {
                accept_theory += std::min(p.count(e.id) ? p[e.id] : 0.0, (double) e.p);
            }

            for (const bool broken : { false, true }) {
                auto verifier_params = params(seed, candidates);
                common_sampler_ptr verifier(common_sampler_init(model.get(), verifier_params));
                std::mt19937 draft_rng(seed + 1234);
                std::vector<double> q_w;
                for (const auto & e : q) {
                    q_w.push_back(e.p);
                }
                std::discrete_distribution<int> draw_q(q_w.begin(), q_w.end());
                std::map<llama_token, int> count;
                int accepted = 0;
                for (int t = 0; t < n; ++t) {
                    const llama_token d = q[draw_q(draft_rng)].id;
                    const auto res = common_sampler_sample_and_accept_n_rejection_rows(verifier.get(),
                            [&](size_t) { return row.row(candidates); }, { d }, { broken ? q_wrong : q });
                    check(!res.empty(), "verify returned no token");
                    count[res[0]]++;
                    accepted += res.size() == 2;
                }
                double chi2 = 0.0;
                int outside = 0;
                for (const auto & [id, c] : count) {
                    if (!p.count(id)) {
                        outside += c;
                    }
                }
                for (const auto & [id, pr] : p) {
                    const double expected = n*pr;
                    const double delta = (count.count(id) ? count[id] : 0) - expected;
                    chi2 += delta*delta/expected;
                }
                const double df = p.size() - 1;
                const double pv = outside ? 0.0 : chi2_p(chi2, df);
                const double acc = double(accepted)/n;
                const double acc_z = (acc - accept_theory)/std::sqrt(accept_theory*(1.0 - accept_theory)/n);
                const bool law_ok = pv > 1e-6 && std::fabs(acc_z) < 5.0;
                std::printf("M: %s row, %s q: %d draws, support %zu, outside-support %d, chi2 %.2f df %.0f p %.3g, accept %.4f vs theory %.4f (z %.2f): %s\n",
                        candidates ? "candidate" : "full", broken ? "wrong (negative control)" : "true", n, p.size(), outside, chi2, df, pv,
                        acc, accept_theory, acc_z, law_ok ? "law holds" : "law violated");
                failures += broken ? law_ok : !law_ok;
            }
        }
        check(failures == 0, "rejection law check failed (or a negative control escaped)");
        std::printf("PASS: target law preserved for full and candidate rows; both negative controls rejected\n");
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
