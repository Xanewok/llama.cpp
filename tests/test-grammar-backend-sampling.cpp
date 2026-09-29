// Backend (GPU) sampling with a lazy tool-call grammar and the reasoning budget must reproduce the host
// path's output distribution. Rows are simulated: a "backend" row carries a draw from the plain chain plus
// its candidates, as llama_context delivers them; a "host" row carries only the full logits.
//
// usage: test-grammar-backend-sampling <model.gguf with a Qwen3-Coder style chat template> [n_draws]
// Only the vocab and the template are loaded.

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "reasoning-budget.h"
#include "sampling.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

struct setup {
    llama_model *          model = nullptr;
    const llama_vocab *    vocab = nullptr;
    int32_t                n_vocab = 0;
    common_params_sampling sparams;
};

// mirrors server-common.cpp (oaicompat params) and server-schema.cpp (trigger and budget fields)
static common_params_sampling codex_like_params(llama_model * model) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    auto tmpls = common_chat_templates_init(model, "");

    common_chat_templates_inputs inputs;
    inputs.messages = { common_chat_msg{} };
    inputs.messages[0].role    = "user";
    inputs.messages[0].content = "List the files in the repository and fix the failing test.";
    inputs.tools = {
        { "exec_command", "Runs a shell command and returns its output.",
          R"({"type":"object","properties":{"cmd":{"type":"string"},"workdir":{"type":"string"},"yield_time_ms":{"type":"number"}},"required":["cmd"]})" },
        { "apply_patch", "Applies a patch to files in the workspace.",
          R"({"type":"object","properties":{"input":{"type":"string"}},"required":["input"]})" },
    };
    inputs.tool_choice      = COMMON_CHAT_TOOL_CHOICE_AUTO;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    inputs.enable_thinking  = true;

    const auto cp = common_chat_templates_apply(tmpls.get(), inputs);

    common_params_sampling p;
    p.top_k = 20;
    p.top_p = 0.95f;
    p.min_p = 0.05f;
    p.temp  = 1.0f;
    p.seed  = 1234;
    p.backend_sampling = true;

    p.grammar      = { COMMON_GRAMMAR_TYPE_TOOL_CALLS, cp.grammar };
    p.grammar_lazy = cp.grammar_lazy;
    p.generation_prompt = cp.generation_prompt;
    for (const auto & t : cp.preserved_tokens) {
        const auto ids = common_tokenize(vocab, t, false, true);
        if (ids.size() == 1) {
            p.preserved_tokens.insert(ids[0]);
        }
    }
    for (const auto & t : cp.grammar_triggers) {
        if (t.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
            const auto ids = common_tokenize(vocab, t.value, false, true);
            if (ids.size() == 1) {
                common_grammar_trigger tt;
                tt.type  = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                tt.value = t.value;
                tt.token = ids[0];
                p.grammar_triggers.push_back(tt);
                continue;
            }
        }
        p.grammar_triggers.push_back(t);
    }

    if (!cp.thinking_end_tags.empty()) {
        p.reasoning_budget_tokens = -1;
        p.reasoning_budget_start  = common_tokenize(vocab, cp.thinking_start_tag, false, true);
        for (const auto & tag : cp.thinking_end_tags) {
            p.reasoning_budget_end.push_back(common_tokenize(vocab, tag, false, true));
        }
        p.reasoning_budget_forced = p.reasoning_budget_end.front();
    }

    printf("chat format %d, grammar %zu chars, lazy %d, triggers %zu, thinking start '%s', end tags %zu, generation_prompt '%s'\n",
            (int) cp.format, cp.grammar.size(), (int) cp.grammar_lazy, p.grammar_triggers.size(),
            cp.thinking_start_tag.c_str(), cp.thinking_end_tags.size(), cp.generation_prompt.c_str());
    for (const auto & t : p.grammar_triggers) {
        printf("  trigger type %d value '%s' token %d\n", (int) t.type, t.value.c_str(), t.token);
    }

    return p;
}

// kept set and probabilities of the chain without its final dist, on logits already masked by the caller
static std::vector<llama_token_data> chain_dist(llama_sampler * chain, std::vector<llama_token_data> cur) {
    llama_token_data_array arr = { cur.data(), cur.size(), -1, false };

    const int32_t n = llama_sampler_chain_n(chain);
    for (int32_t i = 0; i + 1 < n; ++i) {
        llama_sampler_apply(llama_sampler_chain_get(chain, i), &arr);
    }

    std::vector<llama_token_data> res(arr.data, arr.data + arr.size);

    float max_l = -INFINITY;
    for (const auto & e : res) {
        max_l = std::max(max_l, e.logit);
    }
    double sum = 0.0;
    for (auto & e : res) {
        e.p = e.logit == -INFINITY ? 0.0f : expf(e.logit - max_l);
        sum += e.p;
    }
    for (auto & e : res) {
        e.p /= sum;
    }

    return res;
}

static std::vector<llama_token_data> full_array(const std::vector<float> & logits) {
    std::vector<llama_token_data> cur(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        cur[i] = { (llama_token) i, logits[i], 0.0f };
    }
    return cur;
}

// what the backend delivers for one row: a draw from the plain chain and the chain's candidates
struct backend_row_data {
    llama_token              sampled;
    std::vector<llama_token> ids;
    std::vector<float>       logits;
    std::vector<float>       probs;
};

static backend_row_data backend_row_sim(llama_sampler * chain, const std::vector<llama_token_data> & c, std::mt19937 & rng) {
    backend_row_data r;
    for (const auto & e : c) {
        r.ids.push_back(e.id);
        r.logits.push_back(e.logit);
        r.probs.push_back(e.p);
    }
    GGML_UNUSED(chain);

    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double run = 0.0;
    r.sampled = c.back().id;
    for (const auto & e : c) {
        run += e.p;
        if (run >= u) {
            r.sampled = e.id;
            break;
        }
    }
    return r;
}

static common_sampler_row make_row(const backend_row_data * b, const std::vector<float> & logits, int * n_full_reads) {
    common_sampler_row row;
    if (b) {
        row.sampled     = b->sampled;
        row.cand_ids    = b->ids.data();
        row.cand_logits = b->logits.data();
        row.cand_probs  = b->probs.data();
        row.n_cand      = b->ids.size();
    }
    row.logits = [&logits, n_full_reads]() -> const float * {
        if (n_full_reads) {
            (*n_full_reads)++;
        }
        return logits.data();
    };
    return row;
}

static bool grammar_ok(const llama_sampler * grmr_src, llama_token id) {
    llama_sampler * g = llama_sampler_clone(grmr_src);
    llama_token_data d = { id, 1.0f, 0.0f };
    llama_token_data_array a = { &d, 1, -1, false };
    llama_sampler_apply(g, &a);
    llama_sampler_free(g);
    return d.logit != -INFINITY;
}

// chi-square goodness of fit, Wilson-Hilferty; returns the upper-tail p-value
static double chi2_p(const std::map<llama_token, int> & obs, const std::map<llama_token, double> & ref, int n) {
    double chi2 = 0.0;
    int    df   = -1;
    double p_other = 0.0;
    int    o_other = 0;
    for (const auto & [id, p] : ref) {
        const auto it = obs.find(id);
        const int  o  = it == obs.end() ? 0 : it->second;
        if (p*n < 5.0) {
            p_other += p;
            o_other += o;
            continue;
        }
        chi2 += (o - p*n)*(o - p*n)/(p*n);
        df++;
    }
    for (const auto & [id, o] : obs) {
        if (ref.find(id) == ref.end()) {
            o_other += o;
            if (p_other == 0.0) {
                return 0.0; // a token with zero reference probability was produced
            }
        }
    }
    if (p_other*n >= 5.0) {
        chi2 += (o_other - p_other*n)*(o_other - p_other*n)/(p_other*n);
        df++;
    }
    if (df <= 0) {
        return 1.0;
    }
    const double k = df;
    const double z = (std::cbrt(chi2/k) - (1.0 - 2.0/(9.0*k)))/std::sqrt(2.0/(9.0*k));
    return 0.5*std::erfc(z/std::sqrt(2.0));
}

// P(x) = c(x) 1[x in G] + (1 - c(G)) c'(x), the host path's law (see common_sampler_sample_row)
static std::map<llama_token, double> host_law(common_sampler * gs_state, const std::vector<float> & logits, bool grammar_on, const llama_sampler * grmr) {
    llama_sampler * chain = common_sampler_get(gs_state);

    const auto c = chain_dist(chain, full_array(logits));

    std::map<llama_token, double> P;
    if (!grammar_on) {
        for (const auto & e : c) {
            if (e.p > 0) {
                P[e.id] += e.p;
            }
        }
        return P;
    }

    double cG = 0.0;
    for (const auto & e : c) {
        if (e.p > 0 && grammar_ok(grmr, e.id)) {
            P[e.id] += e.p;
            cG += e.p;
        }
    }

    auto masked = full_array(logits);
    {
        llama_sampler * g = llama_sampler_clone(grmr);
        llama_token_data_array a = { masked.data(), masked.size(), -1, false };
        llama_sampler_apply(g, &a);
        llama_sampler_free(g);
    }
    const auto c2 = chain_dist(chain, masked);
    for (const auto & e : c2) {
        if (e.p > 0) {
            P[e.id] += (1.0 - cG)*e.p;
        }
    }
    return P;
}

static void accept_text(common_sampler * gs, const llama_vocab * vocab, const std::string & text) {
    for (auto t : common_tokenize(vocab, text, false, true)) {
        common_sampler_accept(gs, t, true);
    }
}

// logits with the given tokens lifted to the top, the rest Gumbel-like noise
static std::vector<float> make_logits(int32_t n_vocab, std::mt19937 & rng, const std::vector<std::pair<llama_token, float>> & lifts) {
    std::vector<float> l(n_vocab);
    std::normal_distribution<float> nd(0.0f, 1.5f);
    for (auto & v : l) {
        v = nd(rng);
    }
    for (const auto & [id, v] : lifts) {
        l[id] = v;
    }
    return l;
}

struct state_case {
    const char *  name;
    std::string   text;       // accepted after the generation prompt
    bool          grammar_on; // grammar masks in this state
};

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_draws]\n", argv[0]);
        return 1;
    }
    const int n_draws = argc > 2 ? atoi(argv[2]) : 4000;

    llama_backend_init();
    common_log_set_verbosity_thold(-1); // one INF line per full-row resample otherwise

    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    printf("n_vocab %d\n", n_vocab);

    common_params_sampling sparams = codex_like_params(model);
    CHECK(sparams.grammar_lazy, "expected a lazy tool-call grammar");

    auto tok1 = [&](const char * s) {
        const auto ids = common_tokenize(vocab, s, false, true);
        return ids.empty() ? LLAMA_TOKEN_NULL : ids[0];
    };

    // tokens the model plausibly puts near the top in each state; valid and invalid ones mixed
    const llama_token t_tool   = tok1("<tool_call>");
    const llama_token t_think_end = tok1("</think>");
    const llama_token t_eos    = llama_vocab_eot(vocab) != LLAMA_TOKEN_NULL ? llama_vocab_eot(vocab) : llama_vocab_eos(vocab);
    const llama_token t_exec   = tok1("exec");
    const llama_token t_apply  = tok1("apply");
    const llama_token t_ls     = tok1("ls");
    const llama_token t_hello  = tok1("Hello");
    const llama_token t_lt     = tok1("<");
    const llama_token t_nl     = tok1("\n");

    const std::vector<state_case> cases = {
        { "thinking (budget counting, grammar gated)", "I should look at the repo first.", false },
        { "prose after </think> (lazy grammar awaiting trigger)", "I will inspect the tree.\n</think>\n\nLet me", false },
        { "tool call, function name (grammar active, structural)", "Checking.\n</think>\n\n<tool_call>\n<function=", true },
        { "tool call, parameter value (grammar active, free text)", "Checking.\n</think>\n\n<tool_call>\n<function=exec_command>\n<parameter=cmd>\nls -la", true },
    };

    std::mt19937 rng_logits(42);
    std::mt19937 rng_backend(4242);

    for (const auto & sc : cases) {
        common_params_sampling p = sparams;
        common_sampler * base = common_sampler_init(model, p);
        CHECK(p.backend_sampling, "backend sampling was disabled for the grammar request");
        accept_text(base, vocab, sc.text);

        // common_sampler keeps its grammar private: an equivalent one in the same state, for the reference law
        llama_sampler * g_ref = nullptr;
        {
            std::vector<const char *> pats;
            std::vector<std::string>  pat_s;
            std::vector<llama_token>  toks;
            for (const auto & t : p.grammar_triggers) {
                if (t.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
                    toks.push_back(t.token);
                } else if (t.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    pat_s.push_back(regex_escape(t.value));
                } else {
                    pat_s.push_back(t.value);
                }
            }
            for (const auto & s : pat_s) {
                pats.push_back(s.c_str());
            }
            g_ref = llama_sampler_init_grammar_lazy_patterns(vocab, common_grammar_value(p.grammar).c_str(), "root",
                    pats.data(), pats.size(), toks.data(), toks.size());
            // mirror common_sampler_accept: the grammar sees only tokens after the thinking block, plus the
            // replayed end tag
            const std::string after = sc.text.find("</think>") != std::string::npos ? sc.text.substr(sc.text.find("</think>")) : "";
            for (auto t : common_tokenize(vocab, after, false, true)) {
                llama_sampler_accept(g_ref, t);
            }
        }

        std::vector<std::pair<llama_token, float>> lifts = {
            // above the noise maximum (~7.5 over the vocab), so c mixes valid and invalid tokens
            { t_tool, 9.0f }, { t_think_end, 8.5f }, { t_eos, 9.2f }, { t_exec, 9.5f }, { t_apply, 8.8f },
            { t_ls, 9.2f }, { t_hello, 9.0f }, { t_lt, 8.6f }, { t_nl, 8.9f },
        };
        const auto logits = make_logits(n_vocab, rng_logits, lifts);

        const auto P = host_law(base, logits, sc.grammar_on, g_ref);
        const auto c = chain_dist(common_sampler_get(base), full_array(logits));

        double cG = 0.0;
        for (const auto & e : c) {
            if (!sc.grammar_on || grammar_ok(g_ref, e.id)) {
                cG += e.p;
            }
        }

        // host path, backend path
        std::map<llama_token, int> obs_host, obs_back;
        int n_full_reads = 0;
        int n_same_as_backend = 0;

        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < n_draws; ++k) {
            obs_host[common_sampler_sample_row(base, make_row(nullptr, logits, nullptr))]++;
        }
        const auto t1 = std::chrono::steady_clock::now();
        for (int k = 0; k < n_draws; ++k) {
            const auto b = backend_row_sim(common_sampler_get(base), c, rng_backend);
            const llama_token id = common_sampler_sample_row(base, make_row(&b, logits, &n_full_reads));
            obs_back[id]++;
            n_same_as_backend += id == b.sampled;
        }
        const auto t2 = std::chrono::steady_clock::now();

        const double p_host = chi2_p(obs_host, P, n_draws);
        const double p_back = chi2_p(obs_back, P, n_draws);

        const double ms_host = std::chrono::duration<double, std::milli>(t1 - t0).count()/n_draws;
        const double ms_back = std::chrono::duration<double, std::milli>(t2 - t1).count()/n_draws;

        printf("\n[%s]\n  c(G) = %.4f, |supp P| = %zu, host chi2 p = %.3g, backend chi2 p = %.3g\n",
                sc.name, cG, P.size(), p_host, p_back);
        printf("  backend rows: %d/%d full-row reads (expected ~%.0f), %d kept the backend draw\n",
                n_full_reads, n_draws, (1.0 - cG)*n_draws, n_same_as_backend);
        printf("  host ms/row %.3f, backend ms/row %.4f\n", ms_host, ms_back);

        CHECK(p_host > 1e-4, "host path deviates from its own law in '%s'", sc.name);
        CHECK(p_back > 1e-4, "backend path deviates from the host law in '%s'", sc.name);
        if (!sc.grammar_on) {
            CHECK(n_full_reads == 0 && n_same_as_backend == n_draws, "a no-op grammar state touched the full row in '%s'", sc.name);
        }

        // speculative verify, one draft token: the first emitted token must follow the same law
        if (sc.grammar_on) {
            // a draft proposal that puts mass on valid and invalid tokens alike
            std::vector<llama_token_data> q;
            {
                const size_t m = std::min<size_t>(c.size(), 4);
                for (size_t i = 0; i < m; ++i) {
                    q.push_back({ c[i].id, c[i].logit, 1.0f/m });
                }
            }
            std::discrete_distribution<int> qd(q.size(), 0.0, 1.0, [&](double) { return 1.0; });

            std::map<llama_token, int> obs_vh, obs_vb;
            for (int pass = 0; pass < 2; ++pass) {
                for (int k = 0; k < n_draws; ++k) {
                    // a clone would repeat base's rng streams every trial; a fresh seed gives independent trials
                    common_params_sampling pt = sparams;
                    pt.seed = 1000 + 2*k + pass;
                    common_sampler * gs = common_sampler_init(model, pt);
                    accept_text(gs, vocab, sc.text);
                    const llama_token d = q[qd(rng_backend)].id;
                    const auto b0 = backend_row_sim(nullptr, c, rng_backend);
                    const auto res = common_sampler_sample_and_accept_n_rejection_rows(gs,
                        [&](size_t i) -> common_sampler_row {
                            if (pass == 0) {
                                return make_row(nullptr, logits, nullptr);
                            }
                            GGML_UNUSED(i); // the bonus row reuses row 0; only res[0] is scored
                            return make_row(&b0, logits, nullptr);
                        }, { d }, { q });
                    (pass == 0 ? obs_vh : obs_vb)[res[0]]++;
                    common_sampler_free(gs);
                }
            }
            const double pv_h = chi2_p(obs_vh, P, n_draws);
            const double pv_b = chi2_p(obs_vb, P, n_draws);
            printf("  verify (1 draft): host chi2 p = %.3g, backend chi2 p = %.3g\n", pv_h, pv_b);
            CHECK(pv_h > 1e-4, "host verify deviates in '%s'", sc.name);
            CHECK(pv_b > 1e-4, "backend verify deviates in '%s'", sc.name);
        }

        llama_sampler_free(g_ref);
        common_sampler_free(base);
    }

    // reasoning budget forcing: the forced token, whatever the backend drew
    {
        common_params_sampling p = sparams;
        common_sampler * gs = common_sampler_init(model, p);
        accept_text(gs, vocab, "thinking");
        CHECK(common_sampler_reasoning_budget_force(gs), "could not force the reasoning budget");

        const auto logits = make_logits(n_vocab, rng_logits, { { t_hello, 5.0f } });
        const auto c = chain_dist(common_sampler_get(gs), full_array(logits));
        const auto b = backend_row_sim(nullptr, c, rng_backend);
        int n_full = 0;
        const llama_token id_b = common_sampler_sample_row(gs, make_row(&b, logits, &n_full));
        const llama_token id_h = common_sampler_sample_row(gs, make_row(nullptr, logits, nullptr));
        printf("\n[reasoning budget forcing] backend drew %d, backend path %d, host path %d, forced %d, full reads %d\n",
                b.sampled, id_b, id_h, p.reasoning_budget_forced.empty() ? -1 : p.reasoning_budget_forced[0], n_full);
        CHECK(id_b == id_h && !p.reasoning_budget_forced.empty() && id_b == p.reasoning_budget_forced[0], "forcing differs");
        common_sampler_free(gs);
    }

    // host-path cost pieces on the full vocab
    {
        common_params_sampling p = sparams;
        common_sampler * gs = common_sampler_init(model, p);
        accept_text(gs, vocab, "Checking.\n</think>\n\n<tool_call>\n<function=exec_command>\n<parameter=cmd>\nls");
        const auto logits = make_logits(n_vocab, rng_logits, {});

        const int n = 200;
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < n; ++k) {
            common_sampler_sample_row(gs, make_row(nullptr, logits, nullptr));
        }
        auto t1 = std::chrono::steady_clock::now();
        for (int k = 0; k < n; ++k) {
            common_sampler * c2 = common_sampler_clone(gs);
            common_sampler_free(c2);
        }
        auto t2 = std::chrono::steady_clock::now();

        // full-vocab grammar mask, the host resample's dominant cost
        llama_sampler * g = nullptr;
        {
            std::vector<llama_token> toks;
            for (const auto & t : p.grammar_triggers) {
                if (t.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
                    toks.push_back(t.token);
                }
            }
            g = llama_sampler_init_grammar_lazy_patterns(vocab, common_grammar_value(p.grammar).c_str(), "root", nullptr, 0, toks.data(), toks.size());
            for (auto t : common_tokenize(vocab, "<tool_call>\n<function=exec_command>\n<parameter=cmd>\nls", false, true)) {
                llama_sampler_accept(g, t);
            }
        }
        const int n_mask = 10;
        auto t3 = std::chrono::steady_clock::now();
        for (int k = 0; k < n_mask; ++k) {
            auto arr = full_array(logits);
            llama_token_data_array a = { arr.data(), arr.size(), -1, false };
            llama_sampler_apply(g, &a);
        }
        auto t4 = std::chrono::steady_clock::now();

        printf("\n[host cost, n_vocab %d] sample_row (set_full + chain + 1-token grammar check) %.3f ms, "
               "common_sampler_clone %.3f ms, full-vocab grammar mask %.2f ms\n", n_vocab,
               std::chrono::duration<double, std::milli>(t1 - t0).count()/n,
               std::chrono::duration<double, std::milli>(t2 - t1).count()/n,
               std::chrono::duration<double, std::milli>(t4 - t3).count()/n_mask);

        llama_sampler_free(g);
        common_sampler_free(gs);
    }

    llama_model_free(model);

    printf("\n%s (%d failures)\n", n_fail ? "FAILED" : "OK", n_fail);
    return n_fail ? 1 : 0;
}
