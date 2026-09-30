#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

struct probe {
    int picks = 0;
    bool corrupt_pick = false;
    bool corrupt_embedding = false;
};

static bool observe(ggml_tensor * t, bool ask, void * data) {
    auto & p = *(probe *) data;
    const bool pick = std::strncmp(t->name, "mtp_chain_tok_", 14) == 0;
    const bool embedding = t->op == GGML_OP_GET_ROWS && t->src[0] &&
            std::strcmp(t->src[0]->name, "mtp_chain_tok_embd") == 0;
    if (ask) {
        return pick || (embedding && p.corrupt_embedding);
    }
    if (pick) {
        ++p.picks;
        if (p.corrupt_pick) {
            float wrong = -1;
            ggml_backend_tensor_set(t, &wrong, 0, sizeof(wrong));
        }
    }
    if (embedding && p.corrupt_embedding) {
        std::vector<float> row(ggml_nelements(t), 0.0f);
        ggml_backend_tensor_set(t, row.data(), 0, ggml_nbytes(t));
    }
    return true;
}

struct draft_result {
    llama_tokens tokens;
    std::vector<std::vector<llama_token_data>> q;
    double ms = 0;
    bool used_chain = false;
};

static draft_result draft(common_speculative * spec, llama_context * ctx, const llama_tokens & prompt,
                          uint32_t seed, float temp, int round) {
    llama_synchronize(ctx);
    const int before = llama_perf_context(ctx).n_p_eval;
    draft_result r;
    auto & dp = common_speculative_get_draft_params(spec, 0);
    dp.drafting = true;
    dp.n_max = round == 2 ? 2 : -1;
    dp.pos0 = prompt.size();
    dp.id_last = 100;
    dp.prompt = &prompt;
    dp.result = &r.tokens;
    dp.result_q = &r.q;
    dp.seed = seed;
    dp.temp = temp;
    auto start = std::chrono::steady_clock::now();
    common_speculative_draft(spec);
    r.used_chain = llama_perf_context(ctx).n_p_eval > before;
    r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return r;
}

static bool equal(const draft_result & a, const draft_result & b, bool verbose = true) {
    if (a.tokens != b.tokens || a.q.size() != b.q.size()) {
        return false;
    }
    for (size_t r = 0; r < a.q.size(); ++r) {
        if (a.q[r].size() != b.q[r].size()) {
            return false;
        }
        for (size_t i = 0; i < a.q[r].size(); ++i) {
            const auto & x = a.q[r][i];
            const auto & y = b.q[r][i];
            if (x.id != y.id || x.logit != y.logit || x.p != y.p) {
                if (verbose) std::fprintf(stderr, "q mismatch row=%zu i=%zu ids=%d/%d logits=%.9g/%.9g p=%.9g/%.9g\n",
                             r, i, x.id, y.id, x.logit, y.logit, x.p, y.p);
                return false;
            }
        }
    }
    return true;
}

static llama_tokens verify(llama_model * model, const draft_result & d, uint32_t seed) {
    common_params_sampling sp;
    sp.seed = seed;
    sp.temp = 1;
    sp.samplers = {COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE};
    sp.top_k = 10;
    common_sampler_ptr s {common_sampler_init(model, sp)};
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> row(nv, -INFINITY);
    for (size_t j = 0; j < model->draft_vocab_ids.size(); ++j) {
        row[model->draft_vocab_ids[j]] = std::sin(float(j)*0.13f);
    }
    return common_sampler_sample_and_accept_n_rejection_rows(s.get(), [&](size_t) {
        common_sampler_row r;
        r.n_vocab = nv;
        r.logits = [&]() { return row.data(); };
        return r;
    }, d.tokens, d.q, true);
}

static int run(int argc, char ** argv) {
    require(argc >= 2, "usage: test-draft-chain MODEL [single|tp] [seeds] [normal|wrong-pick|wrong-embedding]");
    require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    const bool tp = argc > 2 && std::string(argv[2]) == "tp";
    const int seeds = argc > 3 ? std::atoi(argv[3]) : 40;
    const std::string control = argc > 4 ? argv[4] : "normal";
    llama_log_set([](ggml_log_level level, const char * message, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) { std::fputs(message, stderr); }
    }, nullptr);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mtp = true;
    mp.progress_callback = [](float, void *) { return true; };
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devices[] = {cpu, cpu, nullptr};
    if (tp) {
        mp.devices = devices;
        mp.n_gpu_layers = -1;
        mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    }
    std::vector<llama_token> ids;
    // Unsorted and well above the BF16 exact-integer range.
    for (int i = 248299; i > 0; i -= 10) { ids.push_back(i); }
    mp.draft_vocab_ids = ids.data();
    mp.n_draft_vocab = ids.size();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    require(model, "model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = cp.n_ubatch = 32;
    cp.n_threads = cp.n_threads_batch = tp ? 4 : 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    llama_context * target = llama_init_from_model(model, cp);
    require(target, "target init failed");
    llama_set_embeddings_nextn(target, true, false);
    llama_tokens prompt = {100, 101, 102, 103};
    common_batch prefill = common_batch_get_one(target, prompt);
    require(llama_process(target, LLAMA_PROCESS_TYPE_DECODE, prefill.get()) == 0, "prefill failed");
    probe stats;
    stats.corrupt_pick = control == "wrong-pick";
    stats.corrupt_embedding = control == "wrong-embedding";
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    llama_context * plain = llama_init_from_model(model, cp);
    cp.cb_eval = control == "normal" ? nullptr : observe;
    cp.cb_eval_user_data = &stats;
    llama_context * chain = llama_init_from_model(model, cp);
    require(plain && chain, "draft init failed");
    int compared = 0;
    int chained = 0;
    for (int k : {3, 4, 5}) {
        for (float p_min : {0.0f, 0.4f, 1.0f}) {
            common_params_speculative params;
            params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
            params.draft.ctx_tgt = target;
            params.draft.ctx_dft = plain;
            params.draft.n_max = k;
            params.draft.p_min = p_min;
            params.draft.probabilistic = true;
            common_speculative_ptr a {common_speculative_init(params, 1)};
            params.draft.ctx_dft = chain;
            params.draft.mtp_chain = true;
            common_speculative_ptr b {common_speculative_init(params, 1)};
            double a_ms = 0, b_ms = 0;
            int n = 0;
            for (int seed = 1; seed <= seeds; ++seed) {
                const float temp = seed % 3 == 0 ? 0.4f : seed % 3 == 1 ? 1.0f : 1.7f;
                llama_memory_clear(llama_get_memory(plain), true);
                llama_memory_clear(llama_get_memory(chain), true);
                require(common_speculative_process(a.get(), prefill) && common_speculative_process(b.get(), prefill), "catch-up failed");
                common_speculative_begin(a.get(), 0, prompt);
                common_speculative_begin(b.get(), 0, prompt);
                for (int round = 0; round < 3; ++round) {
                    require(llama_memory_seq_rm(llama_get_memory(plain), 0, prompt.size(), -1), "plain trim failed");
                    require(llama_memory_seq_rm(llama_get_memory(chain), 0, prompt.size(), -1), "chain trim failed");
                    auto x = draft(a.get(), plain, prompt, seed, temp, round);
                    auto y = draft(b.get(), chain, prompt, seed, temp, round);
                    require(equal(x, y), "draft tokens/q differ");
                    require(verify(model, x, seed) == verify(model, y, seed), "verification outputs differ");
                    auto wrong = y;
                    if (!wrong.q.empty()) {
                        wrong.q[0][0].p = -1;
                        require(!equal(x, wrong, false), "q negative control escaped comparator");
                    }
                    if (round > 0) { a_ms += x.ms; b_ms += y.ms; ++n; }
                    chained += y.used_chain;
                    ++compared;
                }
            }
            std::printf("M: %s k=%d p_min=%.1f %d steady rounds: plain %.6f, chain %.6f ms/draft; exact tokens/q/verify PASS\n",
                        tp ? "Meta(CPU,CPU)" : "CPU", k, p_min, n, a_ms/n, b_ms/n);
            std::fflush(stdout);
        }
    }
    require(chained == compared, "chain did not execute in every round");
    require(control == "normal" || stats.picks > 0, "fault injector did not run");
    std::printf("M: %d compared rounds, %d chain decodes, %d observed fault-injection picks; %s control\n", compared, chained, stats.picks, control.c_str());
    llama_free(chain);
    llama_free(plain);
    llama_free(target);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

int main(int argc, char ** argv) {
    try { return run(argc, argv); } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
