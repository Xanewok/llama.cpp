#include "common.h"
#include "llama-cpp.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 2);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
    GGML_ASSERT(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = cp.n_ubatch = 16;
    cp.n_threads = cp.n_threads_batch = 4;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    GGML_ASSERT(ctx);
    const auto * vocab = llama_model_get_vocab(model.get());
    const int32_t nv = llama_vocab_n_tokens(vocab);
    const auto allowed = common_tokenize(vocab, "a", false, true);
    GGML_ASSERT(allowed.size() == 1);
    common_batch batch(ctx.get());
    batch.add(allowed[0], 0, 0, true);
    GGML_ASSERT(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
    llama_synchronize(ctx.get());
    std::vector<float> logits(nv);
    for (int32_t i = 0; i < nv; ++i) {
        logits[i] = -float(i);
    }
    const auto set_row = [&]() {
        std::memcpy(llama_get_logits_ith(ctx.get(), 0), logits.data(), nv * sizeof(float));
    };
    common_params_sampling p;
    p.top_k = 4;
    p.top_p = 0.95f;
    p.min_p = 0.05f;
    p.temp = 1.0f;
    p.backend_sampling = false;
    for (bool tied : {false, true}) {
        p.top_k = tied ? 20 : 4;
        p.top_p = tied ? 1.0f : 0.95f;
        p.min_p = tied ? 0.0f : 0.05f;
        for (int32_t i = 0; i < nv; ++i) {
            logits[i] = -float(i);
        }
        if (tied) {
            // Keep six tied triples inside k and a triple across k; scattered ids expose heap/partial_sort order.
            for (int i = 0; i < 21; ++i) {
                const llama_token id = 1000 + (7919 * i) % 30011;
                GGML_ASSERT(id < nv);
                logits[id] = i < 18 ? 12.0f - 0.25f * (i / 3) : 10.0f;
            }
        }
        for (bool bias : {false, true}) {
            p.logit_bias = bias ? std::vector<llama_logit_bias>{{nv - 1, float(nv)}} : std::vector<llama_logit_bias>{};
            for (uint32_t seed : {42u, 1729u}) {
                p.seed = seed;
                common_sampler_ptr sampler(common_sampler_init(model.get(), p));
                common_sampler_ptr reference(common_sampler_init(model.get(), p));
                std::vector<llama_token_data> full(nv);
                for (int32_t i = 0; i < nv; ++i) {
                    full[i] = {i, logits[i], 0.0f};
                }
                llama_token_data_array row = {full.data(), full.size(), -1, false};
                llama_sampler_apply(common_sampler_get(reference.get()), &row);
                set_row();
                const llama_token got = common_sampler_sample(sampler.get(), ctx.get(), 0);
                GGML_ASSERT(row.selected >= 0 && got == row.data[row.selected].id);
                const auto * actual = common_sampler_get_candidates(sampler.get(), false);
                GGML_ASSERT(actual->size == row.size && actual->selected == row.selected);
                for (size_t i = 0; i < row.size; ++i) {
                    GGML_ASSERT(actual->data[i].id == row.data[i].id);
                    GGML_ASSERT(actual->data[i].logit == row.data[i].logit && actual->data[i].p == row.data[i].p);
                }
            }
        }
    }
    p.logit_bias.clear();
    p.top_k = 4;
    p.top_p = 0.95f;
    p.min_p = 0.05f;
    for (int32_t i = 0; i < nv; ++i) {
        logits[i] = -float(i);
    }
    logits[allowed[0]] = -float(nv);
    for (bool lazy : {false, true}) {
        p.grammar_lazy = lazy;
        p.grammar = {COMMON_GRAMMAR_TYPE_USER, lazy ? "root ::= \"!a\"" : "root ::= \"a\""};
        const auto trigger = common_tokenize(vocab, "!", false, true);
        GGML_ASSERT(trigger.size() == 1);
        p.grammar_triggers = {{COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN, "!", trigger[0]}};
        for (bool grammar_first : {false, true}) {
            common_sampler_ptr sampler(common_sampler_init(model.get(), p));
            if (lazy) {
                common_sampler_accept(sampler.get(), trigger[0], true);
            }
            set_row();
            GGML_ASSERT(common_sampler_sample(sampler.get(), ctx.get(), 0, grammar_first) == allowed[0]);
        }
    }
    p.grammar = {};
    p.grammar_lazy = false;
    p.reasoning_budget_tokens = -1;
    p.reasoning_control = true;
    p.reasoning_budget_start = {allowed[0]};
    p.reasoning_budget_end = {{allowed[0]}};
    p.reasoning_budget_forced = {allowed[0]};
    common_sampler_ptr sampler(common_sampler_init(model.get(), p));
    common_sampler_accept(sampler.get(), allowed[0], true);
    GGML_ASSERT(common_sampler_reasoning_budget_force(sampler.get()));
    set_row();
    GGML_ASSERT(common_sampler_sample(sampler.get(), ctx.get(), 0) == allowed[0]);
}
