#pragma once

#include "llama.h"

#include <vector>
#include <functional>

struct llama_vocab;
struct llama_grammar;

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    uint32_t n_nodes = 0;

    struct info {
        bool is_backend;

        llama_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for llama_sampler_sample to avoid repeated allocations
    std::vector<llama_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

uint32_t llama_sampler_backend_n_nodes(const llama_sampler * sampler);
void llama_sampler_backend_begin(llama_sampler * sampler);

// Restore graph inputs without copying or advancing sampler state. Empty for unsupported bindings.
std::function<void()> llama_sampler_backend_bindings(llama_sampler * sampler);

// k after empty samplers and pointwise logit biases, else -1. n_prefix excludes top-k.
int32_t llama_sampler_chain_leading_top_k(const llama_sampler * sampler, int32_t * n_prefix = nullptr);
void llama_sampler_chain_backend_apply_range(llama_sampler * sampler, ggml_context * ctx, ggml_cgraph * gf, llama_sampler_data * data, int32_t begin, int32_t end);

struct llama_sampler * llama_sampler_init_dry_testing(
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llama_token>> & seq_breakers);
