#pragma once

#include "llama-graph.h"
#include "llama-memory.h"

#include <random>

struct llama_context;
struct llama_model;

struct llama_syncfree_state {
    bool armed = false;
    bool ready = false;
    bool cache = false;
    bool greedy = false;
    int32_t k = 0;
    int32_t n_candidates = 0;
    uint32_t seed = 0;
    uint64_t rounds = 0;
    std::mt19937 rng;
    std::vector<float> packet;
    std::vector<int32_t> accepted;
    std::vector<llama_token> draft;
    std::vector<std::vector<llama_token_data>> q;
    llama_ubatch ubatch = {};
    llama_memory_context_ptr mctx;
    llama_context * ctx_dft = nullptr;
    llama_context * owner = nullptr;
    ggml_tensor * cell_positions = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_buffer_ptr host_buffer;
    ggml_tensor * token = nullptr;
    ggml_tensor * hidden = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * accept = nullptr;
    ggml_tensor * packet_tensor = nullptr;
    std::vector<ggml_tensor *> sampled;
    std::vector<ggml_tensor *> probs;
    std::vector<ggml_tensor *> logits;
    std::vector<ggml_tensor *> candidates;
};

void llama_syncfree_build(const llama_model & model, const llm_graph_params & params);
ggml_tensor * llama_syncfree_mask(ggml_context * ctx, llama_syncfree_state * sf,
        ggml_tensor * mask, ggml_tensor * cell_pos);
