#pragma once

#include "llama-graph.h"
#include "llama-memory.h"

#include <random>

struct llama_context;
struct llama_model;

inline bool llama_syncfree_text_positions(const llama_ubatch & batch, int32_t begin, int32_t count) {
    for (uint32_t axis = 0; axis < batch.n_pos; ++axis) {
        for (int32_t i = 0; i < count; ++i) {
            if (batch.pos[axis*batch.n_tokens + begin + i] != (axis < 3 ? batch.pos[begin] + i : 0)) {
                return false;
            }
        }
    }
    return true;
}

inline ggml_tensor * llama_syncfree_step_idxs(ggml_context * ctx, ggml_tensor * idxs,
        int64_t n_chain, int64_t n_seq, int64_t step) {
    // SET_ROWS accepts strided indices; keep the I64 packet as a view.
    return n_seq == 1 ? ggml_view_1d(ctx, idxs, 1, step*idxs->nb[0]) :
        ggml_transpose(ctx, ggml_view_2d(ctx, idxs, 1, n_seq, n_chain*idxs->nb[0], step*idxs->nb[0]));
}

struct llama_syncfree_seq {
    bool armed = false;
    bool ready = false;
    bool cache = false;
    bool greedy = false;
    int32_t k = 0;
    uint32_t seed = 0;
    std::mt19937 rng;
    std::vector<float> packet;
    std::vector<int32_t> accepted;
    std::vector<llama_token> draft;
    std::vector<std::vector<llama_token_data>> q;
};

struct llama_syncfree_state {
    bool armed = false;
    bool ready = false;
    int32_t k = 0;
    int32_t n_candidates = 0;
    uint32_t logged_np = 0;
    bool reported_not_armed = false;
    std::vector<llama_syncfree_seq> seqs;
    std::vector<llama_seq_id> active;
    std::vector<float> packet;
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
    std::vector<ggml_backend_event_ptr> completion;
    bool completion_checked = false;
    bool completion_supported = false;
    bool completion_pending = false;
};

void llama_syncfree_build(const llama_model & model, const llm_graph_params & params);
ggml_tensor * llama_syncfree_mask(ggml_context * ctx, llama_syncfree_state * sf,
        ggml_tensor * mask, ggml_tensor * cell_pos);
