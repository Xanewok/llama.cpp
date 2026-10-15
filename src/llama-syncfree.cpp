#include "llama-syncfree.h"
#include "llama-context.h"
#include "llama-ext.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "llama-sampler.h"
#include "models/models.h"
#include "../ggml/src/ggml-syncfree.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>

bool llama_syncfree_enabled() {
    static const bool enabled = [] { const char * p = std::getenv("LLAMA_SYNCFREE"); return p && std::strcmp(p, "1") == 0; }();
    return enabled;
}

void llama_syncfree_reset(llama_context * ctx) {
    auto & sf = ctx->syncfree;
    sf.armed = sf.ready = sf.cache = false;
    sf.k = 0;
}

bool llama_syncfree_arm(llama_context * ctx, llama_seq_id seq, const std::vector<llama_token> & draft,
        const std::vector<std::vector<llama_token_data>> & q, float temp, uint32_t seed) {
    auto & sf = ctx->syncfree;
    sf.armed = sf.ready = sf.cache = false;
    auto * dft = ctx->mtp_fold_dft;
    if (!llama_syncfree_enabled() || seq != 0 || ctx->n_seq_max() != 1 || !dft ||
            draft.size() < 2 || draft.size() > LLAMA_MTP_CHAIN_MAX ||
            !ctx->get_cparams().flash_attn || !dft->get_cparams().flash_attn ||
            ctx->get_cparams().cb_eval || !dft->set_mtp_chain(draft.size())) {
        return false;
    }
    if (ctx->sampling.samplers.size() != 1 || !ctx->sampling.samplers.count(0)) {
        return false;
    }
    const int target_k = llama_sampler_chain_leading_top_k(ctx->sampling.samplers.at(0));
    if (target_k < 1 || target_k > 40) {
        return false;
    }
    for (const auto id : draft) {
        if (id < 0 || id >= (1 << 24)) {
            return false;
        }
    }
    if (dft->sampling.samplers.size() != 1 || !dft->sampling.samplers.count(0)) {
        return false;
    }
    auto * chain = dft->sampling.samplers.at(0);
    const int n = llama_sampler_chain_n(chain);
    if (n < 3 || llama_sampler_chain_leading_top_k(chain) != 10 ||
            std::strcmp(llama_sampler_name(llama_sampler_chain_get(chain, n - 1)), "dist") != 0) {
        return false;
    }
    sf.ctx_dft = dft;
    sf.owner = ctx;
    sf.greedy = temp <= 0.0f;
    if (!sf.greedy) {
        if (q.size() != draft.size()) {
            return false;
        }
        for (const auto & row : q) {
            if (row.empty() || row.size() > 10) {
                return false;
            }
            for (const auto & token : row) {
                if (token.id < 0 || token.id >= (1 << 24)) {
                    return false;
                }
            }
        }
    }
    if (sf.k == 0 || sf.seed != seed) {
        sf.seed = seed;
        sf.rng.seed(seed == LLAMA_DEFAULT_SEED ? std::random_device{}() : seed ^ 0x53465245u);
    }
    sf.k = draft.size();
    sf.packet.assign(2 + 23*sf.k, 0.0f);
    sf.packet[1] = sf.greedy ? 1.0f : 0.0f;
    for (int i = 0; i < sf.k; ++i) {
        sf.packet[2 + i] = draft[i];
        for (int j = 0; j < 10; ++j) {
            sf.packet[2 + sf.k + i*10 + j] = -1.0f;
            if (!sf.greedy && j < (int) q[i].size()) {
                sf.packet[2 + sf.k + i*10 + j] = q[i][j].id;
                sf.packet[2 + 11*sf.k + i*10 + j] = q[i][j].p;
            }
        }
    }
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (int i = 0; i < 2*sf.k; ++i) {
        sf.packet[2 + 21*sf.k + i] = uniform(sf.rng);
    }
    sf.armed = true;
    return true;
}

bool llama_context::syncfree_prepare(const llama_ubatch & verify) {
    auto & sf = syncfree;
    sf.ready = false;
    sf.mctx.reset();
    if (!sf.armed || !mctx_fold_cur || verify.n_tokens != (uint32_t) sf.k + 1 ||
            verify.n_seqs_unq != 1 || verify.seq_id_unq[0] != 0 ||
            n_outputs != verify.n_tokens || sampling.samplers.size() != 1 ||
            sf.ctx_dft->sampling.samplers.size() != 1 || !loras->empty()) {
        sf.armed = false;
        return false;
    }
    if (!verify.token || verify.n_pos != model.hparams.n_pos_per_embd() || (verify.n_pos != 1 && verify.n_pos != 4)) {
        sf.armed = false;
        return false;
    }
    for (uint32_t axis = 0; axis < verify.n_pos; ++axis) {
        for (uint32_t i = 0; i < verify.n_tokens; ++i) {
            if (verify.pos[axis*verify.n_tokens + i] != verify.pos[0] + (llama_pos) i ||
                    verify.n_seq_id[i] != 1 || verify.seq_id[i][0] != 0) {
                sf.armed = false;
                return false;
            }
        }
    }
    if (verify.pos[0] < 0 || verify.pos[0] > (1 << 23)) {
        sf.armed = false;
        return false;
    }
    if (!sf.host_buffer) {
        auto * device = model.dev_output();
        auto * buft = device ? ggml_backend_dev_host_buffer_type(device) : nullptr;
        sf.host_buffer.reset(ggml_backend_buft_alloc_buffer(buft ? buft : ggml_backend_cpu_buffer_type(), 4096));
        if (!sf.host_buffer) {
            sf.armed = false;
            return false;
        }
    }
    sf.packet[0] = verify.pos[0];
    sf.ubatch = verify;
    auto data = std::make_shared<llama_ubatch::data_t>();
    sf.ubatch.data = data;
    sf.ubatch.n_tokens = sf.ubatch.n_seq_tokens = sf.k;
    sf.ubatch.n_seqs = sf.ubatch.n_seqs_unq = sf.ubatch.b_equal_seqs = 1;
    sf.ubatch.type = nullptr;
    sf.ubatch.decision_order = nullptr;
    data->token.assign(sf.k, 0);
    data->embd_state.assign((size_t) sf.k*model.hparams.n_embd, 0.0f);
    data->pos.resize(sf.k*verify.n_pos);
    data->n_seq_id.assign(sf.k, 1);
    data->seq_id_data.assign(sf.k, 0);
    data->seq_id.resize(sf.k);
    data->seq_id_unq = {0};
    data->seq_idx.assign(LLAMA_MAX_SEQ, 0);
    data->output.assign(sf.k, 1);
    data->batch_idxs.resize(sf.k);
    const llama_pos scratch_pos = verify.pos[0] + sf.k + 1;
    for (int i = 0; i < sf.k; ++i) {
        data->seq_id[i] = &data->seq_id_data[i];
        data->batch_idxs[i] = i;
        for (uint32_t axis = 0; axis < verify.n_pos; ++axis) {
            data->pos[axis*sf.k + i] = scratch_pos + i;
        }
    }
    sf.ubatch.token = data->token.data();
    sf.ubatch.embd_state = data->embd_state.data();
    sf.ubatch.pos = data->pos.data();
    sf.ubatch.n_seq_id = data->n_seq_id.data();
    sf.ubatch.seq_id = data->seq_id.data();
    sf.ubatch.seq_id_unq = data->seq_id_unq.data();
    sf.ubatch.seq_idx = data->seq_idx.data();
    sf.ubatch.output = data->output.data();
    sf.mctx = sf.ctx_dft->mtp_fold_prepare(sf.ubatch);
    if (!sf.mctx || !sf.mctx->apply()) {
        sf.ctx_dft->get_memory()->seq_rm(0, scratch_pos, -1);
        sf.armed = false;
        return false;
    }
    for (const auto & entry : sf.ctx_dft->sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }
    return true;
}

llm_graph_params llama_context::syncfree_graph_params(llm_graph_result * res) const {
    auto & sf = const_cast<llama_syncfree_state &>(syncfree);
    auto p = sf.ctx_dft->graph_params(res, sf.ubatch, sf.mctx.get(), LLM_GRAPH_TYPE_DECODER_MTP);
    p.n_outputs = sf.k;
    p.mctx_fold = nullptr;
    p.fold_h_prev = nullptr;
    p.syncfree = &sf;
    p.syncfree_draft = true;
    return p;
}

static void syncfree_accept_cpu(ggml_tensor * dst, int, int, void *) {
    const int k = dst->src[2]->ne[0] - 1;
    ggml_syncfree_accept((const float *) dst->src[0]->data, (const int32_t *) dst->src[1]->data,
        (const int32_t *) dst->src[2]->data, (const float *) dst->src[3]->data,
        (int32_t *) dst->data, k, (dst->ne[0] - 4 - k)/k, dst->src[0]->ne[0]);
}

static void syncfree_mask_cpu(ggml_tensor * dst, int, int, void *) {
    const float * mask = (const float *) dst->src[0]->data;
    const int32_t * pos = (const int32_t *) dst->src[1]->data;
    const int32_t * accept = (const int32_t *) dst->src[2]->data;
    const float * packet = (const float *) dst->src[3]->data;
    const int k = (dst->src[3]->ne[0] - 2)/23;
    float * out = (float *) dst->data;
    for (int64_t i = 0; i < ggml_nelements(dst); ++i) {
        const int p = pos[i % dst->ne[0]];
        out[i] = p >= accept[2] && p < (int) packet[0] + k + 1 ? -INFINITY : mask[i];
    }
}

ggml_tensor * llama_syncfree_mask(ggml_context * ctx, llama_syncfree_state * sf,
        ggml_tensor * mask, ggml_tensor * cell_pos) {
    ggml_tensor * args[] = { mask, cell_pos, sf->accept, sf->packet_tensor };
    auto * out = ggml_custom_4d(ctx, GGML_TYPE_F32, mask->ne[0], mask->ne[1], mask->ne[2], mask->ne[3],
        args, 4, syncfree_mask_cpu, 1, nullptr);
    ggml_set_name(out, "syncfree_mask");
    ggml_backend_sched_set_tensor_backend(sf->sched, out, ggml_backend_sched_get_backend(sf->sched, 0));
    return out;
}

class llm_graph_input_syncfree : public llm_graph_input_i {
public:
    llama_syncfree_state * sf;
    std::vector<llm_graph_input_ptr> children;

    explicit llm_graph_input_syncfree(llama_syncfree_state * sf) : sf(sf) {}

    void set_input(const llama_ubatch *) override {
        ggml_backend_tensor_set(sf->packet_tensor, sf->packet.data(), 0, sf->packet.size()*sizeof(float));
        for (const auto & input : children) {
            input->set_input(&sf->ubatch);
        }
        const auto * kv = dynamic_cast<const llama_kv_cache *>(sf->ctx_dft->get_memory());
        const auto & cells = kv->get_cells(0);
        std::vector<int32_t> positions(sf->cell_positions->ne[0], -1);
        for (size_t i = 0; i < positions.size() && i < cells.size(); ++i) {
            if (!cells.is_empty(i) && cells.seq_has(i, 0)) {
                positions[i] = cells.pos_get(i);
            }
        }
        ggml_backend_tensor_set(sf->cell_positions, positions.data(), 0, positions.size()*sizeof(int32_t));
    }

    bool can_reuse(const llm_graph_params & params) override {
        if (params.syncfree != sf || !sf->armed) {
            return false;
        }
        auto p = sf->owner->syncfree_graph_params(params.res);
        bool reuse = sf->packet_tensor->ne[0] == (int64_t) sf->packet.size();
        for (const auto & input : children) {
            reuse = input->can_reuse(p) && reuse;
        }
        return reuse;
    }
};

void llama_syncfree_build(const llama_model & model, const llm_graph_params & params) {
    auto * sf = params.syncfree;
    auto * res = params.res;
    auto * ctx = res->get_ctx();
    auto * gf = res->get_gf();
    sf->sched = params.sched;
    const auto target_sampled = res->t_sampled;
    const auto target_probs = res->t_sampled_probs;
    const auto target_logits = res->t_sampled_logits;
    const auto target_candidates = res->t_candidates;
    auto * target_h = res->t_h_nextn;
    auto * target_output = res->t_logits;
    GGML_ASSERT(target_sampled.size() == (size_t) sf->k + 1);

    ggml_tensor * p = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * sampled = nullptr;
    for (int i = 0; i <= sf->k; ++i) {
        auto * row_p = target_probs[i];
        auto * row_ids = target_candidates[i];
        if (sf->greedy) {
            row_p = ggml_cast(ctx, target_sampled[i], GGML_TYPE_F32);
            row_ids = target_sampled[i];
        }
        GGML_ASSERT(row_p && row_ids && target_sampled[i]);
        row_p = ggml_reshape_2d(ctx, row_p, ggml_nelements(row_p), 1);
        row_ids = ggml_reshape_2d(ctx, row_ids, ggml_nelements(row_ids), 1);
        p = p ? ggml_concat(ctx, p, row_p, 1) : row_p;
        ids = ids ? ggml_concat(ctx, ids, row_ids, 1) : row_ids;
        auto * token = ggml_reshape_1d(ctx, target_sampled[i], 1);
        sampled = sampled ? ggml_concat(ctx, sampled, token, 0) : token;
    }
    sf->packet_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sf->packet.size());
    ggml_set_input(sf->packet_tensor);
    ggml_tensor * args[] = { p, ids, sampled, sf->packet_tensor };
    sf->accept = ggml_custom_4d(ctx, GGML_TYPE_I32, 4 + sf->k + sf->ubatch.n_pos*sf->k, 1, 1, 1,
        args, 4, syncfree_accept_cpu, 1, nullptr);
    ggml_set_name(sf->accept, "syncfree_accept");
    ggml_backend_sched_set_tensor_backend(params.sched, sf->accept, ggml_backend_sched_get_backend(params.sched, 0));
    ggml_set_output(sf->accept);
    ggml_build_forward_expand(gf, sf->accept);
    sf->token = ggml_view_1d(ctx, sf->accept, 1, sizeof(int32_t));
    auto * h_idx = ggml_view_1d(ctx, sf->accept, 1, 0);
    sf->hidden = ggml_get_rows(ctx, target_h, h_idx);
    sf->positions = ggml_view_1d(ctx, sf->accept, sf->ubatch.n_pos*sf->k, (4 + sf->k)*sizeof(int32_t));

    auto wrapper = std::make_unique<llm_graph_input_syncfree>(sf);
    const size_t input_begin = res->inputs.size();
    auto p_draft = sf->owner->syncfree_graph_params(res);
    p_draft.sched = params.sched;
    p_draft.backend_cpu = params.backend_cpu;
    p_draft.cb = params.cb;
    llama_model_qwen35::graph_mtp draft_graph(model, p_draft);
    sf->sampled = res->t_sampled;
    sf->probs = res->t_sampled_probs;
    sf->logits = res->t_sampled_logits;
    sf->candidates = res->t_candidates;
    sf->n_candidates = ggml_nelements(sf->probs[0]);
    GGML_ASSERT(sf->n_candidates >= 1 && sf->n_candidates <= 10);
    for (const auto & rows : {sf->sampled, sf->probs, sf->logits, sf->candidates}) {
        for (auto * t : rows) {
            GGML_ASSERT(t);
            ggml_set_output(t);
        }
    }
    for (size_t i = input_begin; i < res->inputs.size(); ++i) {
        wrapper->children.push_back(std::move(res->inputs[i]));
    }
    res->inputs.resize(input_begin);
    res->add_input(std::move(wrapper));
    res->t_sampled = target_sampled;
    res->t_sampled_probs = target_probs;
    res->t_sampled_logits = target_logits;
    res->t_candidates = target_candidates;
    res->t_h_nextn = target_h;
    res->t_logits = target_output;
    res->set_params(params);
}

void llama_context::syncfree_readback() {
    auto & sf = syncfree;
    auto * accepted = (int32_t *) ggml_backend_buffer_get_base(sf.host_buffer.get());
    auto * draft = accepted + 4 + sf.k + sf.ubatch.n_pos*sf.k;
    auto * probs = (float *) (draft + sf.k);
    auto * logits = probs + sf.k*10;
    auto * candidates = (int32_t *) (logits + sf.k*10);
    auto copy = [&](ggml_tensor * t, void * out) {
        auto * backend = ggml_backend_sched_get_tensor_backend(get_sched_graph(), t);
        tensor_get_async(backend, t, out, 0, ggml_nbytes(t));
    };
    copy(sf.accept, accepted);
    for (int i = 0; i < sf.k; ++i) {
        GGML_ASSERT(ggml_nelements(sf.probs[i]) == sf.n_candidates && ggml_nelements(sf.candidates[i]) == sf.n_candidates &&
            ggml_nelements(sf.logits[i]) == sf.n_candidates && ggml_nelements(sf.sampled[i]) == 1);
        copy(sf.sampled[i], &draft[i]);
        copy(sf.probs[i], &probs[i*10]);
        copy(sf.logits[i], &logits[i*10]);
        copy(sf.candidates[i], &candidates[i*10]);
    }
    sf.ready = true;
    sf.armed = false;
    if (++sf.rounds == 1) {
        LLAMA_LOG_INFO("syncfree: fused np=1 k=%d accept+draft queued before target sync\n", sf.k);
    }
}

bool llama_syncfree_accept(llama_context * ctx, std::vector<llama_token> & accepted) {
    auto & sf = ctx->syncfree;
    if (!sf.ready) {
        return false;
    }
    auto * words = (int32_t *) ggml_backend_buffer_get_base(sf.host_buffer.get());
    const int n_accept = 4 + sf.k + sf.ubatch.n_pos*sf.k;
    sf.accepted.assign(words, words + n_accept);
    sf.draft.assign(words + n_accept, words + n_accept + sf.k);
    auto * probs = (float *) (words + n_accept + sf.k);
    auto * logits = probs + sf.k*10;
    auto * candidates = (int32_t *) (logits + sf.k*10);
    const int a = sf.accepted[0];
    GGML_ASSERT(a >= 0 && a <= sf.k);
    accepted.assign(sf.accepted.begin() + 3, sf.accepted.begin() + 4 + a);
    sf.q.assign(sf.k, {});
    for (int i = 0; i < sf.k; ++i) {
        for (int j = 0; j < sf.n_candidates; ++j) {
            sf.q[i].push_back({candidates[i*10 + j], logits[i*10 + j], probs[i*10 + j]});
        }
    }
    sf.ctx_dft->get_memory()->seq_rm(0, (llama_pos) sf.packet[0] + sf.k + 1, -1);
    sf.ready = false;
    sf.cache = true;
    return true;
}

bool llama_syncfree_draft(llama_context * ctx, llama_pos pos, llama_token bonus,
        std::vector<llama_token> & draft, std::vector<std::vector<llama_token_data>> * q) {
    auto & sf = ctx->syncfree;
    const bool valid = sf.cache && pos == sf.accepted[2] && bonus == sf.accepted[1];
    sf.cache = false;
    if (!valid) {
        return false;
    }
    draft = sf.draft;
    if (q) {
        *q = sf.q;
    }
    return true;
}
