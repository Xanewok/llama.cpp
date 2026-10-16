#include "llama-syncfree.h"
#include "ggml-round-trace.h"
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

bool llama_syncfree_samplers_supported(const llama_sampler * target, const llama_sampler * draft) {
    return llama_sampler_chain_leading_top_k(target) > 0 && llama_sampler_chain_supports_syncfree(target, 40) &&
        llama_sampler_chain_leading_top_k(draft) == 10 && llama_sampler_chain_supports_syncfree(draft, 10);
}

void llama_syncfree_reset(llama_context * ctx, llama_seq_id seq) {
    auto & sf = ctx->syncfree;
    if (seq < 0) {
        sf.armed = sf.ready = false;
        for (auto & state : sf.seqs) {
            state = {};
        }
    } else if ((size_t) seq < sf.seqs.size()) {
        sf.seqs[seq] = {};
    }
}

void llama_syncfree_disarm(llama_context * ctx) {
    auto & sf = ctx->syncfree;
    sf.armed = sf.ready = false;
    for (auto & state : sf.seqs) {
        state.armed = state.ready = false;
    }
}

bool llama_syncfree_arm(llama_context * ctx, llama_seq_id seq, const std::vector<llama_token> & draft,
        const std::vector<std::vector<llama_token_data>> & q, float temp, uint32_t seed) {
    auto & sf = ctx->syncfree;
    auto * dft = ctx->mtp_fold_dft;
    if (!llama_syncfree_enabled() || seq < 0 || seq >= (llama_seq_id) ctx->n_seq_max() || !dft ||
            draft.size() < 2 || draft.size() > LLAMA_MTP_CHAIN_MAX ||
            !ctx->get_cparams().flash_attn || !dft->get_cparams().flash_attn ||
            ctx->get_cparams().cb_eval || !dft->set_mtp_chain(draft.size()) ||
            !ctx->sampling.samplers.count(seq) || !dft->sampling.samplers.count(seq)) {
        return false;
    }
    const auto & model = ctx->get_model();
    const auto & nextn = model.layers[model.hparams.n_layer()].nextn;
    auto * table = nextn.embed_tokens ? nextn.embed_tokens : model.tok_embd;
    auto * backend = ggml_backend_sched_get_backend(ctx->get_sched(), 0);
    if (!table || !table->buffer || !ggml_backend_supports_buft(backend, ggml_backend_buffer_get_type(table->buffer))) {
        return false;
    }
    if (!llama_syncfree_samplers_supported(ctx->sampling.samplers.at(seq), dft->sampling.samplers.at(seq))) {
        return false;
    }
    for (const auto id : draft) {
        if (id < 0 || id >= (1 << 24)) {
            return false;
        }
    }
    sf.ctx_dft = dft;
    sf.owner = ctx;
    sf.seqs.resize(ctx->n_seq_max());
    auto & state = sf.seqs[seq];
    state.armed = state.ready = state.cache = false;
    state.greedy = temp <= 0.0f;
    if (!state.greedy) {
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
    if (state.k == 0 || state.seed != seed) {
        state.seed = seed;
        state.rng.seed(seed == LLAMA_DEFAULT_SEED ? std::random_device{}() : seed ^ 0x53465245u);
    }
    state.k = draft.size();
    state.packet.assign(2 + 23*state.k, 0.0f);
    state.packet[1] = state.greedy ? 1.0f : 0.0f;
    const int k = state.k;
    for (int i = 0; i < k; ++i) {
        state.packet[2 + i] = draft[i];
        for (int j = 0; j < 10; ++j) {
            state.packet[2 + k + i*10 + j] = -1.0f;
            if (!state.greedy && j < (int) q[i].size()) {
                state.packet[2 + k + i*10 + j] = q[i][j].id;
                state.packet[2 + 11*k + i*10 + j] = q[i][j].p;
            }
        }
    }
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (int i = 0; i < 2*k; ++i) {
        state.packet[2 + 21*k + i] = uniform(state.rng);
    }
    state.armed = true;
    sf.armed = true;
    return true;
}

bool llama_context::syncfree_prepare(const llama_ubatch & verify) {
    auto & sf = syncfree;
    sf.ready = false;
    sf.mctx.reset();
    auto fail = [&]() {
        sf.armed = false;
        for (auto & state : sf.seqs) {
            state.armed = false;
        }
        return false;
    };
    if (!sf.armed || !mctx_fold_cur || verify.n_seqs_unq == 0 ||
            n_outputs != verify.n_tokens || !loras->empty() ||
            !verify.token || verify.n_pos != model.hparams.n_pos_per_embd() || (verify.n_pos != 1 && verify.n_pos != 4)) {
        return fail();
    }
    const int ns = verify.n_seqs_unq;
    sf.active.assign(verify.seq_id_unq, verify.seq_id_unq + ns);
    sf.packet.clear();
    sf.k = 0;
    for (int s = 0; s < ns; ++s) {
        const auto seq = sf.active[s];
        if (seq < 0 || (size_t) seq >= sf.seqs.size() || !sf.seqs[seq].armed ||
                !sampling.samplers.count(seq) || !sf.ctx_dft->sampling.samplers.count(seq)) {
            return fail();
        }
        auto & state = sf.seqs[seq];
        if (s == 0) {
            sf.k = state.k;
        }
        if (state.greedy != sf.seqs[sf.active[0]].greedy || sf.k != state.k || verify.n_tokens != (uint32_t) ns*(sf.k + 1) ||
                llama_sampler_chain_leading_top_k(sampling.samplers.at(seq)) !=
                    llama_sampler_chain_leading_top_k(sampling.samplers.at(sf.active[0]))) {
            return fail();
        }
        const int begin = s*(sf.k + 1);
        const llama_pos base = verify.pos[begin];
        if (base < 0 || base > (1 << 23)) {
            return fail();
        }
        for (int i = 0; i <= sf.k; ++i) {
            if (verify.n_seq_id[begin + i] != 1 || verify.seq_id[begin + i][0] != seq ||
                    (i > 0 && verify.token[begin + i] != (llama_token) state.packet[2 + i - 1])) {
                return fail();
            }
            for (uint32_t axis = 0; axis < verify.n_pos; ++axis) {
                if (verify.pos[axis*verify.n_tokens + begin + i] != base + i) {
                    return fail();
                }
            }
        }
        state.packet[0] = base;
        sf.packet.insert(sf.packet.end(), state.packet.begin(), state.packet.end());
    }
    const size_t host_bytes = (size_t) ns*(4 + sf.k + verify.n_pos*sf.k + 31*sf.k)*sizeof(int32_t);
    if (!sf.host_buffer || ggml_backend_buffer_get_size(sf.host_buffer.get()) < host_bytes) {
        if (sf.host_buffer) {
            synchronize();
        }
        auto * device = model.dev_output();
        auto * buft = device ? ggml_backend_dev_host_buffer_type(device) : nullptr;
        sf.host_buffer.reset(ggml_backend_buft_alloc_buffer(buft ? buft : ggml_backend_cpu_buffer_type(), host_bytes));
        if (!sf.host_buffer) {
            return fail();
        }
    }
    if (!sf.completion_checked) {
        sf.completion_checked = true;
        sf.completion_supported = true;
        for (const auto & backend : backends) {
            auto * dev = ggml_backend_get_device(backend.get());
            const bool synchronous = backend.get() == backend_cpu;
            ggml_backend_event_ptr event(synchronous ? nullptr : ggml_backend_event_new(dev));
            sf.completion_supported &= synchronous || bool(event);
            sf.completion.push_back(std::move(event));
        }
    }
    sf.ubatch = verify;
    auto data = std::make_shared<llama_ubatch::data_t>();
    sf.ubatch.data = data;
    const int nt = ns*sf.k;
    sf.ubatch.n_tokens = nt;
    sf.ubatch.n_seq_tokens = sf.k;
    sf.ubatch.n_seqs = sf.ubatch.n_seqs_unq = ns;
    sf.ubatch.b_equal_seqs = 1;
    sf.ubatch.type = nullptr;
    sf.ubatch.decision_order = nullptr;
    data->token.assign(nt, 0);
    data->embd_state.assign((size_t) nt*model.hparams.n_embd, 0.0f);
    data->pos.resize(nt*verify.n_pos);
    data->n_seq_id.assign(nt, 1);
    data->seq_id_data.resize(nt);
    data->seq_id.resize(nt);
    data->seq_id_unq = sf.active;
    data->seq_idx.assign(LLAMA_MAX_SEQ, -1);
    data->output.assign(nt, 1);
    data->batch_idxs.resize(nt);
    for (int s = 0; s < ns; ++s) {
        const auto seq = sf.active[s];
        data->seq_idx[seq] = s;
        const llama_pos scratch_pos = (llama_pos) sf.seqs[seq].packet[0] + sf.k + 1;
        for (int i = 0; i < sf.k; ++i) {
            const int row = s*sf.k + i;
            data->seq_id_data[row] = seq;
            data->seq_id[row] = &data->seq_id_data[row];
            data->batch_idxs[row] = row;
            for (uint32_t axis = 0; axis < verify.n_pos; ++axis) {
                data->pos[axis*nt + row] = scratch_pos + i;
            }
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
        for (auto seq : sf.active) {
            sf.ctx_dft->get_memory()->seq_rm(seq, (llama_pos) sf.seqs[seq].packet[0] + sf.k + 1, -1);
        }
        sf.mctx.reset();
        return fail();
    }
    for (const auto & entry : sf.ctx_dft->sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }
    return true;
}

llm_graph_params llama_context::syncfree_graph_params(llm_graph_result * res) const {
    auto & sf = const_cast<llama_syncfree_state &>(syncfree);
    auto p = sf.ctx_dft->graph_params(res, sf.ubatch, sf.mctx.get(), LLM_GRAPH_TYPE_DECODER_MTP);
    p.n_outputs = sf.k*sf.active.size();
    p.mctx_fold = nullptr;
    p.fold_h_prev = nullptr;
    p.syncfree = &sf;
    p.syncfree_draft = true;
    return p;
}

static void syncfree_accept_cpu(ggml_tensor * dst, int, int, void *) {
    const int k = dst->src[2]->ne[0] - 1;
    ggml_syncfree_accept_batch((const float *) dst->src[0]->data, (const int32_t *) dst->src[1]->data,
        (const int32_t *) dst->src[2]->data, (const float *) dst->src[3]->data,
        (int32_t *) dst->data, k, (dst->ne[0] - 4 - k)/k, dst->src[0]->ne[0], dst->ne[1]);
}

static void syncfree_mask_cpu(ggml_tensor * dst, int, int, void *) {
    const float * mask = (const float *) dst->src[0]->data;
    const int32_t * pos = (const int32_t *) dst->src[1]->data;
    const int32_t * accept = (const int32_t *) dst->src[2]->data;
    const float * packet = (const float *) dst->src[3]->data;
    const int k = (dst->src[3]->ne[0] - 2)/23;
    float * out = (float *) dst->data;
    const int ns = dst->src[3]->ne[1];
    const int stride = dst->src[2]->ne[0];
    for (int64_t i = 0; i < ggml_nelements(dst); ++i) {
        const int seq = i/(dst->ne[0]*k);
        GGML_ASSERT(seq < ns);
        out[i] = ggml_syncfree_mask_value(mask[i], pos[seq*dst->ne[0] + i % dst->ne[0]],
            accept + seq*stride, packet + seq*(2 + 23*k), k);
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
    const std::vector<llama_seq_id> active;
    const bool greedy;

    explicit llm_graph_input_syncfree(llama_syncfree_state * sf) :
        sf(sf), active(sf->active), greedy(sf->seqs[sf->active[0]].greedy) {}

    void set_input(const llama_ubatch *) override {
        ggml_backend_tensor_set(sf->packet_tensor, sf->packet.data(), 0, sf->packet.size()*sizeof(float));
        for (const auto & input : children) {
            input->set_input(&sf->ubatch);
        }
        const auto * kv = dynamic_cast<const llama_kv_cache *>(sf->ctx_dft->get_memory());
        const int nkv = sf->cell_positions->ne[0];
        std::vector<int32_t> positions(nkv*sf->active.size(), -1);
        for (size_t s = 0; s < sf->active.size(); ++s) {
            const auto seq = sf->active[s];
            const auto & cells = kv->get_cells(seq);
            for (size_t i = 0; i < (size_t) nkv && i < cells.size(); ++i) {
                if (!cells.is_empty(i) && cells.seq_has(i, seq)) {
                    positions[s*nkv + i] = cells.pos_get(i);
                }
            }
        }
        ggml_backend_tensor_set(sf->cell_positions, positions.data(), 0, positions.size()*sizeof(int32_t));
    }

    bool can_reuse(const llm_graph_params & params) override {
        if (params.syncfree != sf || !sf->armed || active != sf->active || greedy != sf->seqs[sf->active[0]].greedy) {
            return false;
        }
        auto p = sf->owner->syncfree_graph_params(params.res);
        bool reuse = sf->packet_tensor->ne[0]*sf->packet_tensor->ne[1] == (int64_t) sf->packet.size() &&
            sf->packet_tensor->ne[1] == (int64_t) sf->active.size();
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
    const int ns = sf->active.size();
    const int k = sf->k;
    const bool greedy = sf->seqs[sf->active[0]].greedy;
    GGML_ASSERT(target_sampled.size() == (size_t) ns*(k + 1));

    ggml_tensor * p = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * sampled = nullptr;
    for (int i = 0; i < ns*(k + 1); ++i) {
        auto * row_p = target_probs[i];
        auto * row_ids = target_candidates[i];
        if (greedy) {
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
    p = ggml_reshape_3d(ctx, p, p->ne[0], k + 1, ns);
    ids = ggml_reshape_3d(ctx, ids, ids->ne[0], k + 1, ns);
    sampled = ggml_reshape_2d(ctx, sampled, k + 1, ns);
    sf->packet_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 + 23*k, ns);
    ggml_set_input(sf->packet_tensor);
    ggml_tensor * args[] = { p, ids, sampled, sf->packet_tensor };
    sf->accept = ggml_custom_4d(ctx, GGML_TYPE_I32, 4 + k + sf->ubatch.n_pos*k, ns, 1, 1,
        args, 4, syncfree_accept_cpu, 1, nullptr);
    ggml_set_name(sf->accept, "syncfree_accept");
    ggml_backend_sched_set_tensor_backend(params.sched, sf->accept, ggml_backend_sched_get_backend(params.sched, 0));
    ggml_set_output(sf->accept);
    ggml_build_forward_expand(gf, sf->accept);
    sf->token = ggml_cont_1d(ctx, ggml_view_2d(ctx, sf->accept, 1, ns, sf->accept->nb[1], sizeof(int32_t)), ns);
    sf->hidden = nullptr;
    for (int s = 0; s < ns; ++s) {
        auto * h_idx = ggml_view_1d(ctx, sf->accept, 1, s*sf->accept->nb[1]);
        auto * h_seq = ggml_view_2d(ctx, target_h, target_h->ne[0], k + 1, target_h->nb[1], s*(k + 1)*target_h->nb[1]);
        auto * h = ggml_get_rows(ctx, h_seq, h_idx);
        sf->hidden = sf->hidden ? ggml_concat(ctx, sf->hidden, h, 1) : h;
    }
    sf->positions = nullptr;
    for (uint32_t axis = 0; axis < sf->ubatch.n_pos; ++axis) {
        auto * pos = ggml_cont_1d(ctx, ggml_view_2d(ctx, sf->accept, k, ns, sf->accept->nb[1],
            (4 + k + axis*k)*sizeof(int32_t)), k*ns);
        sf->positions = sf->positions ? ggml_concat(ctx, sf->positions, pos, 0) : pos;
    }

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
    const int ns = sf.active.size();
    const int nt = ns*sf.k;
    auto * accepted = (int32_t *) ggml_backend_buffer_get_base(sf.host_buffer.get());
    auto * draft = accepted + ns*(4 + sf.k + sf.ubatch.n_pos*sf.k);
    auto * probs = (float *) (draft + nt);
    auto * logits = probs + nt*10;
    auto * candidates = (int32_t *) (logits + nt*10);
    auto copy = [&](ggml_tensor * t, void * out) {
        auto * backend = ggml_backend_sched_get_tensor_backend(get_sched_graph(), t);
        tensor_get_async(backend, t, out, 0, ggml_nbytes(t));
    };
    copy(sf.accept, accepted);
    for (int i = 0; i < nt; ++i) {
        GGML_ASSERT(ggml_nelements(sf.probs[i]) == sf.n_candidates && ggml_nelements(sf.candidates[i]) == sf.n_candidates &&
            ggml_nelements(sf.logits[i]) == sf.n_candidates && ggml_nelements(sf.sampled[i]) == 1);
        copy(sf.sampled[i], &draft[i]);
        copy(sf.probs[i], &probs[i*10]);
        copy(sf.logits[i], &logits[i*10]);
        copy(sf.candidates[i], &candidates[i*10]);
    }
    sf.ready = true;
    sf.armed = false;
    for (auto seq : sf.active) {
        sf.seqs[seq].ready = true;
        sf.seqs[seq].armed = false;
    }
    if (sf.completion_supported && ggml_backend_sched_get_n_copies(get_sched_graph()) == 1) {
        for (size_t i = 0; i < sf.completion.size(); ++i) {
            if (sf.completion[i]) {
                ggml_backend_event_record(sf.completion[i].get(), backends[i].get());
            }
        }
        sf.completion_pending = true;
    }
    if (sf.logged_np != (uint32_t) ns) {
        sf.logged_np = ns;
        LLAMA_LOG_INFO("syncfree: fused np=%d k=%d accept+draft queued before target sync, output=%s\n",
            ns, sf.k, sf.completion_pending ? "events" : "backend");
    }
}

bool llama_context::syncfree_synchronize() {
    auto & sf = syncfree;
    if (!sf.completion_pending) {
        return false;
    }
    ggml_round_trace_scope trace("syncfree_output_wait");
    // The events follow every output copy. A later enqueue invalidates this completion point.
    for (const auto & event : sf.completion) {
        if (event) {
            ggml_backend_event_synchronize(event.get());
        }
    }
    return true;
}

bool llama_syncfree_accept(llama_context * ctx, llama_seq_id seq, std::vector<llama_token> & accepted) {
    auto & sf = ctx->syncfree;
    if (!sf.ready || seq < 0 || (size_t) seq >= sf.seqs.size() || !sf.seqs[seq].ready) {
        return false;
    }
    auto & state = sf.seqs[seq];
    const auto it = std::find(sf.active.begin(), sf.active.end(), seq);
    GGML_ASSERT(it != sf.active.end());
    const int s = it - sf.active.begin();
    const int ns = sf.active.size();
    const int nt = ns*sf.k;
    auto * words = (int32_t *) ggml_backend_buffer_get_base(sf.host_buffer.get());
    const int n_accept = 4 + sf.k + sf.ubatch.n_pos*sf.k;
    state.accepted.assign(words + s*n_accept, words + (s + 1)*n_accept);
    auto * draft = words + ns*n_accept;
    state.draft.assign(draft + s*sf.k, draft + (s + 1)*sf.k);
    auto * probs = (float *) (draft + nt);
    auto * logits = probs + nt*10;
    auto * candidates = (int32_t *) (logits + nt*10);
    const int a = state.accepted[0];
    GGML_ASSERT(a >= 0 && a <= sf.k);
    accepted.assign(state.accepted.begin() + 3, state.accepted.begin() + 4 + a);
    state.q.assign(sf.k, {});
    for (int i = 0; i < sf.k; ++i) {
        for (int j = 0; j < sf.n_candidates; ++j) {
            const int row = (s*sf.k + i)*10 + j;
            state.q[i].push_back({candidates[row], logits[row], probs[row]});
        }
    }
    sf.ctx_dft->get_memory()->seq_rm(seq, (llama_pos) state.packet[0] + sf.k + 1, -1);
    state.ready = false;
    state.cache = true;
    return true;
}

bool llama_syncfree_draft(llama_context * ctx, llama_seq_id seq, llama_pos pos, llama_token bonus,
        std::vector<llama_token> & draft, std::vector<std::vector<llama_token_data>> * q) {
    auto & sf = ctx->syncfree;
    if (seq < 0 || (size_t) seq >= sf.seqs.size()) {
        return false;
    }
    auto & state = sf.seqs[seq];
    const bool valid = state.cache && pos == state.accepted[2] && bonus == state.accepted[1];
    state.cache = false;
    if (!valid) {
        return false;
    }
    draft = state.draft;
    if (q) {
        *q = state.q;
    }
    return true;
}
