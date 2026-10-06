// TODO: merge with test-save-load-state.cpp
// TODO: merge with test-state-restore-fragmented.cpp

#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "log.h"
#include "llama-cpp.h"
#include "llama.h"

#include "../src/llama-io.h"
#include "../src/llama-memory.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <set>
#include <string>
#include <vector>

enum class test_status {
    PASS,
    FAIL,
    SKIP,
};

static const char * test_status_str(test_status status) {
    switch (status) {
        case test_status::PASS: return "\033[1;32mPASS\033[m";
        case test_status::FAIL: return "\033[1;31mFAIL\033[m";
        case test_status::SKIP: return "\033[1;33mSKIP\033[m";
    }
    return "";
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens) {
    common_batch batch(ctx);
    for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
        batch.add(tokens[pos], pos, 0, pos + 1 == tokens.size());
    }
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq = 0) {
    common_batch batch(ctx);
    batch.add(tok, pos, seq, true);
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

struct cache_buffer_collector : llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

static llama_context_ptr init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill) {
    llama_context_ptr ctx{llama_init_from_model(model, cparams)};
    if (!ctx || fill == 0) {
        return ctx;
    }

    // Use a full ubatch so buffer discovery preserves prefill allocation sizes.
    const uint32_t n_tokens = llama_n_ubatch(ctx.get());
    if (!decode_tokens(ctx.get(), std::vector<llama_token>(n_tokens, 0))) {
        return nullptr;
    }
    llama_synchronize(ctx.get());
    cache_buffer_collector collector;
    llama_get_memory(ctx.get())->state_write(collector);
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    if (collector.buffers.empty()) {
        LOG_ERR("%s: no cache buffers found\n", __func__);
        return nullptr;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

static llama_context_ptr make_ctx(const common_params & params, llama_model * model, uint8_t fill) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return init_ctx(model, cparams, fill);
}

static float logit_diff(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity();
}

static double nmse(const float * a, const float * b, int n) {
    double mse_ab = 0.0;
    double mse_a0 = 0.0;
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return std::numeric_limits<double>::infinity();
        }
        const double diff = (double) a[i] - b[i];
        mse_ab += diff*diff;
        mse_a0 += (double) a[i]*a[i];
    }
    return mse_a0 == 0.0 ? (mse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : mse_ab/mse_a0;
}

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch.
static test_status test_multi_seq_split_replay(const common_params & params, llama_model * model, uint8_t fill) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return init_ctx(model, cparams, fill);
    };

    llama_context_ptr ctx_roll = make_ctx_multi();
    llama_context_ptr ctx_ref  = make_ctx_multi();
    if (!ctx_roll || !ctx_ref) {
        LOG_ERR("%s: failed to init multi-seq contexts\n", __func__);
        return test_status::FAIL;
    }

    if (llama_n_rs_seq(ctx_roll.get()) < n_rollback) {
        LOG_INF("%s: skipping because n_rs_seq is too small\n", __func__);
        return test_status::SKIP;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // decode tokens [p_begin, p_end) of seq s, a batch belongs to one context so it is built per call
    const auto decode_range = [&](llama_context * ctx, uint32_t s, llama_pos p_begin, llama_pos p_end) {
        common_batch batch(ctx);
        for (llama_pos pos = p_begin; pos < p_end; ++pos) {
            batch.add(tok(s, pos), pos, (llama_seq_id) s, false);
        }
        return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
    };

    // both contexts decode the identical [0, p0) prefill; only ctx_roll decodes
    // the tail, which is then rolled back so its restore is pending at replay
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        ok = ok && decode_range(ctx_roll.get(), s, 0, (llama_pos) p0);
        ok = ok && decode_range(ctx_ref.get(),  s, 0, (llama_pos) p0);

        ok = ok && decode_range(ctx_roll.get(), s, (llama_pos) p0, (llama_pos) n_prompt);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0, -1);

        // a second partial removal while one is pending must be refused
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0 - 1, -1);
    }
    if (!ok) {
        LOG_ERR("%s: multi-seq prefill/rollback failed\n", __func__);
        return test_status::FAIL;
    }

    // all seqs replay in a single batch
    const auto decode_replay = [&](llama_context * ctx) {
        common_batch batch(ctx);
        for (uint32_t s = 0; s < n_seqs; ++s) {
            for (uint32_t i = 0; i < n_replay; ++i) {
                const llama_pos pos = p0 + (llama_pos) i;
                batch.add(tok(s, pos), pos, (llama_seq_id) s, true);
            }
        }
        return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
    };
    ok = decode_replay(ctx_roll.get());
    ok = ok && decode_replay(ctx_ref.get());
    if (!ok) {
        LOG_ERR("%s: multi-seq replay decode failed\n", __func__);
        return test_status::FAIL;
    }

    // both contexts decode identical batches, so the logits should match;
    // random dummy models can still drift up to ~1.7e-5, so the bound is 1e-4
    constexpr float nmse_eps = 1e-4f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    double   nmse_ab   = 0.0;
    double   nmse_a0   = 0.0;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            LOG_ERR("%s: missing multi-seq logits at index %u\n", __func__, i);
            return test_status::FAIL;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            const float diff = logit_diff(r, f);
            if (diff > 0.0f && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_ab += d*d;
                nmse_a0 += (double) r*r;
            } else {
                nmse_ab = std::numeric_limits<double>::infinity();
                nmse_a0 = 1.0;
            }
        }
    }
    const double nmse_val = nmse_a0 == 0.0 ? (nmse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_ab/nmse_a0;

    if (nmse_val > nmse_eps) {
        LOG_ERR("%s: multi-seq split replay logits mismatch (max diff %g, nmse %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, nmse_val, seq_first, pos_first);
        return test_status::FAIL;
    }

    LOG_INF("%s: multi-seq split replay matched (max diff %g, nmse %g)\n", __func__, (double) diff_max, nmse_val);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        common_batch batch_tail(ctx_ref.get());
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            batch_tail.add(tok(0, pos + 7), pos, 0, false);
        }
        ok = llama_process(ctx_ref.get(), LLAMA_PROCESS_TYPE_DECODE, batch_tail.get()) == 0;
    }

    float diff_tail = 0.0f;
    double nmse_tail_ab = 0.0;
    double nmse_tail_a0 = 0.0;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        ok = decode_one(ctx_roll.get(), tok(1, pos), pos, 1);
        ok = ok && decode_one(ctx_ref.get(), tok(1, pos), pos, 1);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            diff_tail = std::max(diff_tail, logit_diff(r, f));
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_tail_ab += d*d;
                nmse_tail_a0 += (double) r*r;
            } else {
                nmse_tail_ab = std::numeric_limits<double>::infinity();
                nmse_tail_a0 = 1.0;
            }
        }
    }
    const double nmse_tail = nmse_tail_a0 == 0.0 ? (nmse_tail_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_tail_ab/nmse_tail_a0;

    if (!ok || nmse_tail > nmse_eps) {
        LOG_ERR("%s: seq-1-only decode leaked seq 0 state (ok=%d, max diff %g, nmse %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail, nmse_tail);
        return test_status::FAIL;
    }

    LOG_INF("%s: seq-1-only decode independent of seq 0 (max diff %g, nmse %g)\n", __func__, (double) diff_tail, nmse_tail);
    return test_status::PASS;
}

// Save a rolled-back single-seq state, restore it into fresh and dirty
// contexts, and verify exact logit matches on replay.
static test_status test_rollback(const common_params & params, llama_model * model, uint8_t fill) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_ptr ctx_src = make_ctx(params, model, fill);
    llama_context_ptr ctx_dst = make_ctx(params, model, fill);
    if (!ctx_src || !ctx_dst) {
        LOG_ERR("%s: failed to init contexts\n", __func__);
        return test_status::FAIL;
    }

    if (llama_n_rs_seq(ctx_src.get()) == 0) {
        LOG_INF("%s: skipping because n_rs_seq is disabled\n", __func__);
        return test_status::SKIP;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src.get(), "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src.get());
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        LOG_INF("%s: skipping because n_rs_seq is too small\n", __func__);
        return test_status::SKIP;
    }
    if (tokens.empty()) {
        LOG_ERR("%s: not enough prompt tokens\n", __func__);
        return test_status::FAIL;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src.get(), tokens)) {
        LOG_ERR("%s: failed to decode prompt\n", __func__);
        return test_status::FAIL;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1)) {
        LOG_ERR("%s: rollback failed\n", __func__);
        return test_status::FAIL;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src.get(), 0, 0);
    ckpt.load_tgt(ctx_dst.get(), 0, 0);

    constexpr float nmse_eps = 0.0;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src.get(), tokens[pos], pos) ||
                !decode_one(ctx_dst.get(), tokens[pos], pos)) {
                LOG_ERR("%s: %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src.get(), 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst.get(), 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                LOG_ERR("%s: missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            const double nmse_val = nmse(logits_src, logits_dst, n_vocab);
            int token_first = -1;
            for (int token = 0; token < n_vocab; ++token) {
                if (logit_diff(logits_src[token], logits_dst[token]) > 0.0f && token_first < 0) {
                    token_first = token;
                }
            }
            if (nmse_val > nmse_eps) {
                LOG_ERR("%s: %s logits mismatch at position %d, first token %d, nmse %g\n",
                        __func__, mode, pos, token_first, nmse_val);
                return false;
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return test_status::FAIL;
    }

    // TODO: this test is invalid because RS rollback is only correct once after a ubatch with more than n_rs_seq tokens
    //       this is not the case here. add asserts and guardrails to prevent such attempts
    //if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
    //    !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
    //    fprintf(stderr, "%s : partial rollback failed\n", __func__);
    //    return 1;
    //}

    //constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    //common_prompt_checkpoint ckpt_partial;
    //ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    //ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    //if (!replay_and_compare("partial")) {
    //    return 1;
    //}

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context_ptr ctx_dirty = make_ctx(params, model, fill);
    if (!ctx_dirty) {
        LOG_ERR("%s: failed to init dirty ctx\n", __func__);
        return test_status::FAIL;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty.get(), noise)) {
        LOG_ERR("%s: dirty prompt decode failed\n", __func__);
        return test_status::FAIL;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty.get()), 0, rollback_pos, -1)) {
        LOG_ERR("%s: dirty rollback failed\n", __func__);
        return test_status::FAIL;
    }

    ckpt.load_tgt(ctx_dirty.get(), 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty.get(), tokens[pos], pos)) {
            LOG_ERR("%s: dirty replay failed at position %d\n", __func__, pos);
            return test_status::FAIL;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty.get(), 0);
        if (logits_dirty == nullptr) {
            LOG_ERR("%s: missing dirty logits at position %d\n", __func__, pos);
            return test_status::FAIL;
        }

        const double nmse_dirty = nmse(logits_src_replay[i].data(), logits_dirty, n_vocab);
        int token_first = -1;
        for (int token = 0; token < n_vocab; ++token) {
            if (logit_diff(logits_src_replay[i][token], logits_dirty[token]) > 0.0f && token_first < 0) {
                token_first = token;
            }
        }
        if (nmse_dirty > nmse_eps) {
            LOG_ERR("%s: dirty-ctx logits mismatch at position %d, first token %d, nmse %g\n",
                    __func__, pos, token_first, nmse_dirty);
            return test_status::FAIL;
        }
    }

    LOG_INF("%s: recurrent rollback checkpoint restored successfully\n", __func__);
    return test_status::PASS;
}

// Counts the graph nodes that only exist when a ubatch uses fewer cells than the cache span (n_rs > n_seqs).
struct graph_probe {
    int n_getter = 0; // gated delta net nodes reading their state by row id
    int n_extra  = 0; // copies that move the states of idle sequences out of the way
};

static bool graph_probe_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * probe = (graph_probe *) user_data;
    if (!ask) {
        return true;
    }
    if (t->op == GGML_OP_GATED_DELTA_NET && t->src[6]) {
        probe->n_getter++;
    }
    // build_rs: the gathered states of the rows past the ubatch go back into the cache
    if (t->op == GGML_OP_CPY && ggml_nelements(t) > 0 && t->src[0]->op == GGML_OP_VIEW &&
        t->src[0]->view_src->op == GGML_OP_GET_ROWS) {
        probe->n_extra++;
    }
    return false;
}

// Decode a prompt into seq 0, share its cells with a second sequence, then keep
// decoding both.
static test_status test_shared_seq_reserve(const common_params & params, llama_model * model, uint8_t fill) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // these archs reserve the final pp graph with n_seqs = 1, so every multi-seq
    // graph has a different layout and re-reserves by design
    // see [TAG_RESERVE_DIAG_DECAY] in llama-context.cpp
    char arch_str[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch_str, sizeof(arch_str));
    if (strcmp(arch_str, "kimi-linear") == 0 || strcmp(arch_str, "minimax-01") == 0) {
        LOG_INF("%s: skipping %s, its reserve uses n_seqs = 1\n", __func__, arch_str);
        return test_status::SKIP;
    }

    constexpr uint32_t n_seqs     = 2;
    constexpr uint32_t n_prompt   = 128;
    constexpr uint32_t n_continue = 32;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_seqs;
    cparams.n_rs_seq   = 8;
    cparams.n_ctx      = 512;
    cparams.n_batch    = 256;
    cparams.n_ubatch   = 64;
    cparams.kv_unified = true; // only a unified cache shares cells on seq_cp

    llama_context_ptr ctx = init_ctx(model, cparams, fill);
    if (!ctx) {
        LOG_ERR("%s: failed to init context\n", __func__);
        return test_status::FAIL;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    {
        common_batch batch(ctx.get());
        for (llama_pos pos = 0; pos < (llama_pos) n_prompt; ++pos) {
            batch.add(tok(0, pos), pos, 0, false);
        }
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: prompt decode failed\n", __func__);
            return test_status::FAIL;
        }
    }

    // this is what llama-batched-bench does for -pps
    llama_memory_seq_cp(llama_get_memory(ctx.get()), 0, 1, -1, -1);

    // One sequence per ubatch while the other stays idle. After seq_cp both share a cell, so this does not make n_rs
    // exceed n_seqs (test_idle_slot_state covers that); it covers decoding and reserving on a shared cell.
    for (uint32_t s = 0; s < n_seqs; ++s) {
        if (!decode_one(ctx.get(), tok(s, n_prompt), n_prompt, s)) {
            LOG_ERR("%s: single-seq decode after seq_cp failed\n", __func__);
            return test_status::FAIL;
        }
    }

    for (uint32_t i = 1; i < n_continue; ++i) {
        const llama_pos pos = (llama_pos) (n_prompt + i);

        common_batch batch(ctx.get());
        for (uint32_t s = 0; s < n_seqs; ++s) {
            batch.add(tok(s, pos), pos, (llama_seq_id) s, true);
        }
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: shared-seq decode failed at step %u\n", __func__, i);
            return test_status::FAIL;
        }

        for (uint32_t s = 0; s < n_seqs; ++s) {
            const float * logits = llama_get_logits_ith(ctx.get(), (int) s);
            if (logits == nullptr) {
                LOG_ERR("%s: missing shared-seq logits at index %u\n", __func__, s);
                return test_status::FAIL;
            }
            for (int t = 0; t < n_vocab; ++t) {
                if (!std::isfinite(logits[t])) {
                    LOG_ERR("%s: non-finite shared-seq logit at step %u, seq %u, index %d\n", __func__, i, s, t);
                    return test_status::FAIL;
                }
            }
        }
    }

    LOG_INF("%s: shared-seq decode succeeded (%u tokens after seq_cp)\n", __func__, n_continue*n_seqs);
    return test_status::PASS;
}

// Two slots decode in one ubatch next to a slot that holds a state but does not decode. With the idle slot between the
// active ones their cells are not adjacent, so the cache span exceeds the ubatch (n_rs > n_seqs): the state of the idle
// slot has to survive the gathers, the multi-token verify, the rollback and the replay of its neighbours. It is checked
// where that can be exact, when the idle slot resumes alone against a context that only ever held it, and the logits
// of another slot's state must read as a mismatch. Without the idle slot the same script is the control
// (n_rs == n_seqs); the graph probe proves which shape each run exercised.
//
// Multi-sequence ubatches are not bitwise equal to single-sequence ones on the CPU fixture (nmse up to 4e-3, the same
// on the base commit), so they are not compared with the reference contexts. What an idle slot must not change is
// compared exactly instead: the logits of the active slots equal those of the control layout.
using idle_logits = std::vector<std::pair<std::string, std::vector<float>>>;

static test_status test_idle_slot_state(const common_params & params, llama_model * model, uint8_t fill, bool with_idle, idle_logits & recorded) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    constexpr uint32_t n_active   = 2;
    constexpr uint32_t n_prompt   = 24;
    constexpr uint32_t n_steps    = 4;
    constexpr uint32_t n_rollback = 2;
    constexpr uint32_t n_resume   = 4;
    constexpr float    nmse_eps   = 1e-6f;

    const uint32_t n_slots = with_idle ? 3 : 2;
    const uint32_t n_idle  = 1; // slot index of the idle sequence
    const uint32_t active[n_active] = { 0, with_idle ? 2u : 1u };
    const char * layout = with_idle ? "idle slot between" : "adjacent slots";

    graph_probe probe;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_slots;
    cparams.n_rs_seq   = 4;
    cparams.n_ctx      = 512;
    cparams.n_batch    = 64;
    cparams.n_ubatch   = 32;
    cparams.kv_unified = true;
    cparams.cb_eval           = graph_probe_cb;
    cparams.cb_eval_user_data = &probe;

    llama_context_ptr ctx = init_ctx(model, cparams, fill);
    if (!ctx) {
        LOG_ERR("%s: failed to init context\n", __func__);
        return test_status::FAIL;
    }
    if (llama_n_rs_seq(ctx.get()) < n_rollback) {
        LOG_INF("%s: skipping because n_rs_seq is too small\n", __func__);
        return test_status::SKIP;
    }

    // one context per slot that only ever holds that slot's sequence
    auto cparams_ref = cparams;
    cparams_ref.n_seq_max         = 1;
    cparams_ref.cb_eval           = nullptr;
    cparams_ref.cb_eval_user_data = nullptr;
    std::vector<llama_context_ptr> ref;
    for (uint32_t s = 0; s < n_slots; ++s) {
        ref.push_back(init_ctx(model, cparams_ref, fill));
        if (!ref.back()) {
            LOG_ERR("%s: failed to init the reference context %u\n", __func__, s);
            return test_status::FAIL;
        }
    }

    // the token stream follows the position in the active list, so both layouts feed the same two streams
    const auto tok = [&](uint32_t slot, llama_pos pos) {
        const uint32_t stream = slot == n_idle && with_idle ? 2 : slot == active[1] ? 1 : 0;
        return (llama_token) ((7*(uint32_t) pos + 31*stream + 1) % (uint32_t) n_vocab);
    };

    struct item {
        uint32_t  slot;
        llama_pos pos;
    };

    // all items of a batch ask for logits; item i of the batch is output i
    const auto decode = [&](llama_context * c, const std::vector<item> & items, bool is_ref) {
        common_batch batch(c);
        for (const auto & it : items) {
            batch.add(tok(it.slot, it.pos), it.pos, is_ref ? 0 : (llama_seq_id) it.slot, true);
        }
        return llama_process(c, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
    };

    double nmse_worst = 0.0;
    int    n_bad      = 0;

    // multi: the ubatch held several sequences, so the logits are only recorded for the layout comparison;
    // single-sequence ubatches have to match the reference context
    const auto compare = [&](const char * what, uint32_t slot, llama_pos pos, int out_ctx, int out_ref, bool multi) {
        const float * a = llama_get_logits_ith(ctx.get(), out_ctx);
        const float * b = llama_get_logits_ith(ref[slot].get(), out_ref);
        if (a == nullptr || b == nullptr) {
            LOG_ERR("%s: missing logits (%s, slot %u, pos %d)\n", __func__, what, slot, pos);
            n_bad++;
            return;
        }
        if (multi) {
            recorded.push_back({ std::string(what) + " stream " + std::to_string(slot == active[1] ? 1 : 0) + " pos " + std::to_string(pos),
                                 std::vector<float>(a, a + n_vocab) });
            return;
        }
        const double d = nmse(b, a, n_vocab);
        nmse_worst = std::max(nmse_worst, d);
        if (d > nmse_eps) {
            LOG_ERR("%s: %s: %s, slot %u, pos %d: nmse %g vs a context without the other slots\n", __func__, layout, what, slot, pos, d);
            n_bad++;
        }
    };

    // prompts: one slot at a time, so slot s gets cell s
    for (uint32_t s = 0; s < n_slots; ++s) {
        std::vector<item> items;
        for (llama_pos pos = 0; pos < (llama_pos) n_prompt; ++pos) {
            items.push_back({ s, pos });
        }
        if (!decode(ctx.get(), items, false) || !decode(ref[s].get(), items, true)) {
            LOG_ERR("%s: prompt decode failed\n", __func__);
            return test_status::FAIL;
        }
        compare("prompt", s, n_prompt - 1, (int) n_prompt - 1, (int) n_prompt - 1, false);
    }

    probe = {};

    // the active slots advance together while the idle slot waits
    for (uint32_t i = 0; i < n_steps; ++i) {
        const llama_pos pos = n_prompt + i;
        if (!decode(ctx.get(), { { active[0], pos }, { active[1], pos } }, false)) {
            LOG_ERR("%s: mixed decode failed at step %u\n", __func__, i);
            return test_status::FAIL;
        }
        for (uint32_t a = 0; a < n_active; ++a) {
            if (!decode(ref[active[a]].get(), { { active[a], pos } }, true)) {
                LOG_ERR("%s: reference decode failed\n", __func__);
                return test_status::FAIL;
            }
            compare("active step", active[a], pos, (int) a, 0, true);
        }
    }

    // MTP-style verify: one ubatch of n_rollback + 1 tokens per active slot, then one slot drops the tail of it.
    // Rollback only reaches back into the last ubatch, so the snapshots must come from this one.
    const llama_pos p_verify = n_prompt + n_steps;
    {
        std::vector<item> verify[n_active];
        common_batch batch(ctx.get());
        for (uint32_t a = 0; a < n_active; ++a) {
            for (uint32_t i = 0; i <= n_rollback; ++i) {
                verify[a].push_back({ active[a], p_verify + (llama_pos) i });
                batch.add(tok(active[a], p_verify + (llama_pos) i), p_verify + (llama_pos) i, (llama_seq_id) active[a], true);
            }
        }
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: verify decode failed\n", __func__);
            return test_status::FAIL;
        }
        for (uint32_t a = 0; a < n_active; ++a) {
            if (!decode(ref[active[a]].get(), verify[a], true)) {
                LOG_ERR("%s: reference verify failed\n", __func__);
                return test_status::FAIL;
            }
            for (uint32_t i = 0; i <= n_rollback; ++i) {
                compare("verify", active[a], verify[a][i].pos, (int) (a*(n_rollback + 1) + i), (int) i, true);
            }
        }
    }

    // the second active slot drops the last n_rollback tokens while the idle slot still holds its cell, then replays
    // them in a multi-token ubatch next to the first one, which moves on
    {
        const llama_pos p0 = p_verify + 1;
        if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), (llama_seq_id) active[1], p0, -1) ||
            !llama_memory_seq_rm(llama_get_memory(ref[active[1]].get()), 0, p0, -1)) {
            LOG_ERR("%s: rollback failed\n", __func__);
            return test_status::FAIL;
        }
        std::vector<item> replay[n_active];
        for (uint32_t i = 0; i < n_rollback; ++i) {
            replay[0].push_back({ active[0], (llama_pos) (p_verify + n_rollback + 1 + i) });
            replay[1].push_back({ active[1], p0 + (llama_pos) i });
        }
        common_batch batch(ctx.get());
        for (uint32_t a = 0; a < n_active; ++a) {
            for (const auto & it : replay[a]) {
                batch.add(tok(it.slot, it.pos), it.pos, (llama_seq_id) it.slot, true);
            }
        }
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: replay decode failed\n", __func__);
            return test_status::FAIL;
        }
        for (uint32_t a = 0; a < n_active; ++a) {
            if (!decode(ref[active[a]].get(), replay[a], true)) {
                LOG_ERR("%s: reference replay failed\n", __func__);
                return test_status::FAIL;
            }
            for (uint32_t i = 0; i < n_rollback; ++i) {
                compare("rollback replay", active[a], replay[a][i].pos, (int) (a*n_rollback + i), (int) i, true);
            }
        }
    }

    // both active slots in one more ubatch
    {
        const llama_pos pos0 = p_verify + n_rollback + 1 + n_rollback;
        const llama_pos pos2 = pos0 - (llama_pos) n_rollback;
        const std::vector<item> both = { { active[0], pos0 }, { active[1], pos2 } };
        if (!decode(ctx.get(), both, false)) {
            LOG_ERR("%s: final decode failed\n", __func__);
            return test_status::FAIL;
        }
        for (size_t i = 0; i < both.size(); ++i) {
            if (!decode(ref[both[i].slot].get(), { both[i] }, true)) {
                LOG_ERR("%s: final reference decode failed\n", __func__);
                return test_status::FAIL;
            }
            compare("final", both[i].slot, both[i].pos, (int) i, 0, true);
        }
    }

    const int n_extra_active  = probe.n_extra;
    const int n_getter_active = probe.n_getter;

    // the idle slot resumes where it stopped, alone: its state went through the gathers, the rollback and the
    // relocations of its neighbours, and must still be what its prompt left
    double wrong = 0.0;
    if (with_idle) {
        for (uint32_t i = 0; i < n_resume; ++i) {
            const llama_pos pos = n_prompt + (llama_pos) i;
            if (!decode(ctx.get(), { { n_idle, pos } }, false) || !decode(ref[n_idle].get(), { { n_idle, pos } }, true)) {
                LOG_ERR("%s: resume decode failed\n", __func__);
                return test_status::FAIL;
            }
            compare("idle slot resume", n_idle, pos, 0, 0, false);
        }
        // negative control: the logits of another slot's state must read as a mismatch
        wrong = nmse(llama_get_logits_ith(ref[active[0]].get(), 0), llama_get_logits_ith(ctx.get(), 0), n_vocab);
    }

    if (n_bad > 0) {
        LOG_ERR("%s: %s: %d logit comparisons failed (worst nmse %g)\n", __func__, layout, n_bad, nmse_worst);
        return test_status::FAIL;
    }

    if (with_idle && !(wrong > 100.0*nmse_eps)) {
        LOG_ERR("%s: %s: the comparison cannot tell two slots apart (nmse %g)\n", __func__, layout, wrong);
        return test_status::FAIL;
    }

    // engagement: the graphs under test existed, and the idle slot is what made the span larger than the ubatch
    if (n_getter_active == 0 || (n_extra_active > 0) != with_idle) {
        LOG_ERR("%s: %s: %d getter nodes, %d extra-state copies in the mixed ubatches\n", __func__, layout, n_getter_active, n_extra_active);
        return test_status::FAIL;
    }

    LOG_INF("%s: %s: single-sequence worst nmse %g (another slot reads as %g), %d getter nodes, %d extra-state copies in the mixed ubatches\n",
            __func__, layout, nmse_worst, wrong, n_getter_active, n_extra_active);
    return test_status::PASS;
}

// Runs the script with and without the idle slot; the active slots must see bitwise the same logits in both.
static test_status test_idle_slot_pair(const common_params & params, llama_model * model, uint8_t fill) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    idle_logits control;
    idle_logits idle;
    const test_status s0 = test_idle_slot_state(params, model, fill, false, control);
    const test_status s1 = test_idle_slot_state(params, model, fill, true,  idle);
    if (s0 == test_status::FAIL || s1 == test_status::FAIL) {
        return test_status::FAIL;
    }
    if (s0 == test_status::SKIP || s1 == test_status::SKIP) {
        return test_status::SKIP;
    }

    constexpr double nmse_exact = 1e-9;
    double worst = 0.0;
    int    n_bad = 0;
    if (control.size() != idle.size() || control.empty()) {
        LOG_ERR("%s: the two layouts recorded %zu and %zu logit sets\n", __func__, control.size(), idle.size());
        return test_status::FAIL;
    }
    for (size_t i = 0; i < control.size(); ++i) {
        const double d = nmse(control[i].second.data(), idle[i].second.data(), n_vocab);
        worst = std::max(worst, d);
        if (control[i].first != idle[i].first || d > nmse_exact) {
            LOG_ERR("%s: %s: nmse %g between the layouts\n", __func__, control[i].first.c_str(), d);
            n_bad++;
        }
    }
    if (n_bad > 0) {
        LOG_ERR("%s: %d of %zu active-slot logit sets changed with an idle slot present (worst nmse %g)\n", __func__, n_bad, control.size(), worst);
        return test_status::FAIL;
    }

    // negative control: the two streams must differ, or equal logits across layouts prove nothing
    double apart = 0.0;
    for (size_t i = 0; i + 1 < control.size(); ++i) {
        if (control[i].first.find("stream 0") != std::string::npos && control[i + 1].first.find("stream 1") != std::string::npos) {
            apart = std::max(apart, nmse(control[i].second.data(), control[i + 1].second.data(), n_vocab));
        }
    }
    if (!(apart > 1e-3)) {
        LOG_ERR("%s: the two streams are indistinguishable (nmse %g)\n", __func__, apart);
        return test_status::FAIL;
    }

    LOG_INF("%s: %zu active-slot logit sets identical with and without the idle slot (worst nmse %g, streams differ by %g)\n",
            __func__, control.size(), worst, apart);
    return test_status::PASS;
}

static test_status merge_status(test_status a, test_status b) {
    if (a == test_status::FAIL || b == test_status::FAIL) {
        return test_status::FAIL;
    }
    if (a == test_status::PASS || b == test_status::PASS) {
        return test_status::PASS;
    }
    return test_status::SKIP;
}

struct test_results {
    test_status rollback = test_status::SKIP;
    test_status replay   = test_status::SKIP;
    test_status shared   = test_status::SKIP;
    test_status idle     = test_status::SKIP;
};

// Run every test for an initialized model over both cache fills.
static test_results run_tests(const common_params & params, llama_model * model) {
    test_results res;
    for (uint8_t fill : { 0, 0x3e }) {
        LOG_INF("%s: testing with cache fill 0x%02x\n", __func__, fill);
        const test_status rb = test_rollback(params, model, fill);
        const test_status rp = test_multi_seq_split_replay(params, model, fill);
        const test_status ss = test_shared_seq_reserve(params, model, fill);
        const test_status is = test_idle_slot_pair(params, model, fill);
        res.rollback = merge_status(res.rollback, rb);
        res.replay   = merge_status(res.replay,   rp);
        res.shared   = merge_status(res.shared,   ss);
        res.idle     = merge_status(res.idle,     is);
        if (rb == test_status::FAIL || rp == test_status::FAIL || ss == test_status::FAIL || is == test_status::FAIL) {
            break;
        }
    }
    return res;
}

// Run the tests for a single model file.
// Returns the per-test statuses.
static test_results run_tests_for_model(const std::string & model_path, const struct common_params & base_params) {
    struct common_params params = base_params;
    params.model.path = model_path;

    auto llama_init = common_init_from_params(params, true);
    auto * model = llama_init->model();

    if (model == nullptr) {
        LOG_ERR("%s: failed to init model '%s'\n", __func__, model_path.c_str());
        // a model that cannot be loaded is a failure, not a skip
        return { test_status::FAIL, test_status::FAIL, test_status::FAIL, test_status::FAIL };
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        LOG_INF("%s: skipping for non-recurrent model\n", __func__);
        return {};
    }

    return run_tests(params, model);
}

static void print_usage(int /* argc */, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n  %s -m your_model.gguf\n", argv[0]);
    LOG("\n  %s --models tests/test-models\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    // extract our own --models DIR option before handing the rest to the common arg parser
    std::string models_dir;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --models requires a directory argument\n", __func__);
                return 1;
            }
            models_dir = argv[i + 1];
            i++;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    filtered_argv.push_back(nullptr);
    const int fargc = (int)filtered_argv.size() - 1;

    // in --models mode there is no single model; set a placeholder so the common parser's
    // "--model is required" check passes (each model is set individually inside the loop)
    if (!models_dir.empty()) {
        params.model.path = models_dir;
    }

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    llama_backend_init();

    if (!models_dir.empty()) {
        // run every test over each dummy model in the directory
        if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
            LOG_ERR("%s: models directory '%s' does not exist\n", __func__, models_dir.c_str());
            return 1;
        }

        std::vector<std::string> models;
        for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                models.push_back(entry.path().string());
            }
        }
        std::sort(models.begin(), models.end());

        if (models.empty()) {
            LOG_ERR("%s: no .gguf models found in '%s'\n", __func__, models_dir.c_str());
            return 1;
        }

        size_t name_width = 5; // "Model"
        for (const auto & model_path : models) {
            name_width = std::max(name_width, std::filesystem::path(model_path).filename().string().size());
        }

        // silence everything but the table itself (LOG has verbosity LOG_LEVEL_OUTPUT = 0)
        common_log_set_verbosity_thold(0);

        LOG("%-*s  %-8s  %-11s  %-10s  %s\n", (int) name_width, "Model", "rollback", "split replay", "shared seq", "idle slot");
        common_log_flush(common_log_main());

        size_t n_pass[4] = { 0, 0, 0, 0 };
        size_t n_skip[4] = { 0, 0, 0, 0 };
        size_t n_fail[4] = { 0, 0, 0, 0 };
        for (const auto & model_path : models) {
            const auto name = std::filesystem::path(model_path).filename().string();

            LOG("%-*s", (int) name_width, name.c_str());

            const test_results res = run_tests_for_model(model_path, params);

            // all status strings have the same raw length, so the columns line up;
            // pad the first status to the width of the "rollback" header + separator
            LOG("  %s      %s         %s        %s", test_status_str(res.rollback), test_status_str(res.replay), test_status_str(res.shared), test_status_str(res.idle));
            LOG("\n");
            common_log_flush(common_log_main());

            const test_status all[4] = { res.rollback, res.replay, res.shared, res.idle };
            for (int t = 0; t < 4; ++t) {
                switch (all[t]) {
                    case test_status::PASS: n_pass[t]++; break;
                    case test_status::FAIL: n_fail[t]++; break;
                    case test_status::SKIP: n_skip[t]++; break;
                }
            }
        }

        common_log_set_verbosity_thold(LOG_DEFAULT_LLAMA);
        common_log_flush(common_log_main());

        LOG_INF("%s: rollback:     %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, n_pass[0], n_skip[0], n_fail[0], models.size());
        LOG_INF("%s: split replay: %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, n_pass[1], n_skip[1], n_fail[1], models.size());
        LOG_INF("%s: shared seq:   %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, n_pass[2], n_skip[2], n_fail[2], models.size());
        LOG_INF("%s: idle slot:    %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, n_pass[3], n_skip[3], n_fail[3], models.size());

        return (n_fail[0] + n_fail[1] + n_fail[2] + n_fail[3]) == 0 ? 0 : 1;
    }

    // single-model mode
    const test_results res = run_tests_for_model(params.model.path, params);

    return (res.rollback == test_status::FAIL || res.replay == test_status::FAIL || res.shared == test_status::FAIL ||
            res.idle == test_status::FAIL) ? 1 : 0;
}
