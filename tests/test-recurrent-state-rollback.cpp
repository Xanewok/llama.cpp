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
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <random>
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
    constexpr uint32_t  n_prompt   = 20;
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

    // Keep the anchor at p0 - 1 when rolling back the tail.
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        ok = ok && decode_range(ctx_roll.get(), s, 0, (llama_pos) p0 - 1);
        ok = ok && decode_range(ctx_ref.get(),  s, 0, (llama_pos) p0 - 1);

        ok = ok && decode_range(ctx_roll.get(), s, (llama_pos) p0 - 1, (llama_pos) n_prompt);
        ok = ok && decode_range(ctx_ref.get(),  s, (llama_pos) p0 - 1, (llama_pos) p0);

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

// Counts gated delta net nodes by how they read their state (row ids or a gathered copy) and the recurrent state rows
// that get_rows copies.
struct gdn_probe {
    int     n_indexed       = 0;
    int     n_plain         = 0;
    int     n_indexed_multi = 0;
    int     n_plain_single  = 0;
    int64_t n_gathered      = 0;
    std::set<ggml_tensor *> rotations;
    bool rotation_ok = true;
};

static bool gdn_probe_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * probe = (gdn_probe *) user_data;
    int32_t hint;
    memcpy(&hint, t->op_params + 1, sizeof(hint));
    const bool rotation = t->op == GGML_OP_MUL_MAT && hint == GGML_HINT_SRC0_IS_HADAMARD;
    if (!ask) {
        if (rotation && probe->rotations.insert(t->src[0]).second) {
            const auto * matrix = t->src[0];
            const int64_t n = matrix->ne[0];
            std::vector<float> values(n*n);
            ggml_backend_tensor_get(matrix, values.data(), 0, ggml_nbytes(matrix));
            const float magnitude = 1.0 / sqrtf(n);
            for (int64_t row = 0; row < n; ++row) {
                for (int64_t col = 0; col < n; ++col) {
                    int parity = 0;
                    for (int64_t bits = row & col; bits; bits >>= 1) {
                        parity ^= bits & 1;
                    }
                    const float expected = parity ? -magnitude : magnitude;
                    probe->rotation_ok &= memcmp(&values[row*n + col], &expected, sizeof(float)) == 0;
                }
            }
        }
        return true;
    }
    if (t->op == GGML_OP_GATED_DELTA_NET) {
        (t->src[6] ? probe->n_indexed : probe->n_plain)++;
        if (t->src[6] && t->src[2]->ne[3] > 1) {
            ++probe->n_indexed_multi;
        }
        if (!t->src[6] && t->src[2]->ne[3] == 1) {
            ++probe->n_plain_single;
        }
    }
    const ggml_tensor * src = t->op == GGML_OP_GET_ROWS ? t->src[0] : nullptr;
    if (src && src->view_src) {
        src = src->view_src;
    }
    if (src && strncmp(src->name, "cache_s_l", 9) == 0) {
        probe->n_gathered += t->ne[1];
    }
    return rotation;
}

// Compare getter and gather builds on identical batches; single-sequence batches can use different arithmetic.
static test_status test_getter_gate(const common_params & params, llama_model * model, uint8_t fill,
                                   FILE * logits_out = nullptr, FILE * logits_ref = nullptr, bool gathered_multi = false) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    constexpr uint32_t n_ubatch = 16;

    gdn_probe probe;
    const auto make_ctx_gate = [&](uint32_t n_seq) {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seq;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256*n_seq;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        cparams.cb_eval           = gdn_probe_cb;
        cparams.cb_eval_user_data = &probe;
        return init_ctx(model, cparams, fill);
    };

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((11*(uint32_t) pos + 29*seq + 3) % (uint32_t) n_vocab);
    };

    enum class form { BY_ID, GATHERED };
    // the streams share one batch in order, tokens from out_from on request logits
    struct phase {
        const char *          name;
        std::vector<uint32_t> streams;
        llama_pos             p0, p1, out_from;
        form                  expect;
    };
    struct scenario {
        const char *       name;
        uint32_t           n_seq;
        std::vector<phase> phases;
    };
    const std::vector<scenario> scenarios = {
        { "two slots", 2, {
            { "prefill seq 0",             { 0 },    0, 19, 18, form::BY_ID }, // two ubatches of one sequence
            { "prefill seq 1",             { 1 },    0, 19, 18, form::BY_ID },
            { "verify batch, both seqs",   { 0, 1 }, 19, 25, 19, form::BY_ID }, // n_seqs = n_rs = 2, several tokens each
            { "decode, both seqs",         { 0, 1 }, 25, 26, 25, form::BY_ID }, // one token each
            { "decode, both seqs again",   { 0, 1 }, 26, 27, 26, form::BY_ID },
            { "seq 1 beside an idle slot", { 1 },    27, 28, 27, form::BY_ID },
        } },
        // seq 1 takes the first cell, so the joint batch swaps the cells: each sequence reads the cell the other writes
        { "swapped cells", 2, {
            { "prefill seq 1",             { 1 },    0, 19, 18, form::BY_ID },
            { "prefill seq 0",             { 0 },    0, 19, 18, form::BY_ID },
            { "verify batch, both seqs",   { 0, 1 }, 19, 25, 19, form::BY_ID },
            { "decode, both seqs",         { 0, 1 }, 25, 26, 25, form::BY_ID },
        } },
        // both start from the shared zero state in one batch
        { "joint start", 2, {
            { "prefill both seqs",         { 0, 1 }, 0, 12, 11, form::BY_ID },
            { "decode, both seqs",         { 0, 1 }, 12, 13, 12, form::BY_ID },
        } },
        // seq 2 takes the cell between seqs 0 and 1, so their joint batch holds its state as an extra state the gather
        // must move (without a unified KV cache a ubatch joins only consecutive sequence ids)
        { "extra state", 3, {
            { "prefill seq 0",              { 0 },    0, 19, 18, form::BY_ID },
            { "prefill seq 2",              { 2 },    0, 19, 18, form::BY_ID },
            { "prefill seq 1",              { 1 },    0, 19, 18, form::BY_ID },
            { "verify batch, seqs 0 and 1", { 0, 1 }, 19, 25, 19, form::GATHERED },
            { "seq 2 after the move",       { 2 },    19, 25, 19, form::BY_ID },
            { "decode, seqs 0 and 1",       { 0, 1 }, 25, 26, 25, form::BY_ID },
        } },
    };

    int n_gdn = 0;
    uint32_t scenario_id = 0;
    std::vector<float> reference(n_vocab);
    for (const auto & sc : scenarios) {
        llama_context_ptr ctx_gate = make_ctx_gate(sc.n_seq);
        if (!ctx_gate) {
            LOG_ERR("%s: %s: failed to init context\n", __func__, sc.name);
            return test_status::FAIL;
        }
        if (llama_n_rs_seq(ctx_gate.get()) == 0) {
            LOG_INF("%s: skipping because n_rs_seq is disabled\n", __func__);
            return test_status::SKIP;
        }

        uint32_t phase_id = 0;
        for (const auto & ph : sc.phases) {
            probe = {};
            common_batch batch(ctx_gate.get());
            for (uint32_t s : ph.streams) {
                for (llama_pos pos = ph.p0; pos < ph.p1; ++pos) {
                    batch.add(tok(s, pos), pos, (llama_seq_id) s, pos >= ph.out_from);
                }
            }
            if (llama_process(ctx_gate.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                LOG_ERR("%s: %s: %s: decode failed\n", __func__, sc.name, ph.name);
                return test_status::FAIL;
            }
            llama_synchronize(ctx_gate.get());
            if (!probe.rotation_ok) {
                LOG_ERR("%s: rotation matrix bytes differ from the Hadamard reference\n", __func__);
                return test_status::FAIL;
            }
            if ((ggml_is_quantized(params.cache_type_k) || ggml_is_quantized(params.cache_type_v)) && probe.rotations.empty()) {
                LOG_ERR("%s: quantized KV did not read a rotation matrix\n", __func__);
                return test_status::FAIL;
            }
            LOG_INF("%s: %s: %s: gdn nodes %d by id, %d gathered; state rows gathered %" PRId64 "\n", __func__,
                    sc.name, ph.name, probe.n_indexed, probe.n_plain, probe.n_gathered);

            for (size_t j = 0; j < ph.streams.size(); ++j) {
                const uint32_t s = ph.streams[j];
                const int32_t first = (int32_t) j*(ph.p1 - ph.p0) - ph.p0; // batch index of pos is first + pos
                for (llama_pos pos = ph.out_from; pos < ph.p1; ++pos) {
                    const float * l_gate = llama_get_logits_ith(ctx_gate.get(), first + pos);
                    if (!l_gate || !std::all_of(l_gate, l_gate + n_vocab, [](float v) { return std::isfinite(v); })) {
                        LOG_ERR("%s: %s: %s: seq %u pos %d missing or non-finite logits\n", __func__, sc.name, ph.name, s, (int) pos);
                        return test_status::FAIL;
                    }
                    const uint32_t key[] = { fill, scenario_id, phase_id, s, (uint32_t) pos, (uint32_t) n_vocab };
                    if (logits_out && (fwrite(key, sizeof(key), 1, logits_out) != 1 ||
                                       fwrite(l_gate, sizeof(float), n_vocab, logits_out) != (size_t) n_vocab)) {
                        LOG_ERR("%s: failed to write getter logits\n", __func__);
                        return test_status::FAIL;
                    }
                    if (logits_ref) {
                        uint32_t key_ref[6];
                        if (fread(key_ref, sizeof(key_ref), 1, logits_ref) != 1 || memcmp(key, key_ref, sizeof(key)) != 0 ||
                            fread(reference.data(), sizeof(float), n_vocab, logits_ref) != (size_t) n_vocab ||
                            memcmp(l_gate, reference.data(), n_vocab*sizeof(float)) != 0) {
                            LOG_ERR("%s: %s: %s: seq %u pos %d logits differ from the gathered same-batch reference\n",
                                    __func__, sc.name, ph.name, s, (int) pos);
                            return test_status::FAIL;
                        }
                    }
                }
            }

            ++phase_id;
            n_gdn += probe.n_indexed + probe.n_plain;
            if (probe.n_indexed + probe.n_plain == 0) {
                continue;
            }
            // A phase with several streams can split into single-sequence ubatches.
            if (gathered_multi) {
                if (probe.n_indexed_multi != 0 || probe.n_plain_single != 0 || (probe.n_plain == 0 && probe.n_gathered != 0)) {
                    LOG_ERR("%s: %s: %s: control must gather only for multi-sequence ubatches\n", __func__, sc.name, ph.name);
                    return test_status::FAIL;
                }
                continue;
            }
            if (ph.expect == form::BY_ID && (probe.n_indexed == 0 || probe.n_plain != 0 || probe.n_gathered != 0)) {
                LOG_ERR("%s: %s: %s: ubatches without extra states must read the state by id and gather none\n",
                        __func__, sc.name, ph.name);
                return test_status::FAIL;
            }
            if (ph.expect == form::GATHERED && (probe.n_plain == 0 || probe.n_indexed != 0)) {
                LOG_ERR("%s: %s: %s: ubatches with extra states must keep the gathered state\n", __func__, sc.name, ph.name);
                return test_status::FAIL;
            }
        }
        ++scenario_id;
    }

    if (n_gdn == 0) {
        LOG_INF("%s: skipping because the model has no gated delta net\n", __func__);
        return test_status::SKIP;
    }
    LOG_INF("%s: all scenarios pass%s\n", __func__, logits_ref ? " with exact same-batch logits" : " structure and finite-logit checks");
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

    for (uint32_t i = 0; i < n_continue; ++i) {
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

struct selected_recurrent_state : llama_io_write_i {
    struct span {
        ggml_tensor * tensor;
        size_t offset;
        std::vector<float> data;
    };
    std::map<std::string, span> states;
    size_t size = 0;

    void write(const void *, size_t n) override { size += n; }
    size_t n_bytes() override { return size; }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t n) override {
        const std::string name = ggml_get_name(tensor);
        if (name.find("cache_r_l") == 0 || name.find("cache_s_l") == 0) {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32 && n % sizeof(float) == 0);
            span value = { tensor, offset, std::vector<float>(n / sizeof(float)) };
            ggml_backend_tensor_get(tensor, value.data.data(), offset, n);
            states.emplace(name, std::move(value));
        }
        size += n;
    }
};

static selected_recurrent_state selected_state(llama_context * ctx, llama_seq_id seq = 0) {
    llama_synchronize(ctx);
    selected_recurrent_state state;
    llama_get_memory(ctx)->state_write(state, seq);
    return state;
}

static bool states_match(const selected_recurrent_state & a, const selected_recurrent_state & b) {
    if (a.states.empty() || a.states.size() != b.states.size()) {
        return false;
    }
    for (const auto & [name, value] : a.states) {
        const auto it = b.states.find(name);
        if (it == b.states.end() || value.data.size() != it->second.data.size() ||
            nmse(value.data.data(), it->second.data.data(), value.data.size()) > 1e-4) {
            return false;
        }
    }
    return true;
}

static uint64_t hash_bytes(uint64_t hash, const void * ptr, size_t n) {
    const auto * bytes = static_cast<const uint8_t *>(ptr);
    for (size_t i = 0; i < n; ++i) {
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

// Compare these fingerprints between frozen builds with the same workload and backend.
static test_status test_state_movers(const common_params & params, llama_model * model) {
    const int n_seq = params.n_parallel;
    if (n_seq < 1 || n_seq > 2) {
        return test_status::FAIL;
    }
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = n_seq;
    cparams.n_rs_seq = 8;
    cparams.n_batch = std::max(cparams.n_batch, uint32_t(6 * n_seq));
    cparams.n_ubatch = std::max(cparams.n_ubatch, uint32_t(6 * n_seq));
    auto ctx = init_ctx(model, cparams, 0);
    if (!ctx || llama_n_rs_seq(ctx.get()) < 6) {
        return test_status::FAIL;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto prompt = common_tokenize(ctx.get(), "The quick brown fox", true);
    auto prefill = [&]() {
        common_batch batch(ctx.get());
        for (int seq = 0; seq < n_seq; ++seq) {
            for (size_t i = 0; i < prompt.size(); ++i) {
                batch.add(prompt[i], i, seq, false);
            }
        }
        return llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
    };
    if (!prefill()) {
        return test_status::FAIL;
    }
    std::mt19937 rng(1729);
    std::vector<int> accepts = {1, 2, 3, 4, 5, 6};
    std::shuffle(accepts.begin(), accepts.end(), rng);
    accepts.push_back(1 + rng() % 6);
    accepts.push_back(6);
    llama_pos pos = prompt.size();
    bool wrong_index_detected = false;
    bool dropped_snapshot_detected = false;
    for (size_t trial = 0; trial < accepts.size(); ++trial) {
        const int accepted = accepts[trial];
        common_batch batch(ctx.get());
        for (int seq = 0; seq < n_seq; ++seq) {
            for (int i = 0; i < 6; ++i) {
                batch.add((31 * (pos + i) + 7 + seq) % n_vocab, pos + i, seq, false);
            }
        }
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            return test_status::FAIL;
        }
        std::vector<selected_recurrent_state> actuals;
        for (int seq = 0; seq < n_seq; ++seq) {
            if (accepted < 6 && !llama_memory_seq_rm(llama_get_memory(ctx.get()), seq, pos + accepted, -1)) {
                return test_status::FAIL;
            }
            actuals.push_back(selected_state(ctx.get(), seq));
        }
        const auto & actual = actuals[0];
        if (actual.states.empty()) {
            return test_status::FAIL;
        }
        if (accepted < 6 && !wrong_index_detected) {
            auto wrong = actual;
            for (auto & [name, value] : wrong.states) {
                const size_t plane_stride = value.data.size() * sizeof(float) * value.tensor->ne[1] / (llama_n_rs_seq(ctx.get()) + 1);
                GGML_ASSERT(value.offset >= plane_stride);
                ggml_backend_tensor_get(value.tensor, value.data.data(), value.offset - plane_stride, value.data.size() * sizeof(float));
            }
            wrong_index_detected = !states_match(wrong, actual);
        }
        if (!dropped_snapshot_detected) {
            const auto & value = actual.states.begin()->second;
            const std::vector<float> zero(value.data.size(), 0.0f);
            ggml_backend_tensor_set(value.tensor, zero.data(), value.offset, zero.size() * sizeof(float));
            dropped_snapshot_detected = !states_match(selected_state(ctx.get()), actual);
            ggml_backend_tensor_set(value.tensor, value.data.data(), value.offset, value.data.size() * sizeof(float));
        }
        if (trial == 2) {
            const size_t size = llama_state_seq_get_size(ctx.get(), 0);
            std::vector<uint8_t> checkpoint(size);
            GGML_ASSERT(llama_state_seq_get_data(ctx.get(), checkpoint.data(), size, 0) == size);
            GGML_ASSERT(llama_state_seq_set_data(ctx.get(), checkpoint.data(), size, 0) == size);
            if (!states_match(selected_state(ctx.get()), actual)) {
                return test_status::FAIL;
            }
        }
        pos += accepted;
        common_batch probe(ctx.get());
        for (int seq = 0; seq < n_seq; ++seq) {
            probe.add((31 * pos + 7 + seq) % n_vocab, pos, seq, true);
        }
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, probe.get()) != 0) {
            return test_status::FAIL;
        }
        for (int seq = 0; seq < n_seq; ++seq) {
            uint64_t hash = UINT64_C(14695981039346656037);
            for (const auto & [name, value] : actuals[seq].states) {
                hash = hash_bytes(hash, value.data.data(), value.data.size() * sizeof(float));
            }
            const float * logits = llama_get_logits_ith(ctx.get(), seq);
            const int token = std::max_element(logits, logits + n_vocab) - logits;
            hash = hash_bytes(hash, logits, n_vocab * sizeof(float));
            printf("STATE_MOVERS trial=%zu seq=%d accepted=%d state_logits_hash=%016llx token=%d\n", trial, seq, accepted, (unsigned long long) hash, token);
        }
        ++pos;
        if (trial == 5) {
            for (int seq = 0; seq < n_seq; ++seq) {
                if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), seq, 0, -1)) {
                    return test_status::FAIL;
                }
            }
            if (!prefill()) {
                return test_status::FAIL;
            }
            pos = prompt.size();
        }
    }
    if (!wrong_index_detected || !dropped_snapshot_detected) {
        LOG_ERR("%s: negative control failed\n", __func__);
        return test_status::FAIL;
    }
    printf("STATE_MOVERS negative_controls=PASS\n");
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
    test_status gate     = test_status::SKIP;
};

// Run every test for an initialized model over both cache fills.
static test_results run_tests(const common_params & params, llama_model * model) {
    test_results res;
    for (uint8_t fill : { 0, 0x3e }) {
        LOG_INF("%s: testing with cache fill 0x%02x\n", __func__, fill);
        const test_status rb = test_rollback(params, model, fill);
        const test_status rp = test_multi_seq_split_replay(params, model, fill);
        const test_status ss = test_shared_seq_reserve(params, model, fill);
        const test_status rg = test_getter_gate(params, model, fill);
        res.rollback = merge_status(res.rollback, rb);
        res.replay   = merge_status(res.replay,   rp);
        res.shared   = merge_status(res.shared,   ss);
        res.gate     = merge_status(res.gate,     rg);
        if (rb == test_status::FAIL || rp == test_status::FAIL || ss == test_status::FAIL || rg == test_status::FAIL) {
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
        return { test_status::FAIL, test_status::FAIL, test_status::FAIL };
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
    LOG("\n  %s --state-movers -m your_model.gguf\n", argv[0]);
    LOG("\n  %s --getter-gate [--getter-logits FILE | --getter-reference FILE] [--getter-gathered] -m your_model.gguf\n", argv[0]);
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
    bool state_movers = false;
    bool getter_gate  = false;
    bool getter_gathered = false;
    std::string getter_logits, getter_reference;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--state-movers") == 0) {
            state_movers = true;
        } else if (strcmp(argv[i], "--getter-gate") == 0) {
            getter_gate = true;
        } else if (strcmp(argv[i], "--getter-gathered") == 0) {
            getter_gathered = true;
        } else if (strcmp(argv[i], "--getter-logits") == 0 || strcmp(argv[i], "--getter-reference") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: %s requires a file argument\n", __func__, argv[i]);
                return 1;
            }
            (strcmp(argv[i], "--getter-logits") == 0 ? getter_logits : getter_reference) = argv[i + 1];
            ++i;
        } else if (strcmp(argv[i], "--models") == 0) {
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
    if (state_movers) {
        auto init = common_init_from_params(params, true);
        return init->model() && test_state_movers(params, init->model()) == test_status::PASS ? 0 : 1;
    }

    if ((!getter_logits.empty() || !getter_reference.empty() || getter_gathered) && !getter_gate) {
        LOG_ERR("%s: getter options require --getter-gate\n", __func__);
        return 1;
    }
    if (!getter_logits.empty() && !getter_reference.empty()) {
        LOG_ERR("%s: select a getter output or a reference file\n", __func__);
        return 1;
    }
    if (getter_gate) {
        auto init = common_init_from_params(params, true);
        if (!init->model()) {
            return 1;
        }
        FILE * out = getter_logits.empty() ? nullptr : fopen(getter_logits.c_str(), "wb");
        FILE * ref = getter_reference.empty() ? nullptr : fopen(getter_reference.c_str(), "rb");
        if ((!getter_logits.empty() && !out) || (!getter_reference.empty() && !ref)) {
            LOG_ERR("%s: failed to open getter logits file\n", __func__);
            if (out) { fclose(out); }
            if (ref) { fclose(ref); }
            return 1;
        }
        test_status gate = test_status::SKIP;
        for (uint8_t fill : { 0, 0x3e }) {
            gate = merge_status(gate, test_getter_gate(params, init->model(), fill, out, ref, getter_gathered));
            if (gate == test_status::FAIL) {
                break;
            }
        }
        if (out && fclose(out) != 0) { gate = test_status::FAIL; }
        if (ref) {
            if (fgetc(ref) != EOF || ferror(ref)) { gate = test_status::FAIL; }
            fclose(ref);
        }
        return gate == test_status::PASS ? 0 : 1;
    }

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

        LOG("%-*s  %-8s  %-11s  %-10s  %s\n", (int) name_width, "Model", "rollback", "split replay", "shared seq", "getter gate");
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
            LOG("  %s      %s         %s        %s", test_status_str(res.rollback), test_status_str(res.replay), test_status_str(res.shared), test_status_str(res.gate));
            LOG("\n");
            common_log_flush(common_log_main());

            const test_status all[4] = { res.rollback, res.replay, res.shared, res.gate };
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
        LOG_INF("%s: getter gate:  %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, n_pass[3], n_skip[3], n_fail[3], models.size());

        return (n_fail[0] + n_fail[1] + n_fail[2] + n_fail[3]) == 0 ? 0 : 1;
    }

    // single-model mode
    const test_results res = run_tests_for_model(params.model.path, params);

    return (res.rollback == test_status::FAIL || res.replay == test_status::FAIL || res.shared == test_status::FAIL || res.gate == test_status::FAIL) ? 1 : 0;
}
