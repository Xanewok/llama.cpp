// CPU test of the per-sequence draft vocabulary extension (llama API level).
// Usage: test-draft-vocab-ext MODEL [single|tp]. Exit 0 = pass; any failure throws with a message on stderr.
// LLAMA_DRAFT_VOCAB_EXT_MUTANT=row must make it fail with "extension logit does not match the full head".
#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const std::string & message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

struct row_t {
    int step;
    std::vector<float> logits;          // full vocabulary, -inf outside the head
    std::vector<llama_token> backend;   // backend top-k candidate ids (backend runs only)
};

using hook_t = std::function<void(llama_context *, int step)>;

static llama_context_params small_params(const llama_model * model, llama_context_type type) {
    auto cp = llama_context_default_params();
    cp.ctx_type = type;
    cp.n_ctx = 256;
    cp.n_batch = 8;
    cp.n_ubatch = 2;
    cp.n_seq_max = 2;
    cp.n_outputs_max_per_seq = 4;
    cp.n_threads = model->devices.size() > 1 ? 4 : 8;
    cp.n_threads_batch = cp.n_threads;
    return cp;
}

static std::vector<row_t> run(llama_model * model, bool backend, int steps, const hook_t & hook) {
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int ne = llama_model_n_embd(model);
    llama_context * ctx = llama_init_from_model(model, small_params(model, LLAMA_CONTEXT_TYPE_MTP));
    require(ctx, "context load failed");
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));
    if (backend) {
        require(llama_set_sampler(ctx, 0, chain), "backend sampling refused");
    }
    auto * batch = llama_batch_ext_init(ctx);
    std::vector<float> hidden(ne);
    for (int i = 0; i < ne; ++i) {
        hidden[i] = std::sin(float(i) * 0.13f);
    }
    std::vector<row_t> rows;
    for (int step = 0; step < steps; ++step) {
        if (hook) {
            hook(ctx, step);
        }
        llama_batch_ext_clear(batch);
        const int nr = step == 0 ? 4 : 1;
        for (int r = 0; r < nr; ++r) {
            const llama_pos pos = step == 0 ? r : step + 3;
            const int idx = llama_batch_ext_add_token(batch, 0, 100 + pos);
            require(idx >= 0, "add token failed");
            require(llama_batch_ext_set_pos(batch, idx, &pos), "set pos failed");
            require(llama_batch_ext_set_embd_token(batch, idx, {hidden.data(), 1, size_t(ne)}), "set hidden failed");
            require(llama_batch_ext_set_output_logits(batch, idx, true), "set output failed");
        }
        require(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0, "decode failed");
        llama_synchronize(ctx);
        for (int r = 0; r < nr; ++r) {
            row_t row{step, std::vector<float>(nv, -INFINITY), {}};
            if (backend) {
                const auto * ids = llama_get_sampled_candidates_ith(ctx, r);
                const auto * logits = llama_get_sampled_logits_ith(ctx, r);
                const auto n = llama_get_sampled_logits_count_ith(ctx, r);
                require(ids && logits && n == 10, "missing backend candidates");
                for (uint32_t i = 0; i < n; ++i) {
                    require(ids[i] >= 0 && ids[i] < nv, "backend id out of range");
                    require(row.logits[ids[i]] == -INFINITY, "backend candidate id repeated");
                    row.logits[ids[i]] = logits[i];
                    row.backend.push_back(ids[i]);
                }
            } else {
                const float * logits = llama_get_logits_ith(ctx, r);
                require(logits, "missing raw logits");
                row.logits.assign(logits, logits + nv);
            }
            rows.push_back(std::move(row));
        }
    }
    llama_batch_ext_free(batch);
    if (backend) {
        llama_set_sampler(ctx, 0, nullptr);
    }
    llama_free(ctx);
    llama_sampler_free(chain);
    return rows;
}

static bool logit_close(float got, float ref) {
    return std::fabs(double(got) - ref) < 2e-4 * (1 + std::fabs(ref));
}

// Host rows: exactly n_head finite ids, each equal to the full head's logit; every id of must_have finite.
static void check_host(const std::vector<row_t> & got, const std::vector<row_t> & ref, size_t n_head,
        const std::function<std::set<llama_token>(int step)> & must_have, const char * what) {
    require(got.size() == ref.size(), std::string(what) + ": row count changed");
    for (size_t r = 0; r < got.size(); ++r) {
        size_t n_finite = 0;
        for (size_t id = 0; id < got[r].logits.size(); ++id) {
            const float v = got[r].logits[id];
            if (v == -INFINITY) {
                continue;
            }
            ++n_finite;
            require(logit_close(v, ref[r].logits[id]), std::string(what) + ": extension logit does not match the full head (id " +
                    std::to_string(id) + ")");
        }
        require(n_finite == n_head, std::string(what) + ": head support has " + std::to_string(n_finite) + " ids, expected " +
                std::to_string(n_head) + " (duplicate or lost row)");
        for (llama_token id : must_have(got[r].step)) {
            require(got[r].logits[id] != -INFINITY, std::string(what) + ": placed id " + std::to_string(id) + " missing from the head");
        }
    }
}

int main(int argc, char ** argv) {
    require(argc >= 2, "usage: test-draft-vocab-ext MODEL [single|tp]");
    require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    const bool tp = argc > 2 && std::string(argv[2]) == "tp";
    const int steps = 6;
    llama_log_set([](ggml_log_level level, const char * message, void *) {
        if (level >= GGML_LOG_LEVEL_WARN || std::string(message).find("draft vocab") != std::string::npos ||
                std::string(message).find("draft head") != std::string::npos) {
            std::fputs(message, stderr);
        }
    }, nullptr);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mtp = true;
    mp.use_extra_bufts = false; // a repacked CPU head cannot be read back (the extension refuses it)
    mp.progress_callback = [](float, void *) { return true; };
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devs[] = {cpu, cpu, nullptr};
    if (tp) {
        mp.devices = devs;
        mp.n_gpu_layers = -1;
        mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    }
    llama_model * full = llama_model_load_from_file(argv[1], mp);
    require(full, "full model load failed");
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(full));

    std::vector<llama_token> ids;
    std::vector<uint8_t> in_base(nv, 0);
    for (int i = nv - 1; i >= 0; i -= 10) {
        ids.push_back(i);
        in_base[i] = 1;
    }
    const auto ref = run(full, false, steps, nullptr);

    // ids outside the base head ranked by their best full-head logit: placed, they must reach the backend top-k
    std::vector<float> best(nv, -INFINITY);
    for (const auto & row : ref) {
        for (int id = 0; id < nv; ++id) {
            best[id] = std::max(best[id], row.logits[id]);
        }
    }
    std::vector<llama_token> hot;
    for (int id = 0; id < nv; ++id) {
        if (!in_base[id]) {
            hot.push_back(id);
        }
    }
    std::sort(hot.begin(), hot.end(), [&](llama_token a, llama_token b) { return best[a] != best[b] ? best[a] > best[b] : a < b; });
    hot.resize(96);

    mp.draft_vocab_ids = ids.data();
    mp.n_draft_vocab = ids.size();

    // off: n_draft_vocab_ext = 0 is the plain reduced head
    llama_model * plain = llama_model_load_from_file(argv[1], mp);
    require(plain, "reduced model load failed");
    require(plain->n_draft_vocab_base == ids.size() && plain->draft_vocab_ids.size() == ids.size(), "off: id map changed");
    const auto off = run(plain, false, steps, [&](llama_context * ctx, int step) {
        if (step == 0) {
            require(llama_draft_vocab_ext_capacity(ctx) == 0, "off: capacity must be 0");
            require(llama_set_draft_vocab_ext(ctx, 0, hot.data(), hot.size()) == -1, "off: set must fail");
        }
    });
    // exactly the base ids in every row: with the extension off no out-of-head id can be drafted
    const std::set<llama_token> base_set(ids.begin(), ids.end());
    check_host(off, ref, ids.size(), [&](int) { return base_set; }, "off");

    const uint32_t n_ext = 64;
    mp.n_draft_vocab_ext = n_ext;
    llama_model * ext = llama_model_load_from_file(argv[1], mp);
    require(ext, "extension model load failed");
    require(ext->n_draft_vocab_base == ids.size() && ext->draft_vocab_ids.size() == ids.size() + n_ext, "extension rows not allocated");
    {
        std::set<llama_token> all(ext->draft_vocab_ids.begin(), ext->draft_vocab_ids.end());
        require(all.size() == ext->draft_vocab_ids.size(), "initial head has duplicate ids");
    }

    // load-time filler rows come from the GGUF like the base rows
    const auto fill = run(ext, false, steps, nullptr);
    check_host(fill, ref, ids.size() + n_ext, [](int) { return std::set<llama_token>(); }, "filler");

    // seq 0: base ids and repeats are skipped, quota 64/2 = 32; seq 1 overlaps seq 0; seq 0 is replaced at step 3
    std::vector<llama_token> seq0 = {ids[0], hot[0], hot[0], ids[1]};
    seq0.insert(seq0.end(), hot.begin() + 1, hot.begin() + 40);
    const std::vector<llama_token> seq1(hot.begin() + 20, hot.begin() + 60);
    const std::vector<llama_token> seq0b(hot.begin() + 60, hot.begin() + 96);
    const std::set<llama_token> want0(hot.begin(), hot.begin() + 32);
    const std::set<llama_token> want1(hot.begin() + 20, hot.begin() + 52);
    const std::set<llama_token> want0b(hot.begin() + 60, hot.begin() + 92);
    const hook_t place = [&](llama_context * ctx, int step) {
        if (step == 0) {
            require(llama_draft_vocab_ext_capacity(ctx) == n_ext, "capacity");
            require(llama_set_draft_vocab_ext(ctx, 0, seq0.data(), seq0.size()) == 32, "seq 0 must place its quota");
            require(llama_set_draft_vocab_ext(ctx, 1, seq1.data(), seq1.size()) == 32, "seq 1 must place its quota");
            require(llama_set_draft_vocab_ext(ctx, 2, seq1.data(), seq1.size()) == -1, "seq out of range must fail");
            const llama_token bad = nv;
            require(llama_set_draft_vocab_ext(ctx, 0, &bad, 1) == -1, "id out of range must fail");
        }
        if (step == 3) {
            require(llama_set_draft_vocab_ext(ctx, 0, seq0b.data(), seq0b.size()) == 32, "seq 0 update must place its quota");
        }
    };
    const auto placed = run(ext, false, steps, place);
    check_host(placed, ref, ids.size() + n_ext, [&](int step) {
        std::set<llama_token> s = step < 3 ? want0 : want0b;
        s.insert(want1.begin(), want1.end());
        return s;
    }, "placed");

    // the model keeps the placed ids with their rows: a context created after the placing one reads them, and only
    // one MTP context at a time may hold the extension rows
    std::set<llama_token> last_placed = want0b;
    last_placed.insert(want1.begin(), want1.end());
    const auto again = run(ext, false, steps, [&](llama_context * ctx, int step) {
        if (step == 0) {
            require(llama_draft_vocab_ext_capacity(ctx) == n_ext, "re-created: capacity");
            llama_context * second = llama_init_from_model(ext, small_params(ext, LLAMA_CONTEXT_TYPE_MTP));
            require(!second, "a second MTP context on the extension model must be refused");
            llama_context * target = llama_init_from_model(ext, small_params(ext, LLAMA_CONTEXT_TYPE_DEFAULT));
            require(target, "a target context on the extension model must be allowed");
            llama_free(target);
        }
    });
    check_host(again, ref, ids.size() + n_ext, [&](int) { return last_placed; }, "re-created");

    // backend top-k over base + extension: same logits as the host rows, the global top 10, and it uses extension ids
    const auto backend = run(ext, true, steps, place);
    require(backend.size() == placed.size(), "backend row count changed");
    size_t n_ext_cand = 0;
    for (size_t r = 0; r < backend.size(); ++r) {
        std::vector<float> sorted;
        for (float v : placed[r].logits) {
            if (v != -INFINITY) {
                sorted.push_back(v);
            }
        }
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());
        for (llama_token id : backend[r].backend) {
            require(placed[r].logits[id] != -INFINITY, "backend candidate outside the head");
            require(logit_close(backend[r].logits[id], placed[r].logits[id]), "backend candidate logit changed");
            require(backend[r].logits[id] >= sorted[9] - 1e-4, "backend missed the global top-k");
            n_ext_cand += !in_base[id];
        }
    }
    require(n_ext_cand > 0, "backend top-k never picked an extension id");

    std::printf("M: %s: off = plain reduced head; filler rows, placed rows (2 seqs, quota, update), a re-created context and backend top-k match the full head; "
                "second MTP context refused; "
                "%zu extension ids among %zu backend candidates PASS\n", tp ? "tp" : "single", n_ext_cand, backend.size() * 10);
    llama_model_free(ext);
    llama_model_free(plain);
    llama_model_free(full);
    llama_backend_free();
    return 0;
}
