#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static bool check_deferred = false;

static std::vector<std::vector<float>> run(llama_model * model, bool mtp, bool backend, int steps) {
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int ne = llama_model_n_embd(model);
    auto cp = llama_context_default_params();
    cp.ctx_type = mtp ? LLAMA_CONTEXT_TYPE_MTP : LLAMA_CONTEXT_TYPE_DEFAULT;
    cp.n_ctx = 256;
    cp.n_batch = 8;
    cp.n_ubatch = 2;
    cp.n_outputs_max_per_seq = 4;
    cp.n_threads = model->devices.size() > 1 ? 4 : 8;
    cp.n_threads_batch = cp.n_threads;
    llama_context * ctx = llama_init_from_model(model, cp);
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
    std::vector<std::vector<float>> rows;
    double elapsed = 0;
    for (int step = 0; step < steps; ++step) {
        const auto t0 = std::chrono::steady_clock::now();
        llama_batch_ext_clear(batch);
        const int nr = step == 0 ? 4 : 1;
        for (int r = 0; r < nr; ++r) {
            const llama_pos pos = step == 0 ? r : step + 3;
            const int idx = llama_batch_ext_add_token(batch, 0, 100 + pos);
            require(idx >= 0, "add token failed");
            require(llama_batch_ext_set_pos(batch, idx, &pos), "set pos failed");
            if (mtp) {
                require(llama_batch_ext_set_embd_token(batch, idx, {hidden.data(), 1, size_t(ne)}), "set hidden failed");
            }
            require(llama_batch_ext_set_output_logits(batch, idx, true), "set output failed");
        }
        require(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0, "decode failed");
        llama_synchronize(ctx);
        for (int r = 0; r < nr; ++r) {
            if (backend) {
                const auto * ids = llama_get_sampled_candidates_ith(ctx, r);
                const auto * logits = llama_get_sampled_logits_ith(ctx, r);
                const auto n = llama_get_sampled_logits_count_ith(ctx, r);
                require(ids && logits && n == 10, "missing backend candidates");
                std::vector<float> row(nv, -INFINITY);
                for (uint32_t i = 0; i < n; ++i) {
                    require(ids[i] >= 0 && ids[i] < nv, "backend id out of range");
                    require(std::find(model->draft_vocab_ids.begin(), model->draft_vocab_ids.end(), ids[i]) != model->draft_vocab_ids.end(), "backend id escaped subset");
                    row[ids[i]] = logits[i];
                }
                rows.push_back(std::move(row));
            } else {
                const float * logits = llama_get_logits_ith(ctx, r);
                require(logits, "missing raw logits");
                rows.emplace_back(logits, logits + nv);
                require(std::equal(rows.back().begin(), rows.back().end(), llama_get_logits_ith(ctx, r)), "second logits read changed row");
            }
        }
        if (backend && check_deferred) {
            const float * raw = llama_get_logits(ctx);
            require(raw, "deferred raw logits missing");
            std::vector<bool> in_subset(nv, false);
            for (llama_token id : model->draft_vocab_ids) {
                in_subset[id] = true;
            }
            for (int r = 0; r < nr; ++r) {
                const auto & candidates = rows[rows.size() - nr + r];
                for (int id = 0; id < nv; ++id) {
                    const bool inside = in_subset[id];
                    require(inside ? std::isfinite(raw[r*nv + id]) : raw[r*nv + id] == -INFINITY, "deferred row mask wrong");
                    if (candidates[id] != -INFINITY) {
                        require(candidates[id] == raw[r*nv + id], "deferred candidate logit wrong");
                    }
                }
            }
        }
        if (step > 0) {
            elapsed += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
    }
    std::printf("M: %s %s %s %.6f ms/decode-step (%d steady steps; host+compute+readback)\n",
            model->draft_output ? "subset" : "full", mtp ? "MTP" : "target", backend ? "backend" : "host", elapsed / (steps - 1), steps - 1);
    llama_batch_ext_free(batch);
    if (backend) {
        llama_set_sampler(ctx, 0, nullptr);
    }
    llama_free(ctx);
    llama_sampler_free(chain);
    return rows;
}

int main(int argc, char ** argv) {
    require(argc >= 2, "usage: test-draft-vocab MODEL [tp] [steps]");
    require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    const bool tp = argc > 2 && std::string(argv[2]) == "tp";
    const int steps = argc > 3 ? std::atoi(argv[3]) : 9;
    require(steps > 1, "need steady steps");
    check_deferred = argc > 4 && std::string(argv[4]) == "deferred";
    llama_log_set([](ggml_log_level level, const char * message, void *) { if (level >= GGML_LOG_LEVEL_WARN) std::fputs(message, stderr); }, nullptr);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mtp = true;
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
    // Deliberately unsorted, with gaps and ids outside the BF16 exact-integer range.
    std::vector<llama_token> ids;
    const bool identity = argc > 4 && std::string(argv[4]) == "identity";
    if (identity) {
        ids.resize(nv);
        std::iota(ids.begin(), ids.end(), 0);
    } else {
        for (int i = nv - 1; i >= 0; i -= 10) {
            ids.push_back(i);
        }
    }
    mp.draft_vocab_ids = ids.data();
    mp.n_draft_vocab = ids.size();
    llama_model * sub = llama_model_load_from_file(argv[1], mp);
    require(sub, "subset model load failed");
    std::ofstream list("codex/cpu-vocab-ids.json");
    list << '[';
    for (size_t i = 0; i < ids.size(); ++i) {
        list << (i ? "," : "") << ids[i];
    }
    list << ']';
    list.close();
    const auto & layer = full->layers.back();
    const auto * head = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : full->output;
    require(sub->draft_output->type == head->type, "quantized type changed");
    std::printf("D: compact head %zu bytes, ids %zu bytes; type %s\n", ggml_nbytes(sub->draft_output), ids.size()*sizeof(float), ggml_type_name(head->type));
    const auto target_full = run(full, false, false, 2);
    const auto target_sub = run(sub, false, false, 2);
    require(target_full == target_sub, "target logits changed");
    const auto ref = run(full, true, false, steps);
    const auto got = run(sub, true, false, steps);
    require(ref.size() == got.size(), "row count changed");
    if (identity) {
        for (size_t r = 0; r < ref.size(); ++r) {
            require(std::memcmp(ref[r].data(), got[r].data(), nv * sizeof(float)) == 0, "full-vocab mapping changed logits");
            for (uint32_t seed = 0; seed < 100; ++seed) {
                auto sample = [&](const std::vector<float> & logits) {
                    auto * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(10));
                    llama_sampler_chain_add(sampler, llama_sampler_init_temp(1.0f));
                    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
                    std::vector<llama_token_data> data(nv);
                    for (int id = 0; id < nv; ++id) {
                        data[id] = {id, logits[id], 0};
                    }
                    llama_token_data_array candidates = {data.data(), data.size(), -1, false};
                    llama_sampler_apply(sampler, &candidates);
                    const auto id = candidates.data[candidates.selected].id;
                    llama_sampler_free(sampler);
                    return id;
                };
                require(sample(ref[r]) == sample(got[r]), "full-vocab mapping changed sampled token");
            }
        }
        std::printf("M: identity mapping: %zu MTP rows bit-identical; 100 seeds per row bit-identical\n", ref.size());
    }
    double max_error = 0;
    bool negative_detected = false;
    std::vector<bool> inside(nv, false);
    for (llama_token id : ids) {
        inside[id] = true;
    }
    for (size_t r = 0; r < got.size(); ++r) {
        for (int id = 0; id < nv; ++id) {
            if (inside[id]) {
                const double error = std::fabs(double(got[r][id]) - ref[r][id]);
                max_error = std::max(max_error, error);
                require(error < 2e-4 * (1 + std::fabs(ref[r][id])), "subset logit does not match source row");
                negative_detected |= std::fabs(got[r][id] - ref[r][(id + 1) % nv]) > 0.01;
            } else {
                require(got[r][id] == -INFINITY, "host logits escaped subset");
            }
        }
    }
    require(negative_detected, "wrong id mapping escaped negative control");
    const auto backend = run(sub, true, true, steps);
    for (size_t r = 0; r < backend.size(); ++r) {
        std::vector<float> sorted;
        for (llama_token id : ids) {
            sorted.push_back(got[r][id]);
        }
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());
        int count = 0;
        for (int id = 0; id < nv; ++id) {
            if (backend[r][id] != -INFINITY) {
                require(std::fabs(backend[r][id] - got[r][id]) < 2e-4 * (1 + std::fabs(got[r][id])), "backend candidate logit changed");
                require(backend[r][id] >= sorted[9] - 1e-4, "backend missed global top-k");
                ++count;
            }
        }
        require(count == 10, "backend candidate ids collided");
    }
    std::printf("M: target bit identity, subset logits, outside mask, backend top-k, multi-ubatch reads PASS; max logit error %.9g; mapping negative control detected\n", max_error);
    const llama_token small_ids[] = {17, 40037};
    mp.draft_vocab_ids = small_ids;
    mp.n_draft_vocab = 1;
    if (tp) {
        require(!llama_model_load_from_file(argv[1], mp), "empty TP shard must be rejected");
        mp.n_draft_vocab = 2;
    }
    llama_model * tiny = llama_model_load_from_file(argv[1], mp);
    require(tiny, "small subset failed to load");
    auto cp = llama_context_default_params();
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp.n_ctx = 128;
    cp.n_batch = cp.n_ubatch = 8;
    cp.n_threads = cp.n_threads_batch = tp ? 4 : 8;
    llama_context * tiny_ctx = llama_init_from_model(tiny, cp);
    require(tiny_ctx, "small context failed to load");
    llama_sampler * top = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(top, llama_sampler_init_top_k(10));
    require(!llama_set_sampler(tiny_ctx, 0, top), "small subset must use host top-k");
    llama_sampler_free(top);
    llama_free(tiny_ctx);
    const auto tiny_rows = run(tiny, true, false, 2);
    for (const auto & row : tiny_rows) {
        for (int id = 0; id < nv; ++id) {
            const bool inside = id == small_ids[0] || (tp && id == small_ids[1]);
            require(inside ? std::isfinite(row[id]) : row[id] == -INFINITY, "small subset escaped support");
        }
    }
    std::puts(tp ? "M: two-token subset uses host fallback; empty TP partition rejected" : "M: one-token subset uses host fallback and preserves its support");
    llama_model_free(tiny);
    llama_model_free(sub);
    llama_model_free(full);
    llama_backend_free();
}
