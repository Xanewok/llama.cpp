#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

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
        if (step > 0) {
            elapsed += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
    }
    std::printf("M: %s %s %s %.6f ms/draft-step (%d steady steps; host+compute+readback)\n",
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
    for (int i = nv - 1; i >= 0; i -= 10) {
        ids.push_back(i);
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
    llama_model_free(sub);
    llama_model_free(full);
    llama_backend_free();
}
