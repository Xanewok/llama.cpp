#define main architecture_test_main
#include "../tests/test-llama-archs.cpp"
#undef main
#include "ggml-round-trace.h"
#include "../src/llama-model.h"
#include <fstream>
#include <unistd.h>

static void require_reuse(bool ok, const char * why) {
    if (!ok) { throw std::runtime_error(why); }
}

template<typename T> static void save(std::ofstream & out, const T * data, size_t count = 1) {
    require_reuse(data != nullptr, "missing backend output");
    out.write(reinterpret_cast<const char *>(data), count * sizeof(T));
    require_reuse(bool(out), "output write failed");
}

static void save_sampling(std::ofstream & out, llama_context * ctx, int row, llama_sampler * sampler, const std::vector<llama_token> * subset) {
    const auto token = llama_get_sampled_token_ith(ctx, row);
    require_reuse(token >= 0, "backend token missing");
    const auto ni = llama_get_sampled_candidates_count_ith(ctx, row);
    const auto np = llama_get_sampled_probs_count_ith(ctx, row);
    const auto nl = llama_get_sampled_logits_count_ith(ctx, row);
    const auto * ids = llama_get_sampled_candidates_ith(ctx, row);
    require_reuse(ni > 0 && ni == np && np == nl, "candidate/q/logit dimensions differ");
    if (subset) {
        for (uint32_t i = 0; i < ni; ++i) {
            require_reuse(std::find(subset->begin(), subset->end(), ids[i]) != subset->end(), "proposal escaped subset");
        }
    }
    save(out, &token);
    save(out, &ni);
    save(out, ids, ni);
    save(out, llama_get_sampled_probs_ith(ctx, row), np);
    save(out, llama_get_sampled_logits_ith(ctx, row), nl);
    llama_sampler_accept(sampler, token);
}

static llama_sampler * chain(uint32_t seed) {
    auto * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(20));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(1.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    return smpl;
}

static int run_fixture(int argc, char ** argv) {
    require_reuse(argc == 8, "usage: reuse-cpu MODEL|- META K SEED ROUNDS OUTPUT TRACE(0|1)");
    require_reuse(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    llama_log_set([](ggml_log_level level, const char * text, void *) { if (std::getenv("DEBUG_FIXTURE") || level >= GGML_LOG_LEVEL_WARN) { fputs(text, stderr); } }, nullptr);
    llama_backend_init();
    const bool meta = std::atoi(argv[2]);
    const int k = std::atoi(argv[3]), rounds = std::atoi(argv[5]);
    const uint32_t seed = std::strtoul(argv[4], nullptr, 10);
    require_reuse((k == 3 || k == 5) && rounds > 1, "bad K or rounds");
    const bool synthetic = std::string(argv[1]) == "-";
    auto mp = llama_model_default_params();
    auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devs[] = {cpu, cpu, nullptr};
    mp.n_gpu_layers = meta ? -1 : 0;
    mp.load_mtp = true;
    mp.progress_callback = silent_model_load_progress;
    if (meta) { mp.devices = devs; mp.split_mode = LLAMA_SPLIT_MODE_TENSOR; }
    std::vector<llama_token> subset;
    const int nv = synthetic ? 1024 : 248320;
    for (int id = nv - 1; id >= 0; id -= synthetic ? 2 : 17) { subset.push_back(id); }
    mp.draft_vocab_ids = subset.data();
    mp.n_draft_vocab = subset.size();
    llama_model_ptr model;
    tensor_data_params tensors = { 1729, 0.1f };
    if (synthetic) {
        auto metadata = get_gguf_ctx(LLM_ARCH_QWEN35, false, nv);
        gguf_set_val_u32(metadata.get(), "qwen35.block_count", 3);
        gguf_set_val_u32(metadata.get(), "qwen35.nextn_predict_layers", 1);
        auto generated_mp = mp;
        generated_mp.devices = nullptr;
        generated_mp.n_gpu_layers = 0;
        generated_mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
        generated_mp.draft_vocab_ids = nullptr;
        generated_mp.n_draft_vocab = 0;
        model.reset(llama_model_init_from_user(metadata.get(), set_tensor_data, &tensors, generated_mp));
        require_reuse(bool(model), "synthetic model creation failed");
        const std::string path = "build-tmp/reuse-" + std::to_string(getpid()) + ".gguf";
        gguf_context_ptr disk_metadata(gguf_init_empty());
        gguf_set_kv(disk_metadata.get(), metadata.get());
        for (const auto & entry : model->tensors_by_name) {
            if (ggml_is_contiguous(entry.second) && entry.first.find("nextn.shared_head_head") == std::string::npos) {
                gguf_add_tensor(disk_metadata.get(), entry.second);
            }
        }
        require_reuse(gguf_write_to_file(disk_metadata.get(), path.c_str(), false), "synthetic GGUF write failed");
        model.reset();
        model.reset(llama_model_load_from_file(path.c_str(), mp));
        require_reuse(std::remove(path.c_str()) == 0, "synthetic model cleanup failed");
    } else {
        model.reset(llama_model_load_from_file(argv[1], mp));
    }
    require_reuse(bool(model), "model load failed");
    require_reuse(llama_vocab_n_tokens(llama_model_get_vocab(model.get())) == nv, "unexpected test vocab");
    llama_model_ptr target_model;
    if (!synthetic) {
        auto target_mp = mp;
        target_mp.load_mtp = false;
        target_mp.draft_vocab_ids = nullptr;
        target_mp.n_draft_vocab = 0;
        const char * target_path = meta && std::getenv("CPU_TARGET_MODEL") ? std::getenv("CPU_TARGET_MODEL") : argv[1];
        target_model.reset(llama_model_load_from_file(target_path, target_mp));
        require_reuse(bool(target_model), "target model load failed");
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = cp.n_ubatch = 16;
    cp.n_threads = cp.n_threads_batch = meta ? 4 : 8;
    cp.n_outputs_max = cp.n_outputs_max_per_seq = k + 1;
    llama_context_ptr target(llama_init_from_model(target_model ? target_model.get() : model.get(), cp));
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp.n_outputs_max = cp.n_outputs_max_per_seq = 1;
    llama_context_ptr draft(llama_init_from_model(model.get(), cp));
    require_reuse(target && draft, "context load failed");
    llama_sampler_ptr st(chain(seed)), sd(chain(seed + 1));
    require_reuse(llama_set_sampler(target.get(), 0, st.get()), "target backend sampling refused");
    require_reuse(llama_set_sampler(draft.get(), 0, sd.get()), "draft backend sampling refused");
    auto * bt = llama_batch_ext_init(target.get());
    auto * bd = llama_batch_ext_init(draft.get());
    const int ne = llama_model_n_embd(model.get());
    std::vector<float> hidden(ne);
    for (int i = 0; i < ne; ++i) { hidden[i] = std::sin(float(i) * 0.13f); }
    std::ofstream out(argv[6], std::ios::binary);
    int pt = 0, pd = 0;
    auto decode = [&](llama_context * ctx, llama_batch_ext * batch, const char * phase, int n, bool output, int & pos, bool mtp) {
        llama_batch_ext_clear(batch);
        for (int i = 0; i < n; ++i) {
            const llama_pos p = pos++;
            const int row = llama_batch_ext_add_token(batch, 0, (100 + p) % nv);
            require_reuse(row >= 0 && llama_batch_ext_set_pos(batch, row, &p), "batch position failed");
            if (mtp) { require_reuse(llama_batch_ext_set_embd_token(batch, row, {hidden.data(), 1, size_t(ne)}), "hidden input failed"); }
            require_reuse(llama_batch_ext_set_output_logits(batch, row, output), "output flag failed");
        }
        ggml_round_trace_begin(phase, n);
        const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        ggml_round_trace_end(rc);
        require_reuse(rc == 0, "decode failed");
    };
    const auto begin = ggml_time_us();
    for (int round = 0; round < rounds; ++round) {
        decode(target.get(), bt, "V", k + 1, true, pt, false);
        for (int row = 0; row <= k; ++row) { save_sampling(out, target.get(), row, st.get(), nullptr); }
        decode(draft.get(), bd, "C", k + 1, false, pd, true);
        for (int d = 0; d < k; ++d) {
            const std::string phase = "D" + std::to_string(d + 1);
            decode(draft.get(), bd, phase.c_str(), 1, true, pd, true);
            save_sampling(out, draft.get(), 0, sd.get(), &subset);
        }
    }
    ggml_round_trace_flush();
    printf("M: meta=%d K=%d seed=%u rounds=%d wall_us_per_round=%.3f bytes=%lld\n", meta, k, seed, rounds,
        double(ggml_time_us() - begin) / rounds, (long long) out.tellp());
    llama_batch_ext_free(bt);
    llama_batch_ext_free(bd);
    llama_set_sampler(target.get(), 0, nullptr);
    llama_set_sampler(draft.get(), 0, nullptr);
    target.reset(); draft.reset(); target_model.reset(); model.reset();
    llama_backend_free();
    return 0;
}

int main(int argc, char ** argv) {
    try { return run_fixture(argc, argv); }
    catch (const std::exception & e) { fprintf(stderr, "fixture failure: %s\n", e.what()); return 1; }
}
