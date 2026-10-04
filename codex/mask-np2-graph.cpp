#define main architecture_test_main
#include "../tests/test-llama-archs.cpp"
#undef main
#include <fstream>
#include "../src/llama-model.h"
#include "../src/llama-kv-cache.h"

static void mask_require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

struct mask_probe {
    int compact = 0;
    int streams = 0;
};

static bool probe_mask(ggml_tensor * tensor, bool ask, void * userdata) {
    if (ask) {
        return strcmp(tensor->name, "attn_kq_mask") == 0;
    }
    auto & probe = *(mask_probe *) userdata;
    mask_require(tensor->op == GGML_OP_CONCAT && tensor->ne[3] == probe.streams, "wrong production compact mask shape");
    ++probe.compact;
    return true;
}

int main(int argc, char ** argv) {
    try {
        mask_require(argc == 3, "usage: mask-np2-graph LOGITS EXPECT_COMPACT");
        mask_require(getenv("CUDA_VISIBLE_DEVICES") && !*getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
        common_init();
        llama_log_set([](ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_WARN) {
                fputs(text, stderr);
            }
        }, nullptr);
        auto metadata = get_gguf_ctx(LLM_ARCH_LLAMA, false);
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_dev_t devices[] = {cpu, cpu, nullptr};
        auto mp = llama_model_default_params();
        mp.devices = devices;
        mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        mp.n_gpu_layers = 99;
        tensor_data_params data = {42, 0.1f};
        llama_model_ptr model(llama_model_init_from_user(metadata.get(), set_tensor_data, &data, mp));
        mask_require(bool(model), "graph model creation failed");
        std::ofstream out(argv[1], std::ios::binary);
        mask_require(bool(out), "logits file open failed");
        int total_compact = 0;
        for (const std::vector<int> & depths : {std::vector<int>{8192, 35072}, {33280, 35072}, {33280, 35072, 33792}}) {
            const int ns = depths.size();
            mask_probe probe = {0, ns};
            auto cp = llama_context_default_params();
            cp.n_ctx = ns*35840;
            cp.n_seq_max = ns;
            cp.n_batch = cp.n_ubatch = 64;
            cp.n_outputs_max = 64;
            cp.n_outputs_max_per_seq = 16;
            cp.n_threads = cp.n_threads_batch = 4;
            cp.kv_unified = false;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            cp.type_k = cp.type_v = GGML_TYPE_F16;
            cp.cb_eval = probe_mask;
            cp.cb_eval_user_data = &probe;
            llama_context_ptr context(llama_init_from_model(model.get(), cp));
            mask_require(bool(context), "graph context creation failed");
            auto * cache = dynamic_cast<llama_kv_cache *>(llama_get_memory(context.get()));
            mask_require(cache != nullptr, "missing KV cache");
            for (int s = 0; s < ns; ++s) {
                auto & cells = const_cast<llama_kv_cells &>(cache->get_cells(s));
                for (int j = 0; j < depths[s]; ++j) {
                    cells.pos_set(j, j);
                    cells.seq_add(j, s);
                }
            }
            auto * batch = llama_batch_ext_init(context.get());
            auto next = depths;
            int step = 0;
            for (int rows : {4, 1, 6, 16, 1, 1, 1}) {
                if (step == 5) {
                    mask_require(cache->seq_rm(1, 17, 20), "graph hole removal failed");
                }
                llama_batch_ext_clear(batch);
                for (int s = 0; s < ns; ++s) {
                    for (int r = 0; r < rows; ++r) {
                        const int idx = llama_batch_ext_add_token(batch, s, 3 + s + r);
                        const llama_pos pos[] = {next[s]++};
                        mask_require(llama_batch_ext_set_pos(batch, idx, pos), "graph position setup failed");
                        mask_require(llama_batch_ext_set_output_logits(batch, idx, true), "graph output setup failed");
                    }
                }
                mask_require(llama_process(context.get(), LLAMA_PROCESS_TYPE_DECODE, batch) == 0, "graph decode failed");
                for (int i = 0; i < rows*ns; ++i) {
                    auto * logits = llama_get_logits_ith(context.get(), i);
                    mask_require(logits != nullptr, "missing graph logits");
                    out.write((const char *) logits, 128*sizeof(float));
                }
                ++step;
            }
            llama_batch_ext_free(batch);
            if (atoi(argv[2])) {
                mask_require((depths[0] < 32768) == (probe.compact == 0), "production multistream compact gate failed");
            } else {
                mask_require(probe.compact == 0, "baseline unexpectedly compact");
            }
            total_compact += probe.compact;
        }
        printf("M: production compact callbacks %d\n", total_compact);
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
