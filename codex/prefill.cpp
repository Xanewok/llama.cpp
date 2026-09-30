#include "common.h"
#include "sampling.h"
#include "llama-cpp.h"
#include "../src/llama-context.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        return 2;
    }
    llama_backend_init();
    auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    const bool tp = argc > 2 && std::strcmp(argv[2], "--tp") == 0;
    ggml_backend_dev_t devices[] = { cpu, tp ? cpu : nullptr, nullptr };
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = tp ? 99 : 0;
    mp.devices = devices;
    mp.split_mode = tp ? LLAMA_SPLIT_MODE_TENSOR : LLAMA_SPLIT_MODE_LAYER;
    mp.load_mtp = true;
    llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
    if (!model) {
        return 2;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 2048;
    cp.n_ubatch = 128;
    cp.n_threads = cp.n_threads_batch = tp ? 4 : 8;
    cp.n_outputs_max = cp.n_outputs_max_per_seq = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    if (!ctx) {
        return 2;
    }
    llama_set_embeddings_nextn(ctx.get(), true, false);
    ctx->sched_reserve();
    std::string prompt;
    while (prompt.size() < 12000) {
        prompt += "Explain how a compiler verifies a program and reports useful errors. ";
    }
    auto tokens = common_tokenize(llama_model_get_vocab(model.get()), prompt, true, true);
    tokens.resize(602);
    std::vector<float> reference;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    common_sampler_ptr sampler;
    int run = 0;
    for (bool backend : {false, true, true, false}) {
        llama_set_sampler(ctx.get(), 0, nullptr);
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        common_params_sampling sp;
        sp.seed = 1;
        sp.top_k = 20;
        sp.top_p = 0.95f;
        sp.min_p = 0.05f;
        sp.temp = 1.0f;
        sp.backend_sampling = backend;
        common_sampler_ptr next(common_sampler_init(model.get(), sp));
        if (!llama_set_sampler(ctx.get(), 0, backend ? common_sampler_get_backend(next.get()) : nullptr)) {
            throw std::runtime_error("sampler refused");
        }
        sampler = std::move(next);
        const auto start = ggml_time_us();
        ctx->sched_reserve();
        const auto reserved = ggml_time_us();
        llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); ++i) {
            common_batch_add(batch, tokens[i], i, {0}, i + 1 == tokens.size());
        }
        std::fprintf(stderr, "ARM run=%d backend=%d tp=%d\n", ++run, backend, tp);
        const int rc = llama_decode(ctx.get(), batch);
        llama_synchronize(ctx.get());
        const auto end = ggml_time_us();
        if (rc) {
            throw std::runtime_error("decode failed");
        }
        const float * full = llama_get_logits_ith(ctx.get(), -1);
        if (reference.empty()) {
            reference.assign(full, full + n_vocab);
        } else if (std::memcmp(full, reference.data(), n_vocab * sizeof(float))) {
            throw std::runtime_error("prefill logits differ");
        }
        std::printf("M: run=%d backend=%d tp=%d reserve_us=%lld prefill_us=%lld total_us=%lld exact_logits=1\n",
                run, backend, tp, (long long) (reserved-start), (long long) (end-reserved), (long long) (end-start));
        llama_batch_free(batch);
    }
}
