#include "llama.h"
#include "llama-cpp.h"
#include "llama-ext.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

int main(int argc, char ** argv) {
    require(argc == 2, "model path required");
    require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mtp = true;
    llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
    require(bool(model), "model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = cp.n_ubatch = 64;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context_ptr target(llama_init_from_model(model.get(), cp));
    require(bool(target), "target context failed");
    for (int mode = 0; mode < 4; ++mode) {
        auto dp = cp;
        dp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        if (mode == 1) {
            dp.rope_freq_base = 42.0f;
        } else if (mode == 2) {
            dp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        } else if (mode == 3) {
            dp.op_offload = !cp.op_offload;
        }
        llama_context_ptr draft(llama_init_from_model(model.get(), dp));
        require(bool(draft), "draft context failed");
        require(llama_mtp_fold_init(target.get(), draft.get()) == (mode == 0), "fold eligibility mismatch");
        require(llama_mtp_fold_init(target.get(), nullptr), "fold detach failed");
    }
    std::fprintf(stderr, "M: matching fold context accepted; RoPE, FA and offload mismatches refused\n");
}
