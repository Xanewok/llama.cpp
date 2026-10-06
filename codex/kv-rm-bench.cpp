#include "../src/llama-kv-cache.h"
#include "../src/llama-model.h"
#include "llama-cpp.h"

#include <chrono>
#include <cstdio>

int main() {
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model_ptr model(llama_model_create(LLM_ARCH_LLAMA, mp));
    auto hp = model->hparams;
    hp.no_alloc = true;
    for (int np : {1, 2}) {
        for (int size : {8192, 32768, 102400}) {
            llama_kv_cache cache(*model, hp, GGML_TYPE_F16, GGML_TYPE_F16, false, false, false,
                                 size, np, 256, 0, LLAMA_SWA_TYPE_NONE, nullptr,
                                 [](uint32_t) { return false; }, nullptr, nullptr);
            const int depth = size - 8;
            for (int s = 0; s < np; ++s) {
                auto & cells = const_cast<llama_kv_cells &>(cache.get_cells(s));
                for (int i = 0; i < depth; ++i) {
                    cells.pos_set(i, i);
                    cells.seq_add(i, s);
                }
            }
            for (int trial = 0; trial < 5; ++trial) {
                const auto start = std::chrono::steady_clock::now();
                for (int repeat = 0; repeat < 2000; ++repeat) {
                    for (int s = 0; s < np; ++s) {
                        auto & cells = const_cast<llama_kv_cells &>(cache.get_cells(s));
                        for (int i = depth; i < depth + 4; ++i) {
                            cells.pos_set(i, i);
                            cells.seq_add(i, s);
                        }
                        cache.seq_rm(s, depth, -1);
                        cache.seq_rm(s, depth, -1);
                    }
                }
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count()/2000;
                printf("CPU M np=%d capacity=%d trial=%d two_trims_and_reinsert_us=%.3f\n", np, size, trial, us);
            }
        }
    }
}
