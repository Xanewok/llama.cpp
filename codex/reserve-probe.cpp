#include "llama.h"

#include <cinttypes>
#include <cstdio>
#include <dlfcn.h>

static bool observe_getter(ggml_tensor * tensor, bool ask, void *) {
    if (ask && tensor->op == GGML_OP_GATED_DELTA_NET && tensor->src[6]) {
        const ggml_tensor * ids = tensor->src[6];
        const int64_t n_rs = ids->view_src ? ids->view_src->ne[0] : ids->ne[0];
        if (n_rs > ids->ne[0]) {
            std::fprintf(stderr, "GDN_GETTER_EXTRA n_rs=%" PRId64 " n_seqs=%" PRId64 "\n", n_rs, ids->ne[0]);
        }
    }
    return !ask;
}

extern "C" llama_context * llama_init_from_model(llama_model * model, llama_context_params params) {
    auto original = reinterpret_cast<decltype(&llama_init_from_model)>(dlsym(RTLD_NEXT, "llama_init_from_model"));
    if (params.n_rs_seq > 0 && !params.cb_eval) {
        params.cb_eval = observe_getter;
    }
    return original(model, params);
}
