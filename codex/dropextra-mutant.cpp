#include "ggml.h"

#include <atomic>
#include <cstdio>
#include <dlfcn.h>

struct ggml_compute_params;

// Drops build_rs's relocation of the states past the ubatch (the copy of gathered rows back into the cache).
extern "C" void ggml_compute_forward_cpy(const ggml_compute_params * params, ggml_tensor * dst) {
    using compute_fn = void (*)(const ggml_compute_params *, ggml_tensor *);
    static auto original = reinterpret_cast<compute_fn>(dlsym(RTLD_NEXT, "ggml_compute_forward_cpy"));
    const ggml_tensor * src = dst->src[0];
    if (ggml_nelements(dst) > 0 && src->op == GGML_OP_VIEW && src->view_src && src->view_src->op == GGML_OP_GET_ROWS) {
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true)) {
            std::fprintf(stderr, "DROPEXTRA_MUTANT active\n");
        }
        return;
    }
    original(params, dst);
}
