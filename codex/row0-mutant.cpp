#include "ggml.h"

#include <atomic>
#include <cstdio>
#include <dlfcn.h>
#include <vector>

struct ggml_compute_params;

#ifdef __APPLE__
// dyld interposing: the original stays reachable under its own name; needs libggml-cpu linked -interposable.
extern "C" void ggml_compute_forward_gated_delta_net(const ggml_compute_params * params, ggml_tensor * dst);
extern "C" void row0_mutant_gated_delta_net(const ggml_compute_params * params, ggml_tensor * dst);
__attribute__((used, section("__DATA,__interpose"))) static const struct { void * replacement; void * replacee; } row0_interpose = {
    reinterpret_cast<void *>(row0_mutant_gated_delta_net), reinterpret_cast<void *>(ggml_compute_forward_gated_delta_net) };
extern "C" void row0_mutant_gated_delta_net(const ggml_compute_params * params, ggml_tensor * dst) {
    const auto original = ggml_compute_forward_gated_delta_net;
#else
extern "C" void ggml_compute_forward_gated_delta_net(const ggml_compute_params * params, ggml_tensor * dst) {
    using compute_fn = void (*)(const ggml_compute_params *, ggml_tensor *);
    static auto original = reinterpret_cast<compute_fn>(dlsym(RTLD_NEXT, "ggml_compute_forward_gated_delta_net"));
#endif
    if (!dst->src[6]) {
        original(params, dst);
        return;
    }
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        std::fprintf(stderr, "ROW0_MUTANT active\n");
    }
    // Keep the gather branch's ids intact while every indexed read selects row 0.
    std::vector<int32_t> rows(dst->src[6]->ne[0], 0);
    ggml_tensor ids = *dst->src[6];
    ids.data = rows.data();
    ggml_tensor mutant = *dst;
    mutant.src[6] = &ids;
    original(params, &mutant);
}
