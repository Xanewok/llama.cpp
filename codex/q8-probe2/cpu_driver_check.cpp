#define main unused_probe_main
#include "../../build-tmp/q8-probe2/driver.cpp"
#undef main
#include "ggml-cpu.h"

int main() {
    ggml_init_params params = {8 * ggml_tensor_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 128, 256, 2, 1);
    ggml_tensor * v = ggml_dup_tensor(ctx, k);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 64);
    ggml_set_name(k, "probe_src_1");
    ggml_set_name(v, "probe_src_2");
    ggml_set_name(mask, "probe_src_3");
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    GGML_ASSERT(buffer != nullptr);

    init_tensor_uniform(k);
    std::vector<uint8_t> first(ggml_nbytes(k));
    std::vector<uint8_t> second(first.size());
    ggml_backend_tensor_get(k, first.data(), 0, first.size());
    init_tensor_uniform(k);
    ggml_backend_tensor_get(k, second.data(), 0, second.size());
    GGML_ASSERT(first == second);
    init_tensor_uniform(v);
    ggml_backend_tensor_get(v, second.data(), 0, second.size());
    GGML_ASSERT(first != second);

    first.resize(ggml_nbytes(mask));
    second.resize(first.size());
    init_tensor_kq_mask(mask);
    ggml_backend_tensor_get(mask, first.data(), 0, first.size());
    init_tensor_kq_mask(mask);
    ggml_backend_tensor_get(mask, second.data(), 0, second.size());
    GGML_ASSERT(first == second);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    printf("M: CPU deterministic K/V and mask repeat byte-for-byte; distinct K/V seed control passed\n");
    return 0;
}
