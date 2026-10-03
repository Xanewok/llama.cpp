#include "../ggml/src/ggml-backend-impl.h"
#define REUSE_FIXTURE_ENTRY input_reference_main
#include "reuse-cpu.cpp"
#undef REUSE_FIXTURE_ENTRY

static ggml_backend_t (*cpu_init)(ggml_backend_dev_t, const char *);
static size_t owned_uploads = 0, owned_queries = 0;
static bool cpu_owned(ggml_backend_t, const ggml_backend_tensor_transfer * transfers, size_t count) {
    if (count == 0) { ++owned_queries; return true; }
    ++owned_uploads;
    ggml_backend_tensor_set_batch(transfers, count);
    return true;
}
static ggml_backend_t init_owned_cpu(ggml_backend_dev_t dev, const char * params) {
    auto * backend = cpu_init(dev, params);
    backend->iface.set_tensor_batch_owned = cpu_owned;
    return backend;
}
int main(int argc, char ** argv) {
    llama_backend_init();
    auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    cpu_init = cpu->iface.init_backend;
    cpu->iface.init_backend = init_owned_cpu;
    const int result = input_reference_main(argc, argv);
    cpu->iface.init_backend = cpu_init;
    require_reuse(result == 0 && owned_queries > 0 && owned_uploads > 0, "owned Meta(CPU,CPU) path was not exercised");
    printf("M: owned uploads=%zu capability queries=%zu\n", owned_uploads, owned_queries);
    return result;
}
