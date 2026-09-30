#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdio>
#include <cstring>

static ggml_backend_meta_split_state mirrored(const ggml_tensor *, void *) {
    ggml_backend_meta_split_state state = {};
    state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    state.n_segments = 1;
    state.nr[0] = 1;
    return state;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devices[] = { cpu, cpu };
    ggml_backend_ptr meta(ggml_backend_dev_init(ggml_backend_meta_device(devices, 2, mirrored, nullptr), nullptr));
    ggml_backend_ptr host(ggml_backend_cpu_init());
    for (size_t i = 0; i < 2; ++i) {
        ggml_backend_cpu_set_n_threads(ggml_backend_meta_simple_backend(meta.get(), i), 4);
    }
    ggml_backend_t backends[] = { meta.get(), host.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, 64, true, true));
    ggml_init_params params = { 4*ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr inputs(ggml_init(params));
    ggml_tensor * a = ggml_new_tensor_1d(inputs.get(), GGML_TYPE_F32, 8);
    ggml_tensor * b = ggml_new_tensor_1d(inputs.get(), GGML_TYPE_F32, 8);
    ggml_set_input(a);
    ggml_set_input(b);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(inputs.get(), ggml_backend_cpu_buffer_type()));
    params.mem_size = 8*ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false);
    ggml_context_ptr compute(ggml_init(params));
    ggml_tensor * out = ggml_add(compute.get(), a, b);
    ggml_cgraph * graph = ggml_new_graph_custom(compute.get(), 64, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_sched_set_tensor_backend(sched.get(), out, meta.get());
    if (!ggml_backend_sched_alloc_graph(sched.get(), graph) || ggml_backend_sched_get_n_copies(sched.get()) <= 1) {
        return 1;
    }
    for (int round = 0; round < 12; ++round) {
        float x[8], y[8], want[8], got[8];
        for (int i = 0; i < 8; ++i) {
            x[i] = round + i;
            y[i] = 2*round - i;
            want[i] = x[i] + y[i];
        }
        ggml_backend_tensor_set(a, x, 0, sizeof(x));
        ggml_backend_tensor_set(b, y, 0, sizeof(y));
        if (ggml_backend_sched_graph_compute_async(sched.get(), graph) != GGML_STATUS_SUCCESS) {
            return 2;
        }
        memset(a->data, 0, sizeof(x));
        memset(b->data, 0, sizeof(y));
        ggml_backend_sched_synchronize(sched.get());
        ggml_backend_tensor_get(out, got, 0, sizeof(got));
        if (memcmp(want, got, sizeof(got))) {
            return 3;
        }
    }
    printf("M Meta(CPU,CPU): pipeline copy slots=%d, 12 reused computes: bit-identical\n", ggml_backend_sched_get_n_copies(sched.get()));
    sched.reset();
    return 0;
}
