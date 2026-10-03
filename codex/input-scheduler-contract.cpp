#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"
#include <cstdio>
#include <stdexcept>

static int uploads = 0, fences = 0;
static ggml_backend_buffer_type_t consumer_buft;
static bool accepts_host = false;
static bool accepts_buffer(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return accepts_host || buft == consumer_buft; }
static void fence(ggml_backend_t) { ++fences; }
static bool owned(ggml_backend_t, const ggml_backend_tensor_transfer * transfers, size_t count) {
    if (count) { ++uploads; ggml_backend_tensor_set_batch(transfers, count); }
    return true;
}
static bool not_host(ggml_backend_buffer_type_t) { return false; }
static void require(bool value, const char * message) { if (!value) { throw std::runtime_error(message); } }
static bool refuse_upload(ggml_backend_t, const ggml_backend_tensor_transfer *, size_t count) { return count == 0; }
static void meta_contract() {
    ggml_backend_load_all();
    auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devs[] = {cpu, cpu};
    auto split = [](const ggml_tensor *, void *) {
        return ggml_backend_meta_split_state{GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    };
    auto * dev = ggml_backend_meta_device(devs, 2, split, nullptr);
    auto * backend = ggml_backend_dev_init(dev, nullptr);
    auto * a = ggml_backend_meta_simple_backend(backend, 0);
    auto * b = ggml_backend_meta_simple_backend(backend, 1);
    a->iface.set_tensor_batch_owned = b->iface.set_tensor_batch_owned = owned;
    a->iface.synchronize = b->iface.synchronize = fence;
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(input);
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(dev));
    require(buffer != nullptr, "meta allocation failed");
    float data[] = {4, 3, 2, 1}, result[4];
    ggml_backend_tensor_transfer transfer{input, data, 0, sizeof(data)};
    const int before = uploads;
    require(backend->iface.set_tensor_batch_owned(backend, &transfer, 1), "meta owned upload failed");
    require(uploads == before + 2, "meta did not upload both ranks");
    ggml_backend_tensor_get(input, result, 0, sizeof(result));
    require(result[0] == 4 && result[3] == 1, "meta output mismatch");
    b->iface.set_tensor_batch_owned = nullptr;
    require(!backend->iface.set_tensor_batch_owned(backend, &transfer, 1), "mixed capability accepted");
    require(uploads == before + 2, "mixed capability partially wrote");
    b->iface.set_tensor_batch_owned = refuse_upload;
    fences = 0;
    require(backend->iface.set_tensor_batch_owned(backend, &transfer, 1), "late rank refusal lost fallback");
    require(fences == 2, "late rank refusal did not drain both ranks");
    ggml_backend_tensor_get(input, result, 0, sizeof(result));
    require(result[0] == 4 && result[3] == 1, "late rank fallback mismatch");
    ggml_backend_buffer_free(buffer); ggml_free(ctx); ggml_backend_free(backend);
    puts("M: Meta(CPU,CPU) owned dispatch, capability preflight, late-rank drain/fallback: pass");
}
int main() {
    try {
        ggml_backend_t source = ggml_backend_cpu_init(), consumer = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(consumer, 1);
        ggml_backend_buffer_type buft = *ggml_backend_cpu_buffer_type();
        buft.iface.is_host = not_host;
        consumer_buft = &buft;
        ggml_backend_device device = *ggml_backend_get_device(consumer);
        device.iface.supports_buft = accepts_buffer;
        consumer->device = &device;
        consumer->iface.set_tensor_batch_owned = owned;
        consumer->iface.synchronize = fence;
        ggml_backend_t backends[] = {consumer, source};
        ggml_backend_buffer_type_t bufts[] = {consumer_buft, ggml_backend_cpu_buffer_type()};
        auto * sched = ggml_backend_sched_new(backends, bufts, 2, 128, false, true);
        auto * ctx = ggml_init({1024*1024, nullptr, true});
        auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_set_input(input);
        auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        auto * projected = ggml_scale(ctx, input, 3);
        auto * output = ggml_scale(ctx, projected, 2);
        ggml_set_output(output);
        auto * graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_sched_set_tensor_backend(sched, projected, source);
        ggml_backend_sched_set_tensor_backend(sched, output, consumer);
        require(ggml_backend_sched_alloc_graph(sched, graph), "allocation failed");
        require(ggml_backend_sched_can_stage_inputs(sched, graph), "owned capability missed");
        float data[] = {1, 2, 3, 4};
        ggml_backend_tensor_set(input, data, 0, sizeof(data));
        fences = 0;
        require(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS, "compute failed");
        require(uploads == 1 && fences == 0, "owned upload used a host fence or missed dispatch");
        float result[4];
        ggml_backend_tensor_get(output, result, 0, sizeof(result));
        require(result[0] == 6 && result[3] == 24, "owned output mismatch");
        accepts_host = true;
        require(!ggml_backend_sched_can_stage_inputs(sched, graph), "direct host consumer accepted");
        accepts_host = false;
        consumer->iface.set_tensor_batch_owned = nullptr;
        require(!ggml_backend_sched_can_stage_inputs(sched, graph), "unsupported backend accepted");
        require(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS, "fallback failed");
        require(fences > 0, "unsupported backend lost its fence");
        consumer->iface.set_tensor_batch_owned = owned;
        auto * input_buft = input->buffer->buft;
        auto saved_host = input_buft->iface.is_host;
        input_buft->iface.is_host = not_host;
        require(!ggml_backend_sched_can_stage_inputs(sched, graph), "direct device input accepted");
        input_buft->iface.is_host = saved_host;
        ggml_backend_sched_reset(sched);
        require(!ggml_backend_sched_can_stage_inputs(sched, graph), "unallocated graph accepted");
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        consumer->device = ggml_backend_get_device(source);
        ggml_backend_free(consumer); ggml_backend_free(source);
        puts("M: scheduler dispatch, fence removal, unsupported/direct-input/unallocated fallback: pass");
        meta_contract();
        return 0;
    } catch (const std::exception & e) { fprintf(stderr, "%s\n", e.what()); return 1; }
}
