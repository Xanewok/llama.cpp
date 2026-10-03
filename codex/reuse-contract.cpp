#include "llama-sampler.h"
#include "llama-graph.h"
#include "llama-batch.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-impl.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>

static void require_contract(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

static void bindings(bool broken) {
    const uint32_t seed = 1729;
    auto * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
    require_contract(sampler->iface->backend_init(sampler, ggml_backend_cpu_buffer_type(), 2), "sampler init failed");
    const ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context_ptr a(ggml_init(params)), b(ggml_init(params));
    auto build = [&](ggml_context * ctx, int rows) {
        sampler->iface->backend_reset(sampler);
        auto * gf = ggml_new_graph_custom(ctx, 128, false);
        for (int i = 0; i < rows; ++i) {
            auto * logits = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
            ggml_set_input(logits);
            llama_sampler_data data = { logits, nullptr, nullptr, nullptr };
            sampler->iface->backend_apply(sampler, ctx, gf, &data);
            ggml_build_forward_expand(gf, data.sampled);
        }
    };
    build(a.get(), 2);
    llm_graph_input_sampling input_a({ {0, sampler} });
    ggml_backend_buffer_ptr ba(ggml_backend_alloc_ctx_tensors_from_buft(a.get(), ggml_backend_cpu_buffer_type()));
    build(b.get(), 1);
    llm_graph_input_sampling input_b({ {0, sampler} });
    ggml_backend_buffer_ptr bb(ggml_backend_alloc_ctx_tensors_from_buft(b.get(), ggml_backend_cpu_buffer_type()));
    auto * u0 = ggml_get_tensor(a.get(), "uniform_0");
    auto * u1 = ggml_get_tensor(a.get(), "uniform_1");
    const float sentinel = -1;
    ggml_backend_tensor_set(u0, &sentinel, 0, sizeof(sentinel));
    ggml_backend_tensor_set(u1, &sentinel, 0, sizeof(sentinel));
    llama_ubatch ubatch = {};
    int8_t outputs[] = { 1, 1 };
    llama_seq_id seq = 0;
    llama_seq_id * seqs[] = { &seq, &seq };
    ubatch.n_tokens = 2;
    ubatch.output = outputs;
    ubatch.seq_id = seqs;
    llama_sampler_backend_begin(sampler);
    (broken ? input_b : input_a).set_input(&ubatch);
    float x0, x1;
    ggml_backend_tensor_get(u0, &x0, 0, sizeof(x0));
    ggml_backend_tensor_get(u1, &x1, 0, sizeof(x1));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0, 1);
    const float e0 = dist(rng), e1 = dist(rng);
    require_contract(x0 == e0 && x1 == e1, "stale bindings or RNG draw count");
    llama_sampler_accept(sampler, 0);
    llama_sampler_accept(sampler, 0);
    llama_sampler_backend_begin(sampler);
    input_b.set_input(&ubatch);
    float x2;
    ggml_backend_tensor_get(ggml_get_tensor(b.get(), "uniform_0"), &x2, 0, sizeof(x2));
    require_contract(x2 == float(dist(rng)), "restoring bindings changed RNG state");
    llama_sampler_free(sampler);
    puts("M: two/one-output binding alternation preserves uniform bytes and committed RNG draws");
}

static void epochs() {
    auto * backend = ggml_backend_cpu_init();
    ggml_backend_t backends[] = { backend };
    ggml_backend_sched_ptr parent(ggml_backend_sched_new(backends, nullptr, 1, 128, false, true));
    ggml_backend_sched_ptr child(ggml_backend_sched_new_shared(parent.get()));
    require_contract(bool(child), "shared scheduler missing");
    const ggml_init_params params = {1024 * 1024, nullptr, true};
    ggml_context_ptr ca(ggml_init(params)), cb(ggml_init(params));
    auto build = [](ggml_context * ctx, int size) {
        auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
        ggml_set_input(input);
        auto * output = ggml_scale(ctx, input, 2);
        ggml_set_output(output);
        auto * graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(graph, output);
        return graph;
    };
    auto * a = build(ca.get(), 8);
    auto * b = build(cb.get(), 8192);
    require_contract(ggml_backend_sched_alloc_graph(parent.get(), a), "allocate A failed");
    require_contract(ggml_backend_sched_graph_is_alloc(parent.get(), a), "A not valid");
    require_contract(ggml_backend_sched_graph_compute_async(parent.get(), b) == GGML_STATUS_ALLOC_FAILED, "UID mismatch escaped compute guard");
    require_contract(ggml_backend_sched_reserve(child.get(), b), "reserve B failed");
    require_contract(!ggml_backend_sched_graph_is_alloc(parent.get(), a), "stale epoch reported valid");
    require_contract(ggml_backend_sched_graph_compute_async(parent.get(), a) == GGML_STATUS_ALLOC_FAILED, "stale epoch escaped public compute guard");
    ggml_backend_sched_reset(parent.get());
    parent.reset(); child.reset();
    ggml_backend_sched_ptr parallel(ggml_backend_sched_new(backends, nullptr, 1, 128, true, true));
    require_contract(!ggml_backend_sched_new_shared(parallel.get()), "pipeline scheduler unexpectedly shares storage");
    parallel.reset();
    ggml_backend_free(backend);
    puts("M: public compute rejects wrong UID and stale allocation epoch");
}

static void meta_input_owner() {
    llama_backend_init();
    auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devs[] = { cpu, cpu };
    auto split = [](const ggml_tensor *, void *) {
        return ggml_backend_meta_split_state { GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1 };
    };
    auto * dev = ggml_backend_meta_device(devs, 2, split, nullptr);
    auto * backend = ggml_backend_dev_init(dev, nullptr);
    auto * buft = ggml_backend_dev_buffer_type(dev);
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(buft, 512));
    const ggml_init_params params = {1024 * 1024, nullptr, true};
    ggml_context_ptr ca(ggml_init(params)), cb(ggml_init(params));
    auto allocate = [&](ggml_tensor * tensor, size_t offset) {
        ggml_backend_tensor_alloc(buffer.get(), tensor, (char *) ggml_backend_buffer_get_base(buffer.get()) + offset);
    };
    auto graph = [](ggml_context * ctx, ggml_tensor * tensor) {
        auto * gf = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(gf, tensor);
        gf->uid = ggml_graph_next_uid();
        return gf;
    };
    auto * input = ggml_new_tensor_1d(ca.get(), GGML_TYPE_F32, 8);
    ggml_set_input(input);
    allocate(input, 0);
    auto * oa = ggml_scale(ca.get(), input, 2);
    auto * ga = graph(ca.get(), oa);
    allocate(oa, 64);
    auto pa = ggml_backend_graph_plan_create(backend, ga);
    auto * ob = ggml_scale(cb.get(), input, 3);
    auto * gb = graph(cb.get(), ob);
    allocate(ob, 128);
    auto pb = ggml_backend_graph_plan_create(backend, gb);
    ggml_backend_graph_plan_free(backend, pa);
    const float values[] = {1, 2, 3, 4, 5, 6, 7, 8};
    ggml_backend_tensor_set(input, values, 0, sizeof(values));
    for (int iteration = 0; iteration < 40; ++iteration) {
        ggml_context_ptr ctx(ggml_init(params));
        auto * temporary = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 8);
        ggml_set_input(temporary);
        allocate(temporary, 320);
        auto * output = ggml_scale(ctx.get(), temporary, 4);
        auto * gf = graph(ctx.get(), output);
        allocate(output, 384);
        auto plan = ggml_backend_graph_plan_create(backend, gf);
        ggml_backend_tensor_set(temporary, values, 0, sizeof(values));
        require_contract(ggml_backend_graph_plan_compute(backend, plan) == GGML_STATUS_SUCCESS, "temporary plan compute failed");
        ggml_backend_synchronize(backend);
        ggml_backend_graph_plan_free(backend, plan);
        require_contract(ggml_backend_graph_plan_compute(backend, pb) == GGML_STATUS_SUCCESS, "borrowed input plan compute failed");
        ggml_backend_synchronize(backend);
        float actual[8];
        ggml_backend_tensor_get(ob, actual, 0, sizeof(actual));
        for (int i = 0; i < 8; ++i) { require_contract(actual[i] == 3 * values[i], "borrowed input owner recycled"); }
    }
    auto * fallback = ggml_backend_cpu_init();
    ggml_backend_t backends[] = { backend, fallback };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, 128, false, true));
    require_contract(ggml_backend_sched_alloc_graph(sched.get(), gb), "meta scheduler allocation failed");
    ggml_context_ptr measure(ggml_init(params));
    auto * mi = ggml_new_tensor_1d(measure.get(), GGML_TYPE_F32, 8192);
    ggml_set_input(mi);
    auto * mo = ggml_scale(measure.get(), mi, 2);
    auto * mg = graph(measure.get(), mo);
    ggml_backend_sched_set_tensor_backend(sched.get(), mi, fallback);
    ggml_backend_sched_set_tensor_backend(sched.get(), mo, fallback);
    require_contract(ggml_backend_sched_reserve(sched.get(), mg), "CPU reserve after meta allocation failed");
    sched.reset();
    ggml_backend_free(fallback);
    ggml_backend_graph_plan_free(backend, pb);
    buffer.reset();
    ggml_backend_free(backend);
    llama_backend_free();
    puts("M: Meta(CPU,CPU) retains a borrowed input owner through plan eviction and 40 container rotations");
}

int main(int argc, char ** argv) {
    try {
        require_contract(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
        const std::string mode = argc > 1 ? argv[1] : "all";
        if (mode == "epoch" || mode == "all") { epochs(); }
        if (mode == "meta" || mode == "all") { meta_input_owner(); }
        if (mode == "binding" || mode == "all" || mode == "stale-binding") { bindings(mode == "stale-binding"); }
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "contract failure: %s\n", error.what());
        return 1;
    }
}
