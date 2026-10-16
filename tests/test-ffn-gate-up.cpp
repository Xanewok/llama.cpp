#include "../src/llama-model-loader.h"
#include "../src/llama-model.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-graph.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

struct file_closer {
    void operator()(FILE * file) const { fclose(file); }
};

static constexpr int64_t k = 512;
static constexpr int64_t m = 640; // An unequal 256/384 split after rounding to 128 rows.

static std::vector<uint8_t> weight_data(ggml_type type, int seed) {
    std::vector<float> data(k*m);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = 0.1f*std::sin(float(i*13 + seed));
    }
    std::vector<uint8_t> result(ggml_row_size(type, k)*m);
    require(ggml_quantize_chunk(type, data.data(), result.data(), 0, m, k, nullptr) == result.size(), "quantize size");
    return result;
}

static FILE * fixture(ggml_type gate_type, ggml_type up_type,
        const std::vector<uint8_t> & gate, const std::vector<uint8_t> & up) {
    ggml_context_ptr ctx(ggml_init({4*ggml_tensor_overhead(), nullptr, true}));
    gguf_context_ptr gguf(gguf_init_empty());
    gguf_set_val_str(gguf.get(), "general.architecture", "qwen35");
    ggml_tensor * g = ggml_new_tensor_2d(ctx.get(), gate_type, k, m);
    ggml_tensor * u = ggml_new_tensor_2d(ctx.get(), up_type, k, m);
    ggml_set_name(g, "blk.0.ffn_gate.weight");
    ggml_set_name(u, "blk.0.ffn_up.weight");
    gguf_add_tensor(gguf.get(), g);
    gguf_add_tensor(gguf.get(), u);
    gguf_set_tensor_data(gguf.get(), g->name, gate.data());
    gguf_set_tensor_data(gguf.get(), u->name, up.data());
    FILE * file = tmpfile();
    require(file && gguf_write_to_file_ptr(gguf.get(), file, false), "write fixture");
    rewind(file);
    return file;
}

static ggml_backend_device ranks[2];

static ggml_backend_t init_rank(ggml_backend_dev_t dev, const char * params) {
    auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto * backend = cpu->iface.init_backend(cpu, params);
    backend->device = dev;
    ggml_backend_cpu_set_n_threads(backend, 1);
    return backend;
}

static std::vector<float> evaluate(ggml_backend_t backend, ggml_tensor * gate, ggml_tensor * up,
        ggml_tensor * merged, int columns) {
    llm_graph_result graph_result(32);
    llama_adapter_loras loras;
    llm_graph_params params = {};
    params.arch = LLM_ARCH_QWEN35;
    params.hparams.n_layer_all = 1;
    params.hparams.n_embd = k;
    params.ubatch.n_tokens = columns;
    params.loras = &loras;
    params.res = &graph_result;
    llm_graph_context builder(params);
    auto * ctx = graph_result.get_ctx();
    auto * graph = graph_result.get_gf();
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, columns);
    ggml_set_name(input, "input");
    auto * result = builder.build_ffn(input, up, nullptr, nullptr, gate, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, LLM_FFN_SILU, LLM_FFN_PAR, 0, merged);
    ggml_build_forward_expand(graph, result);
    if (merged && columns > 1) {
        require(ggml_graph_n_nodes(graph) == 2 && result->op == GGML_OP_GLU && result->src[0]->op == GGML_OP_MUL_MAT && !result->src[1],
                "one matmul and one GLU");
    }
    if (merged && columns == 1) {
        auto ** nodes = ggml_graph_nodes(graph);
        const int n = ggml_graph_n_nodes(graph);
        require(nodes[n-3]->op == GGML_OP_MUL_MAT && nodes[n-2]->op == GGML_OP_MUL_MAT && nodes[n-1]->op == GGML_OP_GLU,
                "single-column fusion adjacency");
        require(ggml_can_fuse_subgraph(graph, n-3, {GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU}, {n-1}),
                "single-column fusion subgraph");
        require(ggml_are_same_shape(gate, up) && ggml_are_same_stride(gate, up), "MMVQ weight compatibility");
    }
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    require(ggml_gallocr_alloc_graph(allocator.get(), graph), "allocate graph");
    std::vector<float> data(k*columns);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::cos(float(i*7 + 11));
    }
    ggml_backend_tensor_set(input, data.data(), 0, data.size()*sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute graph");
    std::vector<float> output(m*columns);
    ggml_backend_tensor_get(result, output.data(), 0, output.size()*sizeof(float));
    return output;
}

static void run(ggml_type type, llama_load_mode mode, bool split, bool recurrent) {
    auto gate_data = weight_data(type, 1);
    auto up_data = weight_data(type, 3);
    std::unique_ptr<FILE, file_closer> file(fixture(type, type, gate_data, up_data));
    std::vector<std::string> splits;
    llama_model_loader loader(nullptr, nullptr, nullptr, "", splits, file.get(), mode, true, false, false, nullptr, nullptr);
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35, llama_model_default_params()));
    model->hparams.n_layer_all = 1;
    model->hparams.n_ff_arr[0] = m;
    model->hparams.n_head_arr[0] = model->hparams.n_head_kv_arr[0] = 1;
    model->hparams.n_embd_head_k_full = model->hparams.n_embd_head_v_full = 128;
    model->hparams.is_recr_impl[0] = recurrent;
    model->hparams.ssm_n_group = model->hparams.ssm_dt_rank = 1;
    model->hparams.ssm_d_state = 128;
    const char * env = std::getenv("LLAMA_FFN_GATE_UP_CAT");
    if (!env || std::strcmp(env, "1") != 0) {
        auto * base = static_cast<llama_model_base *>(model.get());
        base->ml = &loader;
        llama_layer layer;
        require(!base->create_tensor_gate_up(layer, 0, k, m, 0) && loader.ctx_map.empty(), "default-off loader");
    }
    ggml_context_ptr down_ctx(ggml_init({ggml_tensor_overhead(), nullptr, true}));
    auto * down = ggml_new_tensor_2d(down_ctx.get(), type, m, k);
    model->tensors_by_name.emplace_back("blk.0.ffn_down.weight", down);
    llama_meta_device_get_split_state_userdata ud = {2, model.get()};
    auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devices[] = {&ranks[0], &ranks[1]};
    auto * dev = split ? ggml_backend_meta_device(devices, 2, llama_meta_device_get_split_state, &ud) : cpu;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    auto * buft = ggml_backend_dev_buffer_type(dev);
    buft_list_t bufts = {{dev, buft}};
    LLM_TN tn(LLM_ARCH_QWEN35);
    auto * merged = loader.create_tensor(model->hparams, &bufts, &bufts, &bufts, &bufts,
            tn(LLM_TENSOR_FFN_GATE_UP, "weight", 0), {k, 2*m}, 0, {"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"});
    loader.done_getting_tensors();
    require(loader.ctx_map.size() == 1, "one weight context");
    auto * ctx = loader.ctx_map.begin()->second.get();
    auto * gate = ggml_get_tensor(ctx, "blk.0.ffn_gate.weight");
    auto * up = ggml_get_tensor(ctx, "blk.0.ffn_up.weight");
    require(gate && up && gate->view_src == merged && up->view_src == merged, "loader weight views");
    ggml_context_ptr size_ctx(ggml_init({ggml_tensor_overhead(), nullptr, true}));
    auto * size_weight = ggml_new_tensor_2d(size_ctx.get(), type, k, 2*m);
    ggml_set_name(size_weight, merged->name);
    require(ggml_backend_alloc_ctx_tensors_from_buft_size(size_ctx.get(), buft) == ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft),
            "views allocate no weight storage");
    if (split) {
        const auto ss = llama_meta_device_get_split_state(merged, &ud);
        require(ss.axis == GGML_BACKEND_SPLIT_AXIS_1 && ss.n_segments == 1 && ss.nr[0] == 2 && ss.ne[0] == 256 && ss.ne[1] == 384,
                "paired unequal split");
    }
    ggml_backend_buffer_ptr weights(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft));
    require(bool(weights), "allocate weights");
    ggml_backend_buffer_set_usage(weights.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    loader.init_mappings(false);
    llama_buf_map buffers = {{0, weights.get()}};
    require(loader.load_all_data(ctx, buffers, nullptr, nullptr, nullptr), "load concat");
    require(loader.size_done == gate_data.size() + up_data.size(), "load progress");
    std::vector<uint8_t> raw(gate_data.size() + up_data.size());
    ggml_backend_tensor_get(merged, raw.data(), 0, raw.size());
    require(!memcmp(raw.data(), gate_data.data(), gate_data.size()) && !memcmp(raw.data() + gate_data.size(), up_data.data(), up_data.size()),
            "loader preserves source bytes and row order");
    for (auto * view : {gate, up}) {
        const auto & expected = view == gate ? gate_data : up_data;
        ggml_backend_tensor_get(view, raw.data(), 0, expected.size());
        require(!memcmp(raw.data(), expected.data(), expected.size()), "weight view bytes");
    }
    model->layers.resize(1);
    model->layers[0].ffn_gate_up = merged;
    model->layers[0].ffn_gate = gate;
    model->layers[0].ffn_up = up;
    llama_model_saver saver(model.get());
    saver.add_tensors_from_model();
    require(gguf_get_n_tensors(saver.gguf_ctx) == 2 && gguf_find_tensor(saver.gguf_ctx, merged->name) < 0,
            "saved GGUF retains only original tensors");
    ggml_backend_ptr reference(ggml_backend_cpu_init());
    ggml_backend_cpu_set_n_threads(reference.get(), 1);
    ggml_context_ptr ref_ctx(ggml_init({2*ggml_tensor_overhead(), nullptr, true}));
    auto * ref_gate = ggml_new_tensor_2d(ref_ctx.get(), type, k, m);
    auto * ref_up = ggml_new_tensor_2d(ref_ctx.get(), type, k, m);
    ggml_backend_buffer_ptr ref_weights(ggml_backend_alloc_ctx_tensors(ref_ctx.get(), reference.get()));
    ggml_backend_tensor_set(ref_gate, gate_data.data(), 0, gate_data.size());
    ggml_backend_tensor_set(ref_up, up_data.data(), 0, up_data.size());
    for (int columns : {1, 6, 12}) {
        const auto expected = evaluate(reference.get(), ref_gate, ref_up, nullptr, columns);
        const auto actual = evaluate(backend.get(), gate, up, merged, columns);
        require(!memcmp(actual.data(), expected.data(), expected.size()*sizeof(float)), "bit-identical FFN output");
    }
    printf("PASS type=%s mode=%d split=%d recurrent=%d columns=1,6,12\n", ggml_type_name(type), int(mode), split, recurrent);
}

static void reject_mixed() {
    auto gate = weight_data(GGML_TYPE_IQ4_XS, 1);
    auto up = weight_data(GGML_TYPE_Q6_K, 3);
    std::unique_ptr<FILE, file_closer> file(fixture(GGML_TYPE_IQ4_XS, GGML_TYPE_Q6_K, gate, up));
    std::vector<std::string> splits;
    llama_model_loader loader(nullptr, nullptr, nullptr, "", splits, file.get(), LLAMA_LOAD_MODE_NONE, true, false, false, nullptr, nullptr);
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35, llama_model_default_params()));
    auto * base = static_cast<llama_model_base *>(model.get());
    base->ml = &loader;
    llama_layer layer;
    require(!base->create_tensor_gate_up(layer, 0, k, m, 0), "mixed types stay separate");
    require(!layer.ffn_gate_up && loader.ctx_map.empty(), "mixed fallback allocates nothing");
}

int main() {
    try {
        setvbuf(stdout, nullptr, _IOLBF, 0);
        ggml_backend_load_all();
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        require(cpu != nullptr, "CPU backend");
        for (auto & rank : ranks) {
            rank = *cpu;
            rank.iface.init_backend = init_rank;
        }
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_IQ4_XS, GGML_TYPE_Q6_K}) {
            for (auto mode : {LLAMA_LOAD_MODE_NONE, LLAMA_LOAD_MODE_MMAP}) {
                run(type, mode, false, false);
                run(type, mode, true, false);
                run(type, mode, true, true);
            }
        }
        reject_mixed();
        puts("test-ffn-gate-up: PASS");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "test-ffn-gate-up: %s\n", e.what());
        return 1;
    }
}
