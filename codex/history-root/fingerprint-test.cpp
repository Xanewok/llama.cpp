#define main history_tool_main
#include "history-root.cpp"
#undef main

static ggml_backend_meta_split_state test_split(const ggml_tensor * tensor, void *) {
    ggml_backend_meta_split_state state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    if (strcmp(tensor->name, "meta-a") == 0 || strcmp(tensor->name, "meta-b") == 0) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_0;
        state.ne[0] = state.ne[1] = 16;
    }
    return state;
}

static void test_meta(ggml_backend_dev_t cpu) {
    ggml_backend_dev_t devices[] = {cpu, cpu};
    auto * device = ggml_backend_meta_device(devices, 2, test_split, nullptr);
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    require(bool(backend), "CPU meta init failed");
    ggml_context_ptr inputs(ggml_init({1024*1024, nullptr, true}));
    ggml_context_ptr outputs(ggml_init({1024*1024, nullptr, true}));
    auto * a = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_F32, 32, 4);
    auto * b = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_F32, 32, 1);
    ggml_set_name(a, "meta-a");
    ggml_set_name(b, "meta-b");
    auto * y = ggml_mul_mat(outputs.get(), a, b);
    auto * z = ggml_scale(outputs.get(), y, 1);
    auto * graph = ggml_new_graph(outputs.get());
    ggml_build_forward_expand(graph, z);
    ggml_backend_buffer_ptr input_buffer(ggml_backend_alloc_ctx_tensors_from_buft(inputs.get(), ggml_backend_dev_buffer_type(device)));
    ggml_backend_buffer_ptr output_buffer(ggml_backend_alloc_ctx_tensors_from_buft(outputs.get(), ggml_backend_dev_buffer_type(device)));
    require(input_buffer && output_buffer, "CPU meta allocation failed");
    ggml_backend_buffer_set_usage(input_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(output_buffer.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    float av[128], bv[32], result[4];
    std::fill(std::begin(av), std::end(av), 1.0f);
    std::fill(std::begin(bv), std::end(bv), 1.0f);
    ggml_backend_tensor_set(a, av, 0, sizeof(av));
    ggml_backend_tensor_set(b, bv, 0, sizeof(bv));
    require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "full CPU meta graph failed");
    ggml_backend_tensor_get(z, result, 0, sizeof(result));
    fprintf(stdout, "CPU meta full result=%g nodes=%d\n", result[0], graph->n_nodes);
    require(result[0] == 32, "full CPU meta result failed");
    auto first = ggml_graph_view(graph, 0, 1);
    auto second = ggml_graph_view(graph, 1, 2);
    require(ggml_backend_graph_compute(backend.get(), &first) == GGML_STATUS_SUCCESS, "bare first node failed");
    require(ggml_backend_graph_compute(backend.get(), &second) == GGML_STATUS_SUCCESS, "bare second node failed");
    ggml_backend_tensor_get(z, result, 0, sizeof(result));
    require(result[0] == 16, "missing AllReduce negative control did not fail");
    complete_meta_node = true;
    require(ggml_backend_graph_compute(backend.get(), &first) == GGML_STATUS_SUCCESS, "completed first node failed");
    require(ggml_backend_graph_compute(backend.get(), &second) == GGML_STATUS_SUCCESS, "completed second node failed");
    complete_meta_node = false;
    ggml_backend_tensor_get(z, result, 0, sizeof(result));
    require(result[0] == 32, "AllReduce completion changed the arithmetic result");
    require(slices(y).size() == 2 && slices(z).size() == 2, "private CPU meta map access failed");
    fprintf(stdout, "meta completion controls passed: full=32 bare=16 completed=32\n");
}

int main() {
    llama_backend_init();
    auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    require(device != nullptr, "CPU device is required");
    ggml_context_ptr context(ggml_init({1024*1024, nullptr, true}));
    auto * padded = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 8, 3);
    auto * view = ggml_view_2d(context.get(), padded, 4, 3, padded->nb[1], 0);
    auto * contiguous = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 4, 3);
    auto * transposed = ggml_transpose(context.get(), contiguous);
    auto * transpose_reference = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 3, 4);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(context.get(), ggml_backend_dev_buffer_type(device)));
    require(bool(buffer), "CPU tensor allocation failed");
    float data[24], expected[12], expected_transpose[12];
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 8; ++col) {
            data[row*8 + col] = col < 4 ? row*4 + col + 1 : -111;
        }
        for (int col = 0; col < 4; ++col) {
            expected[row*4 + col] = row*4 + col + 1;
            expected_transpose[col*3 + row] = row*4 + col + 1;
        }
    }
    ggml_backend_tensor_set(padded, data, 0, sizeof(data));
    ggml_backend_tensor_set(contiguous, expected, 0, sizeof(expected));
    ggml_backend_tensor_set(transpose_reference, expected_transpose, 0, sizeof(expected_transpose));
    const auto first = fingerprint(view);
    require(first == fingerprint(contiguous), "strided row packing failed");
    require(fingerprint(transposed) == fingerprint(transpose_reference), "transposed packing failed");
    for (int row = 0; row < 3; ++row) {
        for (int col = 4; col < 8; ++col) {
            data[row*8 + col] = 999;
        }
    }
    ggml_backend_tensor_set(padded, data, 0, sizeof(data));
    require(first == fingerprint(view), "padding changed the logical fingerprint");
    data[0] = 17;
    ggml_backend_tensor_set(padded, data, 0, sizeof(data));
    require(first != fingerprint(view), "active-byte negative control was missed");
    fprintf(stdout, "fingerprint controls passed: %s\n", first.c_str());
    test_meta(device);
    return 0;
}
