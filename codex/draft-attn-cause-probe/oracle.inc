struct cause_oracle_result {
    std::vector<float> values;
    uint64_t hash;
};

static cause_oracle_result cause_oracle(const ggml_tensor * original, bool round_v = false) {
    GGML_ASSERT(original->src[1]->type == GGML_TYPE_F16 && original->src[2]->type == GGML_TYPE_F16);
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
    ggml_tensor * inputs[4];
    for (int i = 0; i < 4; ++i) {
        const ggml_tensor * src = original->src[i];
        GGML_ASSERT(src != nullptr);
        inputs[i] = ggml_new_tensor(ctx, i == 2 && !round_v ? GGML_TYPE_F32 : src->type, GGML_MAX_DIMS, src->ne);
    }
    ggml_tensor * out = ggml_flash_attn_ext(ctx, inputs[0], inputs[1], inputs[2], inputs[3], 1.0f / std::sqrt(float(inputs[0]->ne[0])), 0, 0);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, round_v ? 1 : 8);
    ggml_backend_cpu_set_use_ref(cpu, !round_v);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    GGML_ASSERT(buffer != nullptr);
    uint64_t hash = 14695981039346656037ull;
    for (int i = 0; i < 4; ++i) {
        const auto values = tensor_to_float(original->src[i]);
        const auto * bytes = (const uint8_t *) values.data();
        for (size_t j = 0; j < values.size() * sizeof(float); ++j) {
            hash = (hash ^ bytes[j]) * 1099511628211ull;
        }
        if (inputs[i]->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(inputs[i], values.data(), 0, values.size() * sizeof(float));
        } else {
            std::vector<ggml_fp16_t> raw(values.size());
            ggml_fp32_to_fp16_row(values.data(), raw.data(), values.size());
            ggml_backend_tensor_set(inputs[i], raw.data(), 0, raw.size() * sizeof(ggml_fp16_t));
        }
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    auto values = tensor_to_float(out);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return {std::move(values), hash};
}
