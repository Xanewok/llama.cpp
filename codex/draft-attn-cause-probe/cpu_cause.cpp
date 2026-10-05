#define main unused_probe_main
#include "../../build-tmp/draft-attn-cause/driver.cpp"
#undef main

int main(int argc, char ** argv) {
    bool crossed = false;
    const int first_seed = argc > 1 ? std::stoi(argv[1]) : 0;
    const int end_seed = argc > 2 ? std::stoi(argv[2]) : 8;
    const bool broken_oracle = argc > 3;
    for (int seed = first_seed; seed < end_seed; ++seed) {
        const auto seed_text = std::to_string(seed);
        setenv("CAUSE_SEED", seed_text.c_str(), 1);
        for (int gqa : {4, 6}) {
            const int dim = 256, kv = 90112;
            std::array<int32_t, GGML_MAX_OP_PARAMS / sizeof(int32_t)> params = {};
            const float scale = 1.0f / 16;
            memcpy(&params[0], &scale, sizeof(scale));
            params[3] = GGML_PREC_F32;
            const size_t row = dim * sizeof(ggml_fp16_t);
            std::vector<input_tensor> sources = {
                {GGML_TYPE_F32, {dim, 1, 2 * gqa, 1}, {}},
                {GGML_TYPE_F16, {dim, kv, 2, 1}, {2, row, 2 * row * kv, 4 * row * kv}},
                {GGML_TYPE_F16, {dim, kv, 2, 1}, {2, row, 2 * row * kv, 4 * row * kv}},
                {GGML_TYPE_F16, {kv, 1, 1, 1}, {}},
            };
            test_generic_op test(GGML_OP_FLASH_ATTN_EXT, GGML_TYPE_F32, {dim, 2 * gqa, 1, 1}, params, sources);
            ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
            ggml_tensor * out = test.build_graph(ctx);
            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, out);
            ggml_backend_t cpu = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(cpu, 1);
            ggml_backend_cpu_set_use_ref(cpu, false);
            ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
            GGML_ASSERT(buffer != nullptr);
            const auto start = std::chrono::steady_clock::now();
            test.initialize_tensors(ctx);
            GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
            const auto native = tensor_to_float(out);
            ggml_backend_cpu_set_n_threads(cpu, 8);
            ggml_backend_cpu_set_use_ref(cpu, true);
            GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
            const auto reference = tensor_to_float(out);
            const auto oracle = cause_oracle(out, broken_oracle);
            test.initialize_tensors(ctx);
            const auto repeated = cause_oracle(out, broken_oracle);
            GGML_ASSERT(oracle.hash == repeated.hash && oracle.values == repeated.values);
            const auto mutant = cause_oracle(out, true);
            const double mutant_err = nmse(native.data(), mutant.values.data(), native.size());
            GGML_ASSERT(mutant.hash == oracle.hash && mutant_err == 0);
            const double err = nmse(oracle.values.data(), native.data(), native.size());
            crossed = crossed || err > test.max_nmse_err();
            GGML_ASSERT(err > 1e-4);
            const double reference_err = nmse(oracle.values.data(), reference.data(), reference.size());
            GGML_ASSERT(reference_err < 1e-10);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            printf("M: seed=%d gqa=%d kv=%d CPU_ERR=%.17g fixed_reference_err=%.17g tolerance=%.17g repeat_hash=%016llx half_oracle_mutant=%.17g seconds=%.3f\n", seed, gqa, kv, err, reference_err, test.max_nmse_err(), (unsigned long long) oracle.hash, mutant_err, seconds);
            ggml_backend_buffer_free(buffer);
            ggml_backend_free(cpu);
            ggml_free(ctx);
        }
    }
    GGML_ASSERT(crossed);
    puts("M: native half accumulation crossed tolerance; fixed reference matched the f32-V oracle without CUDA");
    return 0;
}
