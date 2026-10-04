#define main mask_test_main
#include "../tests/test-kv-mask.cpp"
#undef main
#include <chrono>

template<typename F>
static double median_us(F fn) {
    std::vector<double> samples;
    for (int r = 0; r < 51; ++r) {
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i) {
            fn();
        }
        samples.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count()/20);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size()/2];
}

int main() {
    mask_require(getenv("CUDA_VISIBLE_DEVICES") && !*getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    llama_model_ptr model(llama_model_create(LLM_ARCH_LLAMA, mp));
    auto hp = model->hparams;
    hp.no_alloc = true;
    puts("label,depth,streams,rows_per_stream,prefix,full_bytes,suffix_bytes,full_fill_us,compact_fill_us");
    for (int depth : {8192, 32768, 90112}) {
        llama_kv_cache cache(*model, hp, GGML_TYPE_F16, GGML_TYPE_F16, false, false, false,
                             90624, 2, 256, 0, LLAMA_SWA_TYPE_NONE, nullptr, [](uint32_t) { return false; }, nullptr, nullptr);
        for (int s = 0; s < 2; ++s) {
            auto & cells = const_cast<llama_kv_cells &>(cache.get_cells(s));
            for (int j = 0; j < depth + 4; ++j) {
                cells.pos_set(j, j);
                cells.seq_add(j, s);
            }
        }
        for (int rows : {1, 4, 6}) {
            llama_pos positions[12] = {};
            int32_t counts[12];
            llama_seq_id streams[] = {0, 1};
            llama_seq_id * ids[12];
            for (int s = 0; s < 2; ++s) {
                for (int r = 0; r < rows; ++r) {
                    positions[s*rows + r] = depth + r;
                    counts[s*rows + r] = 1;
                    ids[s*rows + r] = &streams[s];
                }
            }
            llama_ubatch ub = {};
            ub.n_tokens = 2*rows;
            ub.n_seq_tokens = rows;
            ub.n_seqs = ub.n_seqs_unq = 2;
            ub.n_pos = 1;
            ub.b_equal_seqs = true;
            ub.pos = positions;
            ub.n_seq_id = counts;
            ub.seq_id = ids;
            ub.seq_id_unq = streams;
            const int n_kv = depth + 256;
            const uint32_t candidate = cache.get_kq_mask_prefix(ub, n_kv, true);
            const uint32_t prefix = candidate >= 32768 ? candidate : 0;
            ggml_context_ptr ctx(ggml_init({4*ggml_tensor_overhead(), nullptr, true}));
            auto * full = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, n_kv, rows, 1, 2);
            auto * suffix = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, n_kv - prefix, rows, 1, 2);
            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
            const double full_us = median_us([&] { cache.set_input_kq_mask(full, &ub, true); });
            const double compact_us = median_us([&] { cache.set_input_kq_mask(suffix, &ub, true, prefix); });
            printf("M,%d,2,%d,%u,%zu,%zu,%.3f,%.3f\n", depth, rows, prefix, ggml_nbytes(full), ggml_nbytes(suffix), full_us, compact_us);
        }
    }
    llama_backend_free();
}
