// Production-shaped mask round: Qwen35 ubatches carry n_pos = 4 (IMROPE), text positions broadcast.
#define main architecture_test_main
#include "../tests/test-llama-archs.cpp"
#undef main
#include "../src/llama-model.h"
#include "../src/llama-kv-cache.h"
#include <chrono>
#include <fstream>
#include <functional>

static void require(bool ok, const char * message) { if (!ok) { throw std::runtime_error(message); } }

struct qb {
    std::vector<llama_pos> pos; std::vector<int32_t> count; std::vector<llama_seq_id *> ids;
    llama_seq_id seq = 0; llama_ubatch ub = {};
    qb(int n, int start, int n_pos) : pos((size_t) n*n_pos), count(n, 1), ids(n, &seq) {
        for (int i = 0; i < n; ++i) {
            pos[i] = start + i;
            if (n_pos == 4) { pos[i + n] = start + i; pos[i + 2*n] = start + i; pos[i + 3*n] = 0; }
        }
        ub.n_tokens = ub.n_seq_tokens = n; ub.n_seqs = ub.n_seqs_unq = 1; ub.n_pos = n_pos;
        ub.b_equal_seqs = true; ub.pos = pos.data(); ub.n_seq_id = count.data(); ub.seq_id = ids.data(); ub.seq_id_unq = &seq;
    }
};

static double timed(const std::function<void()> & fn, int repeats = 101) {
    std::vector<double> s;
    for (int r = 0; r < repeats; ++r) {
        auto t0 = std::chrono::steady_clock::now(); fn();
        s.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(s.begin(), s.end()); return s[s.size()/2];
}

static llama_kv_cache::slot_info slot(int n, int start) {
    llama_kv_cache::slot_info r = {0, 0, {0}, {{}}};
    for (int i = 0; i < n; ++i) { r.idxs[0].push_back(start + i); }
    return r;
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "usage: bench-mrope MASKS");
        require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
        llama_log_set([](ggml_log_level l, const char * t, void *) { if (l >= GGML_LOG_LEVEL_ERROR) { fputs(t, stderr); } }, nullptr);
        llama_backend_init();
        auto meta = get_gguf_ctx(LLM_ARCH_QWEN35, false, 32);
        gguf_set_val_u32(meta.get(), "qwen35.block_count", 5);
        gguf_set_val_u32(meta.get(), "qwen35.nextn_predict_layers", 1);
        gguf_set_val_u32(meta.get(), "qwen35.embedding_length", 32);
        gguf_set_val_u32(meta.get(), "qwen35.feed_forward_length", 48);
        gguf_set_val_u32(meta.get(), "qwen35.attention.head_count", 1);
        gguf_set_val_u32(meta.get(), "qwen35.attention.head_count_kv", 1);
        gguf_set_val_u32(meta.get(), "qwen35.attention.key_length", 32);
        gguf_set_val_u32(meta.get(), "qwen35.attention.value_length", 32);
        gguf_set_val_u32(meta.get(), "qwen35.rope.dimension_count", 32);
        gguf_set_val_u32(meta.get(), "qwen35.ssm.inner_size", 32);
        gguf_set_val_u32(meta.get(), "qwen35.ssm.state_size", 32);
        gguf_set_val_u32(meta.get(), "qwen35.ssm.time_step_rank", 1);
        gguf_set_val_u32(meta.get(), "qwen35.ssm.group_count", 1);
        uint32_t sections[] = {4, 4, 8, 0};
        gguf_set_arr_data(meta.get(), "qwen35.rope.dimension_sections", GGUF_TYPE_UINT32, sections, 4);
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 0; mp.load_mtp = true;
        tensor_data_params tp = {1729, 0.1f};
        llama_model_ptr model(llama_model_init_from_user(meta.get(), set_tensor_data, &tp, mp));
        require(bool(model), "tiny model creation failed");
        require(model->hparams.n_pos_per_embd() == 4, "expected IMROPE n_pos_per_embd 4");
        std::ofstream out(argv[1], std::ios::binary);
        std::puts("n_pos,depth,round_us");
        for (int n_pos : {1, 4}) {
            for (int depth : {8192, 90112}) {
                auto hp = model->hparams; hp.no_alloc = true; hp.n_layer_all = 65; hp.n_layer_nextn = 1;
                hp.n_head_arr.fill(24); hp.n_head_kv_arr.fill(4); hp.n_embd_head_k_full = hp.n_embd_head_v_full = 256;
                llama_kv_cache cache(*model, hp, GGML_TYPE_F16, GGML_TYPE_F16, false, false, true, 102400, 1, 256,
                                     0, LLAMA_SWA_TYPE_NONE, nullptr, [](uint32_t il) { return il == 64; }, nullptr, nullptr);
                qb hist(depth, 0, n_pos); cache.apply_ubatch(slot(depth, 0), hist.ub);
                qb q6(6, depth - 6, n_pos), q1(1, depth - 1, n_pos);
                ggml_context_ptr ctx(ggml_init({4*ggml_tensor_overhead(), nullptr, true}));
                auto * m6 = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, depth, 6, 1, 1);
                auto * m1 = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, depth, 1, 1, 1);
                ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
                const double us = timed([&] {
                    cache.set_input_kq_mask(m6, &q6.ub, true);
                    for (int d = 0; d < 5; ++d) { cache.set_input_kq_mask(m1, &q1.ub, true); }
                });
                std::printf("%d,%d,%.3f\n", n_pos, depth, us);
                out.write((const char *) m6->data, ggml_nbytes(m6));
                out.write((const char *) m1->data, ggml_nbytes(m1));
            }
        }
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
