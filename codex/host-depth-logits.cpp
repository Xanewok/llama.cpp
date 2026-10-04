#define main architecture_test_main
#include "../tests/test-llama-archs.cpp"
#undef main
#include "../src/llama-model.h"
#include "../src/llama-memory-hybrid.h"
#include <chrono>
#include <fstream>
#include <functional>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

struct batch_storage {
    std::vector<llama_pos> pos;
    std::vector<int32_t> count;
    std::vector<llama_seq_id *> ids;
    llama_seq_id seq = 0;
    llama_ubatch ub = {};
    batch_storage(int n, int start) : pos(n), count(n, 1), ids(n, &seq) {
        for (int i = 0; i < n; ++i) { pos[i] = start + i; }
        ub.n_tokens = ub.n_seq_tokens = n;
        ub.n_seqs = ub.n_seqs_unq = ub.n_pos = 1;
        ub.b_equal_seqs = true;
        ub.pos = pos.data(); ub.n_seq_id = count.data(); ub.seq_id = ids.data(); ub.seq_id_unq = &seq;
    }
};

static double timed(const std::function<void()> & fn, int repeats = 101, int inner = 10) {
    std::vector<double> samples;
    for (int r = 0; r < repeats; ++r) {
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < inner; ++i) { fn(); }
        samples.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count()/inner);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size()/2];
}

static llama_kv_cache::slot_info slot(int n, int start) {
    llama_kv_cache::slot_info result = {0, 0, {0}, {{}}};
    for (int i = 0; i < n; ++i) { result.idxs[0].push_back(start + i); }
    return result;
}

static void write_tensor(std::ofstream & out, ggml_tensor * t) {
    out.write((const char *) t->data, ggml_nbytes(t));
    require(bool(out), "mask write failed");
}


static void logits_test(llama_model & model, int depth, bool draft, std::ofstream & out) {
    auto cp = llama_context_default_params();
    cp.n_ctx = depth + 256; cp.n_batch = cp.n_ubatch = 8;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.n_outputs_max = cp.n_outputs_max_per_seq = 6;
    cp.kv_unified = true; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.ctx_type = draft ? LLAMA_CONTEXT_TYPE_MTP : LLAMA_CONTEXT_TYPE_DEFAULT;
    llama_context_ptr context(llama_init_from_model(&model, cp));
    require(bool(context), "context creation failed");
    auto * mem = llama_get_memory(context.get());
    auto * cache = draft ? dynamic_cast<llama_kv_cache *>(mem) : dynamic_cast<llama_memory_hybrid *>(mem)->get_mem_attn();
    batch_storage history(depth - 6, 0);
    cache->apply_ubatch(slot(depth - 6, 0), history.ub);
    std::vector<float> hidden(model.hparams.n_embd, 0.1f);
    auto * batch = llama_batch_ext_init(context.get());
    int step = 0, next_pos = depth - 6;
    std::vector<int> schedule = {6, 1, 1, 1, 1, 1, 1, 1};
    if (std::getenv("MASK_UPLOAD_META")) {
        // Row changes at 8 and 10 force a fresh compact graph; 9 and 11 mutate inside its prefix and must not reuse it.
        schedule.insert(schedule.end(), {2, 2, 3, 3});
    }
    for (int rows : schedule) {
        if (step == 8) {
            cache->seq_cp(1, 0, depth - 6, depth - 5);
            cache->seq_rm(1, depth - 6, depth - 5);
        }
        if (step == 9) {
            cache->seq_cp(0, 1, 19, 20);
            cache->seq_rm(0, 19, 20);
        }
        if (step == 10) {
            cache->seq_cp(1, 0, 19, 20);
            cache->seq_rm(1, 19, 20);
        }
        // seq_rm moves the head to the freed cells; free more than the ubatch refills so a hole stays.
        if (step == 11) { cache->seq_rm(0, 17, 21); }
        if (step == 6) {
            cache->seq_cp(0, 1, depth - 6, depth - 5);
            cache->seq_rm(0, depth - 6, depth - 5);
        }
        if (step == 7) {
            auto & cells = const_cast<llama_kv_cells &>(cache->get_cells(0));
            cells.rm(depth - 5);
            cells.pos_set(depth - 5, depth - 4);
            cells.seq_add(depth - 5, 0);
        }
        ++step;
        llama_batch_ext_clear(batch);
        const int start = next_pos; next_pos += rows;
        for (int i = 0; i < rows; ++i) {
            int idx = llama_batch_ext_add_token(batch, 0, 3 + i);
            if (draft) { require(llama_batch_ext_set_embd_token(batch, idx, {hidden.data(), 1, hidden.size()}), "draft hidden setup failed"); }
            llama_pos pos[] = {start + i, start + i, start + i, 0};
            require(llama_batch_ext_set_pos(batch, idx, pos) && llama_batch_ext_set_output_logits(batch, idx, true), "batch setup failed");
        }
        require(llama_process(context.get(), LLAMA_PROCESS_TYPE_DECODE, batch) == 0, "decode failed");
        if (!draft && step == 1) {
            const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
            const size_t bytes = llama_state_seq_get_size_ext(context.get(), 0, flags);
            std::vector<uint8_t> state(bytes);
            std::printf("STATE_BYTES,tiny_target,%d,%zu\n", depth, bytes);
            std::printf("M,tiny_target,%d,6,checkpoint_size,%.3f\n", depth, timed([&] { require(llama_state_seq_get_size_ext(context.get(), 0, flags) == bytes, "checkpoint size changed"); }, 31, 1));
            std::printf("M,tiny_target,%d,6,checkpoint_copy,%.3f\n", depth, timed([&] { require(llama_state_seq_get_data_ext(context.get(), state.data(), state.size(), 0, flags) == bytes, "checkpoint copy failed"); }, 31, 1));
        }
        for (int i = 0; i < rows; ++i) {
            auto * logits = llama_get_logits_ith(context.get(), i);
            require(logits != nullptr, "missing logits");
            const int nv = llama_vocab_n_tokens(llama_model_get_vocab(&model));
            for (int j = 0; j < nv; ++j) { require(std::isfinite(logits[j]), "nonfinite logits"); }
            out.write((const char *) logits, nv*sizeof(float));
        }
    }
    llama_batch_ext_free(batch);
}

int main(int argc, char ** argv) {
    try {
        require(argc == 3 || argc == 4, "usage: host-depth MASKS LOGITS [CPU_MODEL]");
        require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
        llama_log_set([](ggml_log_level level, const char * text, void *) { if (level >= GGML_LOG_LEVEL_WARN) { fputs(text, stderr); } }, nullptr);
        llama_backend_init();
        if (argc == 4) {
            auto mp = llama_model_default_params(); mp.n_gpu_layers = 0; mp.load_mtp = true;
            llama_model_ptr real(llama_model_load_from_file(argv[3], mp));
            require(bool(real), "CPU model load failed");
            std::ofstream logits(argv[2], std::ios::binary);
            require(bool(logits), "logits open failed");
            logits_test(*real, 8192, false, logits);
            logits_test(*real, 8192, true, logits);
            std::puts("PASS: real CPU target and MTP logits finite (zero historical KV, metadata depth 8192)");
            return 0;
        }
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
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_dev_t cpu_pair[] = {cpu_dev, cpu_dev, nullptr};
        if (std::getenv("MASK_UPLOAD_META")) {
            gguf_set_val_u32(meta.get(), "qwen35.embedding_length", 64);
            gguf_set_val_u32(meta.get(), "qwen35.feed_forward_length", 96);
            gguf_set_val_u32(meta.get(), "qwen35.attention.head_count", 2);
            gguf_set_val_u32(meta.get(), "qwen35.attention.head_count_kv", 2);
            gguf_set_val_u32(meta.get(), "qwen35.ssm.inner_size", 64);
            gguf_set_val_u32(meta.get(), "qwen35.ssm.time_step_rank", 2);
            gguf_set_val_u32(meta.get(), "qwen35.ssm.group_count", 2);
            mp.devices = cpu_pair; mp.split_mode = LLAMA_SPLIT_MODE_TENSOR; mp.n_gpu_layers = 99;
        }
        tensor_data_params tp = {1729, 0.1f};
        llama_model_ptr model(llama_model_init_from_user(meta.get(), set_tensor_data, &tp, mp));
        require(bool(model), "tiny model creation failed");
        std::ofstream masks(argv[1], std::ios::binary), logits(argv[2], std::ios::binary);
        require(masks && logits, "output open failed");
        std::puts("label,cache,n_kv,n_tokens,operation,median_us");
        for (int depth : {8192, 90112}) {
            for (bool draft : {false, true}) {
                logits_test(*model, depth, draft, logits);
            }
        }
        std::puts("PASS: negative controls, target and MTP logits finite at both depths");
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
