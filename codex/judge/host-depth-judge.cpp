#define main architecture_test_main
#include "../../tests/test-llama-archs.cpp"
#undef main
#include "../../src/llama-model.h"
#include "../../src/llama-memory-hybrid.h"
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

static void compact_test(llama_kv_cache & cache, const batch_storage & query, int depth, int rows) {
    auto ub = query.ub;
    std::vector<llama_pos> positions(4*rows);
    for (int i = 0; i < rows; ++i) {
        positions[i] = query.pos[i];
        positions[rows + i] = positions[2*rows + i] = query.pos[i];
    }
    ub.n_pos = 4;
    ub.pos = positions.data();
    const uint32_t prefix = cache.get_kq_mask_prefix(ub, depth, true);
    require(prefix == uint32_t(depth - 256), "wrong compact prefix");
    ggml_context_ptr ref_ctx(ggml_init({8*ggml_tensor_overhead(), nullptr, true}));
    auto * reference = ggml_new_tensor_4d(ref_ctx.get(), GGML_TYPE_F16, depth, rows, 1, 1);
    ggml_backend_buffer_ptr ref_buffer(ggml_backend_alloc_ctx_tensors_from_buft(ref_ctx.get(), ggml_backend_cpu_buffer_type()));
    ggml_context_ptr ctx(ggml_init({64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true}));
    auto * suffix = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, depth - prefix, rows, 1, 1);
    ggml_set_input(suffix);
    auto * zeros = ggml_fill(ctx.get(), ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, prefix, rows, 1, 1), 0.0f);
    auto * compact = ggml_concat(ctx.get(), zeros, suffix, 0);
    ggml_set_output(compact);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, compact);
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devs[] = {cpu_dev, cpu_dev};
    auto * meta_dev = ggml_backend_meta_device(devs, 2, [](const ggml_tensor *, void *) {
        return ggml_backend_meta_split_state{GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    }, nullptr);
    ggml_backend_ptr meta(ggml_backend_dev_init(meta_dev, nullptr));
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    ggml_backend_cpu_set_n_threads(cpu.get(), 4);
    ggml_backend_t backends[] = {meta.get(), cpu.get()};
    auto * sched = ggml_backend_sched_new(backends, nullptr, 2, 2048, false, true);
    ggml_backend_sched_set_tensor_backend(sched, zeros, meta.get());
    ggml_backend_sched_set_tensor_backend(sched, compact, meta.get());
    require(ggml_backend_sched_alloc_graph(sched, graph), "compact graph allocation failed");
    require(ggml_backend_sched_get_tensor_backend(sched, compact) == meta.get(), "compact mask did not stay on meta backend");
    require(ggml_backend_sched_get_tensor_backend(sched, zeros) == meta.get(), "zero prefix did not stay on meta backend");
    require(ggml_backend_sched_get_tensor_backend(sched, zeros->src[0]) == meta.get(), "zero shape would require a host upload");
    std::vector<char> actual(ggml_nbytes(reference));
    auto check = [&] {
        cache.set_input_kq_mask(reference, &ub, true);
        cache.set_input_kq_mask(suffix, &ub, true, prefix);
        if (std::getenv("MASK_UPLOAD_WRONG_CONTROL")) {
            ((ggml_fp16_t *) suffix->data)[0] = ggml_fp32_to_fp16(-INFINITY);
        }
        require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "compact compute failed");
        ggml_backend_tensor_get(compact, actual.data(), 0, actual.size());
        require(std::memcmp(reference->data, actual.data(), actual.size()) == 0, "compact mask differs from host mask");
    };
    check();
    std::printf("M,compact,%d,%d,host_set,%.3f\n", depth, rows, timed([&] { cache.set_input_kq_mask(suffix, &ub, true, prefix); }));
    std::printf("D,compact,%d,%d,upload_bytes_per_device,%zu\n", depth, rows, ggml_nbytes(suffix));
    auto & cells = const_cast<llama_kv_cells &>(cache.get_cells(0));
    const int cell = depth - 2;
    const auto saved_pos = cells.pos_get(cell);
    const auto saved_ext = cells.ext_get(cell);
    cells.pos_add(cell, ub.pos[0] - saved_pos);
    auto ext = saved_ext; ext.y = ub.pos[rows] + 1;
    cells.ext_set(cell, ext);
    check();
    cells.pos_add(cell, saved_pos - ub.pos[0]); cells.ext_set(cell, saved_ext);
    const auto previous = actual;
    cache.seq_cp(0, 1, 17, 18); cache.seq_rm(0, 17, 18);
    require(cache.get_kq_mask_prefix(ub, depth, true) == 0, "sequence mutation did not invalidate prefix");
    cache.set_input_kq_mask(reference, &ub, true);
    require(std::memcmp(reference->data, previous.data(), previous.size()) != 0, "stale mask negative control missed mutation");
    cache.seq_cp(1, 0, 17, 18); cache.seq_rm(1, 17, 18);
    const auto old_pos = cells.pos_get(18);
    cells.pos_add(18, ub.pos[0] - old_pos); ext = cells.ext_get(18); const auto old_ext = ext;
    ext.y = ub.pos[rows] + 1; cells.ext_set(18, ext);
    require(cache.get_kq_mask_prefix(ub, depth, true) == 0, "M-RoPE mutation did not invalidate prefix");
    cells.pos_add(18, old_pos - ub.pos[0]); cells.ext_set(18, old_ext);
    cells.pos_add(19, depth);
    require(cache.get_kq_mask_prefix(ub, depth, true) == 0, "future position did not invalidate prefix");
    cells.pos_add(19, -depth);
    cache.seq_rm(0, 17, 18);
    require(cells.used_max_p1() == uint32_t(depth) && cells.get_used() == uint32_t(depth - 1), "judge: hole setup");
    require(cache.get_kq_mask_prefix(ub, depth, true) == 0, "judge: hole inside prefix did not invalidate prefix");
    {
        batch_storage refill(1, 17);
        cache.apply_ubatch(slot(1, 17), refill.ub);
    }
    require(cache.get_kq_mask_prefix(ub, depth, true) == prefix, "judge: refilled hole changed prefix");
    cache.seq_rm(0, depth - 256, -1);
    require(cache.get_kq_mask_prefix(ub, depth, true) >= prefix, "tail removal invalidated a visible prefix");
    check();
    batch_storage restore(256, depth - 256);
    cache.apply_ubatch(slot(256, depth - 256), restore.ub);
    ggml_backend_sched_free(sched);
}

static void cache_test(llama_model & model, int depth, bool draft, std::ofstream & out) {
    auto hp = model.hparams;
    hp.no_alloc = true;
    hp.n_layer_all = 65; hp.n_layer_nextn = 1;
    hp.n_head_arr.fill(24); hp.n_head_kv_arr.fill(4);
    hp.n_embd_head_k_full = hp.n_embd_head_v_full = 256;
    llama_kv_cache cache(model, hp, draft ? GGML_TYPE_F16 : GGML_TYPE_Q8_0,
                        draft ? GGML_TYPE_F16 : GGML_TYPE_Q8_0, false, false, true, 102400, 1, 256,
                        0, LLAMA_SWA_TYPE_NONE, nullptr,
                        [draft](uint32_t il) { return draft ? il == 64 : il < 64 && (il + 1)%4 == 0; }, nullptr, nullptr);
    require(cache.get_layer_ids().size() == (draft ? 1 : 16), "wrong layer count");
    batch_storage fill(depth - 6, 0);
    cache.apply_ubatch(slot(depth - 6, 0), fill.ub);
    batch_storage query(6, depth - 6);
    cache.apply_ubatch(slot(6, depth - 6), query.ub);
    ggml_context_ptr ctx(ggml_init({32*ggml_tensor_overhead(), nullptr, true}));
    auto * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 24);
    llm_graph_input_pos pos_input(4); pos_input.pos = pos;
    auto * k = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 6);
    auto * v = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 6);
    auto * m16 = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, depth, 6, 1, 1);
    auto * m32 = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, depth, 6, 1, 1);
    auto * k_rot = cache.build_input_k_rot(ctx.get());
    auto * v_rot = cache.build_input_v_rot(ctx.get());
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
    auto s = slot(6, depth - 6);
    volatile int64_t sink = 0;
    const char * kind = draft ? "draft" : "target";
    for (int rows : {1, 6}) {
        query.ub.n_tokens = query.ub.n_seq_tokens = rows;
        query.pos[0] = rows == 1 ? depth - 1 : depth - 6;
        llama_token dummy[6] = {}; query.ub.token = dummy;
        std::printf("M,%s,%d,%d,position_input,%.3f\n", kind, depth, rows, timed([&] { pos_input.set_input(&query.ub); }));
        query.ub.token = nullptr;
        compact_test(cache, query, depth, rows);
        std::printf("M,%s,%d,%d,mask_f16,%.3f\n", kind, depth, rows, timed([&] { cache.set_input_kq_mask(m16, &query.ub, true); }));
        std::printf("M,%s,%d,%d,mask_f32,%.3f\n", kind, depth, rows, timed([&] { cache.set_input_kq_mask(m32, &query.ub, true); }));
        std::printf("M,%s,%d,%d,find_slot,%.3f\n", kind, depth, rows, timed([&] { sink += cache.find_slot(query.ub, false).size(); }));
        auto sn = slot(rows, depth);
        std::printf("M,%s,%d,%d,prepare,%.3f\n", kind, depth, rows, timed([&] { sink += cache.prepare({query.ub}).size(); }));
        std::vector<double> apply_samples, remove_samples;
        for (int r = 0; r < 101; ++r) {
            auto start = std::chrono::steady_clock::now();
            cache.apply_ubatch(sn, query.ub);
            auto applied = std::chrono::steady_clock::now();
            cache.seq_rm(0, depth - 6, -1);
            auto removed = std::chrono::steady_clock::now();
            apply_samples.push_back(std::chrono::duration<double, std::micro>(applied - start).count());
            remove_samples.push_back(std::chrono::duration<double, std::micro>(removed - applied).count());
            batch_storage restore(6, depth - 6);
            cache.apply_ubatch(s, restore.ub);
        }
        std::sort(apply_samples.begin(), apply_samples.end());
        std::sort(remove_samples.begin(), remove_samples.end());
        std::printf("M,%s,%d,%d,apply_ubatch,%.3f\n", kind, depth, rows, apply_samples[50]);
        std::printf("M,%s,%d,%d,seq_rm_tail,%.3f\n", kind, depth, rows, remove_samples[50]);
        std::printf("M,%s,%d,%d,n_kv,%.3f\n", kind, depth, rows, timed([&] { sink += cache.get_n_kv(s); }, 101, 100));
        std::printf("M,%s,%d,%d,seq_pos,%.3f\n", kind, depth, rows, timed([&] { sink += cache.seq_pos_min(0) + cache.seq_pos_max(0); }, 101, 100));
        if (k_rot) { std::printf("M,%s,%d,%d,k_rotation,%.3f\n", kind, depth, rows, timed([&] { cache.set_input_k_rot(k_rot); })); }
        if (v_rot) { std::printf("M,%s,%d,%d,v_rotation,%.3f\n", kind, depth, rows, timed([&] { cache.set_input_v_rot(v_rot); })); }
        auto sr = slot(rows, depth - 6);
        std::printf("M,%s,%d,%d,kv_indices,%.3f\n", kind, depth, rows, timed([&] { cache.set_input_k_idxs(k, &query.ub, sr); cache.set_input_v_idxs(v, &query.ub, sr); }));
    }
    std::printf("M,%s,%d,11,mask_round_f16,%.3f\n", kind, depth, timed([&] {
        query.ub.n_tokens = query.ub.n_seq_tokens = 6; query.pos[0] = depth - 6;
        cache.set_input_kq_mask(m16, &query.ub, true);
        query.ub.n_tokens = query.ub.n_seq_tokens = 1; query.pos[0] = depth - 1;
        for (int d = 0; d < 5; ++d) { cache.set_input_kq_mask(m16, &query.ub, true); }
    }, 101, 1));
    query.ub.n_tokens = query.ub.n_seq_tokens = 6;
    query.pos[0] = depth - 6;
    for (auto * mask : {m16, m32}) {
        query.ub.n_tokens = query.ub.n_seq_tokens = 1; query.pos[0] = depth - 1;
        cache.set_input_kq_mask(mask, &query.ub, true);
        out.write((const char *) mask->data, ggml_row_size(mask->type, depth));
        query.ub.n_tokens = query.ub.n_seq_tokens = 6; query.pos[0] = depth - 6;
        for (bool causal : {true, false}) {
            cache.set_input_kq_mask(mask, &query.ub, causal); write_tensor(out, mask);
        }
        cache.set_input_kq_mask(mask, &query.ub, true);
        std::vector<char> previous((char *) mask->data, (char *) mask->data + ggml_nbytes(mask));
        cache.seq_cp(0, 1, 17, 18);
        cache.seq_rm(0, 17, 18);
        if (!std::getenv("HOST_DEPTH_STALE_CONTROL")) { cache.set_input_kq_mask(mask, &query.ub, true); }
        require(std::memcmp(previous.data(), mask->data, previous.size()) != 0, "negative control: sequence mutation retained stale mask");
        write_tensor(out, mask);
        cache.seq_cp(1, 0, 17, 18); cache.seq_rm(1, 17, 18);
        const_cast<llama_kv_cells &>(cache.get_cells(0)).pos_add(18, depth);
        cache.set_input_kq_mask(mask, &query.ub, true);
        require(std::memcmp(previous.data(), mask->data, previous.size()) != 0, "negative control: position mutation retained stale mask");
        write_tensor(out, mask);
        const_cast<llama_kv_cells &>(cache.get_cells(0)).pos_add(18, -depth);
        cache.seq_rm(0, 23, 24);
        cache.set_input_kq_mask(mask, &query.ub, true); write_tensor(out, mask);
        batch_storage restore(1, 23); cache.apply_ubatch(slot(1, 23), restore.ub);
        cache.seq_rm(0, depth - 6, -1);
        cache.set_input_kq_mask(mask, &query.ub, true); write_tensor(out, mask);
        cache.apply_ubatch(s, query.ub);
    }
    GGML_UNUSED(sink);
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
    // Rows change at 8 and 10 to force a fresh (compact) graph; 9 and 11 then must not reuse it.
    for (int rows : {6, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3}) {
        if (step == 8) {
            // undo step 6's ownership change so the cache is dense again and a compact graph is built
            cache->seq_cp(1, 0, depth - 6, depth - 5);
            cache->seq_rm(1, depth - 6, depth - 5);
        }
        // Future positions are unreachable here: M-RoPE batch validation requires every new position to exceed
        // the cached maximum. Use ownership and holes, which keep n_kv and the graph shape.
        if (step == 9) { cache->seq_cp(0, 1, 19, 20); cache->seq_rm(0, 19, 20); }   // cell inside the prefix leaves seq 0
        if (step == 10) { cache->seq_cp(1, 0, 19, 20); cache->seq_rm(1, 19, 20); }  // restored: compact again
        // seq_rm moves head to the freed cell, so free more cells than the ubatch refills to keep a hole
        if (step == 11) { cache->seq_rm(0, 17, 21); }               // hole inside the prefix
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
        static const bool sched_log = std::getenv("JUDGE_SCHED_LOG") != nullptr;
        llama_log_set([](ggml_log_level level, const char * text, void *) { if (level >= GGML_LOG_LEVEL_WARN || (sched_log && level == GGML_LOG_LEVEL_DEBUG)) { fputs(text, stderr); } }, nullptr);
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
                cache_test(*model, depth, draft, masks);
                logits_test(*model, depth, draft, logits);
            }
        }
        std::puts("PASS: negative controls, target and MTP logits finite at both depths");
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
