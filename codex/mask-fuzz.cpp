// Judge fuzz: random KV cell states x random query ubatches -> raw mask bytes.
// Run the same binary linked against different libllama variants and compare outputs byte-for-byte.
#define main architecture_test_main
#include "../tests/test-llama-archs.cpp"
#undef main
#include "../src/llama-model.h"
#include "../src/llama-kv-cache.h"
#include <set>
#include <fstream>
#include <random>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

struct ub_store {
    std::vector<llama_pos> pos;
    std::vector<int32_t> nseq;
    std::vector<std::vector<llama_seq_id>> seqs;
    std::vector<llama_seq_id *> ids;
    std::vector<llama_seq_id> unq;
    llama_ubatch ub = {};
    // rows: (seq set, pos, y, x); n_pos 1 or 4
    void build(const std::vector<std::vector<llama_seq_id>> & s, const std::vector<llama_pos> & p,
               const std::vector<llama_pos> & y, const std::vector<llama_pos> & x, int n_pos, uint32_t n_seqs) {
        const int n = (int) p.size();
        seqs = s;
        pos.assign((size_t) n*n_pos, 0);
        for (int i = 0; i < n; ++i) {
            pos[i] = p[i];
            if (n_pos == 4) { pos[i + n] = y[i]; pos[i + 2*n] = x[i]; pos[i + 3*n] = 0; }
        }
        nseq.resize(n); ids.resize(n);
        for (int i = 0; i < n; ++i) { nseq[i] = (int32_t) seqs[i].size(); ids[i] = seqs[i].data(); }
        unq = {0};
        ub = {};
        ub.n_tokens = n; ub.n_seqs = n_seqs; ub.n_seq_tokens = n/n_seqs; ub.n_seqs_unq = 1; ub.n_pos = n_pos;
        ub.b_equal_seqs = n_seqs > 1;
        ub.pos = pos.data(); ub.n_seq_id = nseq.data(); ub.seq_id = ids.data(); ub.seq_id_unq = unq.data();
    }
};

struct stats { long rows = 0, guard_rows = 0, guard_rows_2d = 0, guard_rows_future = 0, guard_rows_dup = 0, guard_shared = 0; };

static void run_case(llama_model & model, std::mt19937 & rng, std::ofstream & out, stats & st, bool unified) {
    auto hp = model.hparams;
    hp.no_alloc = true;
    hp.n_layer_all = 65; hp.n_layer_nextn = 1;
    hp.n_head_arr.fill(24); hp.n_head_kv_arr.fill(4);
    hp.n_embd_head_k_full = hp.n_embd_head_v_full = 256;
    auto R = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
    const uint32_t kv_size = 64 + R(0, 3)*32;
    const uint32_t n_seq_max = unified ? 3 : 2;
    llama_kv_cache cache(model, hp, GGML_TYPE_F16, GGML_TYPE_F16, false, false, unified, kv_size, n_seq_max, 32,
                         0, LLAMA_SWA_TYPE_NONE, nullptr, [](uint32_t il) { return il == 64; }, nullptr, nullptr);
    const uint32_t n_stream = unified ? 1 : 2;
    const bool img = R(0, 3) == 0; // image-like 2D ext in history

    // history per stream
    for (uint32_t s = 0; s < n_stream; ++s) {
        auto & cells = const_cast<llama_kv_cells &>(cache.get_cells(unified ? 0 : s));
        const llama_seq_id base_seq = unified ? 0 : (llama_seq_id) s;
        const int mode = R(0, 2);
        const uint32_t U = R(0, kv_size - 8);
        std::vector<uint32_t> idx; std::vector<std::vector<llama_seq_id>> sq; std::vector<llama_pos> p, y, x;
        llama_pos next = R(0, 5);
        for (uint32_t i = 0; i < U; ++i) {
            uint32_t slot = mode == 2 ? (uint32_t) R(0, kv_size - 1) : i;
            if (std::find(idx.begin(), idx.end(), slot) != idx.end()) { continue; }
            idx.push_back(slot);
            std::vector<llama_seq_id> ss = {base_seq};
            if (unified && R(0, 5) == 0) { ss = {1}; }
            if (unified && R(0, 5) == 0) { ss.push_back(2); }
            sq.push_back(ss);
            llama_pos pp = R(0, 9) == 0 ? next - R(0, 3) : next++; // duplicates / reordering
            if (pp < 0) { pp = 0; }
            p.push_back(pp);
            y.push_back(img ? R(0, 3) : pp); x.push_back(img ? R(0, 3) : pp);
        }
        if (!idx.empty()) {
            ub_store h; h.build(sq, p, y, x, img ? 4 : 1, 1);
            llama_kv_cache::slot_info si = {s, s, {(llama_seq_id) s}, {idx}};
            cache.apply_ubatch(si, h.ub);
        }
        // mutations
        const int nm = R(0, 3);
        for (int m = 0; m < nm; ++m) {
            const llama_pos pmax = std::max<llama_pos>(0, cache.seq_pos_max(base_seq));
            switch (R(0, 6)) {
                case 0: { llama_pos a = R(0, pmax + 1); cache.seq_rm(base_seq, a, a + R(1, 4)); } break;  // hole
                case 1: if (unified) { cache.seq_cp(0, R(1, 2), -1, -1); } break;                     // fully shared
                case 2: if (unified) { llama_pos a = R(0, pmax + 1); cache.seq_cp(0, 1, a, a + R(1, 6)); } break;
                case 3: { // shift a used cell forward/backward (future cell interleaved, duplicate positions)
                    if (cells.get_used() > 0) {
                        uint32_t c = R(0, cells.used_max_p1() - 1);
                        if (!cells.is_empty(c)) { cells.pos_add(c, R(-3, 12)); }
                    }
                } break;
                case 4: { llama_pos a = R(0, pmax + 1); cache.seq_rm(base_seq, a, -1); } break;           // tail rm
                case 5: { // refill first empty cell with an arbitrary position (re-dense)
                    for (uint32_t c = 0; c < cells.size(); ++c) {
                        if (cells.is_empty(c)) {
                            ub_store f; f.build({{base_seq}}, {R(0, pmax + 4)}, {R(0, 3)}, {R(0, 3)}, img ? 4 : 1, 1);
                            llama_kv_cache::slot_info si = {s, s, {(llama_seq_id) s}, {{c}}};
                            cache.apply_ubatch(si, f.ub);
                            break;
                        }
                    }
                } break;
                case 6: if (R(0, 3) == 0) { // used cell without any sequence
                    for (uint32_t c = 0; c < cells.size(); ++c) {
                        if (cells.is_empty(c)) { cells.pos_set(c, R(0, pmax)); break; }
                    }
                } break;
            }
        }
    }

    // queries
    for (int q = 0; q < 6; ++q) {
        const uint32_t n_tps = R(1, 7);
        const int n_pos = R(0, 1) ? 4 : 1;
        const bool text2d = R(0, 1);
        std::vector<std::vector<llama_seq_id>> sq; std::vector<llama_pos> p, y, x;
        for (uint32_t s = 0; s < n_stream; ++s) {
            for (uint32_t i = 0; i < n_tps; ++i) {
                llama_seq_id sid = unified ? (R(0, 3) == 0 ? R(1, 2) : 0) : (llama_seq_id) s;
                sq.push_back({sid});
                const llama_pos pm = std::max<llama_pos>(0, cache.seq_pos_max(sid));
                llama_pos pp = R(0, 2) ? pm - (llama_pos) n_tps + 1 + (llama_pos) i : R(0, pm + 3);
                if (pp < 0) { pp = 0; }
                p.push_back(pp);
                y.push_back(text2d ? pp : R(0, 3)); x.push_back(text2d ? pp : R(0, 3));
            }
        }
        ub_store qs; qs.build(sq, p, y, x, n_pos, n_stream);
        int64_t umax = 0;
        for (uint32_t s = 0; s < n_stream; ++s) { umax = std::max<int64_t>(umax, cache.get_cells(unified ? 0 : s).used_max_p1()); }
        int64_t n_kv;
        switch (R(0, 3)) {
            case 0: n_kv = std::max<int64_t>(1, umax); break;
            case 1: n_kv = kv_size; break;
            case 2: n_kv = std::max<int64_t>(1, std::min<int64_t>(kv_size, GGML_PAD(umax, 32))); break;
            default: n_kv = std::max<int64_t>(1, umax - R(0, 3)); break; // n_kv below used extent
        }
        for (uint32_t i = 0; i < n_stream*n_tps; ++i) {
            const auto & cells = cache.get_cells(sq[i][0]);
            const bool guard = cells.used_max_p1() == cells.get_used() && cells.seq_pos_get(sq[i][0]).size() == cells.get_used();
            st.rows++;
            if (guard) {
                st.guard_rows++;
                if (n_pos == 4) { st.guard_rows_2d++; }
                const auto & sp = cells.seq_pos_get(sq[i][0]);
                if (!sp.empty() && sp.rbegin()->first > p[i]) { st.guard_rows_future++; }
                std::set<llama_pos> uniq; for (auto & e : sp) { uniq.insert(e.first); }
                if (uniq.size() != sp.size()) { st.guard_rows_dup++; }
                bool shared = false;
                for (auto & e : sp) { if (cells.seq_get_all(e.second).count() > 1) { shared = true; break; } }
                if (shared) { st.guard_shared++; }
            }
        }
        for (ggml_type t : {GGML_TYPE_F16, GGML_TYPE_F32}) {
            for (bool causal : {true, false}) {
                ggml_context_ptr ctx(ggml_init({4*ggml_tensor_overhead(), nullptr, true}));
                auto * m = ggml_new_tensor_4d(ctx.get(), t, n_kv, n_tps, 1, n_stream);
                ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
                std::memset(m->data, 0x5a, ggml_nbytes(m)); // sentinel: unwritten bytes show up as differences
                cache.set_input_kq_mask(m, &qs.ub, causal);
                out.write((const char *) m->data, ggml_nbytes(m));
            }
        }
    }
}

int main(int argc, char ** argv) {
    try {
        require(argc == 3, "usage: adv OUT N_CASES");
        require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
        llama_log_set([](ggml_log_level level, const char * text, void *) { if (level >= GGML_LOG_LEVEL_ERROR) { fputs(text, stderr); } }, nullptr);
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
        std::ofstream out(argv[1], std::ios::binary);
        const int n = std::atoi(argv[2]);
        stats st;
        std::mt19937 rng(12345);
        for (int c = 0; c < n; ++c) { run_case(*model, rng, out, st, c % 3 != 2); }
        std::printf("rows=%ld guard_rows=%ld guard_rows_2d=%ld guard_rows_future=%ld guard_rows_dup=%ld guard_rows_shared=%ld\n",
                    st.rows, st.guard_rows, st.guard_rows_2d, st.guard_rows_future, st.guard_rows_dup, st.guard_shared);
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
