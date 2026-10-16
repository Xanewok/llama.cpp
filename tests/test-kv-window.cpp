#include "../src/llama-kv-cache.h"
#include "../src/llama-model.h"
#include "../src/llama-io.h"
#include "../src/llama-cparams.h"
#include "ggml-cpp.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const std::string & message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

struct seq_state {
    llama_pos next  = 0; // the position the next token takes
    llama_pos start = 0; // first position written since the sequence was last emptied
    llama_pos high  = -1; // highest position written since then, rolled-back draft rows included
};

struct window_case {
    bool     unified;
    int      n_seq;
    uint32_t w;
    uint32_t s;
    uint32_t n_ubatch;
};

static std::string name(const window_case & c) {
    return "unified=" + std::to_string(c.unified) + " n_seq=" + std::to_string(c.n_seq) + " W=" + std::to_string(c.w) +
           " S=" + std::to_string(c.s) + " ub=" + std::to_string(c.n_ubatch);
}

static std::set<llama_pos> positions(const llama_kv_cache & cache, llama_seq_id seq) {
    std::set<llama_pos> res;
    for (const auto & p : cache.get_cells(seq).seq_pos_get(seq)) {
        res.insert(p.first);
    }
    return res;
}

// cells: [0, S) from the start of the sequence, then one gap-free run ending at next - 1
static void check_cells(const llama_kv_cache & cache, llama_seq_id seq, const seq_state & st, const window_case & c) {
    const auto pos = positions(cache, seq);
    if (st.next == st.start) {
        require(pos.empty(), "a restarted sequence holds cells");
        return;
    }
    for (llama_pos p = st.start; p < std::min<llama_pos>(c.s, st.next); ++p) {
        require(pos.count(p) == 1, "sink " + std::to_string(p) + " missing");
    }
    llama_pos lo = st.next - 1;
    while (pos.count(lo - 1) == 1 && lo - 1 >= (llama_pos) c.s) {
        --lo;
    }
    for (llama_pos p : pos) {
        require(p < (llama_pos) c.s || (p >= lo && p < st.next), "position " + std::to_string(p) + " outside sinks and window");
    }
    require(pos.count(st.next - 1) == 1, "last position missing");
}

static void check_mask(llama_kv_cache & cache, const llama_ubatch & ub, const std::vector<seq_state> & st, const window_case & c,
                       int & max_visible) {
    const int n_kv = cache.get_size();
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead(), nullptr, true}));
    const uint32_t streams = c.unified ? 1 : ub.n_seqs_unq;
    auto * mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, n_kv, ub.n_tokens/streams, 1, streams);
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
    cache.set_input_kq_mask(mask, &ub, true);
    const float * data = (const float *) mask->data;
    for (uint32_t i = 0; i < ub.n_tokens; ++i) {
        const llama_seq_id seq = ub.seq_id[i][0];
        const llama_pos    p1  = ub.pos[i];
        const auto & cells = cache.get_cells(seq);
        std::set<llama_pos> visible;
        for (int j = 0; j < n_kv; ++j) {
            bool keep = !cells.is_empty(j) && cells.seq_has(j, seq);
            if (keep) {
                const llama_pos p0 = cells.pos_get(j);
                keep = p0 <= p1 && (p1 - p0 < (llama_pos) c.w || p0 < (llama_pos) c.s);
                if (keep) {
                    visible.insert(p0);
                }
            }
            require(data[size_t(i)*n_kv + j] == (keep ? 0.0f : -INFINITY), "mask differs from the cell oracle at row " +
                    std::to_string(i) + " cell " + std::to_string(j));
        }
        require(visible.size() <= c.w + c.s, "row sees " + std::to_string(visible.size()) + " cells, more than W + S");
        max_visible = std::max<int>(max_visible, visible.size());
        // a draft row written and rolled back evicts at most the positions it could not see, all <= high - W
        const llama_pos edge = std::max<llama_pos>(st[seq].start, std::max<llama_pos>(p1, st[seq].high) - (llama_pos) c.w + 1);
        for (llama_pos p = std::max(edge, p1 - (llama_pos) c.w + 1); p <= p1; ++p) {
            require(visible.count(p) == 1, "window position " + std::to_string(p) + " not visible from " + std::to_string(p1));
        }
        for (llama_pos p = st[seq].start; p < std::min<llama_pos>(c.s, p1 + 1); ++p) {
            require(visible.count(p) == 1, "sink " + std::to_string(p) + " not visible from " + std::to_string(p1));
        }
    }
}

static void write(llama_kv_cache & cache, llama_seq_id seq, uint32_t n, std::vector<seq_state> & st, const window_case & c,
                  int & max_visible) {
    std::vector<llama_pos>      pos(n);
    std::vector<int32_t>        counts(n, 1);
    std::vector<llama_seq_id>   seqs(n, seq);
    std::vector<llama_seq_id *> ids(n);
    for (uint32_t i = 0; i < n; ++i) {
        pos[i] = st[seq].next + i;
        ids[i] = &seqs[i];
    }
    llama_ubatch ub = {};
    ub.n_tokens = ub.n_seq_tokens = n;
    ub.n_seqs = ub.n_seqs_unq = ub.n_pos = 1;
    ub.b_equal_seqs = true;
    ub.pos = pos.data();
    ub.n_seq_id = counts.data();
    ub.seq_id = ids.data();
    ub.seq_id_unq = &seq;
    std::vector<std::set<llama_pos>> before;
    for (int s = 0; s < c.n_seq; ++s) {
        before.push_back(positions(cache, s));
    }
    const auto expected_slot = cache.find_slot(ub, false);
    auto sinfos = cache.prepare({ub});
    require(!sinfos.empty(), "no slot for " + std::to_string(n) + " tokens at position " + std::to_string(st[seq].next));
    for (int s = 0; s < c.n_seq; ++s) {
        require(before[s] == positions(cache, s), "prepare changed the retained positions");
    }
    require(cache.find_slot(ub, false).idxs == expected_slot.idxs, "prepare changed a stream's search head");
    cache.apply_ubatch(sinfos[0], ub);
    st[seq].next += n;
    st[seq].high = std::max(st[seq].high, st[seq].next - 1);
    check_mask(cache, ub, st, c, max_visible);
}

static void truncate(llama_kv_cache & cache, llama_seq_id seq, llama_pos p0, std::vector<seq_state> & st, const window_case & c) {
    require(cache.seq_rm(seq, p0, -1), "seq_rm failed");
    // the caller continues at p0, as the server does after a prefix match
    st[seq].next = std::min(st[seq].next, p0);
    if (cache.seq_pos_max(seq) < 0) {
        st[seq].start = st[seq].next;
        st[seq].high  = st[seq].next - 1;
    } else {
        require(cache.seq_pos_max(seq) == st[seq].next - 1, "truncation left a gap before the next position");
    }
    check_cells(cache, seq, st[seq], c);
}

static void write_pair(llama_kv_cache & cache, std::vector<seq_state> & st, const window_case & c, int & max_visible) {
    constexpr uint32_t rows = 6;
    llama_seq_id order[] = {1, 0};
    std::vector<llama_pos> pos(2*rows);
    std::vector<int32_t> counts(2*rows, 1);
    std::vector<llama_seq_id *> ids(2*rows);
    for (uint32_t s = 0; s < 2; ++s) {
        for (uint32_t r = 0; r < rows; ++r) {
            pos[s*rows + r] = st[order[s]].next + r;
            ids[s*rows + r] = &order[s];
        }
    }
    llama_ubatch ub = {};
    ub.n_tokens = 2*rows;
    ub.n_seq_tokens = rows;
    ub.n_seqs = ub.n_seqs_unq = 2;
    ub.n_pos = 1;
    ub.b_equal_seqs = true;
    ub.pos = pos.data();
    ub.n_seq_id = counts.data();
    ub.seq_id = ids.data();
    ub.seq_id_unq = order;
    const auto before0 = positions(cache, 0);
    const auto before1 = positions(cache, 1);
    const auto slots = cache.prepare({ub});
    require(!slots.empty(), "joint batch did not fit");
    require(before0 == positions(cache, 0) && before1 == positions(cache, 1), "joint prepare changed retained positions");
    cache.apply_ubatch(slots[0], ub);
    for (auto & seq : st) {
        seq.next += rows;
        seq.high = std::max(seq.high, seq.next - 1);
    }
    check_mask(cache, ub, st, c, max_visible);
}

static int run_case(const llama_model & model, const window_case & c, uint32_t seed) {
    auto hp = model.hparams;
    hp.no_alloc = true;
    const uint32_t kv_size = (c.w + c.s)*(c.unified ? c.n_seq : 1) + c.n_ubatch;
    llama_kv_cache cache(model, hp, GGML_TYPE_F16, GGML_TYPE_F16, false, false, c.unified, kv_size, c.n_seq, 1, c.w,
                         LLAMA_SWA_TYPE_STANDARD, nullptr, [](uint32_t) { return false; }, nullptr, nullptr, "", c.s);
    std::mt19937 rng(seed);
    std::vector<seq_state> st(c.n_seq);
    int max_visible = 0;
    int steps = 0;
    for (int round = 0; round < 300; ++round) {
        const llama_seq_id seq = rng() % c.n_seq;
        const uint32_t op = rng() % 10;
        if (op < 2) {
            write(cache, seq, 1 + rng() % c.n_ubatch, st, c, max_visible);
        } else if (op < 9) {
            // a speculative round: k draft rows one at a time, then keep 0..k of them
            const llama_pos base = st[seq].next;
            const uint32_t k = 1 + rng() % 5;
            for (uint32_t i = 0; i < k; ++i) {
                write(cache, seq, 1, st, c, max_visible);
            }
            truncate(cache, seq, base + rng() % (k + 1), st, c);
        } else if (st[seq].next > (llama_pos) (c.s + c.w + 8)) {
            // a prefix reuse: below the sinks, below the window or inside it
            const llama_pos lo = st[seq].next - (llama_pos) c.w/2;
            const llama_pos cut = rng() % 3 == 0 ? (llama_pos) (rng() % (c.s + 1)) : rng() % 2 ? lo + rng() % (c.w/2) : c.s + 1;
            truncate(cache, seq, cut, st, c);
        }
        for (llama_seq_id s = 0; s < c.n_seq; ++s) {
            check_cells(cache, s, st[s], c);
        }
        ++steps;
    }
    require(max_visible == (int) (c.w + c.s), "no row saw a full window plus sinks (max " + std::to_string(max_visible) + ")");
    return steps;
}

struct state_writer : llama_io_write_i {
    std::vector<uint8_t> data;
    void write(const void * src, size_t size) override {
        const auto * bytes = static_cast<const uint8_t *>(src);
        data.insert(data.end(), bytes, bytes + size);
    }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        const size_t begin = data.size();
        data.resize(begin + size);
        ggml_backend_tensor_get(tensor, data.data() + begin, offset, size);
    }
    size_t n_bytes() override { return data.size(); }
};

struct state_reader : llama_io_read_i {
    const std::vector<uint8_t> & data;
    size_t offset = 0;
    explicit state_reader(const std::vector<uint8_t> & data) : data(data) {}
    void read(void * dst, size_t size) override {
        require(offset <= data.size() && size <= data.size() - offset, "short checkpoint");
        std::memcpy(dst, data.data() + offset, size);
        offset += size;
    }
    void read_tensor(ggml_tensor * tensor, size_t dst_offset, size_t size) override {
        require(offset <= data.size() && size <= data.size() - offset, "short checkpoint tensor");
        ggml_backend_tensor_set(tensor, data.data() + offset, dst_offset, size);
        offset += size;
    }
    size_t n_bytes() override { return offset; }
};

static float payload(llama_seq_id seq, llama_pos pos, int dim, bool value) {
    return float(10000*seq + 10*pos + dim + (value ? 5 : 0));
}

static void kv_payload(llama_kv_cache & cache, bool trans, bool store) {
    ggml_context_ptr ctx(ggml_init({16*ggml_tensor_overhead(), nullptr, true}));
    llama_kv_cache::slot_info slot = {};
    slot.s0 = 0;
    slot.s1 = cache.get_n_stream() - 1;
    auto * k = cache.get_k(ctx.get(), 0, cache.get_size(), slot);
    auto * v = cache.get_v(ctx.get(), 0, cache.get_size(), slot);
    for (llama_seq_id seq = 0; seq < (llama_seq_id) cache.get_n_seq_max(); ++seq) {
        for (const auto & entry : cache.get_cells(seq).seq_pos_get(seq)) {
            for (int d = 0; d < 4; ++d) {
                for (bool value : {false, true}) {
                    auto * t = value ? v : k;
                    const size_t off = cache.get_stream(seq)*t->nb[3] +
                        (value && trans ? entry.second*t->nb[0] + d*t->nb[2] : d*t->nb[0] + entry.second*t->nb[2]);
                    float x = payload(seq, entry.first, d, value);
                    if (store) {
                        ggml_backend_tensor_set(t, &x, off, sizeof(x));
                    } else {
                        float actual;
                        ggml_backend_tensor_get(t, &actual, off, sizeof(actual));
                        require(actual == x, "checkpoint changed K/V payload");
                    }
                }
            }
        }
    }
}

static void test_rollback(const llama_model & model, bool unified, int n_seq, bool trans) {
    auto hp = model.hparams;
    hp.no_alloc = false;
    hp.n_layer_all = 1;
    hp.n_embd = hp.n_embd_head_k_full = hp.n_embd_head_v_full = 4;
    hp.n_head_arr[0] = hp.n_head_kv_arr[0] = 1;
    const window_case c = {unified, n_seq, 8, 3, 16};
    const uint32_t size = (c.w + c.s)*(unified ? n_seq : 1) + c.n_ubatch;
    llama_kv_cache cache(model, hp, GGML_TYPE_F32, GGML_TYPE_F32, trans, false, unified, size, n_seq, 1, c.w,
                         LLAMA_SWA_TYPE_STANDARD, nullptr, nullptr, nullptr, nullptr, "", c.s);
    std::vector<seq_state> st(n_seq);
    int max_visible = 0;
    for (llama_seq_id seq = 0; seq < n_seq; ++seq) {
        write(cache, seq, 16, st, c, max_visible);
    }
    if (n_seq == 2) {
        for (int round = 0; round < 20; ++round) {
            write_pair(cache, st, c, max_visible);
        }
    }
    for (int round = 0; round < 100; ++round) {
        for (llama_seq_id seq = 0; seq < n_seq; ++seq) {
            const auto base = st[seq];
            for (int i = 0; i < 5; ++i) {
                write(cache, seq, 1, st, c, max_visible);
            }
            truncate(cache, seq, base.next, st, c);
            const auto pos = positions(cache, seq);
            for (llama_pos p = base.next - c.w + 1; p < base.next; ++p) {
                require(pos.count(p), "rollback lost window row " + std::to_string(p));
            }
            write(cache, seq, 1, st, c, max_visible);
        }
    }
    const llama_seq_id saved_seq = n_seq - 1;
    const llama_seq_id other_seq = 0;
    bool rotated = false;
    for (uint32_t i = 0; i < 2*size && !rotated; ++i) {
        write(cache, saved_seq, 1, st, c, max_visible);
        std::vector<uint32_t> indices;
        for (const auto & entry : cache.get_cells(saved_seq).seq_pos_get(saved_seq)) {
            if (entry.first >= st[saved_seq].next - (llama_pos) c.w) {
                indices.push_back(entry.second);
            }
        }
        rotated = !std::is_sorted(indices.begin(), indices.end());
    }
    require(rotated, "checkpoint did not straddle the physical ring boundary");
    kv_payload(cache, trans, true);
    state_writer checkpoint;
    cache.state_write(checkpoint, saved_seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const auto saved = st[saved_seq];
    auto partner_window = [&]() {
        std::set<llama_pos> res;
        for (llama_pos p : positions(cache, other_seq)) {
            if (p < (llama_pos) c.s || (p < st[other_seq].next && st[other_seq].next - p < (llama_pos) c.w)) {
                res.insert(p);
            }
        }
        return res;
    };
    const auto partner = n_seq > 1 ? partner_window() : std::set<llama_pos>{};
    for (uint32_t i = 0; i < 3*size; ++i) {
        write(cache, saved_seq, 1, st, c, max_visible);
    }
    kv_payload(cache, trans, true);
    state_reader reader(checkpoint.data);
    cache.state_read(reader, saved_seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    require(reader.n_bytes() == checkpoint.n_bytes(), "checkpoint was not consumed");
    require(n_seq == 1 || partner == partner_window(), "restore changed the other slot's visible window");
    kv_payload(cache, trans, false);
    st[saved_seq] = saved;
    for (int round = 0; round < 100; ++round) {
        const auto base = st[saved_seq];
        for (int i = 0; i < 5; ++i) {
            write(cache, saved_seq, 1, st, c, max_visible);
        }
        truncate(cache, saved_seq, base.next, st, c);
        const auto pos = positions(cache, saved_seq);
        for (llama_pos p = base.next - c.w + 1; p < base.next; ++p) {
            require(pos.count(p), "restored rollback lost window row " + std::to_string(p));
        }
        write(cache, saved_seq, 1, st, c, max_visible);
    }
}

static void test_size(const llama_model & model) {
    for (bool unified : {false, true}) {
        for (uint32_t n_seq : {1u, 2u}) {
            for (uint32_t window : {0u, 4096u, 8192u, 16384u, UINT32_MAX}) {
                llama_cparams cp = {};
                cp.n_ctx_seq = 102400;
                cp.n_seq_max = n_seq;
                cp.n_kv_pad = 4096;
                cp.n_ubatch = 512;
                cp.flash_attn = true;
                cp.kv_unified = unified;
                cp.swa_window = window;
                cp.swa_sinks = 16;
                llama_memory_params params = {GGML_TYPE_F16, GGML_TYPE_F16, true, LLAMA_CONTEXT_TYPE_MTP, nullptr};
                std::unique_ptr<llama_memory_i> mem(model.create_memory(params, cp));
                auto * cache = dynamic_cast<llama_kv_cache *>(mem.get());
                require(cache != nullptr, "window did not create a KV cache");
                const uint64_t cells = (uint64_t(window) + 16)*(unified ? n_seq : 1) + 512;
                const uint32_t expected = window ? GGML_PAD(std::min<uint64_t>(102400, cells), 256) : 102400;
                require(cache->get_size() == expected, "wrong compact cache size");
                require(cache->get_n_swa() == window, "wrong effective window");
            }
        }
    }
    llama_cparams cp = {};
    cp.n_ctx_seq = UINT32_MAX;
    cp.n_seq_max = 1;
    cp.n_ubatch = 512;
    cp.swa_window = cp.swa_sinks = INT32_MAX;
    llama_memory_params params = {GGML_TYPE_F16, GGML_TYPE_F16, true, LLAMA_CONTEXT_TYPE_MTP, nullptr};
    bool rejected = false;
    try {
        std::unique_ptr<llama_memory_i> mem(model.create_memory(params, cp));
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()) == "attention window cache exceeds the cell count range";
    }
    require(rejected, "padded cache capacity overflow was not rejected");
}

int main() {
    try {
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        llama_model_ptr model(llama_model_create(LLM_ARCH_LLAMA, mp));
        require(bool(model), "model creation failed");
        test_size(*model);
        for (bool unified : {false, true}) {
            for (int n_seq : {1, 2}) {
                for (bool trans : {false, true}) {
                    test_rollback(*model, unified, n_seq, trans);
                }
            }
        }
        int n_cases = 0;
        for (bool unified : {false, true}) {
            for (int n_seq : {1, 2}) {
                for (auto ws : std::vector<std::pair<uint32_t, uint32_t>>{{8, 0}, {8, 3}, {32, 4}, {64, 16}}) {
                    for (uint32_t n_ubatch : {1u, 6u, 16u}) {
                        const window_case c = {unified, n_seq, ws.first, ws.second, n_ubatch};
                        try {
                            run_case(*model, c, 1729 + n_cases);
                        } catch (const std::exception & e) {
                            throw std::runtime_error(name(c) + ": " + e.what());
                        }
                        ++n_cases;
                    }
                }
            }
        }
        printf("test-kv-window: %d windowed cache cases OK\n", n_cases);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "test-kv-window: FAIL %s\n", error.what());
        return 1;
    }
}
