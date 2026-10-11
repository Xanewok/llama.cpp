// A windowed KV cache with sinks (llama_context_params::swa_window / swa_sinks) driven like a speculative draft context:
// prefill ubatches, single-row draft steps rolled back with seq_rm, truncations. The mask is checked against an oracle
// computed from the cell contents alone, and the cells against the positions the window must still hold.
#include "../src/llama-kv-cache.h"
#include "../src/llama-model.h"
#include "ggml-cpp.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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
    auto * mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, n_kv, ub.n_tokens, 1, 1);
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
    auto sinfos = cache.prepare({ub});
    require(!sinfos.empty(), "no slot for " + std::to_string(n) + " tokens at position " + std::to_string(st[seq].next));
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

int main() {
    try {
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        llama_model_ptr model(llama_model_create(LLM_ARCH_LLAMA, mp));
        require(bool(model), "model creation failed");
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
