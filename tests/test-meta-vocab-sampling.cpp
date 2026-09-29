// CPU-only check of the meta-backend ops that llama's build_sampling uses under tensor split:
// a vocab-split logits matrix is reduced to per-device top-k candidates (TOP_K and GET_ROWS on a split tensor are
// device-local) and gathered to every device (ggml_backend_meta_gather). Two CPU backends stand in for two GPUs.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

static constexpr int64_t N_EMBD  = 64;
static constexpr int64_t N_VOCAB = 1000;
static constexpr int64_t N_ROWS  = 3; // output rows; build_sampling adds one dummy row
static constexpr int64_t K       = 8; // per device
static constexpr int64_t V0      = 600; // device 0 slice, deliberately not half

static ggml_backend_meta_split_state get_split_state(const ggml_tensor * tensor, void * /*ud*/) {
    ggml_backend_meta_split_state ss;
    memset(&ss, 0, sizeof(ss));
    ss.nr[0]      = 1;
    ss.n_segments = 1;

    const std::string name = tensor->name;
    if (name == "w") {
        ss.axis  = GGML_BACKEND_SPLIT_AXIS_1;
        ss.ne[0] = V0;
        ss.ne[1] = N_VOCAB - V0;
    } else if (name == "ids") {
        ss.axis  = GGML_BACKEND_SPLIT_AXIS_0;
        ss.ne[0] = V0;
        ss.ne[1] = N_VOCAB - V0;
    } else {
        ss.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    }
    return ss;
}

int main() {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        fprintf(stderr, "no CPU device\n");
        return 1;
    }
    ggml_backend_dev_t devs[2] = { cpu, cpu };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devs, 2, get_split_state, nullptr);
    ggml_backend_t     backend  = ggml_backend_dev_init(meta_dev, nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(meta_dev);

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // static tensors: LM head and vocab ids
    ggml_init_params wp = { 4*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_tensor * w   = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, N_EMBD, N_VOCAB);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, N_VOCAB);
    ggml_set_name(w, "w");
    ggml_set_name(ids, "ids");
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);

    std::vector<float> w_data(N_EMBD*N_VOCAB);
    for (auto & x : w_data) {
        x = nd(rng);
    }
    std::vector<float> ids_data(N_VOCAB);
    for (int64_t i = 0; i < N_VOCAB; ++i) {
        ids_data[i] = (float) i;
    }
    ggml_backend_tensor_set(w,   w_data.data(),   0, ggml_nbytes(w));
    ggml_backend_tensor_set(ids, ids_data.data(), 0, ggml_nbytes(ids));

    // graph: the same op sequence as llm_graph_context::build_sampling
    ggml_init_params gp = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, N_ROWS);
    ggml_set_input(h);

    ggml_tensor * logits   = ggml_mul_mat(ctx, w, h);
    ggml_tensor * logits_t = ggml_pad(ctx, logits, 0, 1, 0, 0);
    const int64_t n_cols   = logits_t->ne[1];
    const int64_t n_cand   = 2*K;

    ggml_tensor * idx  = ggml_top_k(ctx, logits_t, n_cand);
    ggml_tensor * vals = ggml_get_rows(ctx, ggml_reshape_3d(ctx, logits_t, 1, N_VOCAB, n_cols), idx);
    ggml_tensor * vid  = ggml_get_rows(ctx,
            ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, ids, 1, N_VOCAB, 1), 1, N_VOCAB, n_cols, 1), idx);
    vals = ggml_reshape_3d(ctx, vals, n_cand, 1, n_cols);
    vid  = ggml_reshape_3d(ctx, vid,  n_cand, 1, n_cols);
    ggml_tensor * cand = ggml_backend_meta_gather(ctx, ggml_concat(ctx, vals, vid, 1));
    ggml_set_output(cand);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cand);

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "graph allocation failed\n");
        return 1;
    }

    std::vector<float> h_data(N_EMBD*N_ROWS);
    for (auto & x : h_data) {
        x = nd(rng);
    }
    ggml_backend_tensor_set(h, h_data.data(), 0, ggml_nbytes(h));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        return 1;
    }

    std::vector<float> out(ggml_nelements(cand));
    ggml_backend_tensor_get(cand, out.data(), 0, ggml_nbytes(cand));

    int n_fail = 0;
    for (int64_t r = 0; r < n_cols; ++r) {
        std::vector<float> ref(N_VOCAB, 0.0f); // the dummy row is all zeros
        if (r < N_ROWS) {
            for (int64_t v = 0; v < N_VOCAB; ++v) {
                double acc = 0.0;
                for (int64_t e = 0; e < N_EMBD; ++e) {
                    acc += (double) w_data[v*N_EMBD + e] * h_data[r*N_EMBD + e];
                }
                ref[v] = (float) acc;
            }
        }

        const float * c_val = out.data() + r*2*n_cand;
        const float * c_id  = c_val + n_cand;

        std::set<int64_t> got;
        for (int64_t c = 0; c < n_cand; ++c) {
            const int64_t id = (int64_t) c_id[c];
            if (id < 0 || id >= N_VOCAB || (float) id != c_id[c]) {
                fprintf(stderr, "row %lld: bad id %f\n", (long long) r, c_id[c]);
                n_fail++;
                continue;
            }
            if (std::fabs(c_val[c] - ref[id]) > 1e-3f*(1.0f + std::fabs(ref[id]))) {
                fprintf(stderr, "row %lld: logit of %lld is %f, expected %f\n", (long long) r, (long long) id, c_val[c], ref[id]);
                n_fail++;
            }
            got.insert(id);
        }
        if (r == N_ROWS) {
            continue; // ties everywhere, only ids and values are checked
        }

        // expected: top-K of each device slice
        auto top_of = [&](int64_t beg, int64_t end) {
            std::vector<int64_t> v(end - beg);
            for (int64_t i = beg; i < end; ++i) {
                v[i - beg] = i;
            }
            std::partial_sort(v.begin(), v.begin() + K, v.end(), [&](int64_t a, int64_t b) { return ref[a] > ref[b]; });
            v.resize(K);
            return v;
        };
        std::set<int64_t> want;
        for (int64_t id : top_of(0, V0)) {
            want.insert(id);
        }
        for (int64_t id : top_of(V0, N_VOCAB)) {
            want.insert(id);
        }
        if (got != want) {
            fprintf(stderr, "row %lld: candidate set differs from the per-slice top-%lld\n", (long long) r, (long long) K);
            n_fail++;
        }

        // the point of it all: the global top-K is among the candidates
        for (int64_t id : top_of(0, N_VOCAB)) {
            if (!got.count(id)) {
                fprintf(stderr, "row %lld: global top-%lld token %lld missing\n", (long long) r, (long long) K, (long long) id);
                n_fail++;
            }
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    ggml_backend_free(backend);

    printf("%s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
