// GGML_META_AR_IN_GRAPH=1 must reproduce the split meta plan with device-epoch AllReduces between the graphs
// (GGML_CUDA_AR_P2P_KERNEL=1) bit for bit, while the decode-sized AllReduces run inside captured CUDA graphs that are
// replayed, interleaved with a copy-engine sized graph that keeps the split.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-impl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string g_log;

static void log_capture(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    g_log += text;
    fputs(text, stderr);
}

static void require(bool ok, const char * message) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static size_t count(const std::string & text, const std::string & needle) {
    size_t n = 0;
    for (size_t pos = text.find(needle); pos != std::string::npos; pos = text.find(needle, pos + 1)) {
        ++n;
    }
    return n;
}

static uint32_t next(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// Non-integer values, so the BF16 wire rounds both operands.
static std::vector<float> random_values(size_t n, uint32_t seed, float scale) {
    std::vector<float> v(n);
    for (auto & x : v) {
        x = (float(int(next(seed) % 2000001u)) - 1000000.0f) * 1.0e-6f * scale;
    }
    return v;
}

constexpr int64_t D  = 512;  // residual width
constexpr int64_t F  = 1024; // hidden width
constexpr int     L  = 5;    // layers = AllReduces per forward
constexpr int     L0 = 2;    // this layer lives on device 0 only: device 1's shard is inactive and contributes zeros

// Megatron-style split: up projections by output rows, down projections by input columns, so each down projection is
// a PARTIAL sum that the meta backend AllReduces.
static ggml_backend_meta_split_state split_state(const ggml_tensor * t, void *) {
    const bool solo = strstr(t->name, "-solo") != nullptr;
    if (strncmp(t->name, "up", 2) == 0) {
        return { GGML_BACKEND_SPLIT_AXIS_1, { solo ? F : F/2, solo ? 0 : F/2 }, {1}, 1 };
    }
    if (strncmp(t->name, "down", 4) == 0) {
        return { GGML_BACKEND_SPLIT_AXIS_0, { solo ? F : F/2, solo ? 0 : F/2 }, {1}, 1 };
    }
    if (strncmp(t->name, "out", 3) == 0) {
        return { GGML_BACKEND_SPLIT_AXIS_1, { D/2, D/2 }, {1}, 1 };
    }
    return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1 };
}

struct weights {
    ggml_tensor * up[L];
    ggml_tensor * down[L];
    ggml_tensor * out;
};

struct graph {
    ggml_context *            ctx  = nullptr;
    ggml_cgraph *             gf   = nullptr;
    ggml_tensor *             x    = nullptr;
    ggml_tensor *             res  = nullptr; // final residual: MIRRORED, read from device 0
    ggml_tensor *             out  = nullptr; // split by rows: each half is computed from its own device's residual
    ggml_gallocr_t            ga   = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
};

static graph build(ggml_backend_t backend, const weights & w, int64_t rows, uint64_t uid) {
    graph g;
    ggml_init_params params = { (8*L + 16)*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(params);
    require(g.ctx != nullptr, "graph context");
    g.x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, rows);
    ggml_set_name(g.x, "x");
    ggml_set_input(g.x);
    ggml_tensor * cur = g.x;
    for (int l = 0; l < L; ++l) {
        ggml_tensor * h = ggml_mul_mat(g.ctx, w.up[l], cur);
        ggml_tensor * y = ggml_mul_mat(g.ctx, w.down[l], h);
        cur = ggml_add(g.ctx, cur, y);
    }
    g.res = cur;
    ggml_set_output(g.res);
    g.out = ggml_mul_mat(g.ctx, w.out, cur);
    ggml_set_output(g.out);
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    g.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(ggml_gallocr_alloc_graph(g.ga, g.gf), "graph allocation");
    // A plan is reused across computes only for a graph with a stable non-zero uid (as the scheduler's split graphs).
    g.gf->uid = uid;
    g.plan = ggml_backend_graph_plan_create(backend, g.gf);
    require(g.plan != nullptr, "meta plan");
    return g;
}

static void release(ggml_backend_t backend, graph & g) {
    ggml_backend_graph_plan_free(backend, g.plan);
    ggml_gallocr_free(g.ga);
    ggml_free(g.ctx);
}

static std::vector<uint8_t> read(const ggml_tensor * t) {
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    return bytes;
}

int main() {
    ggml_log_set(log_capture, nullptr);
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (!reg || ggml_backend_reg_dev_count(reg) < 2) {
        puts("SKIP: two CUDA devices required");
        return 77;
    }
    set_env("GGML_CUDA_ALLREDUCE", "internal");
    set_env("GGML_CUDA_AR_P2P", "1");
    set_env("GGML_CUDA_AR_P2P_KERNEL", "1");
    // An inherited threshold could move every AllReduce to one path; 1 MiB puts graph A in graph and graph B on the
    // copy engine.
    set_env("GGML_CUDA_AR_COPY_THRESHOLD", "1048576");
    set_env("GGML_CUDA_AR_BF16_THRESHOLD", "1");

    ggml_backend_dev_t devs[2] = { ggml_backend_reg_dev_get(reg, 0), ggml_backend_reg_dev_get(reg, 1) };
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2, split_state, nullptr);
    require(meta != nullptr, "meta device");

    ggml_backend_t backends[2]; // [0] split plan (reference), [1] in-graph
    for (int arm = 0; arm < 2; ++arm) {
        set_env("GGML_META_AR_IN_GRAPH", arm == 0 ? "0" : "1");
        backends[arm] = ggml_backend_dev_init(meta, nullptr);
        require(backends[arm] != nullptr, "meta backend");
    }
    if (count(g_log, "allreduce: P2P kernel sync (device epochs, no host slot wait)") != 2) {
        puts("SKIP: P2P staging unavailable (needs a P2P-capable pair)");
        return 77;
    }

    ggml_init_params wparams = { (2*L + 1)*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_w = ggml_init(wparams);
    weights w;
    for (int l = 0; l < L; ++l) {
        w.up[l]   = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, D, F);
        w.down[l] = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, F, D);
        ggml_format_name(w.up[l],   "up-%d%s",   l, l == L0 ? "-solo" : "");
        ggml_format_name(w.down[l], "down-%d%s", l, l == L0 ? "-solo" : "");
    }
    w.out = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, D, D);
    ggml_set_name(w.out, "out");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx_w, backends[0]);
    require(wbuf != nullptr, "weight buffer");
    uint32_t seed = 0x9e3779b9u;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_w); t != nullptr; t = ggml_get_next_tensor(ctx_w, t)) {
        const std::vector<float> v = random_values(ggml_nelements(t), next(seed), 0.05f);
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    }

    // A: decode-sized (6 rows: 6 KiB on the BF16 wire), every AllReduce in graph; B: 1024 rows (1 MiB), copy engine.
    const int64_t rows[2] = { 6, 1024 };
    graph g[2][2]; // [arm][graph]
    for (int arm = 0; arm < 2; ++arm) {
        for (int k = 0; k < 2; ++k) {
            g[arm][k] = build(backends[arm], w, rows[k], (uint64_t(1) << 62) + 2*arm + k);
        }
    }

    // Round 0 runs the split layout (first compute of a plan), round 1 captures, the rest replay with new inputs.
    const int rounds = 8;
    size_t mismatches = 0;
    for (int round = 0; round < rounds; ++round) {
        std::vector<uint8_t> res[2][2], out[2][2];
        for (int arm = 0; arm < 2; ++arm) {
            // Both graphs queued before one synchronization: the in-graph replay and the host-driven copy-engine
            // AllReduces share the compute streams.
            for (int k = 0; k < 2; ++k) {
                const std::vector<float> x = random_values(ggml_nelements(g[arm][k].x), 0x51ed2700u + 977u*round + 31u*k, 1.0f);
                ggml_backend_tensor_set(g[arm][k].x, x.data(), 0, ggml_nbytes(g[arm][k].x));
                require(ggml_backend_graph_plan_compute(backends[arm], g[arm][k].plan) == GGML_STATUS_SUCCESS, "compute");
            }
            ggml_backend_synchronize(backends[arm]);
            for (int k = 0; k < 2; ++k) {
                res[arm][k] = read(g[arm][k].res);
                out[arm][k] = read(g[arm][k].out);
            }
        }
        for (int k = 0; k < 2; ++k) {
            if (res[0][k] != res[1][k] || out[0][k] != out[1][k]) {
                fprintf(stderr, "round %d graph %c: in-graph differs from the split plan (residual %s, rows split %s)\n",
                        round, 'A' + k, res[0][k] == res[1][k] ? "equal" : "differs", out[0][k] == out[1][k] ? "equal" : "differs");
                ++mismatches;
            }
        }
    }
    require(mismatches == 0, "in-graph AllReduce changed the result");

    const std::string planned = "AllReduce in graph: 5 of 5 AllReduces, 6 -> 1 graphs per device";
    const std::string engaged = "AllReduce in graph engaged: 1 graphs per device per forward (split: 6)";
    require(count(g_log, planned) == 1, "graph A plan did not merge 6 graphs into 1");
    require(count(g_log, engaged) == 1, "graph A never ran its in-graph layout");
    // Exactly those two lines: the reference arm and graph B (copy engine) must keep the split.
    require(count(g_log, "AllReduce in graph") == 2, "an in-graph layout outside graph A of the in-graph arm");
    require(count(g_log, "GGML_META_AR_IN_GRAPH ignored") == 0 && count(g_log, "in-graph AllReduce needs") == 0,
            "in-graph path declined");

    for (int arm = 0; arm < 2; ++arm) {
        for (int k = 0; k < 2; ++k) {
            release(backends[arm], g[arm][k]);
        }
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx_w);
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
    printf("PASS: in-graph AllReduce bitwise == split plan, %d rounds x 2 graphs (%d replays), 6 -> 1 graphs per device, ranks equal\n",
           rounds, rounds - 2);
    return 0;
}
