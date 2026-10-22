#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-impl.h"
#include "../ggml/src/ggml-backend-impl.h"

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
    ggml_tensor * tweak = nullptr;
};

static graph build(ggml_backend_t backend, const weights & w, int64_t rows, uint64_t uid, int layers = L, int extra = 0) {
    graph g;
    ggml_init_params params = { (8*L + 16 + extra)*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(params);
    require(g.ctx != nullptr, "graph context");
    g.x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, rows);
    ggml_set_name(g.x, "x");
    ggml_set_input(g.x);
    ggml_tensor * cur = g.x;
    for (int l = 0; l < layers; ++l) {
        ggml_tensor * h = ggml_mul_mat(g.ctx, w.up[l], cur);
        ggml_tensor * y = ggml_mul_mat(g.ctx, w.down[l], h);
        if (l == 1) {
            ggml_set_name(y, "force-split");
        }
        cur = ggml_add(g.ctx, cur, y);
    }
    for (int i = 0; i < extra; ++i) {
        cur = ggml_scale(g.ctx, cur, 0.99f);
    }
    g.tweak = ggml_scale(g.ctx, cur, 1.0f);
    g.res = g.tweak;
    ggml_set_output(g.res);
    g.out = ggml_mul_mat(g.ctx, w.out, g.res);
    ggml_set_output(g.out);
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    g.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(ggml_gallocr_alloc_graph(g.ga, g.gf), "graph allocation");
    // A plan is reused across computes only for a graph with a stable non-zero uid (as the scheduler's split graphs).
    g.gf->uid = uid;
    return g;
}

static void release(graph & g) {
    ggml_gallocr_free(g.ga);
    ggml_free(g.ctx);
}

static std::vector<uint8_t> read(const ggml_tensor * t) {
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    return bytes;
}

// Suppress only the optional procedures for the reference; no production test knob.
static void * (*get_proc)(ggml_backend_reg_t, const char *);
static ggml_backend_comm_init_t comm_init;
static ggml_backend_comm_graph_ready_t graph_ready;
static ggml_backend_comm_allreduce_graph_t graph_node;
static bool split_reference = false;
static void * comm_contexts[2] = {};
static size_t n_contexts = 0;
static size_t ready_calls = 0;
static size_t ready_true = 0;

static void * init_spy(ggml_backend_t * backends, size_t n) {
    void * ctx = comm_init(backends, n);
    require(n_contexts < 2, "unexpected comm context");
    comm_contexts[n_contexts++] = ctx;
    return ctx;
}

static bool ready_spy(void * ctx, ggml_cgraph ** graphs) {
    ++ready_calls;
    const bool ready = graph_ready(ctx, graphs);
    ready_true += ready;
    return ready;
}

static bool node_spy(void * ctx, ggml_tensor ** tensors, ggml_tensor ** nodes) {
    // Keep one boundary split to exercise retained merged-graph slots after a smaller rebuild.
    if (strcmp(tensors[0]->name, "force-split") == 0) {
        return false;
    }
    return graph_node(ctx, tensors, nodes);
}

static void * lookup(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *) init_spy;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_graph") == 0) {
        return split_reference ? nullptr : (void *) node_spy;
    }
    if (strcmp(name, "ggml_backend_comm_graph_ready") == 0) {
        return split_reference || graph_ready == nullptr ? nullptr : (void *) ready_spy;
    }
    return get_proc(reg, name);
}

struct copy_pair {
    ggml_context * ctx;
    ggml_tensor * tensors[2];
    ggml_backend_buffer_t buffers[2];
};

static copy_pair make_copy_pair(ggml_backend_t backend) {
    copy_pair p = {};
    p.ctx = ggml_init({2*ggml_tensor_overhead(), nullptr, true});
    for (size_t i = 0; i < 2; ++i) {
        p.tensors[i] = ggml_new_tensor_1d(p.ctx, GGML_TYPE_F32, D*1024);
        ggml_set_name(p.tensors[i], "copy-engine-input");
        p.buffers[i] = ggml_backend_alloc_buffer(ggml_backend_meta_simple_backend(backend, i), ggml_nbytes(p.tensors[i]));
        require(p.buffers[i] != nullptr, "copy-engine buffer");
        require(ggml_backend_tensor_alloc(p.buffers[i], p.tensors[i], ggml_backend_buffer_get_base(p.buffers[i])) == GGML_STATUS_SUCCESS,
                "copy-engine tensor allocation");
        ggml_set_input(p.tensors[i]);
        p.tensors[i]->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }
    return p;
}

// Run in a separate process: the expected trap makes the device context unusable.
static int timeout_case(ggml_backend_reg_t reg) {
    ggml_backend_t backends[2];
    ggml_context * ctx = ggml_init({4*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    ggml_tensor * tensors[2];
    ggml_tensor * nodes[2];
    for (size_t i = 0; i < 2; ++i) {
        backends[i] = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, i), nullptr);
        tensors[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
        ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(backends[i], ggml_nbytes(tensors[i]));
        require(buf != nullptr, "timeout buffer");
        require(ggml_backend_tensor_alloc(buf, tensors[i], ggml_backend_buffer_get_base(buf)) == GGML_STATUS_SUCCESS, "timeout tensor");
        tensors[i]->flags = GGML_TENSOR_FLAG_COMPUTE;
        nodes[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        std::vector<float> zeros(D, 0.0f);
        ggml_backend_tensor_set(tensors[i], zeros.data(), 0, ggml_nbytes(tensors[i]));
    }
    const auto init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    const auto make = (ggml_backend_comm_allreduce_graph_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_graph");
    void * comm = init(backends, 2);
    if (make == nullptr || !make(comm, tensors, nodes)) {
        puts("SKIP: graph-recordable P2P AllReduce unavailable");
        return 77;
    }
    const auto reduce = (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");
    require(reduce(comm, tensors), "timeout warmup");
    for (auto backend : backends) {
        ggml_backend_synchronize(backend);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    graph->nodes[0] = nodes[0];
    graph->n_nodes = 1;
    // Rank 1 is deliberately never submitted.
    require(ggml_backend_graph_compute_async(backends[0], graph) == GGML_STATUS_SUCCESS, "timeout graph submit");
    ggml_backend_synchronize(backends[0]);
    require(false, "missing peer did not fail");
    return 1;
}

int main(int argc, char ** argv) {
    ggml_log_set(log_capture, nullptr);
    set_env("GGML_CUDA_ALLREDUCE", "internal");
    set_env("GGML_CUDA_P2P", "1");
    set_env("GGML_CUDA_AR_COPY_THRESHOLD", "1048576");
    set_env("GGML_CUDA_AR_BF16_THRESHOLD", "1");
    set_env("GGML_CUDA_GRAPH_DEBUG", "1");
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (!reg || ggml_backend_reg_dev_count(reg) < 2) {
        puts("SKIP: two CUDA devices required");
        return 77;
    }
    if (argc == 2 && strcmp(argv[1], "--timeout") == 0) {
        return timeout_case(reg);
    }
    get_proc = reg->iface.get_proc_address;
    comm_init = (ggml_backend_comm_init_t) get_proc(reg, "ggml_backend_comm_init");
    graph_ready = (ggml_backend_comm_graph_ready_t) get_proc(reg, "ggml_backend_comm_graph_ready");
    const auto make = (ggml_backend_comm_allreduce_graph_t) get_proc(reg, "ggml_backend_comm_allreduce_graph");
    graph_node = make;
    if (graph_ready == nullptr || make == nullptr) {
        puts("SKIP: CUDA graph procedures unavailable");
        return 77;
    }
    ggml_backend_dev_t devs[2] = { ggml_backend_reg_dev_get(reg, 0), ggml_backend_reg_dev_get(reg, 1) };
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2, split_state, nullptr);
    ggml_backend_t backends[2];
    reg->iface.get_proc_address = lookup;
    for (int arm = 0; arm < 2; ++arm) {
        split_reference = arm == 0;
        backends[arm] = ggml_backend_dev_init(meta, nullptr);
        require(backends[arm] != nullptr, "meta backend");
    }
    reg->iface.get_proc_address = get_proc;
    copy_pair copies[2] = { make_copy_pair(backends[0]), make_copy_pair(backends[1]) };
    // Query actual support, including compiled/runtime graph disablement and peer access.
    ggml_tensor probe[2] = { *copies[1].tensors[0], *copies[1].tensors[1] };
    ggml_tensor * probes[2] = { &probe[0], &probe[1] };
    for (auto & t : probe) {
        t.ne[0] = D;
        t.nb[1] = t.nb[2] = t.nb[3] = D*sizeof(float);
    }
    if (!make(comm_contexts[1], probes, nullptr)) {
        puts("SKIP: graph-recordable P2P AllReduce unavailable (graphs disabled or peer access unavailable)");
        return 77;
    }
    ggml_init_params wparams = { (2*L + 1)*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_w = ggml_init(wparams);
    weights w;
    for (int l = 0; l < L; ++l) {
        w.up[l] = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, D, F);
        w.down[l] = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, F, D);
        ggml_format_name(w.up[l], "up-%d%s", l, l == L0 ? "-solo" : "");
        ggml_format_name(w.down[l], "down-%d%s", l, l == L0 ? "-solo" : "");
    }
    w.out = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, D, D);
    ggml_set_name(w.out, "out");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx_w, backends[0]);
    require(wbuf != nullptr, "weight buffer");
    uint32_t seed = 0x9e3779b9u;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_w); t != nullptr; t = ggml_get_next_tensor(ctx_w, t)) {
        const auto v = random_values(ggml_nelements(t), next(seed), 0.05f);
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    }
    const auto reduce = (ggml_backend_comm_allreduce_tensor_t) get_proc(reg, "ggml_backend_comm_allreduce_tensor");
    const int layers[] = { L, 1, L, L };
    const int extras[] = { 0, 40, 0, 0 };
    const int64_t rows[] = { 6, 6, 6, 1024 };
    for (int phase = 0; phase < 4; ++phase) {
        graph graphs[2];
        for (int arm = 0; arm < 2; ++arm) {
            graphs[arm] = build(backends[arm], w, rows[phase], ggml_graph_next_uid(), layers[phase], extras[phase]);
        }
        const size_t captures_before = count(g_log, "CUDA AllReduce graph capture");
        const size_t ready_before = ready_true;
        size_t captures_stable = 0;
        for (int round = 0; round < 8; ++round) {
            if (phase == 2 && round == 4) {
                for (auto & g : graphs) {
                    ggml_set_op_params_f32(g.tweak, 0, 0.75f);
                    g.gf->uid = ggml_graph_next_uid();
                }
            }
            std::vector<uint8_t> res[2], out[2], copied[2];
            for (int arm = 0; arm < 2; ++arm) {
                graph & g = graphs[arm];
                const auto x = random_values(ggml_nelements(g.x), 0x51ed2700u + 977u*round, 1.0f);
                ggml_backend_tensor_set(g.x, x.data(), 0, ggml_nbytes(g.x));
                require(ggml_backend_graph_compute_async(backends[arm], g.gf) == GGML_STATUS_SUCCESS, "compute");
                // The copy-engine reduction shares this comm pipeline and the same compute streams with graph replay.
                for (int rank = 0; rank < 2; ++rank) {
                    const auto v = random_values(D*1024, 77u + 31u*round + rank, 1.0f);
                    ggml_backend_tensor_set(copies[arm].tensors[rank], v.data(), 0, ggml_nbytes(copies[arm].tensors[rank]));
                }
                require(reduce(comm_contexts[arm], copies[arm].tensors), "interleaved copy-engine reduction");
                ggml_backend_synchronize(backends[arm]);
                res[arm] = read(g.res);
                out[arm] = read(g.out);
                copied[arm] = read(copies[arm].tensors[0]);
                require(copied[arm] == read(copies[arm].tensors[1]), "copy-engine ranks differ");
            }
            require(res[0] == res[1] && out[0] == out[1] && copied[0] == copied[1], "merged differs from split");
            const size_t captures = count(g_log, "CUDA AllReduce graph capture");
            if (round == 2 || (phase == 2 && round == 6)) {
                captures_stable = captures;
            }
            if (round == 3 || round == 7) {
                require(captures == captures_stable, "stable graph recaptured instead of replaying");
            }
            if (phase == 2 && round == 4) {
                require(captures == captures_stable, "changed graph captured before stabilization");
            }
        }
        if (phase < 3) {
            require(ready_true > ready_before, "merged layout never became ready");
            require(count(g_log, "CUDA AllReduce graph capture") > captures_before, "merged graph was never captured");
            if (phase == 0) {
                require(count(g_log, "CUDA AllReduce graph capture") >= captures_before + 4, "two merged segments were not captured on both ranks");
            }
            if (phase == 2) {
                require(count(g_log, "CUDA AllReduce graph capture") >= captures_before + 4, "property change did not recapture both ranks");
            }
        } else {
            require(count(g_log, "CUDA AllReduce graph capture") == captures_before, "copy-engine graph merged");
        }
        for (auto & g : graphs) {
            release(g);
        }
    }
    require(ready_calls > ready_true, "cold/changed graphs never used split stabilization");
    for (int arm = 0; arm < 2; ++arm) {
        for (auto buf : copies[arm].buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(copies[arm].ctx);
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx_w);
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
    puts("PASS: bitwise split identity, replay, copy interleaving, shrink/grow arena and property recapture");
    return 0;
}
