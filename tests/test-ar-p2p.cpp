#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "../ggml/src/ggml-impl.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef GGML_TEST_CUDA_GRAPH
#include <cuda_runtime_api.h>
#endif

static void require(bool ok, const char * message) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static float input(size_t i, int rank, int iteration) {
    return float(int((i + 7 * rank + 3 * iteration) % 31) - 15);
}

static void encode(uint8_t * bytes, ggml_type type, size_t i, float value) {
    if (type == GGML_TYPE_F32) {
        memcpy(bytes + i * sizeof(value), &value, sizeof(value));
    } else if (type == GGML_TYPE_F16) {
        const ggml_fp16_t half = ggml_fp32_to_fp16(value);
        memcpy(bytes + i * sizeof(half), &half, sizeof(half));
    } else {
        const ggml_bf16_t half = ggml_fp32_to_bf16(value);
        memcpy(bytes + i * sizeof(half), &half, sizeof(half));
    }
}

static void encode(std::vector<uint8_t> & bytes, ggml_type type, size_t i, float value) {
    encode(bytes.data(), type, i, value);
}

static bool equal(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    return a == b;
}

static uint32_t next(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// Fractional inputs exercise BF16 wire rounding when enabled.
static std::vector<float> random_values(size_t n, uint32_t seed, float scale) {
    std::vector<float> v(n);
    for (auto & x : v) {
        x = (float(int(next(seed) % 2000001u)) - 1000000.0f) * 1.0e-6f * scale;
    }
    return v;
}

constexpr int64_t D  = 512;  // residual width
constexpr int64_t F  = 1024; // hidden width
constexpr int     L  = 5;
constexpr int     L0 = 2;    // rank 1 must contribute zeros for this layer

// Row-split up projections and column-split down projections produce PARTIAL sums.
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
    ggml_tensor *            tweak = nullptr;
    ggml_tensor *         gathered = nullptr;
};

// Use distinct non-zero UIDs, as scheduler split graphs do.
static uint64_t next_uid() {
    static uint64_t uid = 0;
    return ++uid;
}

static graph build(ggml_backend_t backend, const weights & w, int64_t rows, uint64_t uid, int layers = L, int extra = 0, bool gather = false) {
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
        cur = ggml_add(g.ctx, cur, y);
        if (gather && l == 1) {
            ggml_tensor * boundary = ggml_backend_meta_gather(g.ctx, ggml_mul_mat(g.ctx, w.out, cur));
            ggml_set_name(boundary, "sampling_candidates");
            g.gathered = ggml_cont(g.ctx, boundary);
            ggml_set_output(g.gathered);
            cur = g.gathered;
        }
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

static void * (*get_proc)(ggml_backend_reg_t, const char *);
static ggml_backend_comm_graph_compute_t graph_compute;
static bool split_reference = false;
static size_t graph_calls = 0;
static size_t graph_runs = 0;
static std::vector<size_t> graph_run_sizes;

static bool compute_spy(void * ctx, ggml_cgraph ** graphs, size_t n_graphs) {
    ++graph_calls;
    for (size_t g = 0; g + 1 < n_graphs; ++g) {
        for (size_t r = 0; r < 2; ++r) {
            ggml_cgraph * graph = graphs[2*g + r];
            require(strcmp(graph->nodes[graph->n_nodes - 1]->name, "sampling_candidates") != 0, "gather inside merged run");
        }
    }
    const bool handled = graph_compute(ctx, graphs, n_graphs);
    graph_runs += handled;
    if (handled) {
        graph_run_sizes.push_back(n_graphs);
    }
    return handled;
}

static void * lookup(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_comm_graph_compute") == 0) {
        return split_reference ? nullptr : (void *) compute_spy;
    }
    return get_proc(reg, name);
}

// ggml_graph_view is internal to ggml-base and not exported; same view: a node range sharing the parent's bookkeeping.
static ggml_cgraph graph_view(ggml_cgraph * graph, int i0, int i1) {
    ggml_cgraph view = *graph;
    view.size      = 0;
    view.n_nodes   = i1 - i0;
    view.n_leafs   = 0;
    view.nodes     = graph->nodes + i0;
    view.grads     = nullptr;
    view.grad_accs = nullptr;
    view.leafs     = nullptr;
    view.uid       = 0;
    return view;
}

static bool graph_qualified() {
#ifdef GGML_TEST_CUDA_GRAPH
    if (std::getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr) {
        return false;
    }
    // The collective capture is refused while the ggml graph is captured in chunks.
    const char * chunk = std::getenv("GGML_CUDA_GRAPH_CHUNK_SIZE");
    if (chunk == nullptr || std::atoi(chunk) != 0) {
        return false;
    }
    int count = 0;
    require(cudaGetDeviceCount(&count) == cudaSuccess, "CUDA device query failed");
    if (count < 2) {
        return false;
    }
    for (int rank = 0; rank < 2; ++rank) {
        int major = 0, peer = 0;
        require(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, rank) == cudaSuccess, "CUDA architecture query failed");
        require(cudaDeviceCanAccessPeer(&peer, rank, 1 - rank) == cudaSuccess, "CUDA peer query failed");
        if (major < 7 || !peer) {
            return false;
        }
    }
    return true;
#else
    return false;
#endif
}

#ifdef GGML_TEST_CUDA_GRAPH
struct reduction_graph {
    ggml_context * ctx[2];
    ggml_backend_buffer_t buffers[2];
    ggml_tensor * inputs[2];
    ggml_tensor * sums[2];
    ggml_tensor * outputs[2];
    ggml_cgraph steps[4];
    ggml_cgraph * graphs[4];

    reduction_graph(ggml_backend_t * backends, ggml_type type, size_t n) {
        for (int rank = 0; rank < 2; ++rank) {
            ctx[rank] = ggml_init({3 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
            require(ctx[rank] != nullptr, "reduction graph context");
            inputs[rank] = ggml_new_tensor_1d(ctx[rank], type, n);
            sums[rank] = ggml_dup(ctx[rank], inputs[rank]);
            outputs[rank] = ggml_dup(ctx[rank], sums[rank]);
            sums[rank]->flags |= GGML_TENSOR_FLAG_COMPUTE;
            outputs[rank]->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_cgraph * full = ggml_new_graph(ctx[rank]);
            ggml_build_forward_expand(full, outputs[rank]);
            require(full->n_nodes == 2, "reduction graph boundaries");
            steps[rank] = graph_view(full, 0, 1);
            steps[2 + rank] = graph_view(full, 1, 2);
            graphs[rank] = &steps[rank];
            graphs[2 + rank] = &steps[2 + rank];
            buffers[rank] = ggml_backend_alloc_ctx_tensors(ctx[rank], backends[rank]);
            require(buffers[rank] != nullptr, "reduction graph buffer");
        }
    }

    ~reduction_graph() {
        for (int rank = 0; rank < 2; ++rank) {
            ggml_backend_buffer_free(buffers[rank]);
            ggml_free(ctx[rank]);
        }
    }
};

static void split_compute(reduction_graph & g, ggml_backend_t * backends, void * comm, ggml_backend_comm_allreduce_tensor_t reduce) {
    for (int step = 0; step < 2; ++step) {
        for (int rank = 0; rank < 2; ++rank) {
            require(ggml_backend_graph_compute_async(backends[rank], g.graphs[2 * step + rank]) == GGML_STATUS_SUCCESS, "split compute");
        }
        if (step == 0) {
            require(reduce(comm, g.sums), "split reduction declined");
        }
    }
}

static void replay_cases(ggml_backend_reg_t reg, ggml_backend_t * backends, void * comm, ggml_backend_comm_allreduce_tensor_t reduce) {
    auto compute = (ggml_backend_comm_graph_compute_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_graph_compute");
    require(compute != nullptr, "qualified pair requires ggml_backend_comm_graph_compute");
    const char * threshold = std::getenv("GGML_CUDA_AR_COPY_THRESHOLD");
    const char * bf16 = std::getenv("GGML_CUDA_AR_BF16_THRESHOLD");
    require(threshold && strcmp(threshold, "2097152") == 0, "set GGML_CUDA_AR_COPY_THRESHOLD=2097152 for multi-chunk and copy-engine coverage");
    require(bf16 && strcmp(bf16, "1") == 0, "set GGML_CUDA_AR_BF16_THRESHOLD=1 for F32 wire coverage");
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        for (size_t n : {size_t(256), size_t(524296)}) {
            reduction_graph reference(backends, type, n), captured(backends, type, n);
            const size_t bytes = ggml_nbytes(captured.inputs[0]);
            std::vector<uint8_t> zero(bytes, 0);
            for (int rank = 0; rank < 2; ++rank) {
                ggml_backend_tensor_set(captured.inputs[rank], zero.data(), 0, bytes);
                ggml_backend_tensor_set(reference.inputs[rank], zero.data(), 0, bytes);
            }
            require(!compute(comm, captured.graphs, 2), "cold reduction graph bypassed warmup");
            split_compute(captured, backends, comm, reduce);
            require(compute(comm, captured.graphs, 2), "qualified reduction graph never engaged");
            split_compute(reference, backends, comm, reduce);
            split_compute(reference, backends, comm, reduce);
            for (int rank = 0; rank < 2; ++rank) {
                ggml_backend_synchronize(backends[rank]);
            }

            constexpr int replays = 6;
            uint8_t * host = nullptr;
            require(cudaMallocHost((void **) &host, replays * 6 * bytes) == cudaSuccess, "pinned replay buffers");
            uint8_t * data[replays][2], * actual[replays][2][2];
            std::vector<uint8_t> expected[replays];
            for (int round = 0; round < replays; ++round) {
                expected[round].resize(bytes);
                for (int rank = 0; rank < 2; ++rank) {
                    data[round][rank] = host + (round * 6 + rank) * bytes;
                    for (size_t i = 0; i < n; ++i) {
                        encode(data[round][rank], type, i, input(i, rank, round));
                    }
                }
                for (size_t i = 0; i < n; ++i) {
                    encode(expected[round], type, i, input(i, 0, round) + input(i, 1, round));
                }
            }
            for (int round = 0; round < replays; ++round) {
                for (int arm = 0; arm < 2; ++arm) {
                    reduction_graph & g = arm == 0 ? reference : captured;
                    for (int rank = 0; rank < 2; ++rank) {
                        ggml_backend_tensor_set_async(backends[rank], g.inputs[rank], data[round][rank], 0, bytes);
                    }
                    if (arm == 0) {
                        split_compute(g, backends, comm, reduce);
                    } else {
                        require(compute(comm, g.graphs, 2), "queued replay fell back to split");
                    }
                    for (int rank = 0; rank < 2; ++rank) {
                        actual[round][arm][rank] = host + (round * 6 + 2 + arm * 2 + rank) * bytes;
                        ggml_backend_tensor_get_async(backends[rank], g.outputs[rank], actual[round][arm][rank], 0, bytes);
                    }
                }
            }
            for (int rank = 0; rank < 2; ++rank) {
                ggml_backend_synchronize(backends[rank]);
            }
            for (int round = 0; round < replays; ++round) {
                for (int rank = 0; rank < 2; ++rank) {
                    require(memcmp(actual[round][0][rank], actual[round][1][rank], bytes) == 0 && memcmp(actual[round][1][rank], expected[round].data(), bytes) == 0, "queued reduction bytes differ");
                }
            }
            require(cudaFreeHost(host) == cudaSuccess, "free pinned replay buffers");

            reduction_graph large(backends, type, 1048576);
            const size_t large_bytes = ggml_nbytes(large.inputs[0]);
            std::vector<uint8_t> ones(large_bytes), twos(large_bytes);
            for (size_t i = 0; i < 1048576; ++i) {
                encode(ones, type, i, 1.0f);
                encode(twos, type, i, 2.0f);
            }
            for (int transition = 0; transition < 3; ++transition) {
                for (int rank = 0; rank < 2; ++rank) {
                    ggml_backend_tensor_set_async(backends[rank], large.inputs[rank], ones.data(), 0, large_bytes);
                }
                require(!compute(comm, large.graphs, 2), "copy-engine graph must use split fallback");
                split_compute(large, backends, comm, reduce);
                require(compute(comm, captured.graphs, 2), "capture failed after copy-engine reduction");
                for (int rank = 0; rank < 2; ++rank) {
                    ggml_backend_synchronize(backends[rank]);
                    require(read(large.outputs[rank]) == twos, "copy-engine transition bytes differ");
                    require(read(captured.outputs[rank]) == expected[replays - 1], "captured transition bytes differ");
                }
            }
        }
    }
    puts("PASS: captured F32/F16/BF16, multi-chunk epochs, queued replays and captured/copy-engine transitions");
}
#endif

static void graph_cases(ggml_backend_reg_t reg) {
    get_proc = reg->iface.get_proc_address;
    graph_compute = (ggml_backend_comm_graph_compute_t) get_proc(reg, "ggml_backend_comm_graph_compute");
    require(graph_compute != nullptr, "qualified pair requires collective graph procedure");
    require(!graph_compute(nullptr, nullptr, 0), "empty collective submitted work");
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
    ggml_context * ctx_w = ggml_init({ (2*L + 1)*ggml_tensor_overhead(), nullptr, true });
    require(ctx_w != nullptr, "weight context");
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
    const int layers[] = { L, 1, L, L };
    const int extras[] = { 0, 40, 0, 0 };
    for (int phase = 0; phase < 4; ++phase) {
        graph graphs[2];
        std::vector<uint8_t> res_before_change;
        for (int arm = 0; arm < 2; ++arm) {
            graphs[arm] = build(backends[arm], w, 6, next_uid(), layers[phase], extras[phase], phase == 3);
        }
        for (int round = 0; round < 8; ++round) {
            if (phase == 2 && round == 4) {
                for (auto & g : graphs) {
                    ggml_set_op_params_f32(g.tweak, 0, 0.75f);
                    g.gf->uid = next_uid();
                }
            }
            const size_t runs_before = graph_runs;
            graph_run_sizes.clear();
            std::vector<uint8_t> res[2], out[2], gathered[2];
            for (int arm = 0; arm < 2; ++arm) {
                graph & g = graphs[arm];
                const int input_round = phase == 2 && round == 4 ? round - 1 : round;
                const auto x = random_values(ggml_nelements(g.x), 0x51ed2700u + 977u*input_round, 1.0f);
                ggml_backend_tensor_set(g.x, x.data(), 0, ggml_nbytes(g.x));
                require(ggml_backend_graph_compute_async(backends[arm], g.gf) == GGML_STATUS_SUCCESS, "compute");
                ggml_backend_synchronize(backends[arm]);
                res[arm] = read(g.res);
                out[arm] = read(g.out);
                if (g.gathered != nullptr) {
                    gathered[arm] = read(g.gathered);
                }
            }
            require(res[0] == res[1] && out[0] == out[1] && gathered[0] == gathered[1], "collective graph differs from split");
            if (phase == 2 && round == 3) {
                res_before_change = res[0];
            }
            if (phase == 2 && round == 4) {
                for (size_t i = 0; i < res[0].size(); i += sizeof(float)) {
                    float before, after;
                    memcpy(&before, res_before_change.data() + i, sizeof(float));
                    memcpy(&after, res[0].data() + i, sizeof(float));
                    require(after == 0.75f * before, "changed scale parameter was not applied");
                }
            }
            if ((phase == 0 && round == 0) || (phase == 2 && round == 4)) {
                require(graph_runs == runs_before, "cold/changed properties bypassed warmup");
            }
            if (round >= 2 && !(phase == 2 && round == 4)) {
                require(graph_runs == runs_before + (phase == 3 ? 2 : 1), "stable collective fell back to split");
                if (phase == 3) {
                    require(graph_run_sizes == std::vector<size_t>({3, 4}), "gather did not delimit two merged runs");
                }
            }
        }
        for (auto & g : graphs) {
            release(g);
        }
    }
    require(graph_calls > graph_runs, "cold graph bypassed warmup");
    require(graph_runs > 0, "qualified meta graph never engaged");
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx_w);
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
    puts("PASS: bitwise split identity, replay, inactive rank, arena rebuild, property change and gather-delimited runs");
}

int main(int argc, char ** argv) {
    const char * output = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--output FILE]\n", argv[0]);
            return 1;
        }
    }
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (!reg || ggml_backend_reg_dev_count(reg) < 2) {
        puts("SKIP: two CUDA devices required");
        return 77;
    }
    const char * provider = std::getenv("GGML_CUDA_ALLREDUCE");
    require(provider && strcmp(provider, "internal") == 0, "set GGML_CUDA_ALLREDUCE=internal");
    const bool graphs = std::getenv("GGML_CUDA_P2P") != nullptr;
    if (graphs && !graph_qualified()) {
        puts("SKIP: CUDA graphs enabled and two Volta-or-newer devices with bidirectional P2P required");
        return 77;
    }
    auto init = reinterpret_cast<ggml_backend_comm_init_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init"));
    auto reduce = reinterpret_cast<ggml_backend_comm_allreduce_tensor_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor"));
    auto free_comm = reinterpret_cast<ggml_backend_comm_free_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free"));
    require(init && reduce && free_comm, "CUDA comm API missing");
    ggml_backend_t backends[2];
    for (int rank = 0; rank < 2; ++rank) {
        backends[rank] = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, rank), nullptr);
        require(backends[rank] != nullptr, "backend init");
    }
    void * comm = init(backends, 2);
    require(comm != nullptr, "internal comm init");
    FILE * file = output ? fopen(output, "wb") : nullptr;
    require(!output || file, "open output");
    size_t cases = 0;
    // F32 counts with n % 8 == 4 reach BF16-wire scalar tails; all inputs must occupy a multiple of 16 bytes.
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        for (size_t n : {size_t(8), size_t(256), size_t(260), size_t(262136), size_t(262140), size_t(262144), size_t(262152), size_t(524280), size_t(524288), size_t(524292), size_t(524296)}) {
            if (type != GGML_TYPE_F32 && n % 8 != 0) {
                continue;
            }
            ggml_init_params params = {2 * ggml_tensor_overhead() + 256, nullptr, true};
            ggml_context * ctx = ggml_init(params);
            require(ctx != nullptr, "tensor context");
            ggml_tensor * tensors[2];
            ggml_backend_buffer_t buffers[2];
            const size_t bytes = n * ggml_type_size(type);
            require(bytes % 16 == 0, "AllReduce input alignment");
            for (int rank = 0; rank < 2; ++rank) {
                tensors[rank] = ggml_new_tensor_1d(ctx, type, n);
                buffers[rank] = ggml_backend_alloc_buffer(backends[rank], bytes + 256);
                require(buffers[rank] != nullptr, "tensor buffer");
                require(ggml_backend_tensor_alloc(buffers[rank], tensors[rank], ggml_backend_buffer_get_base(buffers[rank])) == GGML_STATUS_SUCCESS, "tensor allocation");
            }
            std::vector<uint8_t> data[2] = {std::vector<uint8_t>(bytes), std::vector<uint8_t>(bytes)};
            std::vector<uint8_t> expected(bytes), actual(bytes);
            // Integer operands and sums are exact in all three formats, including F32 with BF16 wire conversion.
            for (int inactive = -1; inactive < 2; ++inactive) {
                for (int rank = 0; rank < 2; ++rank) {
                    tensors[rank]->flags = rank == inactive ? 0 : GGML_TENSOR_FLAG_COMPUTE;
                    for (size_t i = 0; i < n; ++i) {
                        encode(data[rank], type, i, input(i, rank, inactive + 1));
                    }
                    ggml_backend_tensor_set(tensors[rank], data[rank].data(), 0, bytes);
                }
                for (size_t i = 0; i < n; ++i) {
                    const float sum = (inactive == 0 ? 0 : input(i, 0, inactive + 1)) +
                                      (inactive == 1 ? 0 : input(i, 1, inactive + 1));
                    encode(expected, type, i, sum);
                }
                require(reduce(comm, tensors), "AllReduce declined input");
                for (int rank = 0; rank < 2; ++rank) {
                    ggml_backend_synchronize(backends[rank]);
                    ggml_backend_tensor_get(tensors[rank], actual.data(), 0, bytes);
                    if (!equal(actual, expected)) {
                        fprintf(stderr, "type=%s n=%zu inactive=%d rank=%d\n", ggml_type_name(type), n, inactive, rank);
                        require(false, "AllReduce bytes differ from exact sum");
                    }
                    if (file) {
                        require(fwrite(actual.data(), 1, bytes, file) == bytes, "write output");
                    }
                }
                ++cases;
            }
            for (int rank = 0; rank < 2; ++rank) {
                ggml_backend_buffer_free(buffers[rank]);
            }
            ggml_free(ctx);
        }
    }
    if (file) {
        require(fclose(file) == 0, "close output");
    }
#ifdef GGML_TEST_CUDA_GRAPH
    if (graphs) {
        replay_cases(reg, backends, comm, reduce);
    }
#endif
    free_comm(comm);
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
    if (graphs) {
        graph_cases(reg);
    }
    printf("PASS: %zu AllReduce cases\n", cases);
    return 0;
}
