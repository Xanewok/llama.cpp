#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "../ggml/src/ggml-impl.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static float input(size_t i, int rank, int iteration) {
    return float(int((i + 7 * rank + 3 * iteration) % 31) - 15);
}

static void encode(std::vector<uint8_t> & bytes, ggml_type type, size_t i, float value) {
    if (type == GGML_TYPE_F32) {
        memcpy(bytes.data() + i * sizeof(value), &value, sizeof(value));
    } else if (type == GGML_TYPE_F16) {
        const ggml_fp16_t half = ggml_fp32_to_fp16(value);
        memcpy(bytes.data() + i * sizeof(half), &half, sizeof(half));
    } else {
        const ggml_bf16_t half = ggml_fp32_to_bf16(value);
        memcpy(bytes.data() + i * sizeof(half), &half, sizeof(half));
    }
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
};

// Use distinct non-zero UIDs, as scheduler split graphs do.
static uint64_t next_uid() {
    static uint64_t uid = 0;
    return ++uid;
}

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

static bool compute_spy(void * ctx, ggml_cgraph ** graphs, size_t n_graphs) {
    ++graph_calls;
    ggml_cgraph * incomplete[] = { graphs[0], nullptr };
    require(!graph_compute(ctx, incomplete, 2), "incomplete rank list submitted work");
    const bool handled = graph_compute(ctx, graphs, n_graphs);
    graph_runs += handled;
    return handled;
}

static void * lookup(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_comm_graph_compute") == 0) {
        return split_reference ? nullptr : (void *) compute_spy;
    }
    return get_proc(reg, name);
}

static void graph_cases(ggml_backend_reg_t reg) {
    get_proc = reg->iface.get_proc_address;
    graph_compute = (ggml_backend_comm_graph_compute_t) get_proc(reg, "ggml_backend_comm_graph_compute");
    if (graph_compute == nullptr) {
        puts("SKIP: collective graph procedure unavailable");
        return;
    }
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
    bool graph_supported = false;
    const int layers[] = { L, 1, L };
    const int extras[] = { 0, 40, 0 };
    for (int phase = 0; phase < 3; ++phase) {
        graph graphs[2];
        for (int arm = 0; arm < 2; ++arm) {
            graphs[arm] = build(backends[arm], w, 6, next_uid(), layers[phase], extras[phase]);
        }
        for (int round = 0; round < 8; ++round) {
            if (phase == 2 && round == 4) {
                for (auto & g : graphs) {
                    ggml_set_op_params_f32(g.tweak, 0, 0.75f);
                    g.gf->uid = next_uid();
                }
            }
            const size_t runs_before = graph_runs;
            std::vector<uint8_t> res[2], out[2];
            for (int arm = 0; arm < 2; ++arm) {
                graph & g = graphs[arm];
                const auto x = random_values(ggml_nelements(g.x), 0x51ed2700u + 977u*round, 1.0f);
                ggml_backend_tensor_set(g.x, x.data(), 0, ggml_nbytes(g.x));
                require(ggml_backend_graph_compute_async(backends[arm], g.gf) == GGML_STATUS_SUCCESS, "compute");
                ggml_backend_synchronize(backends[arm]);
                res[arm] = read(g.res);
                out[arm] = read(g.out);
            }
            require(res[0] == res[1] && out[0] == out[1], "collective graph differs from split");
            if ((phase == 0 && round == 0) || (phase == 2 && round == 4)) {
                require(graph_runs == runs_before, "cold/changed properties bypassed warmup");
            }
            if (phase == 0 && round == 2) {
                graph_supported = graph_runs > 0;
            }
            if (graph_supported && round >= 2 && !(phase == 2 && round == 4)) {
                require(graph_runs == runs_before + 1, "stable collective fell back to split");
            }
        }
        for (auto & g : graphs) {
            release(g);
        }
    }
    require(graph_calls > graph_runs, "cold graph bypassed warmup");
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx_w);
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
    puts(graph_runs ? "PASS: bitwise split identity, replay, inactive rank, arena rebuild and property change" :
                      "SKIP: collective graph capture unsupported; split fallback matched");
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
    free_comm(comm);
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
    graph_cases(reg);
    printf("PASS: %zu AllReduce cases\n", cases);
    return 0;
}
