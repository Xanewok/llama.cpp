// Runs the input staging code of ggml-cuda.cu (sliced into STAGE_SNIPPET by the Mac gate) against cuda_stub.h.
// usage: stage_sim SCENARIO   order-slow1 | order-sym | order-slow0 | order-unmeasured | ring-lazy | ring-random | perrank | declines
//                             | scatter-{perrank,stage}-{lazy,random} | scatter-fallback
#include "cuda_stub.h"
#include "ggml-cpp.h"

#include <random>

#include STAGE_SNIPPET

#define REQUIRE(cond, ...)                                                    \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "FAIL %s:%d %s: ", __FILE__, __LINE__, #cond);    \
            fprintf(stderr, "" __VA_ARGS__);                                  \
            fprintf(stderr, "\n");                                            \
            return 1;                                                         \
        }                                                                     \
    } while (0)

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static void fill_random(std::vector<uint8_t> & v) {
    for (auto & b : v) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        b = (uint8_t) (rng_state >> 24);
    }
}

struct fake_rank {
    ggml_backend backend{};
    ggml_backend_cuda_context ctx;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context * gctx = nullptr;
    std::vector<ggml_tensor *> a, b;
    std::vector<std::vector<uint8_t>> mem;
};

static void make_rank(fake_rank & r, int device, int n_pairs, size_t size_a, size_t size_b) {
    r.ctx.device = device;
    r.backend.iface.get_name = sim_backend_name;
    r.backend.context = &r.ctx;
    ggml_backend_buffer_i iface{};
    r.buffer = ggml_backend_buffer_init(ggml_backend_cuda_buffer_type(device), iface, nullptr, 0);
    ggml_init_params params = { (size_t) (2*n_pairs + 2)*ggml_tensor_overhead(), nullptr, true };
    r.gctx = ggml_init(params);
    for (int i = 0; i < n_pairs; ++i) {
        for (int which = 0; which < 2; ++which) {
            const size_t n = (which == 0 ? size_a : size_b) + (size_t) i*512;
            ggml_tensor * t = ggml_new_tensor_1d(r.gctx, GGML_TYPE_I8, (int64_t) n);
            r.mem.emplace_back(n, 0);
            t->data = r.mem.back().data();
            t->buffer = r.buffer;
            (which == 0 ? r.a : r.b).push_back(t);
        }
    }
}

struct batch {
    std::vector<uint8_t> a, b, c;
    std::vector<uint8_t> expect_a, expect_b;
    batch(size_t size_a, size_t size_b) : a(size_a), b(size_b / 2), c(128) {
        fill_random(a);
        fill_random(b);
        fill_random(c);
        expect_a = a;
        memcpy(expect_a.data() + 64, c.data(), c.size());
        expect_b.assign(size_b, 0);
        memcpy(expect_b.data() + 256, b.data(), b.size());
    }
    void transfers(ggml_tensor * dst_a, ggml_tensor * dst_b, ggml_backend_tensor_transfer * out) const {
        out[0] = { dst_a, a.data(), 0, a.size() };
        out[1] = { dst_b, b.data(), 256, b.size() };
        out[2] = { dst_b, nullptr, 0, 0 };
        out[3] = { dst_a, c.data(), 64, c.size() };
    }
    void scribble() {
        memset(a.data(), 0xA5, a.size());
        memset(b.data(), 0xA5, b.size());
        memset(c.data(), 0xA5, c.size());
    }
};

static void progress_random(std::mt19937 & gen) {
    for (auto * s : sim().streams) {
        for (int k = (int) (gen() % 3); k > 0; --k) {
            sim_step(s);
        }
    }
}

static void drain_all() {
    for (auto * s : sim().streams) {
        sim_drain(s);
    }
}

// Disjoint uploads (one destination tensor per entry, so no two entries overlap): the packed copy + scatter path.
// Entry counts cross the per-launch table size, sizes and offsets hit the 16-byte, 4-byte and byte copy paths, and the
// largest upload grows over the rounds so the device region is reallocated while earlier uploads are still queued.
struct disjoint_round {
    std::vector<std::vector<uint8_t>> src;
    std::vector<size_t> offs;
};

struct disjoint_rank {
    ggml_context * gctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<std::vector<ggml_tensor *>> dst;          // [round][entry]
    std::vector<std::vector<std::vector<uint8_t>>> mem;   // [round][entry], the "device" bytes
    std::vector<std::vector<std::vector<uint8_t>>> expect;
};

static size_t pick_size(std::mt19937 & gen, int round) {
    switch (gen() % 4) {
        case 0:  return 16*(1 + gen() % (8 + 16*round));
        case 1:  return 4*(1 + gen() % 300);
        case 2:  return 1 + gen() % 700;
        default: return 1 + gen() % 15;
    }
}

static size_t pick_offset(std::mt19937 & gen) {
    switch (gen() % 3) {
        case 0:  return 0;
        case 1:  return 4*(gen() % 8);
        default: return gen() % 37;
    }
}

static int run_scatter(const std::string & scenario) {
    const bool stage_mode = scenario.find("stage") != std::string::npos;
    const bool lazy = scenario.find("lazy") != std::string::npos;
    const bool fallback = scenario == "scatter-fallback";
    constexpr int n_rounds = 24;
    std::mt19937 gen(scenario.size()*7919 + 3);
    std::vector<disjoint_round> rounds(n_rounds);
    for (int i = 0; i < n_rounds; ++i) {
        const int n = i == 0 ? 1 : i == 1 ? 32 : i == 2 ? 33 : 1 + (int) (gen() % 70);
        for (int e = 0; e < n; ++e) {
            std::vector<uint8_t> v(pick_size(gen, i));
            fill_random(v);
            rounds[i].src.push_back(std::move(v));
            rounds[i].offs.push_back(pick_offset(gen));
        }
    }
    fake_rank base[2];
    disjoint_rank rk[2];
    for (int r = 0; r < 2; ++r) {
        make_rank(base[r], r, 1, 64, 64);
        ggml_backend_buffer_i iface{};
        rk[r].buffer = ggml_backend_buffer_init(ggml_backend_cuda_buffer_type(r), iface, nullptr, 0);
        size_t n_total = 0;
        for (auto & rd : rounds) {
            n_total += rd.src.size();
        }
        ggml_init_params params = { (n_total + 1)*ggml_tensor_overhead(), nullptr, true };
        rk[r].gctx = ggml_init(params);
        rk[r].dst.resize(n_rounds);
        rk[r].mem.resize(n_rounds);
        rk[r].expect.resize(n_rounds);
        for (int i = 0; i < n_rounds; ++i) {
            for (size_t e = 0; e < rounds[i].src.size(); ++e) {
                const size_t size = rounds[i].offs[e] + rounds[i].src[e].size() + gen() % 9;
                ggml_tensor * t = ggml_new_tensor_1d(rk[r].gctx, GGML_TYPE_I8, (int64_t) size);
                rk[r].mem[i].emplace_back(size, 0);
                t->data = rk[r].mem[i].back().data();
                t->buffer = rk[r].buffer;
                rk[r].dst[i].push_back(t);
                std::vector<uint8_t> want(size, 0);
                memcpy(want.data() + rounds[i].offs[e], rounds[i].src[e].data(), rounds[i].src[e].size());
                rk[r].expect[i].push_back(std::move(want));
            }
        }
    }
    if (fallback) {
        sim().malloc_fails = true;
    }
    ggml_backend_t raw[2] = { &base[0].backend, &base[1].backend };
    void * stage = stage_mode ? ggml_backend_cuda_input_stage_init(raw, 2) : nullptr;
    REQUIRE(!stage_mode || stage != nullptr);
    // The link probe of the stage init copies too.
    int expect_memcpy = sim().memcpy_calls, expect_launch = 0;
    for (int i = 0; i < n_rounds; ++i) {
        const size_t n = rounds[i].src.size();
        std::vector<ggml_backend_tensor_transfer> tr[2];
        for (int r = 0; r < 2; ++r) {
            for (size_t e = 0; e < n; ++e) {
                tr[r].push_back({ rk[r].dst[i][e], rounds[i].src[e].data(), rounds[i].offs[e], rounds[i].src[e].size() });
            }
        }
        if (stage_mode) {
            const ggml_backend_tensor_transfer * per_rank[2] = { tr[0].data(), tr[1].data() };
            REQUIRE(ggml_backend_cuda_input_stage_set_batch(stage, raw, 2, per_rank, n), "round %d refused", i);
        } else {
            for (int r = 0; r < 2; ++r) {
                REQUIRE(ggml_backend_cuda_set_tensor_batch_owned(raw[r], tr[r].data(), n), "rank %d round %d refused", r, i);
            }
        }
        expect_memcpy += 2;
        expect_launch += 2*(int) ((n + GGML_CUDA_INPUT_SCATTER_MAX - 1) / GGML_CUDA_INPUT_SCATTER_MAX);
        for (auto & v : rounds[i].src) {
            memset(v.data(), 0xA5, v.size());
        }
        if (!lazy) {
            progress_random(gen);
        }
    }
    drain_all();
    for (int r = 0; r < 2; ++r) {
        for (int i = 0; i < n_rounds; ++i) {
            for (size_t e = 0; e < rk[r].mem[i].size(); ++e) {
                REQUIRE(rk[r].mem[i][e] == rk[r].expect[i][e], "rank %d round %d entry %zu differs (size %zu offset %zu)", r, i, e,
                        rounds[i].src[e].size(), rounds[i].offs[e]);
            }
        }
    }
    if (fallback) {
        REQUIRE(sim().scatter_launches == 0 && sim().batch_calls == 2*n_rounds, "launches %d batches %d", sim().scatter_launches, sim().batch_calls);
    } else {
        // Per rank and call: one host-to-device copy and one launch per GGML_CUDA_INPUT_SCATTER_MAX entries, no batch copy.
        REQUIRE(sim().batch_calls == 0, "batch copies %d", sim().batch_calls);
        REQUIRE(sim().scatter_launches == expect_launch, "launches %d, expected %d", sim().scatter_launches, expect_launch);
        REQUIRE(sim().memcpy_calls == expect_memcpy, "copies %d, expected %d", sim().memcpy_calls, expect_memcpy);
    }
    if (stage) {
        ggml_backend_cuda_input_stage_free(stage);
    }
    printf("sim %s: OK (copies %d, launches %d, batches %d, event syncs %d)\n", scenario.c_str(), sim().memcpy_calls,
           sim().scatter_launches, sim().batch_calls, sim().event_syncs);
    return 0;
}

int main(int argc, char ** argv) {
    REQUIRE(argc == 2);
    const std::string scenario = argv[1];
    if (scenario.rfind("scatter-", 0) == 0) {
        return run_scatter(scenario);
    }
    if (scenario == "order-slow1") {
        sim().link_ms[1] = 2.0;
    } else if (scenario == "order-slow0") {
        sim().link_ms[0] = 2.0;
    } else if (scenario == "order-sym") {
        sim().link_ms[1] = 1.14;
    } else if (scenario == "order-unmeasured") {
        sim().malloc_fails = true;
    }

    constexpr int n_rounds = 24;
    fake_rank rank[2];
    for (int r = 0; r < 2; ++r) {
        make_rank(rank[r], r, n_rounds, 4096, 2048);
    }
    ggml_backend_t raw[2] = { &rank[0].backend, &rank[1].backend };

    if (scenario.rfind("order-", 0) == 0) {
        void * stage = ggml_backend_cuda_input_stage_init(raw, 2);
        REQUIRE(stage != nullptr);
        ggml_backend_cuda_input_stage_free(stage);
        return 0;
    }

    if (scenario == "perrank") {
        std::mt19937 gen(7);
        std::vector<batch> batches;
        for (int i = 0; i < n_rounds; ++i) {
            batches.emplace_back(4096 + i*512, 2048 + i*512);
            for (int r = 0; r < 2; ++r) {
                ggml_backend_tensor_transfer tr[4];
                batches.back().transfers(rank[r].a[i], rank[r].b[i], tr);
                REQUIRE(ggml_backend_cuda_set_tensor_batch_owned(raw[r], tr, 4));
            }
            batches.back().scribble();
            progress_random(gen);
        }
        drain_all();
        for (int r = 0; r < 2; ++r) {
            for (int i = 0; i < n_rounds; ++i) {
                REQUIRE(!memcmp(rank[r].a[i]->data, batches[i].expect_a.data(), batches[i].expect_a.size()), "rank %d round %d A", r, i);
                REQUIRE(!memcmp(rank[r].b[i]->data, batches[i].expect_b.data(), batches[i].expect_b.size()), "rank %d round %d B", r, i);
            }
        }
        printf("sim perrank: OK\n");
        return 0;
    }

    void * stage = ggml_backend_cuda_input_stage_init(raw, 2);
    REQUIRE(stage != nullptr);

    if (scenario == "declines") {
        batch bt(4096, 2048);
        ggml_backend_tensor_transfer tr[2][4];
        const ggml_backend_tensor_transfer * per_rank[2] = { tr[0], tr[1] };
        for (int r = 0; r < 2; ++r) {
            bt.transfers(rank[r].a[0], rank[r].b[0], tr[r]);
        }
        const uint8_t other = 1;
        tr[1][1].data = &other;
        REQUIRE(!ggml_backend_cuda_input_stage_set_batch(stage, raw, 2, per_rank, 4));
        tr[1][1].data = bt.b.data();
        ggml_backend_t swapped[2] = { raw[1], raw[0] };
        REQUIRE(!ggml_backend_cuda_input_stage_set_batch(stage, swapped, 2, per_rank, 4));
        REQUIRE(!ggml_backend_cuda_input_stage_set_batch(stage, raw, 1, per_rank, 4));
        ggml_backend_tensor_transfer none[2][1] = { { { rank[0].a[0], nullptr, 0, 0 } }, { { rank[1].a[0], nullptr, 0, 0 } } };
        const ggml_backend_tensor_transfer * none_rank[2] = { none[0], none[1] };
        REQUIRE(ggml_backend_cuda_input_stage_set_batch(stage, raw, 2, none_rank, 1), "an empty batch is a no-op");
        for (auto * s : sim().streams) {
            REQUIRE(s->enqueued == 0, "a declined or empty batch enqueued work");
        }
        ggml_backend_cuda_input_stage_free(stage);
        printf("sim declines: OK\n");
        return 0;
    }

    // ring-lazy: no stream runs until the host is forced to wait (events), the harshest case for slot reuse.
    // ring-random: streams progress by a random amount after every batch.
    std::mt19937 gen(11);
    std::vector<batch> batches;
    for (int i = 0; i < n_rounds; ++i) {
        batches.emplace_back(4096 + i*512, 2048 + i*512);
        ggml_backend_tensor_transfer tr[2][4];
        const ggml_backend_tensor_transfer * per_rank[2] = { tr[0], tr[1] };
        for (int r = 0; r < 2; ++r) {
            batches.back().transfers(rank[r].a[i], rank[r].b[i], tr[r]);
        }
        REQUIRE(ggml_backend_cuda_input_stage_set_batch(stage, raw, 2, per_rank, 4), "round %d refused", i);
        batches.back().scribble();
        if (scenario == "ring-random") {
            progress_random(gen);
        }
    }
    drain_all();
    for (int r = 0; r < 2; ++r) {
        for (int i = 0; i < n_rounds; ++i) {
            REQUIRE(!memcmp(rank[r].a[i]->data, batches[i].expect_a.data(), batches[i].expect_a.size()), "rank %d round %d A differs", r, i);
            REQUIRE(!memcmp(rank[r].b[i]->data, batches[i].expect_b.data(), batches[i].expect_b.size()), "rank %d round %d B differs", r, i);
        }
    }
    REQUIRE(sim().default_allocs == 0 || scenario == "never", "the staging slot must be portable pinned memory (%d default allocs)", sim().default_allocs);
    REQUIRE(sim().portable_allocs >= 2, "portable allocs %d", sim().portable_allocs);
    ggml_backend_cuda_input_stage_free(stage);
    REQUIRE(sim().scatter_launches == 0, "overlapping writes reached the unordered scatter");
    printf("sim %s: OK (event syncs %d)\n", scenario.c_str(), sim().event_syncs);
    return 0;
}
