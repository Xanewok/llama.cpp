// Runs the input staging code of ggml-cuda.cu (sliced into STAGE_SNIPPET by the Mac gate) against cuda_stub.h.
// usage: stage_sim SCENARIO   order-slow1 | order-sym | order-slow0 | order-unmeasured | ring-lazy | ring-random | perrank | declines
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

int main(int argc, char ** argv) {
    REQUIRE(argc == 2);
    const std::string scenario = argv[1];
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
    printf("sim %s: OK (event syncs %d)\n", scenario.c_str(), sim().event_syncs);
    return 0;
}
