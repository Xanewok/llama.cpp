// Shared input staging of the meta backend. Part 1 (CPU): the meta glue on two fake ranks. Part 2: the CUDA group upload
// on CUDA0 and CUDA1. Part 3: disjoint uploads (the packed copy + device scatter path), per rank and shared. Parts 2 and 3
// are skipped without two CUDA devices unless INPUT_STAGE_REQUIRE_CUDA=1.
#include "ggml-alloc.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d %s: ", __FILE__, __LINE__, #cond);  \
            fprintf(stderr, "" __VA_ARGS__);                                \
            fprintf(stderr, "\n");                                          \
            return false;                                                   \
        }                                                                   \
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

// One batch: a full write of A, a partial write of B at an offset, an empty write, and a later write overlapping the
// first (transfers complete in list order). The host sources are overwritten once the call returns.
struct batch {
    std::vector<uint8_t> a, b, c;
    size_t b_offset = 256;
    size_t c_offset = 64;
    std::vector<uint8_t> expect_a, expect_b;

    batch(size_t size_a, size_t size_b) {
        a.resize(size_a);
        b.resize(size_b / 2);
        c.resize(128);
        fill_random(a);
        fill_random(b);
        fill_random(c);
        expect_a = a;
        memcpy(expect_a.data() + c_offset, c.data(), c.size());
        expect_b.assign(size_b, 0);
        memcpy(expect_b.data() + b_offset, b.data(), b.size());
    }
    void transfers(ggml_tensor * dst_a, ggml_tensor * dst_b, ggml_backend_tensor_transfer * out) const {
        out[0] = { dst_a, a.data(), 0, a.size() };
        out[1] = { dst_b, b.data(), b_offset, b.size() };
        out[2] = { dst_b, nullptr, 0, 0 };
        out[3] = { dst_a, c.data(), c_offset, c.size() };
    }
    void scribble() {
        memset(a.data(), 0xA5, a.size());
        memset(b.data(), 0xA5, b.size());
        memset(c.data(), 0xA5, c.size());
    }
};

static bool tensor_equals(const ggml_tensor * t, const std::vector<uint8_t> & expect) {
    std::vector<uint8_t> got(expect.size());
    ggml_backend_tensor_get(t, got.data(), 0, got.size());
    return got == expect;
}

//
// Part 1: meta glue on two fake ranks backed by the CPU
//

enum stage_mode { STAGE_ABSENT, STAGE_INIT_NULL, STAGE_DECLINE, STAGE_GOOD, STAGE_MUTANT };

static stage_mode g_mode;
static int g_init, g_free, g_set, g_owned_real;
static ggml_backend_device g_dev[2];
static ggml_backend_reg g_reg;
static ggml_tensor * g_seen[2][4];
static int g_rank_order[2];
static char g_stage_token;

static int rank_of(ggml_backend_t backend) {
    return backend->device == &g_dev[0] ? 0 : 1;
}

static void host_write(const ggml_backend_tensor_transfer & t) {
    if (t.size) {
        ggml_backend_tensor_set(t.tensor, t.data, t.offset, t.size);
    }
}

static bool fake_owned(ggml_backend_t backend, const ggml_backend_tensor_transfer * transfers, size_t n) {
    if (n == 0) {
        return true;
    }
    ++g_owned_real;
    for (size_t i = 0; i < n; ++i) {
        g_seen[rank_of(backend)][i] = transfers[i].tensor;
        host_write(transfers[i]);
    }
    return true;
}

static void * fake_stage_init(ggml_backend_t *, size_t n_backends) {
    ++g_init;
    return g_mode == STAGE_INIT_NULL || n_backends != 2 ? nullptr : &g_stage_token;
}

static void fake_stage_free(void * stage) {
    if (stage != &g_stage_token) {
        abort();
    }
    ++g_free;
}

static bool fake_stage_set(void * stage, ggml_backend_t * backends, size_t n_backends,
        const ggml_backend_tensor_transfer * const * transfers, size_t n_transfers) {
    if (stage != &g_stage_token || n_backends != 2 || n_transfers != 4) {
        abort();
    }
    ++g_set;
    if (g_mode == STAGE_DECLINE) {
        return false;
    }
    for (size_t k = 0; k < 2; ++k) {
        const size_t r = 1 - k; // the slow link first
        g_rank_order[k] = (int) r;
        if (rank_of(backends[r]) != (int) r) {
            abort();
        }
        for (size_t i = 0; i < n_transfers; ++i) {
            if (transfers[r][i].data != transfers[0][i].data || transfers[r][i].size != transfers[0][i].size) {
                abort();
            }
            g_seen[r][i] = transfers[r][i].tensor;
            // Negative control: the mutant drops the last write of rank 0.
            if (g_mode == STAGE_MUTANT && r == 0 && i == 3) {
                continue;
            }
            host_write(transfers[r][i]);
        }
    }
    return true;
}

static void * fake_reg_proc(ggml_backend_reg_t, const char * name) {
    if (g_mode == STAGE_ABSENT) {
        return nullptr;
    }
    if (!strcmp(name, "ggml_backend_input_stage_init")) {
        return (void *) fake_stage_init;
    }
    if (!strcmp(name, "ggml_backend_input_stage_free")) {
        return (void *) fake_stage_free;
    }
    if (!strcmp(name, "ggml_backend_input_stage_set_batch")) {
        return (void *) fake_stage_set;
    }
    return nullptr;
}

static ggml_backend_t fake_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t b = cpu->iface.init_backend(cpu, params);
    b->device = dev;
    b->iface.set_tensor_batch_owned = fake_owned;
    return b;
}

static ggml_backend_meta_split_state split_state(const ggml_tensor * t, void *) {
    if (!strcmp(t->name, "split")) {
        return { GGML_BACKEND_SPLIT_AXIS_0, {t->ne[0]/2, t->ne[0]/2}, {1}, 1 };
    }
    return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1 };
}

static bool run_fake(const char * name, stage_mode mode, int rounds) {
    g_mode = mode;
    g_init = g_free = g_set = g_owned_real = 0;
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    CHECK(cpu != nullptr);
    for (int r = 0; r < 2; ++r) {
        g_dev[r] = *cpu;
        g_dev[r].reg = &g_reg;
        g_dev[r].iface.init_backend = fake_init_backend;
    }
    ggml_backend_dev_t devs[2] = { &g_dev[0], &g_dev[1] };
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2, split_state, nullptr);
    const bool staged = mode == STAGE_GOOD || mode == STAGE_MUTANT || mode == STAGE_DECLINE;
    {
        ggml_backend_ptr backend(ggml_backend_dev_init(meta, nullptr));
        CHECK(backend);
        ggml_init_params params = { 8*ggml_tensor_overhead(), nullptr, true };
        ggml_context_ptr ctx(ggml_init(params));
        ggml_tensor * a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, 4096);
        ggml_tensor * b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, 2048);
        ggml_tensor * s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, 64);
        ggml_set_name(a, "a");
        ggml_set_name(b, "b");
        ggml_set_name(s, "split");
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        CHECK(buffer);

        // A capability probe is not an upload and must not bring up the staging.
        CHECK(backend->iface.set_tensor_batch_owned(backend.get(), nullptr, 0));
        CHECK(g_init == 0 && g_set == 0, "%s: probe touched the staging", name);

        for (int round = 0; round < rounds; ++round) {
            batch bt(4096, 2048);
            ggml_backend_buffer_clear(buffer.get(), 0);
            ggml_backend_tensor_transfer tr[4];
            bt.transfers(a, b, tr);
            memset(g_seen, 0, sizeof(g_seen));
            const int owned0 = g_owned_real;
            CHECK(backend->iface.set_tensor_batch_owned(backend.get(), tr, 4), "%s: batch refused", name);
            bt.scribble();
            if (staged && mode != STAGE_DECLINE) {
                CHECK(g_owned_real == owned0, "%s: per-rank uploads ran beside the staging", name);
                CHECK(g_rank_order[0] == 1 && g_rank_order[1] == 0, "%s: rank order", name);
            } else {
                CHECK(g_owned_real == owned0 + 2, "%s: per-rank fallback not taken", name);
            }
            CHECK(g_seen[0][0] && g_seen[1][0] && g_seen[0][0] != g_seen[1][0] && g_seen[0][1] != g_seen[1][1],
                    "%s: each rank gets its own tensors", name);
            bool match = true;
            for (int r = 0; r < 2; ++r) {
                match &= tensor_equals(g_seen[r][0], bt.expect_a) && tensor_equals(g_seen[r][1], bt.expect_b);
            }
            CHECK(match == (mode != STAGE_MUTANT), "%s: ranks %s", name, match ? "match (the mutant must differ)" : "differ");
        }

        // Split inputs are not mirrored: declined before any staging or per-rank upload.
        const int set0 = g_set, owned0 = g_owned_real;
        std::vector<uint8_t> data(64, 7);
        ggml_backend_tensor_transfer split[1] = { { s, data.data(), 0, data.size() } };
        CHECK(!backend->iface.set_tensor_batch_owned(backend.get(), split, 1) && g_set == set0 && g_owned_real == owned0,
                "%s: split input", name);

        const int want_init = mode == STAGE_ABSENT || rounds == 0 ? 0 : 1;
        CHECK(g_init == want_init, "%s: staging probed %d times", name, g_init);
        CHECK(g_set == (staged ? rounds : 0), "%s: staged batches %d", name, g_set);
        CHECK(g_free == 0, "%s: staging freed with the backend alive", name);
    }
    CHECK(g_free == (mode == STAGE_ABSENT || mode == STAGE_INIT_NULL ? 0 : 1), "%s: staging freed %d times", name, g_free);
    printf("%s: OK (init %d, staged %d, per-rank %d, free %d)\n", name, g_init, g_set, g_owned_real, g_free);
    return true;
}

//
// Part 2: the CUDA group upload
//

static bool run_cuda(bool require) {
    ggml_backend_dev_t d[2] = { ggml_backend_dev_by_name("CUDA0"), ggml_backend_dev_by_name("CUDA1") };
    if (!d[0] || !d[1]) {
        printf("cuda input stage: %s, fewer than two CUDA devices\n", require ? "FAIL" : "SKIP");
        return !require;
    }
    ggml_backend_ptr be[2] = { ggml_backend_ptr(ggml_backend_dev_init(d[0], nullptr)), ggml_backend_ptr(ggml_backend_dev_init(d[1], nullptr)) };
    CHECK(be[0] && be[1]);
    ggml_backend_t raw[2] = { be[0].get(), be[1].get() };
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(d[0]);
    auto init = (ggml_backend_input_stage_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_input_stage_init");
    auto stage_free = (ggml_backend_input_stage_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_input_stage_free");
    auto set = (ggml_backend_input_stage_set_batch_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_input_stage_set_batch");
    CHECK(init && stage_free && set, "CUDA reg does not export the input staging");

    ggml_backend_ptr cpu(ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr));
    ggml_backend_t mixed[2] = { raw[0], cpu.get() };
    CHECK(init(mixed, 2) == nullptr, "a non-CUDA rank must refuse the group");
    CHECK(init(raw, 1) == nullptr, "one rank has nothing to share");
    void * stage = init(raw, 2);
    CHECK(stage != nullptr);

    constexpr int n_rounds = 48;
    constexpr size_t grow = 65536;
    // Per rank: one destination pair per round (written without any synchronization in between), plus a spare pair.
    ggml_context_ptr ctx[2];
    ggml_backend_buffer_ptr buffers[2];
    std::vector<ggml_tensor *> dst_a[2], dst_b[2];
    ggml_tensor * spare_a[2], * spare_b[2];
    for (int r = 0; r < 2; ++r) {
        ggml_init_params params = { (2*n_rounds + 2)*ggml_tensor_overhead(), nullptr, true };
        ctx[r].reset(ggml_init(params));
        for (int i = 0; i < n_rounds; ++i) {
            dst_a[r].push_back(ggml_new_tensor_1d(ctx[r].get(), GGML_TYPE_I8, 4096 + i*grow));
            dst_b[r].push_back(ggml_new_tensor_1d(ctx[r].get(), GGML_TYPE_I8, 2048 + i*1024));
        }
        spare_a[r] = ggml_new_tensor_1d(ctx[r].get(), GGML_TYPE_I8, 4096);
        spare_b[r] = ggml_new_tensor_1d(ctx[r].get(), GGML_TYPE_I8, 2048);
        buffers[r].reset(ggml_backend_alloc_ctx_tensors(ctx[r].get(), raw[r]));
        CHECK(buffers[r]);
        ggml_backend_buffer_clear(buffers[r].get(), 0);
    }

    std::vector<batch> batches;
    for (int i = 0; i < n_rounds; ++i) {
        batches.emplace_back(4096 + i*grow, 2048 + i*1024);
        batch & bt = batches.back();
        ggml_backend_tensor_transfer tr[2][4];
        const ggml_backend_tensor_transfer * per_rank[2] = { tr[0], tr[1] };
        for (int r = 0; r < 2; ++r) {
            bt.transfers(dst_a[r][i], dst_b[r][i], tr[r]);
        }
        CHECK(set(stage, raw, 2, per_rank, 4), "round %d refused", i);
        // The host bytes are consumed before the call returns.
        bt.scribble();
    }

    // Declines leave the destinations untouched.
    {
        batch bt(4096, 2048);
        ggml_backend_tensor_transfer tr[2][4];
        const ggml_backend_tensor_transfer * per_rank[2] = { tr[0], tr[1] };
        for (int r = 0; r < 2; ++r) {
            bt.transfers(spare_a[r], spare_b[r], tr[r]);
        }
        const uint8_t other = 0x11;
        tr[1][1].data = &other;
        CHECK(!set(stage, raw, 2, per_rank, 4), "ranks with different sources must decline");
        tr[1][1].data = bt.b.data();
        ggml_backend_t swapped[2] = { raw[1], raw[0] };
        CHECK(!set(stage, swapped, 2, per_rank, 4), "backends in another order must decline");
        CHECK(!set(stage, raw, 1, per_rank, 4), "a different rank count must decline");
        ggml_backend_synchronize(raw[0]);
        ggml_backend_synchronize(raw[1]);
        const std::vector<uint8_t> zero_a(4096, 0), zero_b(2048, 0);
        for (int r = 0; r < 2; ++r) {
            CHECK(tensor_equals(spare_a[r], zero_a) && tensor_equals(spare_b[r], zero_b), "a declined batch wrote");
        }
    }

    ggml_backend_synchronize(raw[0]);
    ggml_backend_synchronize(raw[1]);
    for (int r = 0; r < 2; ++r) {
        for (int i = 0; i < n_rounds; ++i) {
            CHECK(tensor_equals(dst_a[r][i], batches[i].expect_a), "CUDA%d round %d tensor A differs", r, i);
            CHECK(tensor_equals(dst_b[r][i], batches[i].expect_b), "CUDA%d round %d tensor B differs", r, i);
        }
    }
    stage_free(stage);
    printf("cuda input stage: OK (%d rounds, both ranks byte-identical, declines clean)\n", n_rounds);
    return true;
}

//
// Part 3: disjoint uploads on CUDA0 and CUDA1
//

// One destination tensor per entry, so no two writes overlap. Entry counts cross the scatter's per-launch table, sizes
// and offsets hit its 16-byte, 4-byte and byte paths, and the largest upload grows so the device region is reallocated.
static bool run_cuda_scatter(bool require, bool shared) {
    const char * name = shared ? "cuda scatter shared" : "cuda scatter per rank";
    ggml_backend_dev_t d[2] = { ggml_backend_dev_by_name("CUDA0"), ggml_backend_dev_by_name("CUDA1") };
    if (!d[0] || !d[1]) {
        printf("%s: %s, fewer than two CUDA devices\n", name, require ? "FAIL" : "SKIP");
        return !require;
    }
    ggml_backend_ptr be[2] = { ggml_backend_ptr(ggml_backend_dev_init(d[0], nullptr)), ggml_backend_ptr(ggml_backend_dev_init(d[1], nullptr)) };
    CHECK(be[0] && be[1]);
    ggml_backend_t raw[2] = { be[0].get(), be[1].get() };
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(d[0]);
    auto init = (ggml_backend_input_stage_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_input_stage_init");
    auto stage_free = (ggml_backend_input_stage_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_input_stage_free");
    auto set = (ggml_backend_input_stage_set_batch_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_input_stage_set_batch");
    CHECK(init && stage_free && set);
    CHECK(raw[0]->iface.set_tensor_batch_owned && raw[1]->iface.set_tensor_batch_owned);

    constexpr int n_rounds = 48;
    std::mt19937 gen(shared ? 29 : 17);
    std::vector<std::vector<std::vector<uint8_t>>> src(n_rounds);
    std::vector<std::vector<size_t>> offs(n_rounds), sizes(n_rounds);
    size_t n_total = 0;
    for (int i = 0; i < n_rounds; ++i) {
        const int n = i == 0 ? 1 : i == 1 ? 32 : i == 2 ? 33 : 1 + (int) (gen() % 70);
        for (int e = 0; e < n; ++e) {
            size_t size;
            switch (gen() % 4) {
                case 0:  size = 16*(1 + gen() % (8 + 64*i)); break;
                case 1:  size = 4*(1 + gen() % 300);         break;
                case 2:  size = 1 + gen() % 700;             break;
                default: size = 1 + gen() % 15;              break;
            }
            const size_t off = gen() % 3 == 0 ? 0 : gen() % 2 ? 4*(gen() % 8) : gen() % 37;
            std::vector<uint8_t> v(size);
            fill_random(v);
            src[i].push_back(std::move(v));
            offs[i].push_back(off);
            sizes[i].push_back(off + size + gen() % 9);
        }
        n_total += n;
    }
    ggml_context_ptr ctx[2];
    ggml_backend_buffer_ptr buffers[2];
    std::vector<std::vector<ggml_tensor *>> dst[2];
    for (int r = 0; r < 2; ++r) {
        ggml_init_params params = { (n_total + 1)*ggml_tensor_overhead(), nullptr, true };
        ctx[r].reset(ggml_init(params));
        dst[r].resize(n_rounds);
        for (int i = 0; i < n_rounds; ++i) {
            for (size_t e = 0; e < src[i].size(); ++e) {
                dst[r][i].push_back(ggml_new_tensor_1d(ctx[r].get(), GGML_TYPE_I8, (int64_t) sizes[i][e]));
            }
        }
        buffers[r].reset(ggml_backend_alloc_ctx_tensors(ctx[r].get(), raw[r]));
        CHECK(buffers[r]);
        ggml_backend_buffer_clear(buffers[r].get(), 0);
    }
    void * stage = shared ? init(raw, 2) : nullptr;
    CHECK(!shared || stage);

    std::vector<std::vector<std::vector<uint8_t>>> expect(n_rounds);
    for (int i = 0; i < n_rounds; ++i) {
        const size_t n = src[i].size();
        std::vector<ggml_backend_tensor_transfer> tr[2];
        for (size_t e = 0; e < n; ++e) {
            std::vector<uint8_t> want(sizes[i][e], 0);
            memcpy(want.data() + offs[i][e], src[i][e].data(), src[i][e].size());
            expect[i].push_back(std::move(want));
            for (int r = 0; r < 2; ++r) {
                tr[r].push_back({ dst[r][i][e], src[i][e].data(), offs[i][e], src[i][e].size() });
            }
        }
        if (shared) {
            const ggml_backend_tensor_transfer * per_rank[2] = { tr[0].data(), tr[1].data() };
            CHECK(set(stage, raw, 2, per_rank, n), "round %d refused", i);
        } else {
            for (int r = 0; r < 2; ++r) {
                CHECK(raw[r]->iface.set_tensor_batch_owned(raw[r], tr[r].data(), n), "CUDA%d round %d refused", r, i);
            }
        }
        // The host bytes are consumed before the call returns.
        for (auto & v : src[i]) {
            memset(v.data(), 0xA5, v.size());
        }
    }
    ggml_backend_synchronize(raw[0]);
    ggml_backend_synchronize(raw[1]);
    for (int r = 0; r < 2; ++r) {
        for (int i = 0; i < n_rounds; ++i) {
            for (size_t e = 0; e < expect[i].size(); ++e) {
                CHECK(tensor_equals(dst[r][i][e], expect[i][e]), "%s: CUDA%d round %d entry %zu differs", name, r, i, e);
            }
        }
    }
    if (stage) {
        stage_free(stage);
    }
    printf("%s: OK (%d rounds, %zu disjoint writes per rank, byte-identical)\n", name, n_rounds, n_total);
    return true;
}

int main() {
    ggml_backend_load_all();
    g_reg.iface.get_proc_address = fake_reg_proc;
    bool ok = true;
    ok = ok && run_fake("fake/no-staging-exports", STAGE_ABSENT, 3);
    ok = ok && run_fake("fake/init-refused", STAGE_INIT_NULL, 3);
    ok = ok && run_fake("fake/batch-declined", STAGE_DECLINE, 3);
    ok = ok && run_fake("fake/staged", STAGE_GOOD, 3);
    // Negative control: a staging that drops a write must be caught by the content check.
    ok = ok && run_fake("fake/mutant-drops-a-write", STAGE_MUTANT, 1);
    const char * require = getenv("INPUT_STAGE_REQUIRE_CUDA");
    ok = ok && run_cuda(require && atoi(require) != 0);
    ok = ok && run_cuda_scatter(require && atoi(require) != 0, false);
    ok = ok && run_cuda_scatter(require && atoi(require) != 0, true);
    printf("test-meta-input-stage: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
