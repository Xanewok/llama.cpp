// Host-side bookkeeping of the shared flash-attention KV-max scan (ggml/src/ggml-cuda/fattn-kv-max-cache.h).
// Tensors carry fake data pointers: nothing is allocated or dereferenced.

#include "ggml.h"
#include "ggml-cpp.h"

#include "fattn-kv-max-cache.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); ++n_fail; } } while (0)

struct fixture {
    ggml_context_ptr ctx;
    fixture() {
        ggml_init_params params = { 256*ggml_tensor_overhead(), nullptr, true };
        ctx.reset(ggml_init(params));
    }

    ggml_tensor * mask(uintptr_t addr, int64_t kv, int64_t rows, int64_t ns) {
        ggml_tensor * m = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, kv, rows, 1, ns);
        m->data = (void *) addr;
        return m;
    }

    ggml_tensor * fa(ggml_tensor * m, int64_t kv, int64_t nq, int64_t ns) {
        ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 128, nq, 4, ns);
        ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, 128, kv, 4, ns);
        ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, 128, kv, 4, ns);
        return ggml_flash_attn_ext(ctx.get(), q, k, v, m, 1.0f, 0.0f, 0.0f);
    }

    ggml_tensor * write(uintptr_t addr, size_t n) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, (int64_t) n);
        t->data = (void *) addr;
        return t;
    }
};

static constexpr uintptr_t base = 0x100000;

static void test_sharing() {
    fixture f;
    ggml_tensor * m = f.mask(base, 512, 64, 2);
    ggml_cuda_fattn_kv_max_cache c;
    for (int i = 0; i < 5; ++i) {
        c.add(f.fa(m, 512, 4, 2));
    }
    const size_t n = c.reserve();
    CHECK(n >= 8 && n % 64 == 0);
    CHECK(c.groups.size() == 1);

    std::vector<int> buf(n);
    c.data = buf.data();
    const ggml_tensor * q = f.fa(m, 512, 4, 2)->src[0];
    const ggml_tensor * k = f.fa(m, 512, 4, 2)->src[1];

    bool scan = false;
    int * p0 = c.acquire(q, k, m, 8, scan);
    CHECK(p0 == buf.data() && scan);
    for (int i = 1; i < 5; ++i) {
        int * p = c.acquire(q, k, m, 8, scan);
        CHECK(p == p0 && !scan);
    }
    // the slot belongs to the first ncols1; a different tile height scans locally
    CHECK(c.acquire(q, k, m, 16, scan) == nullptr);
    CHECK(c.acquire(q, k, m, 8, scan) == p0 && !scan);
}

static void test_not_wanted() {
    fixture f;
    ggml_cuda_fattn_kv_max_cache c;
    // one sequence, short batch: the dense scan is not launched
    ggml_tensor * m1 = f.mask(base, 512, 64, 1);
    for (int i = 0; i < 3; ++i) {
        c.add(f.fa(m1, 512, 4, 1));
    }
    // kv not a multiple of the stride
    ggml_tensor * m2 = f.mask(base + 0x10000, 300, 64, 2);
    for (int i = 0; i < 3; ++i) {
        c.add(f.fa(m2, 300, 4, 2));
    }
    // no mask
    for (int i = 0; i < 3; ++i) {
        c.add(f.fa(nullptr, 512, 4, 2));
    }
    CHECK(c.groups.empty());
    CHECK(c.reserve() == 0);
    // long single-sequence batches do scan
    ggml_tensor * m3 = f.mask(base + 0x20000, 512, 1024, 1);
    c.add(f.fa(m3, 512, 1024, 1));
    c.add(f.fa(m3, 512, 1024, 1));
    CHECK(c.groups.size() == 1);
}

static void test_single_user_dropped() {
    fixture f;
    ggml_cuda_fattn_kv_max_cache c;
    c.add(f.fa(f.mask(base, 512, 64, 2), 512, 4, 2));
    CHECK(c.reserve() == 0);
    CHECK(c.groups.empty());
    int dummy[64];
    c.data = dummy;
    bool scan = true;
    ggml_tensor * fa = f.fa(f.mask(base, 512, 64, 2), 512, 4, 2);
    CHECK(c.acquire(fa->src[0], fa->src[1], fa->src[3], 8, scan) == nullptr);
}

static void test_keys() {
    fixture f;
    ggml_cuda_fattn_kv_max_cache c;
    ggml_tensor * ma = f.mask(base, 512, 64, 2);
    ggml_tensor * mb = f.mask(base + 0x10000, 512, 64, 2);
    ggml_tensor * mc = f.mask(base, 1024, 64, 2);              // same bytes, other row stride
    ggml_tensor * md = f.mask(base + 0x30000, 512, 1024, 1);   // one mask broadcast over any number of sequences
    for (int i = 0; i < 2; ++i) {
        c.add(f.fa(ma, 512, 4, 2));
        c.add(f.fa(mb, 512, 4, 2));
        c.add(f.fa(ma, 256, 4, 2));    // other kv extent over the same mask
        c.add(f.fa(ma, 512, 8, 2));    // other batch
        c.add(f.fa(mc, 512, 4, 2));
        c.add(f.fa(md, 512, 1024, 2));
        c.add(f.fa(md, 512, 1024, 1)); // other sequence count
    }
    CHECK(c.groups.size() == 7);
    const size_t n = c.reserve();
    std::vector<int> buf(n);
    c.data = buf.data();
    size_t end = 0;
    for (const auto & g : c.groups) {
        CHECK(g.offset >= end);
        CHECK(g.size >= size_t(g.nq*g.ns));
        end = g.offset + g.size;
    }
    CHECK(end == n);

    bool scan;
    ggml_tensor * fa1 = f.fa(ma, 512, 4, 2);
    ggml_tensor * fa2 = f.fa(ma, 256, 4, 2);
    ggml_tensor * fa3 = f.fa(mb, 512, 4, 2);
    int * p1 = c.acquire(fa1->src[0], fa1->src[1], ma, 8, scan);
    CHECK(p1 && scan);
    int * p2 = c.acquire(fa2->src[0], fa2->src[1], ma, 8, scan);
    CHECK(p2 && p2 != p1 && scan);
    int * p3 = c.acquire(fa3->src[0], fa3->src[1], mb, 8, scan);
    CHECK(p3 && p3 != p1 && p3 != p2 && scan);
    CHECK(c.acquire(fa1->src[0], fa1->src[1], ma, 8, scan) == p1 && !scan);
}

static void test_invalidate() {
    fixture f;
    const size_t mask_bytes = 512*64*2*sizeof(ggml_fp16_t);
    ggml_tensor * m = f.mask(base, 512, 64, 2);
    CHECK(ggml_nbytes(m) == mask_bytes);
    ggml_cuda_fattn_kv_max_cache c;
    for (int i = 0; i < 2; ++i) {
        c.add(f.fa(m, 512, 4, 2));
    }
    std::vector<int> buf(c.reserve());
    c.data = buf.data();
    ggml_tensor * fa = f.fa(m, 512, 4, 2);

    auto valid = [&]() {
        bool scan = true;
        int * p = c.acquire(fa->src[0], fa->src[1], m, 8, scan);
        CHECK(p == buf.data());
        return !scan;
    };

    bool scan;
    c.acquire(fa->src[0], fa->src[1], m, 8, scan);
    CHECK(valid());

    // disjoint writes keep the scan: before, adjacent before, adjacent after, far after
    c.invalidate(f.write(base - 4096, 4096));
    c.invalidate(f.write(base + mask_bytes, 4096));
    c.invalidate(f.write(base + mask_bytes + 100000, 16));
    CHECK(valid());

    // any byte of overlap drops it, and the next user rescans into the same slot
    const struct { uintptr_t addr; size_t n; } hits[] = {
        { base, 1 }, { base - 4095, 4096 }, { base + mask_bytes - 1, 8 }, { base + 7, 2 },
        { base - 100, mask_bytes + 200 }, { base + 1024, 4096 },
    };
    for (const auto & h : hits) {
        c.invalidate(f.write(h.addr, h.n));
        CHECK(!valid());
        CHECK(valid());
    }
}

static void test_limits() {
    fixture f;
    ggml_cuda_fattn_kv_max_cache c;
    for (int g = 0; g < 12; ++g) {
        ggml_tensor * m = f.mask(base + 0x10000*g, 512, 64, 2);
        c.add(f.fa(m, 512, 4, 2));
        c.add(f.fa(m, 512, 4, 2));
    }
    CHECK(c.groups.size() == ggml_cuda_fattn_kv_max_cache::max_groups);

    ggml_cuda_fattn_kv_max_cache big;
    ggml_tensor * mb = f.mask(base, 512, 2048, 600);
    big.add(f.fa(mb, 512, 2048, 600));
    big.add(f.fa(mb, 512, 2048, 600));
    CHECK(big.reserve() == 0);

    ggml_cuda_fattn_kv_max_cache none;
    bool scan = true;
    ggml_tensor * fa = f.fa(f.mask(base, 512, 64, 2), 512, 4, 2);
    CHECK(none.acquire(fa->src[0], fa->src[1], fa->src[3], 8, scan) == nullptr && !scan);
}

int main() {
    test_sharing();
    test_not_wanted();
    test_single_user_dropped();
    test_keys();
    test_invalidate();
    test_limits();
    if (n_fail) {
        printf("test-fattn-kv-max-cache: %d FAILED\n", n_fail);
        return 1;
    }
    printf("test-fattn-kv-max-cache: PASS\n");
    return 0;
}
