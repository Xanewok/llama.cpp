// Runs the CUDA batched top-k kernels (ggml/src/ggml-cuda/top-k-batched.cuh) on the CPU: each block is executed by
// TOPK_BLOCK real threads with a barrier for __syncthreads and real atomics, so the kernel source itself is tested
// against a brute-force reference of its contract (key descending, column ascending, exact tie resolution).
// Blocks run one after another; inter-block ordering is the launch order, as on one stream.

#include <atomic>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

struct emu_dim { unsigned x; };
thread_local emu_dim threadIdx = { 0 };
thread_local emu_dim blockIdx  = { 0 };

class emu_barrier {
    std::mutex              m;
    std::condition_variable cv;
    const int               n;
    int                     count = 0;
    unsigned                gen   = 0;
public:
    explicit emu_barrier(int n) : n(n) {}
    void wait() {
        std::unique_lock<std::mutex> l(m);
        const unsigned g = gen;
        if (++count == n) {
            count = 0;
            ++gen;
            cv.notify_all();
        } else {
            cv.wait(l, [&] { return g != gen; });
        }
    }
};

constexpr int EMU_THREADS = 256;

struct emu_pool {
    emu_barrier           start { EMU_THREADS + 1 };
    emu_barrier           end   { EMU_THREADS + 1 };
    emu_barrier           block { EMU_THREADS };
    std::function<void()> job;
    unsigned              cur  = 0;
    bool                  quit = false;
    std::vector<std::thread> th;

    emu_pool() {
        for (int t = 0; t < EMU_THREADS; ++t) {
            th.emplace_back([this, t] {
                for (;;) {
                    start.wait();
                    if (quit) {
                        return;
                    }
                    threadIdx.x = t;
                    blockIdx.x  = cur;
                    job();
                    end.wait();
                }
            });
        }
    }
    ~emu_pool() {
        quit = true;
        start.wait();
        for (auto & t : th) {
            t.join();
        }
    }
    void launch(int grid, int block_dim, const std::function<void()> & fn) {
        if (block_dim != EMU_THREADS) {
            fprintf(stderr, "emulator runs %d-thread blocks only\n", EMU_THREADS);
            abort();
        }
        job = fn;
        for (int b = 0; b < grid; ++b) {
            cur = b;
            start.wait();
            end.wait();
        }
    }
};

static emu_pool * g_pool;

#define GGML_TOPK_EMULATE
#define GGML_TOPK_SCRATCH_BYTES (3 * 9 * 256 * 4)  // three rows of 9000 columns per chunk: exercises chunking
#define __global__
#define __device__
#define __forceinline__ inline
#define GGML_TOPK_SHARED static
#define __syncthreads() g_pool->block.wait()
#define GGML_TOPK_LAUNCH(kernel, grid, block, stream, ...) g_pool->launch((grid), (block), [&] { kernel(__VA_ARGS__); })

typedef int topk_stream_t;

static inline uint32_t __float_as_uint(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}
static inline int atomicAdd(int * p, int v) {
    return __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
}

#ifdef TOPK_HEADER
#    include TOPK_HEADER
#else
#    include "../ggml/src/ggml-cuda/top-k-batched.cuh"
#endif

static uint32_t ref_key(float v) {
    uint32_t b;
    memcpy(&b, &v, sizeof(b));
    return b ^ ((b >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}

static std::vector<int> ref_topk(const float * row, int n, int k) {
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) {
        idx[i] = i;
    }
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        const uint32_t ka = ref_key(row[a]);
        const uint32_t kb = ref_key(row[b]);
        return ka > kb || (ka == kb && a < b);
    });
    idx.resize(k);
    return idx;
}

static int g_fail = 0;
static int g_cases = 0;

static void check(const char * name, const std::vector<float> & x, int nrows, int ncols, int k) {
    ++g_cases;
    size_t n_states, n_hist;
    topk_batched_scratch(ncols, nrows, n_states, n_hist);
    std::vector<topk_row_state> states(n_states + 1);
    std::vector<int>            hist(n_hist + 1);
    // poison: every output slot must be written
    std::vector<int> dst((size_t) nrows * k, -7);
    topk_batched(x.data(), dst.data(), states.data(), hist.data(), ncols, nrows, k, 0);
    for (int r = 0; r < nrows; ++r) {
        const std::vector<int> ref = ref_topk(x.data() + (size_t) r * ncols, ncols, k);
        for (int i = 0; i < k; ++i) {
            if (dst[(size_t) r * k + i] != ref[i]) {
                printf("FAIL %s ncols=%d nrows=%d k=%d row=%d slot=%d got=%d want=%d\n", name, ncols, nrows, k, r, i,
                       dst[(size_t) r * k + i], ref[i]);
                ++g_fail;
                if (getenv("TOPK_FAILFAST")) {
                    exit(1);  // local: negative controls stop at the first mismatch
                }
                return;
            }
        }
    }
}

static std::mt19937 rng(12345);

static std::vector<float> rows_normal(int nrows, int ncols) {
    std::normal_distribution<float> d(0.0f, 4.0f);
    std::vector<float> x((size_t) nrows * ncols);
    for (auto & v : x) {
        v = d(rng);
    }
    return x;
}

static std::vector<float> rows_ties(int nrows, int ncols, int levels) {
    std::uniform_int_distribution<int> d(0, levels - 1);
    std::vector<float> x((size_t) nrows * ncols);
    for (auto & v : x) {
        v = (float) d(rng);
    }
    return x;
}

static std::vector<float> rows_const(int nrows, int ncols, float c) {
    return std::vector<float>((size_t) nrows * ncols, c);
}

// only `n_finite` random finite entries per row; the rest is the banned value
static std::vector<float> rows_banned(int nrows, int ncols, int n_finite, float ban) {
    std::vector<float> x((size_t) nrows * ncols, ban);
    std::normal_distribution<float> d(0.0f, 4.0f);
    for (int r = 0; r < nrows; ++r) {
        for (int i = 0; i < n_finite; ++i) {
            x[(size_t) r * ncols + std::uniform_int_distribution<int>(0, ncols - 1)(rng)] = d(rng);
        }
    }
    return x;
}

// the k-th key ties across more entries than fit, with k-1 strictly larger ones: ties start at the boundary
static std::vector<float> rows_boundary(int nrows, int ncols, int k, int n_tied) {
    std::vector<float> x((size_t) nrows * ncols);
    for (int r = 0; r < nrows; ++r) {
        std::vector<float> row(ncols);
        for (int i = 0; i < ncols; ++i) {
            row[i] = -1000.0f - (float) (i % 17);
        }
        for (int i = 0; i < k - 1; ++i) {
            row[i] = 100.0f + (float) i;
        }
        for (int i = 0; i < n_tied; ++i) {
            row[k - 1 + i] = 50.0f;
        }
        std::shuffle(row.begin(), row.end(), rng);
        std::copy(row.begin(), row.end(), x.begin() + (size_t) r * ncols);
    }
    return x;
}

static std::vector<float> rows_special(int nrows, int ncols) {
    const float vals[] = { 0.0f, -0.0f, INFINITY, -INFINITY, 1.0f, -1.0f, NAN, -NAN, 1e-45f, -1e-45f, 3.4e38f, -3.4e38f };
    std::uniform_int_distribution<int> d(0, (int) (sizeof(vals) / sizeof(vals[0])) - 1);
    std::vector<float> x((size_t) nrows * ncols);
    for (auto & v : x) {
        v = vals[d(rng)];
    }
    return x;
}

static void run_all() {
    const int narrow[] = { 1, 2, 7, 21, 22, 100, 255, 256, 257, 1000, 1024, 3000, TOPK_SMALL_NCOLS };
    const int wide[]   = { TOPK_SMALL_NCOLS + 1, 9000, 17000, 70000 };
    const int ks[]     = { 1, 2, 5, 21, 64, 513, 1024 };

    for (int ncols : narrow) {
        for (int k : ks) {
            if (k > ncols) {
                continue;
            }
            for (int nrows : { 1, 3 }) {
                if (ncols > 1024 && nrows > 1 && k > 64) {
                    continue;
                }
                check("narrow-normal", rows_normal(nrows, ncols), nrows, ncols, k);
                check("narrow-ties4", rows_ties(nrows, ncols, 4), nrows, ncols, k);
            }
        }
        check("narrow-k=ncols", rows_normal(2, ncols), 2, ncols, std::min(ncols, 1024));
        check("narrow-const", rows_const(2, ncols, 1.5f), 2, ncols, std::min(ncols, 21));
        check("narrow-banned", rows_banned(2, ncols, 3, -INFINITY), 2, ncols, std::min(ncols, 21));
        check("narrow-special", rows_special(2, ncols), 2, ncols, std::min(ncols, 21));
    }
    for (int ncols : { 100, 3000 }) {
        check("narrow-boundary", rows_boundary(2, ncols, 21, 5), 2, ncols, 21);
        check("narrow-boundary-all", rows_boundary(2, ncols, 21, ncols - 20), 2, ncols, 21);
    }

    for (int ncols : wide) {
        for (int k : { 1, 21, 64, 513, 1024 }) {
            for (int nrows : { 2, 7 }) {
                if (ncols > 20000 && nrows > 2) {
                    continue;
                }
                check("wide-normal", rows_normal(nrows, ncols), nrows, ncols, k);
            }
        }
        check("wide-ties4", rows_ties(3, ncols, 4), 3, ncols, 21);
        check("wide-ties-many", rows_ties(2, ncols, 3000), 2, ncols, 64);
        check("wide-const", rows_const(2, ncols, 0.25f), 2, ncols, 21);
        check("wide-const-k1", rows_const(2, ncols, 0.25f), 2, ncols, 1);
        check("wide-banned", rows_banned(4, ncols, 5, -INFINITY), 4, ncols, 21);
        check("wide-banned-big", rows_banned(2, ncols, 100, -INFINITY), 2, ncols, 64);
        check("wide-special", rows_special(3, ncols), 3, ncols, 21);
        check("wide-boundary", rows_boundary(3, ncols, 21, 5), 3, ncols, 21);
        check("wide-boundary-all", rows_boundary(2, ncols, 21, ncols - 20), 2, ncols, 21);
        check("wide-boundary-mid", rows_boundary(2, ncols, 500, 700), 2, ncols, 500);
    }

    // chunking: more rows than the scratch budget holds
    check("chunked-normal", rows_normal(8, 9000), 8, 9000, 21);
    check("chunked-ties", rows_ties(8, 9000, 3), 8, 9000, 21);
    check("chunked-boundary", rows_boundary(8, 9000, 21, 40), 8, 9000, 21);
}

int main() {
    if (const char * m = getenv("TOPK_MUTANT")) {
        topk_mutant = atoi(m);  // local: negative controls, see top-k-batched.cuh
    }
    g_pool = new emu_pool();
    run_all();
    delete g_pool;
    printf("%s: %d cases, %d failed\n", g_fail ? "FAIL" : "PASS", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
