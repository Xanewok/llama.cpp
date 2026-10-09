// CPU model of the chunked-kernel AllReduce protocol: the kernel text of a given revision runs unchanged on std::threads
// (one per CUDA thread, two GPUs concurrently, launches of consecutive calls overlapping as the host allows). Checks
// indexing, flag/token logic, slot reuse across launches without a join between the GPUs, stale-flag wrap, the in-place
// update. Build via ar-duplex-cpumodel.sh.
//
// What it cannot show: the GPU memory model. It runs on x86 (TSO) and the shims below are the kernel's own operations
// (relaxed flag store, seq_cst fence, acquire flag load, plain data), so a missing release/acquire edge or fence in the
// kernel is NOT detected here: the TSO hardware orders the stores anyway. Reordering, L1/L2 staleness, posted-write
// draining and the proxy behaviour of host memory over PCIe are covered only by the CUDA driver/window on real hardware and
// by the SASS review, not by this model.
#include <atomic>
#include <algorithm>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

#define __global__
#define __device__
#define __forceinline__ inline
#define __restrict__ __restrict
#define NO_DEVICE_CODE
#define GGML_CUDA_CC_VOLTA 700
#define __CUDA_ARCH__ 1200

struct dim3_ { int x; };
static thread_local dim3_ threadIdx, blockIdx, blockDim, gridDim;
static thread_local std::barrier<> * tl_barrier;

static inline void __syncthreads() { tl_barrier->arrive_and_wait(); }
static inline void __threadfence_system() { std::atomic_thread_fence(std::memory_order_seq_cst); }
static inline void __nanosleep(int) { std::this_thread::yield(); }
static inline long long clock64() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static inline int max(int a, int b) { return a > b ? a : b; }
static inline int min(int a, int b) { return a < b ? a : b; }
static constexpr int ggml_cuda_get_max_cpy_bytes() { return 16; }

struct bf16 { uint16_t v; };
template <typename T, typename U> static inline T ggml_cuda_cast(U x) {
    if constexpr (std::is_same_v<T, U>) {
        return x;
    } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, bf16>) {
        const uint32_t u = (uint32_t) x.v << 16; float f; memcpy(&f, &u, 4); return f;
    } else if constexpr (std::is_same_v<T, bf16> && std::is_same_v<U, float>) {
        uint32_t u; memcpy(&u, &x, 4); u += 0x7fff + ((u >> 16) & 1); return bf16{(uint16_t) (u >> 16)};
    } else {
        static_assert(sizeof(T) == 0, "cast");
    }
}
template <int nbytes, int alignment = 0> static inline void ggml_cuda_memcpy_1(void * dst, const void * src) { memcpy(dst, src, nbytes); }

// The kernel's own operations, no stronger: flag store = volatile (relaxed.sys), threadfence = fence.sc.sys,
// acquire flag load = ld.acquire.sys, base flag load = volatile (relaxed). Data stays plain memory.
template <class T> static inline void ggml_cuda_ar_signal_set(T * p, T token) { __atomic_store_n(p, token, __ATOMIC_RELAXED); }
template <class T> static inline T ggml_cuda_ar_signal_get_acquire(const T * p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
template <class T> static inline T ggml_cuda_ar_signal_get(const T * p) { return __atomic_load_n(p, __ATOMIC_RELAXED); }

#include KERNEL_HEADER   // constants + ggml_cuda_ar_kernel, extracted from allreduce.cu

#ifndef AR_API_LOCAL
#define AR_API_LOCAL 0
#endif
#ifndef FLAG_T
#define FLAG_T int
#endif
#ifndef MULTI_FLAG
#define MULTI_FLAG 0
#endif

struct launch_cfg { int threads; int round_vecs; int mutant; int stall_ns; int start_delay_ns = 0; uint64_t token0 = 0; bool stale = false; };

// One kernel launch of one GPU: its block threads run concurrently, the call returns when the grid is done.
template <typename T_dst, typename T_wire>
static void launch_gpu(const launch_cfg & c, int g, T_dst * d, std::vector<T_wire> * hbuf, FLAG_T * arrival[2],
                       int count, FLAG_T token, int slot) {
    const int blocks = GGML_CUDA_AR_KERNEL_BLOCKS;
    std::vector<std::thread> ts;
    std::vector<std::unique_ptr<std::barrier<>>> bars;
    for (int b = 0; b < blocks; ++b) bars.emplace_back(new std::barrier<>(c.threads));
    const size_t slot_off = (size_t) slot * (1 << 20) / sizeof(T_wire);
    for (int b = 0; b < blocks; ++b) {
        for (int t = 0; t < c.threads; ++t) {
            ts.emplace_back([&, b, t]() {
                threadIdx.x = t; blockIdx.x = b; blockDim.x = c.threads; gridDim.x = blocks;
                tl_barrier = bars[b].get();
                T_wire * mine = hbuf[g].data() + slot_off;
                const T_wire * other = hbuf[1 - g].data() + slot_off;
                ggml_cuda_ar_kernel<T_dst, T_wire>(d, d, mine, other, count, arrival[g], arrival[1 - g], token
#if AR_API_LOCAL
                    , c.round_vecs > 0 ? c.round_vecs : (1 << 30), c.mutant, g == 1 ? c.stall_ns : 0
#endif
                );
            });
        }
    }
    for (auto & t : ts) t.join();
}

static uint64_t rng = 88172645463325252ull;
static uint32_t rnd() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t) (rng >> 11); }

// Host model: acquire_slot waits for both GPUs' kernel of call k-2 before call k is enqueued (cudaEventSynchronize on
// ev.ker); each GPU runs its own launches in order (stream order) and nothing else joins the two GPUs, so one GPU can be
// in call k+1 while the other is still in call k.
template <typename T_dst, typename T_wire>
static int test(const char * name, const launch_cfg & c, std::vector<int> counts, int calls, bool expect_fail) {
    constexpr int MAXR = 128;
    const size_t flag_words = (size_t) GGML_CUDA_AR_KERNEL_BLOCKS * MAXR;
    std::vector<FLAG_T> flags[2][2];   // [slot][rank]
    for (auto & a : flags) for (auto & f : a) f.assign(flag_words, 0);
    std::vector<T_wire> hbuf[2];
    for (auto & h : hbuf) h.assign(2 * (1 << 20) / sizeof(T_wire), T_wire{});
    std::vector<std::vector<T_dst>> data(2 * (size_t) calls), in(2 * (size_t) calls);
    for (int k = 0; k < calls; ++k) {
        const int count = counts[k % counts.size()];
        for (int g = 0; g < 2; ++g) {
            auto & v = in[2 * k + g];
            v.resize(count);
            for (auto & x : v) x = (T_dst) ((int) (rnd() % 20001) - 10000) / (T_dst) 64;
            data[2 * k + g] = v;
        }
    }
    std::atomic<int> done[2] = { 0, 0 };   // calls finished per GPU
    std::vector<std::thread> gpus;
    std::atomic<int> enq[2] = { 0, 0 };     // calls enqueued per GPU
    for (int g = 0; g < 2; ++g) {
        gpus.emplace_back([&, g]() {
            for (int k = 0; k < calls; ++k) {
                while (enq[g].load(std::memory_order_acquire) <= k) std::this_thread::yield();
                if (g == 1 && c.start_delay_ns > 0) {
                    const long long t0 = clock64();
                    while (clock64() - t0 < c.start_delay_ns) {}
                }
                const int count = counts[k % counts.size()];
                FLAG_T * arrival[2] = { flags[k % 2][0].data(), flags[k % 2][1].data() };
                launch_gpu<T_dst, T_wire>(c, g, data[2 * k + g].data(), hbuf, arrival, count,
                                          (FLAG_T) (c.token0 + k + 1), k % 2);
                done[g].store(k + 1, std::memory_order_release);
            }
        });
    }
    for (int k = 0; k < calls; ++k) {   // host thread
        if (k >= 2) while (done[0].load(std::memory_order_acquire) < k - 1 || done[1].load(std::memory_order_acquire) < k - 1) std::this_thread::yield();
        if (c.stale) {   // what an older epoch left in the rounds this call has not written yet: same value modulo 2^32
            const uint64_t tok = c.token0 + k + 1;
            for (int g = 0; g < 2; ++g) {
                auto & f = flags[k % 2][g];
                for (size_t j = 0; j < f.size(); ++j) if (j % MAXR != 0) f[j] = (FLAG_T) (tok - (1ull << 32));
            }
        }
        enq[0].store(k + 1, std::memory_order_release);
        enq[1].store(k + 1, std::memory_order_release);
    }
    for (auto & t : gpus) t.join();
    int bad = 0, calls_bad = 0;
    for (int k = 0; k < calls; ++k) {
        const int count = counts[k % counts.size()];
        bool call_bad = false;
        for (int g = 0; g < 2; ++g) {
            const auto & got = data[2 * k + g];
            for (int e = 0; e < count; ++e) {
                const T_wire own  = ggml_cuda_cast<T_wire>(in[2 * k + g][e]);
                const T_wire peer = ggml_cuda_cast<T_wire>(in[2 * k + 1 - g][e]);
                const T_dst ref = ggml_cuda_cast<T_dst>(ggml_cuda_cast<float>(own) + ggml_cuda_cast<float>(peer));
                if (memcmp(&ref, &got[e], sizeof(T_dst)) != 0) { ++bad; call_bad = true; }
            }
        }
        calls_bad += call_bad;
    }
    const bool ok = expect_fail ? bad > 0 : bad == 0;
    printf("%s %-26s count=%-7d thr=%-4d rv=%-2d mut=%d stall=%-6d calls=%d bad_elems=%d bad_calls=%d %s\n", ok ? "ok  " : "FAIL",
           name, counts[0], c.threads, c.round_vecs, c.mutant, c.stall_ns, calls, bad, calls_bad, expect_fail ? "(expect fail)" : "");
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    const bool quick = argc > 1 && !strcmp(argv[1], "quick");
    int rc = 0;
    if (argc > 1 && !strcmp(argv[1], "mutated")) {
        // The wait loop is compiled out (script): with rank 1 late, every call must come out wrong.
        for (int n : { 5120 * 6, 5120 * 12 }) {
            rc |= test<float, bf16>("late-rank1+no-wait", launch_cfg{ 128, 1, 0, 0, 300000 }, { n }, 3, true);
        }
        printf("cpumodel-mutated: %s\n", rc ? "FAIL (mutant passed)" : "PASS (mutant failed identity)");
        return rc;
    }
    const int thr[] = { 64, 128, 256 };
    // counts are wire elements; bf16 vector = 8 elements, float vector = 4
    const int counts_bf[] = { 1, 4, 7, 8, 9, 640 * 8, 5120 * 1 * 1 + 4, 5120 * 6, 5120 * 12 + 3, 5120 * 18, 5120 * 64, 262144 };
    const int counts_f[]  = { 4, 5, 20480, 5120 * 6 + 1, 5120 * 12 };
    for (int th : thr) {
        if (quick && th != 128) continue;
        for (int n : counts_bf) {
            if (th == 64 && n > 40000 && quick) continue;
            for (int rv : { 1, 2, 0 }) {
                if (!AR_API_LOCAL && rv != 1) continue;
                launch_cfg c{ th, rv, 0, 0 };
                rc |= test<float, bf16>("f32/bf16", c, { n }, 4, false);
            }
        }
        for (int n : counts_f) rc |= test<float, float>("f32/f32", launch_cfg{ th, 1, 0, 0 }, { n }, 4, false);
    }
    for (int n : { 5120 * 6, 5120 * 12 }) {
        rc |= test<float, bf16>("late-rank1+wait", launch_cfg{ 128, 1, 0, 0, 300000 }, { n }, 3, false);
    }
    // Overlapping launches of consecutive calls, sizes alternating so a slot's flags hold rounds of a larger earlier call.
    for (int th : { 64, 128 }) {
        rc |= test<float, bf16>("slot-reuse-nojoin", launch_cfg{ th, 1, 0, 0, 20000 }, { 5120 * 18, 7, 5120 * 12 + 3, 5120 * 6, 1, 5120 * 18 }, 24, false);
        rc |= test<float, bf16>("slot-reuse-nojoin-nodelay", launch_cfg{ th, 1, 0, 0 }, { 5120 * 12, 5120 * 6, 9, 5120 * 18 }, 24, false);
    }
    // Stale-flag wrap: tokens past 2^32 and untouched rounds holding the value 2^32 calls older.  32-bit per-round flags
    // are exactly the case that must fail; 64-bit flags must pass.  The base has one flag per block that every call
    // rewrites, so stale equality cannot arise there and the case does not apply.
    if (MULTI_FLAG) {
        const bool expect = sizeof(FLAG_T) == 4;
        launch_cfg w{ 128, 1, 0, 0, 300000, 1ull << 32, true };
        rc |= test<float, bf16>("stale-wrap", w, { 5120 * 12 }, 3, expect);
    }
#if AR_API_LOCAL
    // rank 1 stalls before every round: a correct wait still matches, a missing wait must not
    for (int n : { 5120 * 6, 5120 * 12 }) {
        rc |= test<float, bf16>("stall+wait", launch_cfg{ 128, 1, 0, 300000 }, { n }, 3, false);
        rc |= test<float, bf16>("stall+MUTANT", launch_cfg{ 128, 1, 1, 300000 }, { n }, 3, true);
    }
#endif
    printf("cpumodel: %s\n", rc ? "FAIL" : "PASS");
    return rc;
}
