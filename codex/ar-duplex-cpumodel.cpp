// CPU model of the chunked-kernel AllReduce protocol: the kernel text of a given revision runs unchanged on std::threads
// (one per CUDA thread, two GPUs concurrently). Checks indexing, flags, token reuse across calls and the in-place
// update; the memory model of the device is NOT modelled (that is the CUDA driver's job). Build via ar-duplex-cpumodel.sh.
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

static inline void ggml_cuda_ar_signal_set(int * p, int token) { __atomic_store_n(p, token, __ATOMIC_RELEASE); }
static inline int ggml_cuda_ar_signal_get_acquire(const int * p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static inline int ggml_cuda_ar_signal_get(const int * p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }

#include KERNEL_HEADER   // constants + ggml_cuda_ar_kernel, extracted from allreduce.cu

#ifndef AR_API_LOCAL
#define AR_API_LOCAL 0
#endif

struct launch_cfg { int threads; int round_vecs; int mutant; int stall_ns; int start_delay_ns = 0; };

template <typename T_dst, typename T_wire>
static void run_call(const launch_cfg & c, std::vector<T_dst> * data, std::vector<T_wire> * hbuf,
                     int * arrival[2], int count, int token, int slot) {
    const int blocks = GGML_CUDA_AR_KERNEL_BLOCKS;
    std::vector<std::thread> ts;
    std::vector<std::unique_ptr<std::barrier<>>> bars;
    for (int g = 0; g < 2; ++g) for (int b = 0; b < blocks; ++b) bars.emplace_back(new std::barrier<>(c.threads));
    const size_t slot_off = (size_t) slot * (1 << 20) / sizeof(T_wire);
    for (int g = 0; g < 2; ++g) {
        for (int b = 0; b < blocks; ++b) {
            for (int t = 0; t < c.threads; ++t) {
                ts.emplace_back([&, g, b, t]() {
                    threadIdx.x = t; blockIdx.x = b; blockDim.x = c.threads; gridDim.x = blocks;
                    if (g == 1 && c.start_delay_ns > 0) {   // rank 1 starts late, whatever the kernel revision
                        const long long t0 = clock64();
                        while (clock64() - t0 < c.start_delay_ns) {}
                    }
                    tl_barrier = bars[g * blocks + b].get();
                    T_dst * d = data[g].data();
                    T_wire * mine = hbuf[g].data() + slot_off;
                    const T_wire * other = hbuf[1 - g].data() + slot_off;
                    ggml_cuda_ar_kernel<T_dst, T_wire>(d, d, mine, other, count,
                        arrival[g], arrival[1 - g], token
#if AR_API_LOCAL
                        , c.round_vecs > 0 ? c.round_vecs : (1 << 30), c.mutant, g == 1 ? c.stall_ns : 0
#endif
                    );
                });
            }
        }
    }
    for (auto & t : ts) t.join();
}

static uint64_t rng = 88172645463325252ull;
static uint32_t rnd() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t) (rng >> 11); }

template <typename T_dst, typename T_wire>
static int test(const char * name, const launch_cfg & c, int count, int calls, bool expect_fail) {
    constexpr int MAXR = 128;
    const size_t flag_ints = (size_t) GGML_CUDA_AR_KERNEL_BLOCKS * MAXR;
    // one flag array per (slot, rank): the kernel is handed the (slot, rank) base, as the host does
    std::vector<int> flags[2][2];
    for (auto & a : flags) for (auto & f : a) f.assign(flag_ints, 0);
    std::vector<T_wire> hbuf[2];
    for (auto & h : hbuf) h.assign(2 * (1 << 20) / sizeof(T_wire), T_wire{});
    int bad = 0, calls_bad = 0;
    int token = 0;
    for (int call = 0; call < calls; ++call) {
        ++token;
        const int slot = call % 2;
        std::vector<T_dst> data[2], in[2];
        for (int g = 0; g < 2; ++g) {
            in[g].resize(count);
            for (auto & x : in[g]) x = (T_dst) ((int) (rnd() % 20001) - 10000) / (T_dst) 64;
            data[g] = in[g];
        }
        int * arrival[2] = { flags[slot][0].data(), flags[slot][1].data() };
        run_call<T_dst, T_wire>(c, data, hbuf, arrival, count, token, slot);
        bool call_bad = false;
        for (int g = 0; g < 2; ++g) {
            for (int e = 0; e < count; ++e) {
                const T_wire own  = ggml_cuda_cast<T_wire>(in[g][e]);
                const T_wire peer = ggml_cuda_cast<T_wire>(in[1 - g][e]);
                const T_dst ref = ggml_cuda_cast<T_dst>(ggml_cuda_cast<float>(own) + ggml_cuda_cast<float>(peer));
                if (memcmp(&ref, &data[g][e], sizeof(T_dst)) != 0) { ++bad; call_bad = true; }
            }
        }
        calls_bad += call_bad;
    }
    const bool ok = expect_fail ? bad > 0 : bad == 0;
    printf("%s %-28s count=%-7d thr=%-4d rv=%-2d mut=%d stall=%-6d calls=%d bad_elems=%d bad_calls=%d %s\n", ok ? "ok  " : "FAIL",
           name, count, c.threads, c.round_vecs, c.mutant, c.stall_ns, calls, bad, calls_bad, expect_fail ? "(expect fail)" : "");
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    const bool quick = argc > 1 && !strcmp(argv[1], "quick");
    int rc = 0;
    if (argc > 1 && !strcmp(argv[1], "mutated")) {
        // The wait loop is compiled out (script): with rank 1 late, every call must come out wrong.
        for (int n : { 5120 * 6, 5120 * 12 }) {
            rc |= test<float, bf16>("late-rank1+no-wait", launch_cfg{ 128, 1, 0, 0, 300000 }, n, 3, true);
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
                rc |= test<float, bf16>("f32/bf16", c, n, 4, false);
            }
        }
        for (int n : counts_f) rc |= test<float, float>("f32/f32", launch_cfg{ th, 1, 0, 0 }, n, 4, false);
    }
    for (int n : { 5120 * 6, 5120 * 12 }) {
        rc |= test<float, bf16>("late-rank1+wait", launch_cfg{ 128, 1, 0, 0, 300000 }, n, 3, false);
    }
#if AR_API_LOCAL
    // rank 1 stalls before every round: a correct wait still matches, a missing wait must not
    for (int n : { 5120 * 6, 5120 * 12 }) {
        rc |= test<float, bf16>("stall+wait", launch_cfg{ 128, 1, 0, 300000 }, n, 3, false);
        rc |= test<float, bf16>("stall+MUTANT", launch_cfg{ 128, 1, 1, 300000 }, n, 3, true);
    }
#endif
    printf("cpumodel: %s\n", rc ? "FAIL" : "PASS");
    return rc;
}
