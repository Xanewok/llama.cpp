#include "ggml-round-trace.h"
#include "allreduce.cuh"

#if !defined(GGML_USE_MUSA)

#include "convert.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>

// ---------------------------------------------------------------------------
// AllReduce for tensor-parallel inference across two GPUs (CUDA or
// ROCm/HIP).
//
// Provides an in-place sum reduction over matching tensors on two GPUs
// in the same process.  Used by the tensor-split path alongside NCCL;
// targets setups without NVLink/xGMI, where data is exchanged between the
// GPUs by staging it through pinned host memory over PCIe.
//
// Two reduction strategies are selected per call by tensor size:
//
//   * Chunked kernel path (small reductions): a single CUDA kernel both
//     stages data through pinned host memory and performs the local sum.
//     Cross-GPU synchronization happens *inside the kernel* (busy-wait on
//     a host-memory flag), which keeps launch overhead low for the
//     latency-sensitive token-generation case.
//
//   * Copy-engine path (large reductions): the transfer is split into
//     D2H + H2D cudaMemcpyAsync chunks driven by the GPU's copy engine,
//     followed by a small device-side add kernel.  Cross-GPU
//     synchronization happens *outside the kernel*, via CUDA events
//     between streams.  This keeps the compute engine free while large
//     transfers are in flight, which matters for prefill-sized tensors.
//     Reductions larger than the per-call inner cap are processed by an
//     outer chunker that issues sequential inner calls.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-GPU signal mechanism
//
// One 64-bit word per (slot, rank, block, round) in pinned host memory.  Each AR
// call writes a strictly increasing token (= the AR call number, starting at 1)
// into the arrival word of every round it completes.  The peer spins until its
// read of the other's arrival word equals the token it expects for this call --
// a mismatch means the peer hasn't completed that round yet.  Words are zeroed
// once at pipeline init and never reset.  Tokens are 64-bit so that a stale word
// can never equal a later call's token: with 32 bits a word of a round that
// short launches never touch would match again after 2^32 calls.  2^64 calls
// are unreachable.
//
// There is exactly one writer (the owning GPU) and one reader (the peer), so
// we don't need atomics.  Producer: every thread's stores, a block barrier, then
// thread 0 issues __threadfence_system() and the volatile flag store.  The fence
// is cumulative over what the barrier made visible to thread 0, so the other
// threads need no fence of their own (the same pattern as cooperative-groups
// grid sync).  Consumer: thread 0's acquire load of the token followed by a
// block barrier orders the block's reads of the data.
//
// atomicAdd_system() requires hostNativeAtomicSupported, which is unavailable
// on PCIe-attached consumer GPUs without NVLink, so the volatile path is the
// portable choice.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void ggml_cuda_ar_signal_set(uint64_t * p, uint64_t token) {
    *(volatile uint64_t *)p = token;
}
// Acquire load: an observed token orders the loads of the peer's data issued after
// the block barrier, without the full __threadfence_system() whose wait for this
// thread's own posted host stores would serialize the two PCIe directions.
static __device__ __forceinline__ uint64_t ggml_cuda_ar_signal_get_acquire(const uint64_t * p) {
#ifdef GGML_USE_HIP
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    uint64_t v;
    asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
#endif // GGML_USE_HIP
}

// Number of blocks the chunked kernel launches with.  Each block stripes a
// disjoint slice of the data and synchronizes through its own arrival-token
// slots so multiple SMs can pump PCIe stores in parallel.
static constexpr int GGML_CUDA_AR_KERNEL_BLOCKS = 8;

// Threads per block.  One round moves one vector per thread, so a round is
// KERNEL_BLOCKS * KERNEL_THREADS vectors and the kernel overlaps the store of
// round r with the load of round r-1.
static constexpr int GGML_CUDA_AR_KERNEL_THREADS = 256;

// Rounds one launch may have (arrival words per block): a 1 MB launch of 16 B
// vectors has at most 65536 vectors, 128 rounds at 64 threads * 8 blocks.  Each
// block's ints are contiguous and cache-line padded from the next block's, so
// pollers of different blocks never share a line.
static constexpr int GGML_CUDA_AR_MAX_ROUNDS = 128;
static constexpr size_t GGML_CUDA_AR_ARRIVAL_BYTES_PER_BLOCK = GGML_CUDA_AR_MAX_ROUNDS * sizeof(uint64_t);
static_assert(GGML_CUDA_AR_ARRIVAL_BYTES_PER_BLOCK % 64 == 0, "arrival words of a block must fill whole cache lines");

// ---------------------------------------------------------------------------
// Chunked kernel AllReduce -- 2 GPUs, supports float, half, and bfloat16.
//
// Both GPUs run this kernel simultaneously on independent streams.  sendbuf
// and recvbuf live in T_dst (the caller's tensor type); host_mine / host_other
// carry data in T_wire (the on-wire type, possibly narrower than T_dst -- e.g.
// T_dst=F32 with T_wire=BF16 halves the bytes pushed across PCIe).  When
// T_dst == T_wire the casts below are no-ops.
//
// The data is processed in rounds of one T_wire vector per thread (vectors are
// single-instruction wide).  Per round r, in this order:
//
//   store:   cast sendbuf (T_dst) -> T_wire and store round r as vectors into
//            host_mine.
//   load:    (r > 0) wait for the peer's flag for round r-1, read the peer's
//            round r-1 vectors from host_other, cast each element to T_dst, and
//            sum with the local sendbuf value (also rounded through T_wire so
//            that both GPUs truncate identically -- this guarantees
//            bit-equivalent results across the two devices).
//   publish: after a block barrier that follows every thread's stores of round
//            r, thread 0 issues __threadfence_system() and writes the token to
//            the round's flag.
//
// The stores of round r are posted and still draining when the loads of round
// r-1 are issued, so the two PCIe directions carry data at the same time; the
// flag of round r is published after the loads, which keeps the pipeline free of
// a cross-thread hand-off.  A pass after the last round waits for the peer's last
// flag and reads the last round.  With a single round this is the original
// store, signal, wait, load sequence.
//
// Every element goes through exactly the arithmetic of the unpipelined kernel:
// only when it is stored and when it is loaded changes.  A thread loads and
// writes back exactly the vectors it stored (same gtid stripe on both GPUs), so
// the in-place update never overwrites a sendbuf element before it was sent.
//
// Multi-block: blocks stripe vectors across (gridDim.x * blockDim.x) global
// threads to keep multiple SMs issuing PCIe stores in parallel.  Each block has
// its own flags; blocks proceed independently -- no inter-block barrier needed.
// Round r of block b covers the same vectors on both GPUs, so it is published
// and awaited per block.  Tail elements (the leftover < ELEMS_PER_VEC at the end)
// are handled only by block 0, in the last round, to avoid cross-block writes to
// the same slots.
// ---------------------------------------------------------------------------
template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_kernel(
        const T_dst  *              sendbuf,
        T_dst        *              recvbuf,
        T_wire       * __restrict__ host_mine,
        const T_wire *              host_other,   // written by the peer during the launch: no __restrict__, so no ld.global.nc
        int                         count,
        uint64_t *                  arrival_mine,
        uint64_t *                  arrival_other,
        uint64_t                    token,
        int                         round_vecs,    // vectors per thread per round
        int                         mutant,        // test only: 1 = read the peer's round without waiting for its flag
        int                         stall_cycles) {// test only: busy-wait before each round's stores

    // Vector unit for the wire type, sized to the arch's widest single-instruction
    // copy (16 B on Volta+).  Each store iter writes one vector to host memory;
    // each load iter reads one and produces ELEMS_PER_VEC sums.
    constexpr int ELEMS_PER_VEC = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
    constexpr int FLAG_WORDS    = (int)(GGML_CUDA_AR_ARRIVAL_BYTES_PER_BLOCK / sizeof(uint64_t));

    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count / ELEMS_PER_VEC;
    const int tail      = count_vec * ELEMS_PER_VEC;

    // Same on both GPUs (depends on count only); at least one round so that the
    // tail-only case still signals.  A round is `per` strided passes over the
    // vectors; per >= total_iters is the unpipelined single-round kernel.
    const int total_iters = max(1, (count_vec + gnt - 1) / gnt);
    const int per         = min(round_vecs, total_iters);
    const int n_rounds    = (total_iters + per - 1) / per;

    uint64_t       * my_flags    = arrival_mine  + bid * FLAG_WORDS;
    const uint64_t * other_flags = arrival_other + bid * FLAG_WORDS;

    for (int r = 0; r <= n_rounds; ++r) {
        // Store: cast sendbuf (T_dst) -> host_mine (T_wire) as vectors.
        if (r < n_rounds) {
            if (stall_cycles > 0) {
                const long long t0 = clock64();
                while (clock64() - t0 < stall_cycles) {}
            }
            for (int j = 0; j < per; ++j) {
                const int it = r * per + j;
                const int i  = it * gnt + gtid;
                if (it < total_iters && i < count_vec) {
                    const int off = i * ELEMS_PER_VEC;
                    T_wire wire[ELEMS_PER_VEC];
                    #pragma unroll
                    for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                        wire[k] = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
                    }
                    ggml_cuda_memcpy_1<sizeof(wire)>(&host_mine[off], wire);
                }
            }
            if (r == n_rounds - 1 && bid == 0 && tid < count - tail) {
                host_mine[tail + tid] = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
            }
        }

        // Load: the peer's round r-1.  Thread 0 of each block spins on that
        // block's flag for the round; the barrier releases the other threads.
        // No __threadfence_system() here: it would wait for the stores of round r
        // (posted above) to drain before the loads start, which is the
        // serialization this loop removes.  Every line is loaded once per launch,
        // the acquire invalidates L1, and the tail line is only loaded in the last
        // pass.
        if (r > 0) {
            if (tid == 0 && !mutant) {
                const uint64_t * other_slot = other_flags + (r - 1);
                while (ggml_cuda_ar_signal_get_acquire(other_slot) != token) {
#ifdef GGML_USE_HIP
                    // Equals ~100ns at 2500 MHz (sleeps for n * [1,64] clock cycles)
                    __builtin_amdgcn_s_sleep(4);
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
                    __nanosleep(100);
#else
                    NO_DEVICE_CODE;
#endif // GGML_USE_HIP
                }
            }

            __syncthreads();

            // Read peer's T_wire vector, cast both sides through T_wire for
            // bit-equivalence, sum in T_dst precision, and write back to recvbuf.
            for (int j = 0; j < per; ++j) {
                const int it = (r - 1) * per + j;
                const int i  = it * gnt + gtid;
                if (it < total_iters && i < count_vec) {
                    const int off = i * ELEMS_PER_VEC;
                    T_wire wire[ELEMS_PER_VEC];
                    ggml_cuda_memcpy_1<sizeof(wire)>(wire, &host_other[off]);
                    #pragma unroll
                    for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                        const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
                        recvbuf[off + k] = ggml_cuda_cast<T_dst>(
                            ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(wire[k]));
                    }
                }
            }
            if (r == n_rounds && bid == 0 && tid < count - tail) {
                const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
                recvbuf[tail + tid] = ggml_cuda_cast<T_dst>(
                    ggml_cuda_cast<float>(d_low) +
                    ggml_cuda_cast<float>(host_other[tail + tid]));
            }
        }

        // Publish round r.  Every thread's stores of round r precede a block barrier
        // before thread 0 fences: the load phase's barrier for r > 0, an explicit one
        // for r == 0.  Thread 0's release fence is cumulative over the stores that
        // barrier made visible to it, so the other threads need no fence, and none
        // sits between round r+1's stores and round r's loads.
        if (r < n_rounds) {
            if (r == 0) {
                __syncthreads();
            }
            if (tid == 0) {
                __threadfence_system();
                ggml_cuda_ar_signal_set(my_flags + r, token);
                if (r == n_rounds - 1) {
                    __threadfence_system(); // make our last signal visible system-wide
                }
            }
        }
    }
}

// Combined load-convert-add kernel.  The peer's contribution arrives as T_src
// (which may be a lower-precision type than T_dst when the BF16 round-trip is
// active).  For bit-equivalence between the two GPUs, dst is first rounded
// through T_src's precision via ggml_cuda_cast -- peer already truncated its
// own value the same way before sending -- so both sides perform identical
// arithmetic.  When T_dst == T_src the round-trip cast is a no-op.
template <typename T_dst, typename T_src>
static __global__ void ggml_cuda_ar_add_kernel(
        T_dst       * __restrict__ dst,
        const T_src * __restrict__ src,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const T_src d_low = ggml_cuda_cast<T_src>(dst[i]);
        dst[i] = ggml_cuda_cast<T_dst>(
            ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(src[i]));
    }
}

// ---------------------------------------------------------------------------
// Pipeline structure
// ---------------------------------------------------------------------------

// Number of slots in the event / arrival ring.  Two slots is sufficient:
// lockstep guarantees the two GPUs are at most one AR (or chunk) apart, so
// slot[N%2] is always safe to reuse -- peer has already consumed slot[N%2]
// from AR N-2 by the time we get to AR N.  acquire_slot's
// cudaEventSynchronize on ev.ker for both devices makes that consumption
// explicit before we overwrite host_buf[slot] for the new AR.
static constexpr int GGML_CUDA_AR_POOL_SIZE = 2;

// Maximum chunk size (bytes per GPU) handled by one chunked kernel launch.
// Larger tensors are reduced by issuing multiple chunked launches.
static constexpr size_t GGML_CUDA_AR_MAX_BYTES = 1024 * 1024; // 1 MB
// 16 B is the smallest vector the kernel runs with on the supported archs (Volta+).
static_assert(GGML_CUDA_AR_MAX_BYTES / 16 / (GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_KERNEL_THREADS) <= GGML_CUDA_AR_MAX_ROUNDS,
              "a launch must not need more rounds than the arrival words provide");

// Copy-engine path: largest tensor accepted on this path; sets host_large /
// dev_tmp allocation size.
static constexpr size_t GGML_CUDA_AR_COPY_MAX_BYTES = 32 * 1024 * 1024; // 32 MB

// AR wire size at which the copy-engine path takes over from the chunked-
// kernel path.  Override via GGML_CUDA_AR_COPY_THRESHOLD.
static constexpr size_t GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT = 1024 * 1024; // 1 MB
// Per-call CE chunk-size heuristic: chunk_bytes = clamp(nbytes / 4, MIN, MAX).
// The /4 keeps ~4 chunks in flight at any moment (good D2H/H2D overlap with
// the peer); the clamps cover the cases where nbytes/4 is too small (per-
// memcpy fixed cost dominates) or too large (chunk-level pipelining stalls).
// Env var GGML_CUDA_AR_COPY_CHUNK_BYTES can override with a fixed value.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN = 512 * 1024;       // 512 KB
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX = 2 * 1024 * 1024;  // 2 MB
// Absolute floor that an env-var override is allowed to set; this caps the
// per-slot copy-event array.  256 KB -> up to 128 chunks per 32 MB tensor.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN = 256 * 1024;
// Copy-engine stage-2 stream count; override via GGML_CUDA_AR_COPY_STREAMS (1 or 2).
static constexpr int GGML_CUDA_AR_COPY_STREAMS_DEFAULT = 2;
static constexpr int GGML_CUDA_AR_COPY_MAX_CHUNKS =
    static_cast<int>((GGML_CUDA_AR_COPY_MAX_BYTES + GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN - 1) /
                    GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);

struct ggml_cuda_ar_event_slot {
    cudaEvent_t app = nullptr;  // upstream computation complete
    cudaEvent_t cpy[GGML_CUDA_AR_COPY_MAX_CHUNKS] = {};  // copy-engine D2H chunks complete
    cudaEvent_t h2d = nullptr;  // copy-engine H2Ds complete (handoff AR stream -> compute stream)
    cudaEvent_t ker = nullptr;  // AllReduce kernel complete
};

// Mapped pinned host allocation: cudaHostAlloc + cudaHostGetDevicePointer
// in one place, with the host handle preserved for cudaFreeHost.  Used where
// the CPU never touches the buffer -- only the device reads/writes via the
// mapped device pointer.  Required on systems where cudaDevAttrCanUseHost-
// PointerForRegisteredMem is 0 and the host pointer can't be used as a
// device pointer.
struct ggml_cuda_ar_host_mapping {
    uint8_t * host = nullptr;   // cudaFreeHost handle; also the H-side ptr for cudaMemcpyAsync
    uint8_t * dev  = nullptr;   // device-side pointer for kernels / cudaMemset

    cudaError_t alloc(size_t bytes) {
        cudaError_t rc = cudaHostAlloc(reinterpret_cast<void **>(&host), bytes,
                                       cudaHostAllocPortable | cudaHostAllocMapped);
        if (rc != cudaSuccess) {
            host = nullptr;
            return rc;
        }
        rc = cudaHostGetDevicePointer(reinterpret_cast<void **>(&dev), host, 0);
        if (rc != cudaSuccess) {
            CUDA_CHECK(cudaFreeHost(host));
            host = nullptr;
            dev  = nullptr;
        }
        return rc;
    }

    void free() {
        if (host) {
            CUDA_CHECK(cudaFreeHost(host));
            host = nullptr;
            dev  = nullptr;
        }
    }
};

struct ggml_cuda_ar_pipeline {
    int      n_devices;
    int      devices[GGML_CUDA_MAX_DEVICES];
    size_t   buf_bytes;    // bytes per device in host_buf[]
    size_t   copy_bytes;   // bytes per device in host_large[] / dev_tmp[]
    size_t   copy_threshold;
    size_t   copy_chunk_bytes;
    size_t   bf16_threshold; // tensors >= this size (bytes) are reduced via FP32->BF16 round-trip; 0 disables
    uint64_t call_count;
    int      round_threads;  // 0 = auto
    int      round_vecs;     // vectors per thread per kernel round; 0 = one round (unpipelined)
    int      mutant;         // test only
    int      stall_cycles;   // test only: stall applied to rank 1

    // Per-device resources.
    ggml_cuda_ar_host_mapping host_buf[GGML_CUDA_MAX_DEVICES];   // pinned staging (chunked kernel)
    ggml_cuda_ar_host_mapping host_large[GGML_CUDA_MAX_DEVICES]; // pinned staging (copy-engine)
    char *                    dev_tmp[GGML_CUDA_MAX_DEVICES];    // device scratch for copy-engine path
    cudaStream_t             streams[GGML_CUDA_MAX_DEVICES];   // non-blocking
    // Copy-engine stage-2 H2D stream.  Equals streams[i] when GGML_CUDA_AR_COPY_STREAMS=1
    // (D2H and H2D serialized on one stream); a separate non-blocking stream when 2, so
    // the H2D of peer chunk c overlaps the D2H of our remaining chunks.
    cudaStream_t             streams_h2d[GGML_CUDA_MAX_DEVICES];
    ggml_cuda_ar_event_slot  ev_pool[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_POOL_SIZE];

    // Copy-engine: per-device "I finished reading my peer's host_large"
    // event.  Indexed by RECORDER device.  Recorded same-device on streams_h2d[i]
    // after stage 2's last H2D from host_large[peer].  Waited cross-device
    // by peer's stage-1 stream before the next AR overwrites host_large[peer].
    cudaEvent_t              host_large_read_done[GGML_CUDA_MAX_DEVICES];
    bool                     host_large_read_done_valid;

    // Copy-engine: per-device "my add_kernel is done with dev_tmp" event.
    // Recorded on the compute stream after each add_kernel; the AR stream
    // waits on it before the next copy_impl's H2D overwrites dev_tmp.  Lets us
    // single-buffer dev_tmp despite add_kernel running on a separate stream.
    cudaEvent_t              dev_tmp_kernel_done[GGML_CUDA_MAX_DEVICES];
    bool                     dev_tmp_kernel_done_valid;

    // Arrival ring: cache-line padded per-block flag arrays.  Mapped pinned
    // memory; CPU never reads/writes -- only the kernel and cudaMemset.
    // Use ggml_cuda_ar_arrival_ptr() to index.
    ggml_cuda_ar_host_mapping arrival;
};

// Base pointer for the (slot, rank) per-block flag blocks.  The kernel adds
// blockIdx.x * (ARRIVAL_BYTES_PER_BLOCK/sizeof(int)) internally to land on its
// own block's flags.
static uint64_t * ggml_cuda_ar_arrival_ptr(const ggml_cuda_ar_pipeline * p, int slot, int rank) {
    const size_t offset = ((size_t)slot * p->n_devices + rank) *
                          GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_BYTES_PER_BLOCK;
    return reinterpret_cast<uint64_t *>(p->arrival.dev + offset);
}

static uint64_t ggml_cuda_ar_env_u64(const char * name, uint64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t) parsed : default_value;
}

struct ggml_cuda_ar_slot_info {
    int      slot;
    uint64_t token;
};

static ggml_cuda_ar_slot_info ggml_cuda_ar_acquire_slot(ggml_cuda_ar_pipeline * p) {
    const int  slot        = static_cast<int>(p->call_count % GGML_CUDA_AR_POOL_SIZE);
    const bool pool_lapped = p->call_count >= GGML_CUDA_AR_POOL_SIZE;
    p->call_count++;

    if (pool_lapped) {
        for (int i = 0; i < p->n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            ggml_round_trace_scope trace_wait("ar_wait", p->devices[i]);
            CUDA_CHECK(cudaEventSynchronize(p->ev_pool[i][slot].ker));
        }
    }

    return { slot, p->call_count };
}

// Per-AR copy-engine chunk size: env-var override if set, else heuristic
// (clamp(nbytes/4, HEURISTIC_MIN, HEURISTIC_MAX)).
static size_t ggml_cuda_ar_chunk_bytes(const ggml_cuda_ar_pipeline * p, size_t nbytes) {
    if (p->copy_chunk_bytes > 0) {
        return p->copy_chunk_bytes;
    }
    return std::min(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX,
                    std::max(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN, nbytes / 4));
}

static void ggml_cuda_ar_wait_for_compute(
        ggml_cuda_ar_pipeline * p, ggml_backend_cuda_context * cuda_ctx, int rank, int slot) {
    ggml_cuda_ar_event_slot & ev = p->ev_pool[rank][slot];
    CUDA_CHECK(cudaEventRecord(ev.app, cuda_ctx->stream()));
    CUDA_CHECK(cudaStreamWaitEvent(p->streams[rank], ev.app));
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------

ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices) {

    if (n_devices != 2) {
        GGML_LOG_DEBUG("%s: internal AllReduce only supports n_devices=2 (got %zu); "
                       "falling back\n", __func__, n_devices);
        return nullptr;
    }

    // The chunked kernel uses __nanosleep (NVIDIA, sm70+) or
    // __builtin_amdgcn_s_sleep (AMD).
    for (size_t i = 0; i < n_devices; ++i) {
        const int cc = ggml_cuda_info().devices[devices[i]].cc;
        if (cc < GGML_CUDA_CC_VOLTA) {
            GGML_LOG_DEBUG("%s: internal AllReduce requires compute capability >= %d "
                           "(device %d has cc=%d); falling back\n",
                           __func__, GGML_CUDA_CC_VOLTA, devices[i], cc);
            return nullptr;
        }
    }

    auto * p = new ggml_cuda_ar_pipeline{};
    p->n_devices        = n_devices;
    p->copy_bytes       = GGML_CUDA_AR_COPY_MAX_BYTES;
    p->copy_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_THRESHOLD", GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT);
    // 0 = use the per-call heuristic (default).  Non-zero env value forces a
    // fixed chunk size for diagnostics, with a floor at COPY_CHUNK_BYTES_MIN.
    p->copy_chunk_bytes = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_CHUNK_BYTES", 0);
    if (p->copy_chunk_bytes > 0 && p->copy_chunk_bytes < GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN) {
        GGML_LOG_WARN("%s: GGML_CUDA_AR_COPY_CHUNK_BYTES=%zu below minimum %zu; clamping\n",
                      __func__, p->copy_chunk_bytes, GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);
        p->copy_chunk_bytes = GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN;
    }
    // Default 1: BF16 round-trip is always on for F32 inputs (any non-zero
    // ne).  Set GGML_CUDA_AR_BF16_THRESHOLD=0 to disable, or to a larger
    // byte threshold to opt out for small tensors.
    p->bf16_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 1);
    // Local: GGML_CUDA_AR_THREADS (64..1024, multiple of 32; 0 = auto), GGML_CUDA_AR_ROUND_VECS
    // (vectors per thread per round; 0 = one round, the unpipelined kernel), GGML_CUDA_AR_MUTANT=1 (read the
    // peer's round without waiting for its flag: must fail the identity check), GGML_CUDA_AR_STALL_US (rank 1
    // busy-waits ~this many us before each round's stores).
    p->round_threads    = (int) ggml_cuda_ar_env_u64("GGML_CUDA_AR_THREADS", 0);
    if (p->round_threads != 0 && (p->round_threads < 64 || p->round_threads > 1024 || p->round_threads % 32 != 0)) {
        GGML_LOG_WARN("%s: GGML_CUDA_AR_THREADS=%d not 0 or a multiple of 32 in [64,1024]; using auto\n",
                      __func__, p->round_threads);
        p->round_threads = 0;
    }
    p->round_vecs       = (int) ggml_cuda_ar_env_u64("GGML_CUDA_AR_ROUND_VECS", 1);
    p->mutant           = (int) ggml_cuda_ar_env_u64("GGML_CUDA_AR_MUTANT", 0);
    p->stall_cycles     = (int) (ggml_cuda_ar_env_u64("GGML_CUDA_AR_STALL_US", 0) * 2000);
    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i] = devices[i];
    }
    // GGML_CUDA_AR_COPY_STREAMS: 2 = separate H2D stream per GPU, anything else = 1 (shared stream).
    uint64_t copy_streams = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_STREAMS", GGML_CUDA_AR_COPY_STREAMS_DEFAULT);
    if (copy_streams != 1 && copy_streams != 2) {
        GGML_LOG_WARN("%s: GGML_CUDA_AR_COPY_STREAMS=%llu not in {1,2}; using %d\n",
                      __func__, (unsigned long long) copy_streams, GGML_CUDA_AR_COPY_STREAMS_DEFAULT);
        copy_streams = GGML_CUDA_AR_COPY_STREAMS_DEFAULT;
    }
    const bool split_h2d = copy_streams == 2;

    // Per-device streams and event pools.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);

        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaStreamCreateWithFlags failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        p->streams[i] = stream;
        p->streams_h2d[i] = stream;
        if (split_h2d) {
            cudaStream_t stream_h2d = nullptr;
            if (cudaStreamCreateWithFlags(&stream_h2d, cudaStreamNonBlocking) != cudaSuccess) {
                GGML_LOG_ERROR("%s: cudaStreamCreateWithFlags (h2d) failed for device %d\n",
                               __func__, p->devices[i]);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
            p->streams_h2d[i] = stream_h2d;
        }

        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            bool ok =
                cudaEventCreateWithFlags(&p->ev_pool[i][s].app, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].h2d, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].ker, cudaEventDisableTiming) == cudaSuccess;
            for (int c = 0; ok && c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                ok = cudaEventCreateWithFlags(&p->ev_pool[i][s].cpy[c], cudaEventDisableTiming) == cudaSuccess;
            }
            if (!ok) {
                GGML_LOG_ERROR("%s: cudaEventCreate failed for device %d slot %d\n",
                               __func__, p->devices[i], s);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
        }

        if (cudaEventCreateWithFlags(&p->host_large_read_done[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for host_large_read_done failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaEventCreateWithFlags(&p->dev_tmp_kernel_done[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for dev_tmp_kernel_done failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Arrival ring: cache-line padded so each GPU's block flags are on their own lines.
    const size_t arrival_bytes =
        (size_t)GGML_CUDA_AR_POOL_SIZE * n_devices *
        GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_BYTES_PER_BLOCK;
    if (p->arrival.alloc(arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: alloc for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }
    ggml_cuda_set_device(p->devices[0]);
    if (cudaMemset(p->arrival.dev, 0, arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: cudaMemset for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }

    // Per-device pinned staging buffers -- POOL_SIZE-deep ring so the chunked-
    // kernel can write the next slot's data while the peer is still reading
    // the previous slot's. Indexed by (slot * buf_bytes) at the call site.
    p->buf_bytes = GGML_CUDA_AR_MAX_BYTES;
    const size_t host_buf_total = (size_t) GGML_CUDA_AR_POOL_SIZE * p->buf_bytes;
    for (size_t i = 0; i < n_devices; ++i) {
        if (p->host_buf[i].alloc(host_buf_total) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for staging failed (%zu bytes)\n",
                           __func__, host_buf_total);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Copy-engine path: pinned host staging + device scratch, sized for the
    // largest tensor we accept on this path (GGML_CUDA_AR_COPY_MAX_BYTES).
    // dev_tmp is single-buffered; cross-AR safety is enforced by an explicit
    // cross-stream wait in copy_impl on the prior AR's add_kernel-done event.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (p->host_large[i].alloc(p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for large staging failed (%zu bytes)\n",
                           __func__, p->copy_bytes);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&p->dev_tmp[i]), p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaMalloc for copy scratch failed (%zu bytes) on device %d\n",
                           __func__, p->copy_bytes, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    GGML_LOG_INFO("%s: initialized AllReduce pipeline: %zu GPUs, "
                  "%zu KB chunked kernel staging + %zu MB copy-engine staging per GPU, "
                  "copy-engine H2D streams per GPU: %d\n",
                  __func__, n_devices, p->buf_bytes >> 10, p->copy_bytes >> 20, split_h2d ? 2 : 1);

    return p;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * p) {
    if (!p) {
        return;
    }

    // Drain all in-flight kernels before tearing down resources.
    for (int i = 0; i < p->n_devices; ++i) {
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaStreamSynchronize(p->streams[i]));
            if (p->streams_h2d[i] && p->streams_h2d[i] != p->streams[i]) {
                CUDA_CHECK(cudaStreamSynchronize(p->streams_h2d[i]));
            }
        }
    }

    for (int i = 0; i < p->n_devices; ++i) {
        p->host_buf[i].free();
        p->host_large[i].free();
        if (p->dev_tmp[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaFree(p->dev_tmp[i]));
        }
        ggml_cuda_set_device(p->devices[i]);
        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            if (p->ev_pool[i][s].app) { CUDA_CHECK(cudaEventDestroy(p->ev_pool[i][s].app)); }
            for (int c = 0; c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                if (p->ev_pool[i][s].cpy[c]) { CUDA_CHECK(cudaEventDestroy(p->ev_pool[i][s].cpy[c])); }
            }
            if (p->ev_pool[i][s].h2d) { CUDA_CHECK(cudaEventDestroy(p->ev_pool[i][s].h2d)); }
            if (p->ev_pool[i][s].ker) { CUDA_CHECK(cudaEventDestroy(p->ev_pool[i][s].ker)); }
        }
        if (p->host_large_read_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventDestroy(p->host_large_read_done[i]));
        }
        if (p->dev_tmp_kernel_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventDestroy(p->dev_tmp_kernel_done[i]));
        }
        if (p->streams_h2d[i] && p->streams_h2d[i] != p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaStreamDestroy(p->streams_h2d[i]));
        }
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaStreamDestroy(p->streams[i]));
        }
    }
    p->arrival.free();
    delete p;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

// Asymmetric copy_impl: data sent over PCIe in T_src precision (one element of
// nbytes per ne element); accumulated locally into a T_dst buffer.  When
// T_src == T_dst this is the original homogeneous reduction.  When they differ
// (e.g. BF16 wire / F32 accumulator) the add kernel rounds dst through T_src
// for bit-equivalence between GPUs and we skip the otherwise-needed
// post-conversion entirely.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_impl(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne,
        size_t                  nbytes) {
    GGML_ASSERT(p->n_devices == 2);
    GGML_ASSERT(nbytes <= p->copy_bytes);
    GGML_ASSERT(ne <= std::numeric_limits<int>::max());

    const size_t chunk_bytes = ggml_cuda_ar_chunk_bytes(p, nbytes);
    GGML_ASSERT(chunk_bytes > 0);

    const int slot = ggml_cuda_ar_acquire_slot(p).slot;
    const size_t copy_chunks = (nbytes + chunk_bytes - 1) / chunk_bytes;
    GGML_ASSERT(copy_chunks <= GGML_CUDA_AR_COPY_MAX_CHUNKS);

    ggml_backend_cuda_context * cuda_ctx[2] = {};

    // Stage 1: both GPUs copy their local contribution to pinned host memory.
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);

        ggml_cuda_ar_wait_for_compute(p, cuda_ctx[i], i, slot);

        // Wait for peer's H2D from our host_large[i] (recorded in the
        // previous AR's stage 2) to complete before we overwrite host_large[i].
        // host_large_read_done[peer] = peer finished reading host_large[i].
        // No-op on the first AR -- no prior record exists.
        if (p->host_large_read_done_valid) {
            const int peer = 1 - i;
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->host_large_read_done[peer]));
        }

        if (!compute[i]) {
            CUDA_CHECK(cudaMemsetAsync(src_buf[i], 0, nbytes, p->streams[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaMemcpyAsync(
                p->host_large[i].host + offset, reinterpret_cast<char *>(src_buf[i]) + offset, this_bytes,
                cudaMemcpyDeviceToHost, p->streams[i]));
            CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].cpy[c], p->streams[i]));
        }
    }

    // Stage 2: each GPU waits for each peer D2H chunk, pulls that chunk back to
    // local device scratch (dev_tmp), then performs one device-local add over
    // the assembled peer tensor.  The H2Ds run on streams_h2d[i] (copy engine;
    // the D2H stream streams[i] when GGML_CUDA_AR_COPY_STREAMS=1) and the
    // add_kernel runs on the caller's compute stream, so the copy streams stay
    // pure-copy and avoid an in-stream copy->compute engine switch every AR.
    // dev_tmp is single-buffered: the H2D stream waits cross-stream on the
    // prior AR's add_kernel-done event before overwriting it.
    //
    // With a separate H2D stream, stream order no longer places our own D2H
    // before the add_kernel, so the compute stream also waits on our last D2H
    // chunk event: src_buf may alias dst_buf (in-place add) or be pool memory
    // that later compute-stream work reuses.
    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);
        const cudaStream_t h2d_stream = p->streams_h2d[i];

        // Wait for the previous AR's add_kernel (on the compute stream) to
        // finish reading dev_tmp before our H2D overwrites it.  No-op on the
        // first copy_impl call.
        if (p->dev_tmp_kernel_done_valid) {
            CUDA_CHECK(cudaStreamWaitEvent(h2d_stream, p->dev_tmp_kernel_done[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaStreamWaitEvent(h2d_stream, p->ev_pool[peer][slot].cpy[c]));
            CUDA_CHECK(cudaMemcpyAsync(
                p->dev_tmp[i] + offset, p->host_large[peer].host + offset, this_bytes,
                cudaMemcpyHostToDevice, h2d_stream));
        }

        // Mark our reads of host_large[peer] complete so peer's next AR can
        // safely overwrite it.
        CUDA_CHECK(cudaEventRecord(p->host_large_read_done[i], h2d_stream));

        // Hand off from the copy streams to the compute stream: compute stream
        // waits for all H2Ds (and our own D2H, see above), then runs the add_kernel.
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].h2d, h2d_stream));
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->ev_pool[i][slot].h2d));
        if (h2d_stream != p->streams[i]) {
            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->ev_pool[i][slot].cpy[copy_chunks - 1]));
        }

        const int block_size = 256;
        int n_blocks = (int) ((ne + block_size - 1) / block_size);
        if (n_blocks > 1024) {
            n_blocks = 1024;
        }
        ggml_cuda_ar_add_kernel<T_dst, T_src><<<n_blocks, block_size, 0, cuda_ctx[i]->stream()>>>(
            dst_buf[i],
            reinterpret_cast<const T_src *>(p->dev_tmp[i]),
            (int) ne);
        CUDA_CHECK(cudaGetLastError());

        // Record dev_tmp-released on the compute stream so the next copy_impl
        // can wait for the kernel to finish before overwriting dev_tmp.  Also
        // record AR-done as ev.ker for acquire_slot's pool-wraparound sync.
        CUDA_CHECK(cudaEventRecord(p->dev_tmp_kernel_done[i], cuda_ctx[i]->stream()));
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, cuda_ctx[i]->stream()));
    }
    p->host_large_read_done_valid = true;
    p->dev_tmp_kernel_done_valid = true;

    return true;
}

// Outer-level chunker: copy_impl handles up to copy_bytes per call (limited by
// the host_large / dev_tmp allocation size).  When the full AR exceeds that,
// slice the tensor into copy_bytes-sized pieces and call copy_impl repeatedly.
// Each slice goes through its own stage 1 -> stage 2 cycle and acquires its own
// slot, so cross-AR fences and pool wraparound work the same way as for any
// other sequence of small ARs.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_outer(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne) {
    const int64_t outer_max_elems = (int64_t) (p->copy_bytes / sizeof(T_src));
    GGML_ASSERT(outer_max_elems > 0);

    bool ok = true;
    for (int64_t outer_start = 0; outer_start < ne && ok; outer_start += outer_max_elems) {
        const int64_t outer_ne     = std::min(outer_max_elems, ne - outer_start);
        const size_t  outer_nbytes = (size_t) outer_ne * sizeof(T_src);

        T_src * src[GGML_CUDA_MAX_DEVICES] = {};
        T_dst * dst[GGML_CUDA_MAX_DEVICES] = {};
        for (int i = 0; i < p->n_devices; ++i) {
            src[i] = src_buf[i] + outer_start;
            dst[i] = dst_buf[i] + outer_start;
        }
        ok = ggml_cuda_ar_allreduce_copy_impl<T_src, T_dst>(
            p, backends, src, dst, compute, outer_ne, outer_nbytes);
    }
    return ok;
}

// Threads per block of the chunked kernel.  Auto: 128 threads (a round of 16 KB over
// 8 blocks) for the decode-sized reductions this box serves, 256 above (G: 128 gives
// twice the rounds for the 6-18 row decode payloads of a Gen3 x8 link; measured as its own arm).
static int ggml_cuda_ar_kernel_threads(const ggml_cuda_ar_pipeline * p, size_t wire_bytes) {
    if (p->round_threads > 0) {
        return p->round_threads;
    }
    return wire_bytes <= 256 * 1024 ? 128 : GGML_CUDA_AR_KERNEL_THREADS;
}

// Vectors per thread per round.  Raised when the launch would need more rounds than the
// arrival ints provide.
template <typename T_wire>
static int ggml_cuda_ar_kernel_round_vecs(const ggml_cuda_ar_pipeline * p, size_t chunk_elems, int threads) {
    const size_t count_vec   = chunk_elems / (16 / sizeof(T_wire));
    const size_t gnt         = (size_t) GGML_CUDA_AR_KERNEL_BLOCKS * threads;
    const size_t total_iters = std::max<size_t>(1, (count_vec + gnt - 1) / gnt);
    size_t per = p->round_vecs > 0 ? (size_t) p->round_vecs : total_iters;
    while ((total_iters + per - 1) / per > (size_t) GGML_CUDA_AR_MAX_ROUNDS) {
        per *= 2;
    }
    return (int) std::min(per, total_iters);
}

bool ggml_cuda_ar_allreduce(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors) {
    GGML_ASSERT(p != nullptr);
    ggml_round_trace_count("ar_call");

    const int n = p->n_devices;
    GGML_ASSERT(n == 2);

    const ggml_type input_type = tensors[0]->type;
    GGML_ASSERT(input_type == GGML_TYPE_F32 || input_type == GGML_TYPE_F16 || input_type == GGML_TYPE_BF16);

    const int64_t ne = ggml_nelements(tensors[0]);
    GGML_ASSERT(ne > 0);

    const size_t   input_nbytes = ggml_nbytes(tensors[0]);

    // BF16 round-trip: F32 inputs >= bf16_threshold are converted to BF16 for
    // the reduction (chunked or copy-engine), halving on-wire bytes. Matches
    // NCCL's behaviour. The pre-conversion zeroes inactive shards so the
    // inner paths see them as already-prepared compute tensors.
    const bool use_bf16 =
        input_type == GGML_TYPE_F32 &&
        p->bf16_threshold > 0 &&
        input_nbytes >= p->bf16_threshold;

    const ggml_type kernel_type = use_bf16 ? GGML_TYPE_BF16 : input_type;
    const size_t    type_size   = ggml_type_size(kernel_type);
    GGML_ASSERT(p->buf_bytes >= type_size);
    const size_t    nbytes      = (size_t) ne * type_size;

    bool compute_flag[GGML_CUDA_MAX_DEVICES] = {};
    for (int i = 0; i < n; ++i) {
        compute_flag[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    }

    // Decide between copy-engine and chunked kernel paths based on the working
    // type's actual byte count.  No upper bound: copy_outer slices reductions
    // larger than copy_bytes into copy_bytes-sized pieces.
    const bool use_copy_engine =
        p->copy_threshold > 0 &&
        nbytes >= p->copy_threshold;

    // BF16 inactive-shard zeroing: when use_bf16 is on, the combined kernel
    // (chunked kernel path) and the combined add kernel (copy_engine path)
    // both accumulate into the F32 tensor data directly, so an inactive
    // shard's accumulator must start at zero.
    if (use_bf16) {
        for (int i = 0; i < n; ++i) {
            if (!compute_flag[i]) {
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                ggml_cuda_set_device(p->devices[i]);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, (size_t) ne * sizeof(float), cuda_ctx->stream()));
            }
        }
    }

    // Pre-convert F32 -> BF16 into bf16_tmp ONLY for the copy_engine + use_bf16
    // path; the chunked kernel path's combined kernel does the conversion
    // inline as it writes to host_buf.
    ggml_cuda_pool_alloc<nv_bfloat16> bf16_tmp[GGML_CUDA_MAX_DEVICES];
    void * copy_src_ptr[GGML_CUDA_MAX_DEVICES] = {};

    if (use_copy_engine && use_bf16) {
        to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
        for (int i = 0; i < n; ++i) {
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            bf16_tmp[i].pool = &cuda_ctx->pool();
            bf16_tmp[i].alloc(ne);
            ggml_cuda_set_device(p->devices[i]);
            if (compute_flag[i]) {
                to_bf16(tensors[i]->data, bf16_tmp[i].get(), ne, cuda_ctx->stream());
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemsetAsync(bf16_tmp[i].get(), 0, nbytes, cuda_ctx->stream()));
            }
            copy_src_ptr[i] = bf16_tmp[i].get();
        }
    }

    bool ok = true;
    if (use_copy_engine) {
        // After up-front BF16 conversion, the tmp buffers already hold the
        // (possibly zeroed-for-inactive) data, so the inner path can treat
        // every shard as compute.
        bool inner_compute[GGML_CUDA_MAX_DEVICES];
        for (int i = 0; i < n; ++i) {
            inner_compute[i] = use_bf16 ? true : compute_flag[i];
        }

        // Dispatch into copy_impl with explicit src/dst types.  When use_bf16
        // is on, the wire type is BF16 (src = bf16_tmp) and the accumulator
        // is F32 (dst = tensors[i]->data); the combined add kernel rounds dst
        // through BF16 for bit-equivalence and writes F32 directly, so no
        // post-conversion is needed.  Otherwise src == dst (same native type).
        if (use_bf16) {
            GGML_ASSERT(kernel_type == GGML_TYPE_BF16);
            nv_bfloat16 * src[GGML_CUDA_MAX_DEVICES] = {};
            float       * dst[GGML_CUDA_MAX_DEVICES] = {};
            for (int i = 0; i < n; ++i) {
                src[i] = static_cast<nv_bfloat16 *>(copy_src_ptr[i]);
                dst[i] = static_cast<float *>(tensors[i]->data);
            }
            ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, float>(
                p, backends, src, dst, inner_compute, ne);
        } else {
            switch (kernel_type) {
                case GGML_TYPE_F32: {
                    float * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<float *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<float, float>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_BF16: {
                    nv_bfloat16 * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<nv_bfloat16 *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, nv_bfloat16>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_F16: {
                    half * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<half *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<half, half>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                default:
                    GGML_ASSERT(false);
            }
        }
    } else {
        // host_buf carries T_wire-typed data; max_chunk_elems is the count that
        // fits in one host_buf at the wire size.
        const size_t max_chunk_elems = p->buf_bytes / type_size;
        const size_t input_type_size = ggml_type_size(input_type);

        // Chunked kernel path runs entirely on the caller's compute stream:
        // since AR is a barrier here, same-stream ordering subsumes any
        // cross-stream event handshake that the copy-engine path needs, and
        // skips the cross-stream scheduling overhead that was hurting the
        // small-tensor (tg) latency on the AR-stream variant.  Only ev.ker is
        // still recorded at end-of-AR for acquire_slot's pool-wraparound check.
        for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
            const size_t remaining_elems = (size_t) (ne - chunk_start);
            const size_t chunk_elems = remaining_elems < max_chunk_elems ? remaining_elems : max_chunk_elems;
            const size_t chunk_dst_bytes  = chunk_elems * input_type_size;

            const auto [slot, token] = ggml_cuda_ar_acquire_slot(p);
            const bool last_chunk = chunk_start + (int64_t) chunk_elems == ne;

            for (int i = 0; i < n; ++i) {
                const int peer = 1 - i;  // valid for n == 2 only
                ggml_cuda_set_device(p->devices[i]);
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                cudaStream_t stream = cuda_ctx->stream();

                char * data = static_cast<char *>(tensors[i]->data) + chunk_start * (int64_t) input_type_size;

                // Match NCCL/meta-backend semantics: inactive shards contribute
                // zeros.  On the BF16 path the F32 tensor data was already
                // zeroed up-front (above), so per-chunk zeroing isn't needed.
                if (!compute_flag[i] && !use_bf16) {
                    CUDA_CHECK(cudaMemsetAsync(data, 0, chunk_dst_bytes, stream));
                }

#define LAUNCH_AR_KERNEL(T_dst, T_wire) \
                do { \
                    const int ar_threads = ggml_cuda_ar_kernel_threads(p, chunk_elems * sizeof(T_wire)); \
                    ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(ar_threads), 0, stream>>>( \
                        reinterpret_cast<const T_dst *>(data), \
                        reinterpret_cast<T_dst *>(data), \
                        reinterpret_cast<T_wire *>(p->host_buf[i].dev + (size_t) slot * p->buf_bytes), \
                        reinterpret_cast<const T_wire *>(p->host_buf[peer].dev + (size_t) slot * p->buf_bytes), \
                        static_cast<int>(chunk_elems), \
                        ggml_cuda_ar_arrival_ptr(p, slot, i), \
                        ggml_cuda_ar_arrival_ptr(p, slot, peer), \
                        token, \
                        ggml_cuda_ar_kernel_round_vecs<T_wire>(p, chunk_elems, ar_threads), \
                        p->mutant, \
                        i == 1 ? p->stall_cycles : 0); \
                } while (0)

                if (use_bf16) {
                    GGML_ASSERT(input_type == GGML_TYPE_F32);
                    LAUNCH_AR_KERNEL(float, nv_bfloat16);
                } else {
                    switch (input_type) {
                        case GGML_TYPE_F32:  LAUNCH_AR_KERNEL(float,       float);       break;
                        case GGML_TYPE_F16:  LAUNCH_AR_KERNEL(half,        half);        break;
                        case GGML_TYPE_BF16: LAUNCH_AR_KERNEL(nv_bfloat16, nv_bfloat16); break;
                        default: GGML_ASSERT(false);
                    }
                }

#undef LAUNCH_AR_KERNEL
                CUDA_CHECK(cudaGetLastError());

                if (last_chunk) {
                    CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, stream));
                }
            }
        }
    }

    return ok;
}

#else // defined(GGML_USE_MUSA)

// MUSA lacks the host-mapped pinned-memory APIs (cudaHostAllocPortable
// / cudaHostAllocMapped / cudaHostGetDevicePointer) and a device-side
// sleep intrinsic that this implementation relies on, so the internal
// AllReduce is unavailable there. The dispatcher in ggml-cuda.cu treats
// a nullptr pipeline as "init failed" and silently falls back to the meta
// backend's generic AllReduce.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int *, size_t) {
    return nullptr;
}
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline *) {
}
bool ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **) {
    return false;
}

#endif // !defined(GGML_USE_MUSA)
