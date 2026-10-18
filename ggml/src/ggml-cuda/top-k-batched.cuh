#pragma once

// Batched exact top-k over F32 rows: every row in one launch set instead of one CUB call chain per row.
//
// Contract (stronger than GGML_OP_TOP_K's "no particular order"):
//  - Keys order by the IEEE bit pattern (-NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN), the order CUB's
//    DeviceTopK uses, so the selected multiset of values is the one CUB selects.
//  - Row output is sorted by key descending, then column ascending: ties resolve to the lowest columns and the
//    result does not depend on thread scheduling. Without a tie at the k-th key the column set is unique.
//
// Rows with ncols <= TOPK_SMALL_NCOLS take one block each in a single launch. Wider rows use a 4-pass 8-bit radix
// select (histogram + select per pass), a gather and a per-row finalize: 10 launches for any number of rows.

#include <algorithm>
#include <cstdint>
#include <climits>

#define GGML_TOPK_SHARED __shared__
#define GGML_TOPK_LAUNCH(kernel, grid, block, stream, ...) kernel<<<(grid), (block), 0, (stream)>>>(__VA_ARGS__)
typedef cudaStream_t topk_stream_t;

constexpr int TOPK_BLOCK                = 256;
constexpr int TOPK_NBINS                = 256;  // one bin per thread: the histogram and the scans index by threadIdx.x
constexpr int TOPK_MAX_K                = 1024;
constexpr int TOPK_MAX_BLOCKS_PER_ROW   = 64;
constexpr int TOPK_SMALL_NCOLS          = 8192;
#ifndef GGML_TOPK_SCRATCH_BYTES
#    define GGML_TOPK_SCRATCH_BYTES (32 << 20)
#endif
constexpr int TOPK_SCRATCH_BYTES_BUDGET = GGML_TOPK_SCRATCH_BYTES;

static_assert(TOPK_BLOCK == TOPK_NBINS, "a thread owns one histogram bin");

struct topk_row_state {
    uint32_t prefix;     // decided high bits of the k-th key
    uint32_t mask;
    int      rank;       // elements still wanted among those matching prefix (>= 1)
    int      ambiguous;  // more elements tie with the k-th key than rank: the finalize picks the lowest columns
    int      n_greater;
    int      n_equal;
};

static __device__ __forceinline__ uint32_t topk_key(float v) {
    const uint32_t b = __float_as_uint(v);
    return b ^ ((b >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}

static __device__ __forceinline__ bool topk_before(uint32_t ka, int ia, uint32_t kb, int ib) {
    return ka > kb || (ka == kb && ia < ib);
}

// Inclusive suffix sum of one count per thread over s[TOPK_NBINS]. Exactly one thread, the one owning the largest
// bin whose suffix sum reaches rank, gets true; above is the sum over the bins above it. Ends with a barrier.
static __device__ __forceinline__ bool topk_pick_bin(int * s, int count, int rank, int & above) {
    const int tid = threadIdx.x;
    s[tid] = count;
    __syncthreads();
    for (int off = 1; off < TOPK_NBINS; off <<= 1) {
        const int v = tid + off < TOPK_NBINS ? s[tid + off] : 0;
        __syncthreads();
        s[tid] += v;
        __syncthreads();
    }
    above = tid + 1 < TOPK_NBINS ? s[tid + 1] : 0;
    return s[tid] >= rank && above < rank;
}

// out[0..rank) = columns of the first rank elements of the row whose key equals prefix, in column order.
static __device__ void topk_first_equal(const float * row, int ncols, uint32_t prefix, int rank, int * out, int * s) {
    const int tid   = threadIdx.x;
    int       taken = 0;
    for (int base = 0; base < ncols && taken < rank; base += TOPK_BLOCK) {
        const int col = base + tid;
        const int eq  = col < ncols && topk_key(row[col]) == prefix;
        s[tid]        = eq;
        __syncthreads();
        for (int off = 1; off < TOPK_BLOCK; off <<= 1) {
            const int v = tid >= off ? s[tid - off] : 0;
            __syncthreads();
            s[tid] += v;
            __syncthreads();
        }
        const int pos = taken + s[tid] - eq;
        if (eq && pos < rank) {
            out[pos] = col;
        }
        taken += s[TOPK_BLOCK - 1];
        __syncthreads();
    }
}

// idx[0..k) holds the k selected columns of the row, unordered. Writes them to dst sorted by (key desc, column asc).
// p is the next power of two >= k; padding sorts last.
static __device__ void topk_sort_write(const float * row, int ncols, int * idx, uint32_t * key, int * dst, int k, int p) {
    const int tid = threadIdx.x;
    __syncthreads();
    for (int i = tid; i < p; i += TOPK_BLOCK) {
        if (i < k) {
            // a column outside the row can only come from a corrupted state; do not read through it
            key[i] = (unsigned) idx[i] < (unsigned) ncols ? topk_key(row[idx[i]]) : 0;
        } else {
            key[i] = 0;
            idx[i] = INT_MAX;
        }
    }
    __syncthreads();
    for (int size = 2; size <= p; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < p; i += TOPK_BLOCK) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool swap = (i & size) == 0 ? topk_before(key[j], idx[j], key[i], idx[i])
                                                       : topk_before(key[i], idx[i], key[j], idx[j]);
                    if (swap) {
                        const uint32_t tk = key[i];
                        const int      ti = idx[i];
                        key[i] = key[j];
                        idx[i] = idx[j];
                        key[j] = tk;
                        idx[j] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int i = tid; i < k; i += TOPK_BLOCK) {
        dst[i] = idx[i];
    }
}

// One block per row, rows narrow enough that a block scans them in a few passes from cache.
static __global__ void topk_small(const float * __restrict__ src, int * dst, int ncols, int k, int p) {
    GGML_TOPK_SHARED int      shist[TOPK_NBINS];
    GGML_TOPK_SHARED int      sscan[TOPK_BLOCK];
    GGML_TOPK_SHARED int      sidx[TOPK_MAX_K];
    GGML_TOPK_SHARED uint32_t skey[TOPK_MAX_K];
    GGML_TOPK_SHARED int      sbin[3];
    GGML_TOPK_SHARED int      scount[2];

    const int     tid  = threadIdx.x;
    const float * row  = src + (size_t) blockIdx.x * ncols;
    int *         out  = dst + (size_t) blockIdx.x * k;
    uint32_t      prefix = 0;
    uint32_t      mask   = 0;
    int           rank   = k;
    bool          ambiguous = false;

    for (int shift = 24; shift >= 0; shift -= 8) {
        shist[tid] = 0;
        __syncthreads();
        for (int col = tid; col < ncols; col += TOPK_BLOCK) {
            const uint32_t key = topk_key(row[col]);
            if ((key & mask) == prefix) {
                atomicAdd(&shist[(key >> shift) & (TOPK_NBINS - 1)], 1);
            }
        }
        __syncthreads();
        const int count = shist[tid];
        int       above = 0;
        if (topk_pick_bin(sscan, count, rank, above)) {
            sbin[0] = tid;
            sbin[1] = above;
            sbin[2] = count;
        }
        __syncthreads();
        prefix |= (uint32_t) sbin[0] << shift;
        mask |= (uint32_t) (TOPK_NBINS - 1) << shift;
        rank -= sbin[1];
        ambiguous = sbin[2] > rank;
        __syncthreads();
    }

    if (tid == 0) {
        scount[0] = 0;
        scount[1] = 0;
    }
    __syncthreads();
    for (int col = tid; col < ncols; col += TOPK_BLOCK) {
        const uint32_t key = topk_key(row[col]);
        if (key > prefix) {
            const int pos = atomicAdd(&scount[0], 1);
            if (pos < k) {
                sidx[pos] = col;
            }
        } else if (key == prefix && !ambiguous) {
            const int pos = k - rank + atomicAdd(&scount[1], 1);
            if (pos < k) {
                sidx[pos] = col;
            }
        }
    }
    __syncthreads();
    if (ambiguous) {
        topk_first_equal(row, ncols, prefix, rank, sidx + (k - rank), sscan);
    }
    topk_sort_write(row, ncols, sidx, skey, out, k, p);
}

static __global__ void topk_hist(
        const float * __restrict__ src, const topk_row_state * __restrict__ states, int * __restrict__ hist,
        int ncols, int blocks_per_row, int shift) {
    GGML_TOPK_SHARED int h[TOPK_NBINS];

    const int     row  = blockIdx.x / blocks_per_row;
    const int     rb   = blockIdx.x % blocks_per_row;
    const int     tid  = threadIdx.x;
    const float * rsrc = src + (size_t) row * ncols;

    uint32_t prefix = 0;
    uint32_t mask   = 0;
    if (shift != 24) {
        prefix = states[row].prefix;
        mask   = states[row].mask;
    }

    h[tid] = 0;
    __syncthreads();
    for (int col = rb * TOPK_BLOCK + tid; col < ncols; col += blocks_per_row * TOPK_BLOCK) {
        const uint32_t key = topk_key(rsrc[col]);
        if ((key & mask) == prefix) {
            atomicAdd(&h[(key >> shift) & (TOPK_NBINS - 1)], 1);
        }
    }
    __syncthreads();
    hist[((size_t) row * blocks_per_row + rb) * TOPK_NBINS + tid] = h[tid];
}

// Also resets the gather counters, so no init or reset launch exists.
static __global__ void topk_select(
        const int * __restrict__ hist, topk_row_state * __restrict__ states, int blocks_per_row, int shift, int k) {
    GGML_TOPK_SHARED int s[TOPK_NBINS];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // every thread reads the state before the first barrier; the owning thread writes it after the last one
    topk_row_state cur = { 0, 0, k, 0, 0, 0 };
    if (shift != 24) {
        cur = states[row];
    }

    int count = 0;
    for (int rb = 0; rb < blocks_per_row; ++rb) {
        count += hist[((size_t) row * blocks_per_row + rb) * TOPK_NBINS + tid];
    }

    int above = 0;
    if (topk_pick_bin(s, count, cur.rank, above)) {
        cur.prefix |= (uint32_t) tid << shift;
        cur.mask |= (uint32_t) (TOPK_NBINS - 1) << shift;
        cur.rank -= above;
        cur.ambiguous = count > cur.rank;
        cur.n_greater = 0;
        cur.n_equal   = 0;
        states[row]   = cur;
    }
}

static __global__ void topk_gather(
        const float * __restrict__ src, int * __restrict__ dst, topk_row_state * __restrict__ states,
        int ncols, int k, int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int rb  = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;

    const float *         rsrc = src + (size_t) row * ncols;
    int *                 rdst = dst + (size_t) row * k;
    topk_row_state *      st   = &states[row];
    const uint32_t        prefix    = st->prefix;
    const int             rank      = st->rank;
    const int             ambiguous = st->ambiguous;

    for (int col = rb * TOPK_BLOCK + tid; col < ncols; col += blocks_per_row * TOPK_BLOCK) {
        const uint32_t key = topk_key(rsrc[col]);
        if (key > prefix) {
            const int pos = atomicAdd(&st->n_greater, 1);
            if (pos < k) {
                rdst[pos] = col;
            }
        } else if (key == prefix && !ambiguous) {
            const int pos = k - rank + atomicAdd(&st->n_equal, 1);
            if (pos < k) {
                rdst[pos] = col;
            }
        }
    }
}

static __global__ void topk_finalize(
        const float * __restrict__ src, int * dst, const topk_row_state * __restrict__ states, int ncols, int k, int p) {
    GGML_TOPK_SHARED int      sscan[TOPK_BLOCK];
    GGML_TOPK_SHARED int      sidx[TOPK_MAX_K];
    GGML_TOPK_SHARED uint32_t skey[TOPK_MAX_K];

    const int             tid  = threadIdx.x;
    const float *         row  = src + (size_t) blockIdx.x * ncols;
    int *                 out  = dst + (size_t) blockIdx.x * k;
    const topk_row_state  st   = states[blockIdx.x];

    // the last st.rank slots are unwritten when ambiguous
    const int n_have = st.ambiguous ? k - st.rank : k;
    for (int i = tid; i < n_have; i += TOPK_BLOCK) {
        sidx[i] = out[i];
    }
    __syncthreads();
    if (st.ambiguous) {
        topk_first_equal(row, ncols, st.prefix, st.rank, sidx + n_have, sscan);
    }
    topk_sort_write(row, ncols, sidx, skey, out, k, p);
}

// Rows per scratch chunk: bounds the per-block histograms. Valid for ncols > TOPK_SMALL_NCOLS.
static int topk_batched_chunk_rows(int ncols) {
    const int bpr  = std::min((ncols + 1023) / 1024, TOPK_MAX_BLOCKS_PER_ROW);
    const int rows = TOPK_SCRATCH_BYTES_BUDGET / (bpr * TOPK_NBINS * (int) sizeof(int));
    return std::max(rows, 1);
}

// Scratch element counts for topk_batched; zero for the one-launch narrow-row path.
static void topk_batched_scratch(int ncols, int nrows, size_t & n_states, size_t & n_hist) {
    if (ncols <= TOPK_SMALL_NCOLS) {
        n_states = 0;
        n_hist   = 0;
        return;
    }
    const int bpr  = std::min((ncols + 1023) / 1024, TOPK_MAX_BLOCKS_PER_ROW);
    const int rows = std::min(topk_batched_chunk_rows(ncols), nrows);
    n_states = (size_t) rows;
    n_hist   = (size_t) rows * bpr * TOPK_NBINS;
}

static bool topk_batched_supported(int64_t ncols, int64_t nrows, int64_t k) {
    return k >= 1 && k <= TOPK_MAX_K && k <= ncols && ncols <= INT_MAX && nrows >= 1 && nrows <= INT_MAX;
}

// states and hist: sized by topk_batched_scratch; both may be null for ncols <= TOPK_SMALL_NCOLS.
static void topk_batched(
        const float * src, int * dst, topk_row_state * states, int * hist,
        int ncols, int nrows, int k, topk_stream_t stream) {
    int p = 1;
    while (p < k) {
        p <<= 1;
    }

    if (ncols <= TOPK_SMALL_NCOLS) {
        GGML_TOPK_LAUNCH(topk_small, nrows, TOPK_BLOCK, stream, src, dst, ncols, k, p);
        return;
    }

    const int bpr   = std::min((ncols + 1023) / 1024, TOPK_MAX_BLOCKS_PER_ROW);
    const int chunk = topk_batched_chunk_rows(ncols);
    for (int r0 = 0; r0 < nrows; r0 += chunk) {
        const int     n  = std::min(chunk, nrows - r0);
        const float * s  = src + (size_t) r0 * ncols;
        int *         d  = dst + (size_t) r0 * k;
        for (int shift = 24; shift >= 0; shift -= 8) {
            GGML_TOPK_LAUNCH(topk_hist, n * bpr, TOPK_BLOCK, stream, s, states, hist, ncols, bpr, shift);
            GGML_TOPK_LAUNCH(topk_select, n, TOPK_BLOCK, stream, hist, states, bpr, shift, k);
        }
        GGML_TOPK_LAUNCH(topk_gather, n * bpr, TOPK_BLOCK, stream, s, d, states, ncols, k, bpr);
        GGML_TOPK_LAUNCH(topk_finalize, n, TOPK_BLOCK, stream, s, d, states, ncols, k, p);
    }
}
