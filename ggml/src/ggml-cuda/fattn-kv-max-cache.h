#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Shares the flash_attn_mask_to_KV_max scan between the attention ops of one graph evaluation
// that read the same mask (one scan per mask instead of one per layer).
//
// Contract:
// - The entries are valid only between ggml_cuda_fattn_kv_max_cache::reserve() and the end of the
//   node loop that owns the cache; the owner sets ggml_backend_cuda_context::fattn_kv_max_cache for
//   exactly that span and runs every node on the context's main stream.
// - data is one scratch allocation made by the owner before the node loop and held until after it, so
//   the slots are stable through CUDA graph capture and every replay of the captured graph.
// - A slot is valid only while the mask bytes are unchanged: the owner calls invalidate() with every
//   node it executes, and the next user rescans into the same slot.
// - A slot serves one ncols1 (the scan output depends on it); a user with another ncols1 scans locally.

#define GGML_CUDA_FATTN_KV_MAX_STRIDE 256


// Mirrors the dense scan condition in launch_fattn. The sparse path is decided by the kernel
// selection and is not known here; its slots stay unused.
inline bool ggml_cuda_fattn_kv_max_scan_wanted(const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * mask) {
    return mask && mask->type == GGML_TYPE_F16 && K->ne[1] % GGML_CUDA_FATTN_KV_MAX_STRIDE == 0 &&
        (Q->ne[1] >= 1024 || Q->ne[3] > 1);
}

struct ggml_cuda_fattn_kv_max_cache {
    struct group {
        const char * mask = nullptr;
        size_t mask_bytes = 0;
        int64_t mask_ne1 = 0;
        size_t  mask_nb1 = 0;
        size_t  mask_nb3 = 0;
        int64_t nq = 0;      // Q->ne[1]
        int64_t ns = 0;      // Q->ne[3]
        int64_t iter_k = 0;  // K->ne[1] / stride
        size_t offset = 0;   // in ints
        size_t size = 0;     // in ints; sized for ncols1 == 1, the largest output
        int uses = 0;
        int ncols1 = 0;
        bool valid = false;
    };

    std::vector<group> groups;
    int * data = nullptr;

    static constexpr size_t max_groups = 8;
    static constexpr size_t max_ints   = size_t(1) << 20;

    static bool same(const group & g, const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * mask) {
        return g.mask == (const char *) mask->data && g.mask_ne1 == mask->ne[1] &&
            g.mask_nb1 == mask->nb[1] && g.mask_nb3 == mask->nb[3] &&
            g.nq == Q->ne[1] && g.ns == Q->ne[3] && g.iter_k == K->ne[1] / GGML_CUDA_FATTN_KV_MAX_STRIDE;
    }

    void add(const ggml_tensor * fa) {
        const ggml_tensor * Q = fa->src[0];
        const ggml_tensor * K = fa->src[1];
        const ggml_tensor * mask = fa->src[3];
        if (!ggml_cuda_fattn_kv_max_scan_wanted(Q, K, mask)) {
            return;
        }
        for (auto & g : groups) {
            if (same(g, Q, K, mask)) {
                ++g.uses;
                return;
            }
        }
        if (groups.size() >= max_groups) {
            return;
        }
        group g;
        g.mask = (const char *) mask->data;
        g.mask_bytes = ggml_nbytes(mask);
        g.mask_ne1 = mask->ne[1];
        g.mask_nb1 = mask->nb[1];
        g.mask_nb3 = mask->nb[3];
        g.nq = Q->ne[1];
        g.ns = Q->ne[3];
        g.iter_k = K->ne[1] / GGML_CUDA_FATTN_KV_MAX_STRIDE;
        g.uses = 1;
        groups.push_back(g);
    }

    // Returns the scratch size in ints; drops the groups with a single user.
    size_t reserve() {
        size_t total = 0;
        std::vector<group> kept;
        for (auto g : groups) {
            const size_t n = size_t(g.nq) * size_t(g.ns);
            if (g.uses < 2 || n == 0 || n > max_ints - total) {
                continue;
            }
            g.offset = total;
            g.size = (n + 63) / 64 * 64;
            total += g.size;
            kept.push_back(g);
        }
        groups.swap(kept);
        return total;
    }

    // Returns the slot for this attention op, or nullptr when it must scan locally. scan is set when
    // the caller has to launch the scan into the slot.
    int * acquire(const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * mask, int ncols1, bool & scan) {
        scan = false;
        if (!data) {
            return nullptr;
        }
        for (auto & g : groups) {
            if (!same(g, Q, K, mask)) {
                continue;
            }
            if (g.ncols1 != 0 && g.ncols1 != ncols1) {
                return nullptr;
            }
            g.ncols1 = ncols1;
            scan = !g.valid;
            g.valid = true;
            return data + g.offset;
        }
        return nullptr;
    }

    void invalidate(const ggml_tensor * dst) {
        const uintptr_t start = (uintptr_t) dst->data;
        const size_t size = ggml_nbytes(dst);
        for (auto & g : groups) {
            const uintptr_t mask = (uintptr_t) g.mask;
            if (start <= mask ? mask - start < size : start - mask < g.mask_bytes) {
                g.valid = false;
            }
        }
    }
};
