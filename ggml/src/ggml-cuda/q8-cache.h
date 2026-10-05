#pragma once

#include "ggml.h"

#include <array>
#include <cstdint>
#include <unordered_map>

struct ggml_cuda_q8_cache {
    struct entry {
        size_t size = 0;
        size_t offset = 0;
        int uses = 0;
        bool valid = false;
    };

    // Variant 0 is MMVQ; variants 1..3 are the MMQ scale/sum layouts.
    std::unordered_map<const ggml_tensor *, std::array<entry, 4>> entries;
    char * data = nullptr;
    int rows = 0;
    int hits = 0;
    int misses = 0;

    void add(const ggml_tensor * src, int variant, size_t size) {
        rows = rows > ggml_nrows(src) ? rows : ggml_nrows(src);
        auto & e = entries[src][variant];
        e.size = e.size > size ? e.size : size;
        ++e.uses;
    }

    size_t reserve() {
        size_t size = 0;
        for (auto it = entries.begin(); it != entries.end();) {
            bool keep = false;
            for (auto & e : it->second) {
                if (e.uses < 2) {
                    e.size = 0;
                    continue;
                }
                // Bound retained scratch space; other inputs use the ordinary pool path.
                constexpr size_t limit = 64 * 1024 * 1024;
                if (e.size > limit - size) {
                    e.size = 0;
                    continue;
                }
                e.offset = size;
                size += (e.size + 127) / 128 * 128;
                keep = true;
            }
            if (keep) {
                ++it;
            } else {
                it = entries.erase(it);
            }
        }
        return size;
    }

    char * get(const ggml_tensor * src, int variant, size_t size, bool & hit) {
        hit = false;
        const auto it = entries.find(src);
        if (!data || it == entries.end()) {
            return nullptr;
        }
        auto & e = it->second[variant];
        if (size > e.size || e.size == 0) {
            return nullptr;
        }
        hit = e.valid;
        e.valid = true;
        hits += hit;
        misses += !hit;
        return data + e.offset;
    }

    void invalidate(const ggml_tensor * dst) {
        const uintptr_t start = (uintptr_t) dst->data;
        const size_t size = ggml_nbytes(dst);
        for (auto & kv : entries) {
            const uintptr_t src = (uintptr_t) kv.first->data;
            const size_t src_size = ggml_nbytes(kv.first);
            if (start <= src ? src - start < size : start - src < src_size) {
                for (auto & e : kv.second) {
                    e.valid = false;
                }
            }
        }
    }
};
