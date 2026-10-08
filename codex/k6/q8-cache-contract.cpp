// Host-only contract of ggml-cuda/q8-cache.h: eligibility bound, per-evaluation reset, size fallback,
// tensor-identity keys and alias invalidation. It says nothing about CUDA numerics.
#include "q8-cache.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool ok, const char * why) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", why); std::exit(1); }
}

static ggml_tensor shape(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, ggml_type type = GGML_TYPE_F32) {
    ggml_tensor t{};
    t.type = type;
    t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = ne3;
    t.nb[0] = ggml_type_size(type);
    t.nb[1] = t.nb[0]*ne0; t.nb[2] = t.nb[1]*ne1; t.nb[3] = t.nb[2]*ne2;
    return t;
}

int main() {
    for (int64_t rows = 1; rows <= 16; ++rows) {
        auto a = shape(768, rows, 1, 1);
        require(ggml_cuda_q8_reuse_input_ok(&a), "up to 16 rows in one dimension must be eligible");
    }
    auto four_by_four = shape(768, 4, 4, 1);
    auto two_by_two_by_four = shape(768, 2, 2, 4);
    auto seventeen = shape(768, 17, 1, 1);
    auto seventeen_channels = shape(768, 1, 17, 1);
    auto twenty_four = shape(768, 8, 3, 1);
    auto half = shape(768, 6, 1, 1, GGML_TYPE_F16);
    auto strided = shape(768, 6, 1, 1);
    strided.nb[0] = 8;
    auto empty = shape(0, 6, 1, 1);
    auto wide = shape((1 << 20) + 1, 1, 1, 1);
    auto widest = shape(1 << 20, 1, 1, 1);
    require(ggml_cuda_q8_reuse_input_ok(&four_by_four), "16 rows split over two dimensions");
    require(ggml_cuda_q8_reuse_input_ok(&two_by_two_by_four), "16 rows split over three dimensions");
    require(ggml_cuda_q8_reuse_input_ok(&widest), "widest eligible row");
    require(!ggml_cuda_q8_reuse_input_ok(&seventeen), "17 rows must fall back");
    require(!ggml_cuda_q8_reuse_input_ok(&seventeen_channels), "17 channels must fall back");
    require(!ggml_cuda_q8_reuse_input_ok(&twenty_four), "24 total rows must fall back");
    require(!ggml_cuda_q8_reuse_input_ok(&half), "non-F32 input must fall back");
    require(!ggml_cuda_q8_reuse_input_ok(&strided), "element-strided input must fall back");
    require(!ggml_cuda_q8_reuse_input_ok(&empty), "empty row must fall back");
    require(!ggml_cuda_q8_reuse_input_ok(&wide), "over-wide row must fall back");

    for (int rows : {1, 6, 8, 12, 16}) {
        std::vector<float> input(256*rows, 1.0f);
        auto src = shape(256, rows, 1, 1);
        src.data = input.data();
        for (int replay = 0; replay < 3; ++replay) {
            ggml_cuda_q8_cache cache;
            for (int v = 0; v < 4; ++v) {
                cache.add(&src, v, 512);
                cache.add(&src, v, 1024);
            }
            std::vector<char> bytes(cache.reserve());
            require(bytes.size() == 4096, "variant workspace bound");
            cache.data = bytes.data();
            for (int v = 0; v < 4; ++v) {
                bool hit;
                char * first = cache.get(&src, v, 1024, hit);
                require(first && !hit, "an evaluation must start empty");
                require(cache.get(&src, v, 1025, hit) == nullptr && !hit, "a larger consumer must fall back");
                require(cache.get(&src, v, 512, hit) == first && hit, "a shared layout must hit");
            }
            auto view = src;
            view.data = input.data() + 128;
            view.ne[0] = 128;
            view.ne[1] = 1;
            std::memset(view.data, 0, 512);
            cache.invalidate(&view);
            for (int v = 0; v < 4; ++v) {
                bool hit;
                cache.get(&src, v, 512, hit);
                require(!hit, "a stale activation was reused after an alias write");
            }
            auto distinct = src;
            bool hit;
            require(cache.get(&distinct, 0, 512, hit) == nullptr, "a distinct tensor identity must miss");
        }
    }
    {
        // A single consumer is never retained; a group over the scratch bound falls back as a whole.
        std::vector<float> input(256, 1.0f);
        auto src = shape(256, 1, 1, 1);
        src.data = input.data();
        ggml_cuda_q8_cache once;
        once.add(&src, 0, 512);
        require(once.reserve() == 0, "one consumer must not reserve scratch");
        ggml_cuda_q8_cache huge;
        huge.add(&src, 0, size_t(65) << 20);
        huge.add(&src, 0, 512);
        require(huge.reserve() == 0, "a group over the scratch bound must fall back");
    }
    std::puts("M: eligibility bound 16, layouts, evaluation reset, alias invalidation, size and identity fallback PASS");
}
