#include "q8-cache.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void require(bool ok, const char * why) {
    if (!ok) { std::fprintf(stderr, "%s\n", why); std::exit(1); }
}

int main(int argc, char ** argv) {
    const bool stale = argc == 2 && std::string(argv[1]) == "stale";
    for (int rows : {1, 6, 8}) {
        std::vector<float> input(256 * rows, 1.0f);
        ggml_tensor src{};
        src.type = GGML_TYPE_F32;
        src.data = input.data();
        src.ne[0] = 256; src.ne[1] = rows; src.ne[2] = src.ne[3] = 1;
        src.nb[0] = 4; src.nb[1] = 1024; src.nb[2] = src.nb[3] = 1024 * rows;
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
                require(first && !hit, "evaluation must start empty");
                require(cache.get(&src, v, 1025, hit) == nullptr && !hit, "larger consumer must fall back");
                require(cache.get(&src, v, 512, hit) == first && hit, "shared layout must hit");
            }
            // A view writes into the source between consumers.
            auto view = src;
            view.data = input.data() + 128;
            view.ne[0] = 128;
            view.ne[1] = 1;
            std::memset(view.data, 0, 512);
            if (!stale) { cache.invalidate(&view); }
            for (int v = 0; v < 4; ++v) {
                bool hit;
                cache.get(&src, v, 512, hit);
                require(!hit, "stale activation reused after an alias write");
            }
            auto distinct = src;
            bool hit;
            require(cache.get(&distinct, 0, 512, hit) == nullptr, "distinct tensor identity must miss");
        }
    }
    std::puts("M: layouts, widths 1/6/8, evaluation reset, alias invalidation and capacity fallback PASS");
}
