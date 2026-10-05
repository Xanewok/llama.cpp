#include "../ggml/src/ggml-cuda/fattn-bound.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

static void check(bool ok, const char * message) {
    if (!ok) {
        fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

int main(int argc, char ** argv) {
    check(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    const uintptr_t shared_base = argc == 3 && !strcmp(argv[1], "--shared-base") ? std::strtoull(argv[2], nullptr, 0) : 0;
    size_t tuples = 0, loads = 0, tails = 0;
    for (int warp : {32, 64}) {
        const int nbytes = FATTN_KQ_STRIDE / (2*warp)*4;
        for (int offset = 0; offset < 32; ++offset) {
            for (int extra = 0; extra < 32; extra += 2) {
                for (int nk : {1, 31, 32, 63, 64, 255, 256, 257, 4095, 4096, 4097, 8192, 90113, 94208}) {
                    const size_t nbq = 2*nk + extra;
                    for (int nq = 1; nq <= 8; ++nq) {
                        ++tuples;
                        const bool packed = flash_attn_decode_mask_packed(offset, nk, nbq, warp);
                        check(packed == (nk % 256 == 0 && nbq % nbytes == 0 && offset % nbytes == 0), "bad packed dispatch");
                        if (!packed) {
                            continue;
                        }
                        // Checking every tile covers every possible stop, including a fully masked extent.
                        for (int start = 0; start < nk; start += 256) {
                            for (int q = 0; q < nq; ++q) {
                                for (int lane = 0; lane < warp; ++lane) {
                                    const size_t address = offset + flash_attn_decode_mask_offset(q, nbq, start, lane, warp);
                                    check(address % nbytes == 0, "misaligned packed scan load");
                                    check(address >= offset + q*nbq + 2*start && address + nbytes <= offset + q*nbq + 2*(start + 256), "scan crosses tile");
                                    check(address + nbytes <= offset + q*nbq + 2*nk, "scan crosses row");
                                    ++loads;
                                }
                            }
                        }
                    }
                }
            }
        }
        for (int nk = 1; nk <= 4097; ++nk) {
            for (int tail = 0; tail <= std::min(4095, nk); ++tail) {
                const int live = nk - tail;
                const int bound = flash_attn_decode_mask_packed(0, nk, 2*nk, warp) ? (live + 255)/256*256 : nk;
                check(bound >= live && bound <= nk, "tail drops a live column");
                ++tails;
            }
        }
    }
    check(flash_attn_decode_mask_offset(7, size_t(1) << 30, 0, 0, 32) == (size_t(7) << 30), "row offset overflow");

    // VEC reuses KQ for q8_1 data and float2 scales before the mask scan.
    for (int dim : {256, 128, 64}) {
        for (int q = 0; q < 8; ++q) {
            for (int lane = 0; lane < 32; lane += 8) {
                const uintptr_t address = shared_base + q*dim*4 + dim + (lane/8)*8;
                if (address % 8 != 0) {
                    fprintf(stderr, "misaligned VEC float2 scale store: D=%d q=%d lane=%d address=0x%zx (v4 control)\n", dim, q, lane, size_t(address));
                    return 1;
                }
            }
        }
    }
    // Each VEC/TILE shared vector copy starts at a multiple of its transfer size.
    for (int width : {4, 8, 16}) {
        for (int index = 0; index < 4096; index += width) {
            check((shared_base + index) % width == 0, "misaligned shared vector copy");
        }
    }
    printf("M: tuples=%zu scan_loads=%zu tails=%zu; alignment and row/tile bounds pass\n", tuples, loads, tails);
}
