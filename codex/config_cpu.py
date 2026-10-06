"""Generate a host-only contract test from the actual MMQ config declarations."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
source = (ROOT / 'ggml/src/ggml-cuda/mmq.cuh').read_text()
declarations = source[source.index('enum ggml_cuda_mmq_sram_layout'):source.index('#include "mmq-config-pascal-older.cuh"')]
prefix = r'''
#include "ggml.h"
#include <cassert>
#include <cstdio>
#include <initializer_list>
#define __host__
#define __device__
#define GGML_UNUSED(x) (void)(x)
#define GGML_CUDA_CC_BLACKWELL 1200
#define MMQ_ITER_K 256
#define MMQ_ITER_K_FP4 512
#define MMQ_TILE_NE_K 32
#define QI8_0 8
#define QI8_1 8
#define QI6_K 32
static bool amd_mfma_available(int) { return false; }
static bool amd_wmma_available(int) { return false; }
static bool turing_mma_available(int) { return true; }
'''
suffix = r'''
#include "mmq-config-ampere.cuh"
#include "mmq-config-blackwell.cuh"
static bool same(const ggml_cuda_mmq_config & a, const ggml_cuda_mmq_config & b) {
    return a.type == b.type && a.nthreads == b.nthreads && a.occupancy == b.occupancy && a.I == b.I && a.J == b.J &&
        a.sram_layout == b.sram_layout && a.K_vram == b.K_vram && a.stream_k == b.stream_k && a.fallback == b.fallback;
}
int main() {
    int checked = 0;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const auto type = (ggml_type) t;
        const bool touched = type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
            type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q8_0;
        for (int J = 8; J <= 128; J += 8) {
            for (bool fallback : {false, true}) {
                const auto ampere = ggml_cuda_mmq_get_config_ampere(type, J, fallback);
                const auto small = ggml_cuda_mmq_get_config_blackwell(type, J, fallback, 1200);
                const auto other = ggml_cuda_mmq_get_config_blackwell(type, J, fallback, 1210);
                if (type != GGML_TYPE_MXFP4 && type != GGML_TYPE_NVFP4) {
                    assert(same(other, ampere));
#ifndef GGML_CUDA_MMQ_SMALLJ_DISABLE
                    if (touched && J <= 24) {
                        assert(small.nthreads == 128 && small.occupancy == 2 && small.I == 64);
                        assert(small.J == J && small.fallback == fallback && small.stream_k && small.K_vram == 256);
                        assert(small.sram_layout == ggml_cuda_mmq_get_config_ampere(type, 16, fallback).sram_layout);
                    } else
#endif
                    {
                        assert(same(small, ampere));
                    }
                }
                ++checked;
            }
        }
    }
    for (int rows : {3072, 5120, 6144, 8704}) {
        const int tiles = rows / 64;
        const int waves = (tiles + 35) / 36;
        assert(100 * tiles / (36 * waves) >= 66);
    }
    std::printf("MMQ config contracts PASS: %d tuples\n", checked);
}
'''
(HERE / 'config-cpu.cpp').write_text(prefix + declarations + suffix)
negative = HERE / 'negative-config'
negative.mkdir(exist_ok=True)
table = (ROOT / 'ggml/src/ggml-cuda/mmq-config-blackwell.cuh').read_text()
(negative / 'mmq-config-blackwell.cuh').write_text(table.replace('type, 128, 2, 64, J', 'type, 128, 2, 96, J'))
