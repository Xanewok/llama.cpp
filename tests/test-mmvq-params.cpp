#include "../ggml/src/ggml-cuda/mmvq-params.h"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <utility>

static size_t checks = 0;
static size_t kernels = 0;
static size_t cuda_kernels = 0;

static void require(bool ok) {
    if (!ok) {
        std::abort();
    }
}

template <ggml_type type, int ncols, bool small_k, bool halve_iters>
static void check_config(std::set<std::pair<int, int>> & unique, std::set<std::pair<int, int>> & cuda_unique) {
    for (int table = MMVQ_PARAMETERS_GENERIC; table <= MMVQ_PARAMETERS_GB10; ++table) {
        const auto table_id = static_cast<mmvq_parameter_table_id>(table);
        const int nw = calc_nwarps(type, ncols, table_id, small_k, halve_iters);
        const int nr = calc_rows_per_block(ncols, table_id, small_k, nw, type);
        mmvq_select_config<type, ncols, small_k, halve_iters>(table_id, [&](auto config) {
            using config_t = decltype(config);
            require(config_t::nwarps == nw && config_t::rows_per_block == nr);
            require(nw >= 1 && nw <= 8 && nr >= 1 && nr <= 8);
            for (uint32_t nrows = 1; nrows <= 33; ++nrows) {
                const uint32_t blocks = config_t::nblocks(nrows);
                const uint32_t tail_rows = nrows % nr;
                require(blocks * nr + tail_rows == nrows && tail_rows < uint32_t(nr));
            }
            unique.emplace(nw, nr);
            if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_TURING ||
                table_id == MMVQ_PARAMETERS_GB10) {
                cuda_unique.emplace(nw, nr);
            }
            ++checks;
        });
    }
}

template <ggml_type type, int ncols = 1>
static void check_type() {
    std::set<std::pair<int, int>> unique, cuda_unique;
    check_config<type, ncols, false, false>(unique, cuda_unique);
    if constexpr (ncols == 1) {
        check_config<type, ncols, true, false>(unique, cuda_unique);
        check_config<type, ncols, false, true>(unique, cuda_unique);
        kernels += unique.size(); // fused and unfused use the same configurations
    }
    kernels += unique.size();
    std::set<int> tail_warps;
    for (const auto & config : unique) {
        if (config.second > 1) {
            tail_warps.insert(config.first);
        }
    }
    kernels += tail_warps.size() * (ncols == 1 ? 2 : 1);
    std::set<int> cuda_tail_warps;
    for (const auto & config : cuda_unique) {
        if (config.second > 1) {
            cuda_tail_warps.insert(config.first);
        }
    }
    cuda_kernels += (cuda_unique.size() + cuda_tail_warps.size()) * (ncols == 1 ? 2 : 1);
    if constexpr (ncols < MMVQ_MAX_BATCH_SIZE) {
        check_type<type, ncols + 1>();
    }
}

int main() {
    check_type<GGML_TYPE_Q1_0>();
    check_type<GGML_TYPE_Q2_0>();
    check_type<GGML_TYPE_Q4_0>();
    check_type<GGML_TYPE_Q4_1>();
    check_type<GGML_TYPE_Q5_0>();
    check_type<GGML_TYPE_Q5_1>();
    check_type<GGML_TYPE_Q8_0>();
    check_type<GGML_TYPE_MXFP4>();
    check_type<GGML_TYPE_NVFP4>();
    check_type<GGML_TYPE_Q2_K>();
    check_type<GGML_TYPE_Q3_K>();
    check_type<GGML_TYPE_Q4_K>();
    check_type<GGML_TYPE_Q5_K>();
    check_type<GGML_TYPE_Q6_K>();
    check_type<GGML_TYPE_IQ2_XXS>();
    check_type<GGML_TYPE_IQ2_XS>();
    check_type<GGML_TYPE_IQ2_S>();
    check_type<GGML_TYPE_IQ3_XXS>();
    check_type<GGML_TYPE_IQ1_S>();
    check_type<GGML_TYPE_IQ1_M>();
    check_type<GGML_TYPE_IQ4_NL>();
    check_type<GGML_TYPE_IQ4_XS>();
    check_type<GGML_TYPE_IQ3_S>();
    std::printf("M: %zu configurations passed; D: %zu distinct MMVQ specializations across all tables; %zu CUDA specializations\n", checks, kernels, cuda_kernels);
}
