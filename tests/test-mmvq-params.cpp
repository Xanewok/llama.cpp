#include "../ggml/src/ggml-cuda/mmvq-params.h"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <utility>

static size_t checks = 0;
static size_t kernels = 0;

static void require(bool ok) {
    if (!ok) {
        std::abort();
    }
}

template <ggml_type type, int ncols, bool small_k, bool halve_iters>
static void check_config(std::set<std::pair<int, int>> & unique) {
    for (int table = MMVQ_PARAMETERS_GENERIC; table <= MMVQ_PARAMETERS_BLACKWELL; ++table) {
        const auto table_id = static_cast<mmvq_parameter_table_id>(table);
        const int nw = calc_nwarps(type, ncols, table_id, small_k, halve_iters);
        const int nr = calc_rows_per_block(type, ncols, table_id, small_k, nw);
        mmvq_select_config<type, ncols, small_k, halve_iters>(table_id, [&](auto config) {
            using config_t = decltype(config);
            require(config_t::nwarps == nw && config_t::rows_per_block == nr);
            require(nw >= 1 && nw <= 8 && nr >= 1 && nr <= nw * 2);
            for (uint32_t nrows = 1; nrows <= 33; ++nrows) {
                const uint32_t blocks = config_t::nblocks(nrows);
                require(blocks * nr >= nrows && (blocks - 1) * nr < nrows);
            }
            unique.emplace(nw, nr);
            ++checks;
        });
    }
}

template <ggml_type type, int ncols = 1>
static void check_type() {
    std::set<std::pair<int, int>> unique;
    check_config<type, ncols, false, false>(unique);
    if constexpr (ncols == 1) {
        check_config<type, ncols, true, false>(unique);
        check_config<type, ncols, false, true>(unique);
        kernels += unique.size(); // fused and unfused use the same configurations
    }
    kernels += unique.size();
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
    std::printf("M: %zu configurations passed; D: %zu distinct MMVQ specializations across all tables\n", checks, kernels);
}
