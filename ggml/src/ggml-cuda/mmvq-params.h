#pragma once

#include "ggml.h"

#include <cstdint>
#include <type_traits>

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

static constexpr int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <int nwarps_, int rows_per_block_>
struct mmvq_launch_config {
    static constexpr int nwarps = nwarps_;
    static constexpr int rows_per_block = rows_per_block_;

    static constexpr uint32_t nblocks(uint32_t nrows) {
        return nrows / rows_per_block + (nrows % rows_per_block != 0);
    }
};

template <ggml_type type, int ncols, bool small_k = false, bool halve_iters = false, typename F>
static void mmvq_select_config(mmvq_parameter_table_id table, F launch) {
    const auto select = [&](auto table_tag) {
        constexpr auto table_id = decltype(table_tag)::value;
        constexpr int nw = calc_nwarps(type, ncols, table_id, small_k, halve_iters);
        constexpr int nr = calc_rows_per_block(ncols, table_id, small_k, nw);
        launch(mmvq_launch_config<nw, nr>{});
    };
    switch (table) {
        case MMVQ_PARAMETERS_GENERIC: select(std::integral_constant<mmvq_parameter_table_id, MMVQ_PARAMETERS_GENERIC>{}); return;
        case MMVQ_PARAMETERS_TURING: select(std::integral_constant<mmvq_parameter_table_id, MMVQ_PARAMETERS_TURING>{}); return;
        case MMVQ_PARAMETERS_GCN: select(std::integral_constant<mmvq_parameter_table_id, MMVQ_PARAMETERS_GCN>{}); return;
        case MMVQ_PARAMETERS_RDNA2: select(std::integral_constant<mmvq_parameter_table_id, MMVQ_PARAMETERS_RDNA2>{}); return;
        case MMVQ_PARAMETERS_RDNA3_0: select(std::integral_constant<mmvq_parameter_table_id, MMVQ_PARAMETERS_RDNA3_0>{}); return;
        case MMVQ_PARAMETERS_RDNA4: select(std::integral_constant<mmvq_parameter_table_id, MMVQ_PARAMETERS_RDNA4>{}); return;
        case MMVQ_PARAMETERS_GB10: select(std::integral_constant<mmvq_parameter_table_id, MMVQ_PARAMETERS_GB10>{}); return;
    }
    GGML_ABORT("invalid MMVQ parameter table");
}
