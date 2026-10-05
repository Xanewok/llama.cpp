// Keep the tuning selector in this executable; production dispatch has no override.
#define ggml_cuda_should_use_mmvq bench_should_use_mmvq
#define get_mmvq_mmid_max_batch bench_mmid_max_batch
#define ggml_cuda_mul_mat_vec_q bench_mul_mat_vec_q
#define ggml_cuda_op_mul_mat_vec_q bench_op_mul_mat_vec_q
#include "../ggml/src/ggml-cuda/mmvq.cu"
#undef ggml_cuda_should_use_mmvq
#undef get_mmvq_mmid_max_batch
#undef ggml_cuda_mul_mat_vec_q
#undef ggml_cuda_op_mul_mat_vec_q

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

struct bench_shape {
    const char * name;
    int k;
    int m;
};

static const bench_shape shapes[] = {
    { "gate-up",     5120,   8704 },
    { "down",        8704,   5120 },
    { "qkv",         5120,   5120 },
    { "q",           5120,   6144 },
    { "attn-gate",   5120,   3072 },
    { "ssm-out",     3072,   5120 },
    { "head",        5120, 124160 },
};

static const char * variants[] = { "base", "default", "w2r1", "w4r1", "w4r2", "w2r4", "w8r1", "w1r4", "w1r8", "w2r8", "w4r4", "w2r6" };

static bool selected_value(const std::string & selected, const std::string & value) {
    return selected == "all" || ("," + selected + ",").find("," + value + ",") != std::string::npos;
}

struct bench_data {
    void * weights;
    void * x;
    void * y;
    float * dst;
    int k;
    int m;
    int cc;
    int warp_size;
    size_t weight_bytes;
    size_t launch_index;
    int copies;
    cudaStream_t stream;
};

template <ggml_type type, int ncols, int nw = 0, int nr = 0>
static void launch(const bench_data & d) {
    const int nwarps = nw ? nw : calc_nwarps(type, ncols, get_device_table_id(d.cc));
    const int rows = nr ? nr : calc_rows_per_block(ncols, get_device_table_id(d.cc), false, nwarps);
    const dim3 grid((d.m + rows - 1)/rows);
    const dim3 block(d.warp_size, nwarps);
    const uint3 one = init_fastdiv_values(1);
    const ggml_cuda_mm_fusion_args_device fusion{};
    const ggml_cuda_kernel_launch_params params(grid, block, 0, d.stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, ncols, false, false, false, nw, nr>, params,
        d.x, d.y, nullptr, fusion, d.dst, d.k, one, d.k/ggml_blck_size(type), d.k/QK8_1,
        d.m, one, 0, 0, 0, one, 0, 0, 0, 0);
}

template <ggml_type type, int ncols>
static void launch_variant(const bench_data & d, int variant) {
    switch (variant) {
        case 0: launch<type, ncols, (ncols <= 4 ? 4 : 2), (ncols == 1 ? 1 : 2)>(d); break;
        case 1: launch<type, ncols>(d); break;
        case 2: launch<type, ncols, 2, 1>(d); break;
        case 3: launch<type, ncols, 4, 1>(d); break;
        case 4: launch<type, ncols, 4, 2>(d); break;
        case 5: launch<type, ncols, 2, 4>(d); break;
        case 6: launch<type, ncols, 8, 1>(d); break;
        case 7: launch<type, ncols, 1, 4>(d); break;
        case 8: launch<type, ncols, 1, 8>(d); break;
        case 9: launch<type, ncols, 2, 8>(d); break;
        case 10: launch<type, ncols, 4, 4>(d); break;
        case 11: launch<type, ncols, 2, 6>(d); break;
        default: throw std::runtime_error("invalid variant");
    }
}

template <ggml_type type>
static void launch_cols(const bench_data & d, int ncols, int variant) {
    switch (ncols) {
        case 1: launch_variant<type, 1>(d, variant); break;
        case 2: launch_variant<type, 2>(d, variant); break;
        case 4: launch_variant<type, 4>(d, variant); break;
        case 5: launch_variant<type, 5>(d, variant); break;
        case 7: launch_variant<type, 7>(d, variant); break;
        case 6: launch_variant<type, 6>(d, variant); break;
        case 8: launch_variant<type, 8>(d, variant); break;
        default: throw std::runtime_error("invalid column count");
    }
}

static void launch_type(ggml_type type, bench_data & d, int ncols, int variant) {
    d.x = static_cast<char *>(d.weights) + d.weight_bytes*(d.launch_index++ % d.copies);
    switch (type) {
        case GGML_TYPE_Q5_K:   launch_cols<GGML_TYPE_Q5_K>(d, ncols, variant); break;
        case GGML_TYPE_Q6_K:   launch_cols<GGML_TYPE_Q6_K>(d, ncols, variant); break;
        case GGML_TYPE_Q4_K:   launch_cols<GGML_TYPE_Q4_K>(d, ncols, variant); break;
        case GGML_TYPE_IQ4_XS: launch_cols<GGML_TYPE_IQ4_XS>(d, ncols, variant); break;
        case GGML_TYPE_Q8_0:   launch_cols<GGML_TYPE_Q8_0>(d, ncols, variant); break;
        default: throw std::runtime_error("invalid type");
    }
}

static double nmse(const std::vector<float> & actual, const std::vector<float> & expected) {
    double error = 0.0, norm = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            return INFINITY;
        }
        const double delta = double(actual[i]) - expected[i];
        error += delta*delta;
        norm += double(expected[i])*expected[i];
    }
    return error/std::max(norm, 1e-30);
}

static void bench(ggml_type type, const bench_shape & s, int repeats, int trials, int warmup_ms,
                  const std::string & selected, const std::string & column_filter, bool negative) {
    const size_t row_bytes = ggml_row_size(type, s.k);
    const size_t logical_bytes = row_bytes*s.m;
    // The kernel loads every block row before it bounds the output store.
    std::vector<char> weights(row_bytes*((s.m + 23)/24*24), 0);
    std::vector<float> row(s.k);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    for (int r = 0; r < s.m; ++r) {
        for (float & value : row) {
            value = uniform(rng);
        }
        ggml_quantize_chunk(type, row.data(), weights.data() + r*row_bytes, 0, 1, s.k, nullptr);
    }
    const int max_cols = 8;
    std::vector<float> inputs(s.k*max_cols);
    for (float & value : inputs) {
        value = uniform(rng);
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    bench_data d{};
    d.k = s.k;
    d.m = s.m;
    d.cc = 100*prop.major + 10*prop.minor;
    d.warp_size = prop.warpSize;
    d.weight_bytes = weights.size();
    // Model layers evict each other's weights. Keep repeated small shapes out of L2.
    const size_t cache_target = 4*size_t(std::max(prop.l2CacheSize, 1));
    d.copies = int((cache_target + logical_bytes - 1)/logical_bytes);
    CUDA_CHECK(cudaStreamCreate(&d.stream));
    CUDA_CHECK(cudaMalloc(&d.weights, weights.size()*d.copies));
    CUDA_CHECK(cudaMalloc(&d.y, sizeof(block_q8_1)*s.k/QK8_1*max_cols));
    CUDA_CHECK(cudaMalloc(&d.dst, sizeof(float)*s.m*max_cols));
    float * input_device;
    CUDA_CHECK(cudaMalloc(&input_device, sizeof(float)*inputs.size()));
    for (int i = 0; i < d.copies; ++i) {
        CUDA_CHECK(cudaMemcpy(static_cast<char *>(d.weights) + weights.size()*i, weights.data(), weights.size(), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(input_device, inputs.data(), sizeof(float)*inputs.size(), cudaMemcpyHostToDevice));
    quantize_row_q8_1_cuda(input_device, nullptr, d.y, type, s.k, s.k, 0, 0, s.k, max_cols, 1, 1, d.stream);
    CUDA_CHECK(cudaStreamSynchronize(d.stream));
    std::vector<block_q8_1> rhs(s.k/QK8_1*max_cols);
    CUDA_CHECK(cudaMemcpy(rhs.data(), d.y, sizeof(block_q8_1)*rhs.size(), cudaMemcpyDeviceToHost));
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    for (int ncols : {6, 5, 7, 8, 1, 2, 4}) {
        if (!selected_value(column_filter, std::to_string(ncols))) {
            continue;
        }
        std::vector<float> reference(s.m*ncols), actual(s.m*ncols);
        launch_type(type, d, ncols, 0);
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
        CUDA_CHECK(cudaMemcpy(reference.data(), d.dst, sizeof(float)*reference.size(), cudaMemcpyDeviceToHost));
        std::vector<float> cpu, sampled;
        for (int r : {0, 1, s.m/3, s.m/2, s.m - 2, s.m - 1}) {
            ggml_get_type_traits(type)->to_float(weights.data() + r*row_bytes, row.data(), s.k);
            for (int c = 0; c < ncols; ++c) {
                double sum = 0;
                for (int k = 0; k < s.k; ++k) {
                    const block_q8_1 & b = rhs[c*s.k/QK8_1 + k/QK8_1];
                    sum += double(row[k])*__low2float(b.ds)*b.qs[k%QK8_1];
                }
                cpu.push_back(sum);
                sampled.push_back(reference[c*s.m + r]);
            }
        }
        const double cpu_error = nmse(sampled, cpu);
        // Q4_K/Q5_K use the stored input sum with half precision for the min correction.
        if (!(cpu_error < 2e-5)) {
            throw std::runtime_error("baseline disagrees with CPU dequantized dot");
        }
        const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(warmup_ms);
        do {
            for (int i = 0; i < 20; ++i) {
                launch_type(type, d, ncols, 0);
            }
            CUDA_CHECK(cudaStreamSynchronize(d.stream));
        } while (std::chrono::steady_clock::now() < until);
        for (int variant = 0; variant < int(sizeof(variants)/sizeof(variants[0])); ++variant) {
            if (!selected_value(selected, variants[variant])) {
                continue;
            }
            launch_type(type, d, ncols, variant);
            CUDA_CHECK(cudaStreamSynchronize(d.stream));
            CUDA_CHECK(cudaMemcpy(actual.data(), d.dst, sizeof(float)*actual.size(), cudaMemcpyDeviceToHost));
            if (negative) {
                actual[0] += 1e6f;
            }
            const double error = nmse(actual, reference);
            if (!(error < 1e-8)) {
                throw std::runtime_error("variant disagrees with baseline");
            }
            std::vector<float> times;
            for (int trial = 0; trial < trials; ++trial) {
                CUDA_CHECK(cudaEventRecord(begin, d.stream));
                for (int i = 0; i < repeats; ++i) {
                    launch_type(type, d, ncols, variant);
                }
                CUDA_CHECK(cudaEventRecord(end, d.stream));
                CUDA_CHECK(cudaEventSynchronize(end));
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
                times.push_back(ms*1000/repeats);
            }
            std::sort(times.begin(), times.end());
            const float us = times[times.size()/2];
            std::printf("{\"type\":\"%s\",\"shape\":\"%s\",\"k\":%d,\"m\":%d,\"ncols\":%d,\"variant\":\"%s\",\"us_M\":%.6f,\"weight_GB_s_D\":%.6f,\"weight_bytes\":%zu,\"nmse_D\":%.12g,\"cpu_nmse_D\":%.12g,\"repeats\":%d,\"trials\":%d,\"cc\":%d,\"l2_bytes_M\":%d,\"working_set_bytes_D\":%zu,\"copies_D\":%d}\n",
                ggml_type_name(type), s.name, s.k, s.m, ncols, variants[variant], us,
                logical_bytes/double(us)/1000, logical_bytes, error, cpu_error, repeats, trials, d.cc,
                prop.l2CacheSize, logical_bytes*d.copies, d.copies);
            std::fflush(stdout);
        }
    }
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(input_device));
    CUDA_CHECK(cudaFree(d.dst));
    CUDA_CHECK(cudaFree(d.y));
    CUDA_CHECK(cudaFree(d.weights));
    CUDA_CHECK(cudaStreamDestroy(d.stream));
}

int main(int argc, char ** argv) {
    try {
        std::string variant = "all", type_filter = "all", shape_filter = "all", ncols = "all";
        int repeats = 100, trials = 7, warmup_ms = 1000;
        bool negative = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--negative-control") {
                negative = true;
                continue;
            }
            if (i + 1 == argc) {
                throw std::runtime_error("missing option value");
            }
            const std::string value = argv[++i];
            if      (arg == "--variant")   { variant = value; }
            else if (arg == "--type")      { type_filter = value; }
            else if (arg == "--shape")     { shape_filter = value; }
            else if (arg == "--repeats")   { repeats = std::stoi(value); }
            else if (arg == "--trials")    { trials = std::stoi(value); }
            else if (arg == "--warmup-ms") { warmup_ms = std::stoi(value); }
            else if (arg == "--ncols")     { ncols = value; }
            else { throw std::runtime_error("unknown option"); }
        }
        if (repeats < 1 || repeats > 10000 || trials < 1 || trials > 99 || warmup_ms < 0 || warmup_ms > 10000) {
            throw std::runtime_error("invalid repeat, trial or warmup count");
        }
        for (int selection_index = 0; selection_index < 2; ++selection_index) {
            const std::string & selection = selection_index == 0 ? variant : ncols;
            if (selection == "all") {
                continue;
            }
            std::stringstream tokens(selection);
            std::string token;
            if (selection.empty() || selection.back() == ',') {
                throw std::runtime_error("empty selection");
            }
            while (std::getline(tokens, token, ',')) {
                bool found = false;
                if (selection_index == 0) {
                    for (const char * name : variants) {
                        found |= token == name;
                    }
                } else {
                    for (int n : {1, 2, 4, 5, 6, 7, 8}) {
                        found |= token == std::to_string(n);
                    }
                }
                if (!found) {
                    throw std::runtime_error("unknown variant or column count");
                }
            }
        }
        CUDA_CHECK(cudaSetDevice(0));
        bool found = false;
        for (ggml_type type : {GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0}) {
            if (type_filter != "all" && type_filter != ggml_type_name(type)) {
                continue;
            }
            for (const bench_shape & shape : shapes) {
                if (shape_filter != "all" && shape_filter != shape.name) {
                    continue;
                }
                if (std::strcmp(shape.name, "head") == 0 && type != GGML_TYPE_Q6_K) {
                    continue;
                }
                found = true;
                bench(type, shape, repeats, trials, warmup_ms, variant, ncols, negative);
            }
        }
        if (!found) {
            throw std::runtime_error("empty type/shape selection");
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
