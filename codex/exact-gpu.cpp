#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>

static void require(bool ok, const char * why) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); }
}

static std::vector<char> run(ggml_backend_dev_t dev, ggml_type type, int rows, bool strided, bool overwrite, bool norm) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    require(backend, "backend init");
    ggml_context * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    const int columns = norm ? 2560 : 512;
    ggml_tensor * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns + (strided ? 256 : 0), rows);
    ggml_set_input(storage);
    ggml_tensor * input = strided ? ggml_view_2d(ctx, storage, columns, rows, storage->nb[1], 0) : storage;
    ggml_tensor * replacement = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, rows);
    ggml_tensor * output = nullptr;
    ggml_tensor * weight[3] = {};
    if (norm) {
        weight[0] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, columns);
        auto * residual = ggml_add(ctx, input, replacement);
        auto * normalized = ggml_mul(ctx, ggml_rms_norm(ctx, residual, 1e-6f), weight[0]);
        output = ggml_concat(ctx, residual, normalized, 0);
    } else {
        ggml_tensor * part[3];
        for (int i = 0; i < 3; ++i) {
            if (i == 2 && overwrite) {
                ggml_build_forward_expand(graph, ggml_cpy(ctx, replacement, input));
            }
            weight[i] = ggml_new_tensor_2d(ctx, i == 1 ? GGML_TYPE_Q6_K : type, columns, 256);
            part[i] = ggml_mul_mat(ctx, weight[i], input);
            ggml_build_forward_expand(graph, part[i]);
        }
        output = ggml_concat(ctx, ggml_concat(ctx, part[0], part[1], 0), part[2], 0);
    }
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer, "tensor allocation");
    for (auto * w : weight) {
        if (!w) { continue; }
        std::vector<float> data(ggml_nelements(w));
        for (size_t j = 0; j < data.size(); ++j) { data[j] = std::sin(float(j) * 0.017f) * 0.125f; }
        if (w->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(w, data.data(), 0, ggml_nbytes(w));
        } else {
            std::vector<char> quant(ggml_nbytes(w));
            std::vector<float> importance(w->ne[0], 1.0f);
            require(ggml_quantize_chunk(w->type, data.data(), quant.data(), 0, w->ne[1], w->ne[0], importance.data()) == quant.size(), "weight quantization");
            ggml_backend_tensor_set(w, quant.data(), 0, quant.size());
        }
    }
    std::vector<char> result;
    // Stable metadata, changed values: direct execution, capture, then replay.
    for (int step = 0; step < 4; ++step) {
        std::vector<float> values(ggml_nelements(storage));
        for (size_t j = 0; j < values.size(); ++j) { values[j] = std::cos(float(j) * 0.021f + step) + 0.25f * step; }
        ggml_backend_tensor_set(storage, values.data(), 0, ggml_nbytes(storage));
        std::vector<float> other(ggml_nelements(replacement), 0.75f + step);
        ggml_backend_tensor_set(replacement, other.data(), 0, ggml_nbytes(replacement));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute");
        const size_t offset = result.size();
        result.resize(offset + ggml_nbytes(output));
        ggml_backend_tensor_get(output, result.data() + offset, 0, ggml_nbytes(output));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return result;
}

int main(int argc, char ** argv) {
    require(argc == 3, "usage: exact-gpu reference|patch OUTPUT");
    const bool disable = std::strcmp(argv[1], "reference") == 0;
    if (disable) {
        setenv("GGML_CUDA_DISABLE_Q8_REUSE", "1", 1);
        setenv("GGML_CUDA_DISABLE_FUSION", "1", 1);
    } else {
        unsetenv("GGML_CUDA_DISABLE_Q8_REUSE");
        unsetenv("GGML_CUDA_DISABLE_FUSION");
    }
    std::ofstream file(argv[2], std::ios::binary);
    require(bool(file), "output file");
    require(std::getenv("GPUQ_ACTIVE") && std::strcmp(std::getenv("GPUQ_ACTIVE"), "1") == 0, "GPU queue required");
    ggml_backend_load_all();
    int cases = 0;
    for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
        auto * dev = ggml_backend_dev_get(d);
        if (std::strncmp(ggml_backend_dev_name(dev), "CUDA", 4) != 0) { continue; }
        for (auto type : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0}) {
            for (int rows : {1, 6, 8}) {
                for (bool strided : {false, true}) {
                    for (bool overwrite : {false, true}) {
                        const auto data = run(dev, type, rows, strided, overwrite, false);
                        file.write(data.data(), data.size());
                        ++cases;
                    }
                }
            }
        }
        for (int rows : {1, 6, 8}) {
            const auto data = run(dev, GGML_TYPE_F32, rows, false, false, true);
            file.write(data.data(), data.size());
            ++cases;
        }
        std::printf("PASS: device=%s cases=%d\n", ggml_backend_dev_name(dev), cases);
    }
    require(cases > 0, "no CUDA devices");
    file.close();
    require(bool(file), "output write");
    std::printf("PASS: exact cases=%d\n", cases);
}
