#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static float input(size_t i, int rank, int iteration) {
    return float(int((i + 7 * rank + 3 * iteration) % 31) - 15);
}

static void encode(std::vector<uint8_t> & bytes, ggml_type type, size_t i, float value) {
    if (type == GGML_TYPE_F32) {
        memcpy(bytes.data() + i * sizeof(value), &value, sizeof(value));
    } else if (type == GGML_TYPE_F16) {
        const ggml_fp16_t half = ggml_fp32_to_fp16(value);
        memcpy(bytes.data() + i * sizeof(half), &half, sizeof(half));
    } else {
        const ggml_bf16_t half = ggml_fp32_to_bf16(value);
        memcpy(bytes.data() + i * sizeof(half), &half, sizeof(half));
    }
}

static bool equal(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    return a == b;
}

int main(int argc, char ** argv) {
    const char * output = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--output FILE]\n", argv[0]);
            return 1;
        }
    }
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (!reg || ggml_backend_reg_dev_count(reg) < 2) {
        puts("SKIP: two CUDA devices required");
        return 77;
    }
    const char * provider = std::getenv("GGML_CUDA_ALLREDUCE");
    require(provider && strcmp(provider, "internal") == 0, "set GGML_CUDA_ALLREDUCE=internal");
    auto init = reinterpret_cast<ggml_backend_comm_init_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init"));
    auto reduce = reinterpret_cast<ggml_backend_comm_allreduce_tensor_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor"));
    auto free_comm = reinterpret_cast<ggml_backend_comm_free_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free"));
    require(init && reduce && free_comm, "CUDA comm API missing");
    ggml_backend_t backends[2];
    for (int rank = 0; rank < 2; ++rank) {
        backends[rank] = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, rank), nullptr);
        require(backends[rank] != nullptr, "backend init");
    }
    void * comm = init(backends, 2);
    require(comm != nullptr, "internal comm init");
    FILE * file = output ? fopen(output, "wb") : nullptr;
    require(!output || file, "open output");
    size_t cases = 0;
    // F32 counts with n % 8 == 4 reach BF16-wire scalar tails; all inputs must occupy a multiple of 16 bytes.
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        for (size_t n : {size_t(8), size_t(256), size_t(260), size_t(262136), size_t(262140), size_t(262144), size_t(262152), size_t(524280), size_t(524288), size_t(524292), size_t(524296)}) {
            if (type != GGML_TYPE_F32 && n % 8 != 0) {
                continue;
            }
            ggml_init_params params = {2 * ggml_tensor_overhead() + 256, nullptr, true};
            ggml_context * ctx = ggml_init(params);
            require(ctx != nullptr, "tensor context");
            ggml_tensor * tensors[2];
            ggml_backend_buffer_t buffers[2];
            const size_t bytes = n * ggml_type_size(type);
            require(bytes % 16 == 0, "AllReduce input alignment");
            for (int rank = 0; rank < 2; ++rank) {
                tensors[rank] = ggml_new_tensor_1d(ctx, type, n);
                buffers[rank] = ggml_backend_alloc_buffer(backends[rank], bytes + 256);
                require(buffers[rank] != nullptr, "tensor buffer");
                require(ggml_backend_tensor_alloc(buffers[rank], tensors[rank], ggml_backend_buffer_get_base(buffers[rank])) == GGML_STATUS_SUCCESS, "tensor allocation");
            }
            std::vector<uint8_t> data[2] = {std::vector<uint8_t>(bytes), std::vector<uint8_t>(bytes)};
            std::vector<uint8_t> expected(bytes), actual(bytes);
            // Integer operands and sums are exact in all three formats, including F32 with BF16 wire conversion.
            for (int inactive = -1; inactive < 2; ++inactive) {
                for (int rank = 0; rank < 2; ++rank) {
                    tensors[rank]->flags = rank == inactive ? 0 : GGML_TENSOR_FLAG_COMPUTE;
                    for (size_t i = 0; i < n; ++i) {
                        encode(data[rank], type, i, input(i, rank, inactive + 1));
                    }
                    ggml_backend_tensor_set(tensors[rank], data[rank].data(), 0, bytes);
                }
                for (size_t i = 0; i < n; ++i) {
                    const float sum = (inactive == 0 ? 0 : input(i, 0, inactive + 1)) +
                                      (inactive == 1 ? 0 : input(i, 1, inactive + 1));
                    encode(expected, type, i, sum);
                }
                require(reduce(comm, tensors), "AllReduce declined input");
                for (int rank = 0; rank < 2; ++rank) {
                    ggml_backend_synchronize(backends[rank]);
                    ggml_backend_tensor_get(tensors[rank], actual.data(), 0, bytes);
                    if (!equal(actual, expected)) {
                        fprintf(stderr, "type=%s n=%zu inactive=%d rank=%d\n", ggml_type_name(type), n, inactive, rank);
                        require(false, "AllReduce bytes differ from exact sum");
                    }
                    if (file) {
                        require(fwrite(actual.data(), 1, bytes, file) == bytes, "write output");
                    }
                }
                ++cases;
            }
            for (int rank = 0; rank < 2; ++rank) {
                ggml_backend_buffer_free(buffers[rank]);
            }
            ggml_free(ctx);
        }
    }
    if (file) {
        require(fclose(file) == 0, "close output");
    }
    free_comm(comm);
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
    printf("PASS: %zu AllReduce cases\n", cases);
    return 0;
}
