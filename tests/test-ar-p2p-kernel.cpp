// GGML_CUDA_AR_P2P_KERNEL=1 must reproduce the P2P staging AllReduce bit for bit,
// with many reductions queued between host synchronizations.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string g_log;

static void log_capture(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    g_log += text;
    fputs(text, stderr);
}

static void require(bool ok, const char * message) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static uint32_t next(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
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

struct pair {
    ggml_tensor *         t[2];
    std::vector<uint8_t>  input[2];
    std::vector<uint8_t>  out_ref[2];
    int                   inactive;
};

static void * init_comm(ggml_backend_comm_init_t init, ggml_backend_t * backends, const char * kernel) {
    setenv("GGML_CUDA_AR_P2P_KERNEL", kernel, 1);
    g_log.clear();
    void * comm = init(backends, 2);
    require(comm != nullptr, "internal comm init");
    return comm;
}

static void run(ggml_backend_comm_allreduce_tensor_t reduce, void * comm, ggml_backend_t * backends,
                std::vector<pair> & pairs, bool ref, size_t & mismatches) {
    for (auto & p : pairs) {
        for (int rank = 0; rank < 2; ++rank) {
            p.t[rank]->flags = rank == p.inactive ? 0 : GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_tensor_set(p.t[rank], p.input[rank].data(), 0, p.input[rank].size());
        }
    }
    for (auto & p : pairs) {
        require(reduce(comm, p.t), "AllReduce declined input");
    }
    for (int rank = 0; rank < 2; ++rank) {
        ggml_backend_synchronize(backends[rank]);
    }
    std::vector<uint8_t> actual;
    for (auto & p : pairs) {
        for (int rank = 0; rank < 2; ++rank) {
            actual.resize(p.input[rank].size());
            ggml_backend_tensor_get(p.t[rank], actual.data(), 0, actual.size());
            if (ref) {
                p.out_ref[rank] = actual;
            } else if (actual != p.out_ref[rank]) {
                fprintf(stderr, "type=%s n=%lld inactive=%d rank=%d differs\n", ggml_type_name(p.t[0]->type),
                        (long long) ggml_nelements(p.t[0]), p.inactive, rank);
                ++mismatches;
            }
        }
        if (ref && p.out_ref[0] != p.out_ref[1]) {
            fprintf(stderr, "reference ranks differ: type=%s n=%lld\n", ggml_type_name(p.t[0]->type),
                    (long long) ggml_nelements(p.t[0]));
            ++mismatches;
        }
    }
}

int main() {
    ggml_log_set(log_capture, nullptr);
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

    void * comm_ref = init_comm(init, backends, "0");
    if (g_log.find("allreduce: P2P staging (peer device buffers)") == std::string::npos) {
        puts("SKIP: P2P staging unavailable (set GGML_CUDA_AR_P2P=1 on a P2P-capable pair)");
        return 77;
    }
    void * comm_ker = init_comm(init, backends, "1");
    require(g_log.find("allreduce: P2P kernel sync (device epochs, no host slot wait)") != std::string::npos,
            "P2P kernel path not engaged");

    // Decode-sized rows of a 5120-wide model, scalar tails, and copy-engine sizes (>= 1 MiB on the wire) between
    // them, so the queue interleaves both paths; 45 kernel-path reductions per pass flip the epoch parity per pass.
    const size_t sizes[] = {4, 8, 260, 5120, 6 * 5120, 12 * 5120, 262140, 524296};
    ggml_init_params params = {512 * ggml_tensor_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "tensor context");
    std::vector<pair> pairs;
    uint32_t state = 0x9e3779b9u;
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        for (size_t n : sizes) {
            if (type != GGML_TYPE_F32 && n % 8 != 0) {
                continue;
            }
            for (int inactive = -1; inactive < 2; ++inactive) {
                pair p;
                p.inactive = inactive;
                for (int rank = 0; rank < 2; ++rank) {
                    p.t[rank] = ggml_new_tensor_1d(ctx, type, n);
                    p.input[rank].resize(ggml_nbytes(p.t[rank]));
                    for (size_t i = 0; i < n; ++i) {
                        // Non-integer values make the BF16 wire round both operands.
                        const float v = (float(int(next(state) % 2000001u)) - 1000000.0f) * 1.0e-5f;
                        encode(p.input[rank], type, i, v);
                    }
                }
                pairs.push_back(std::move(p));
            }
        }
    }
    ggml_backend_buffer_t buffers[2];
    for (int rank = 0; rank < 2; ++rank) {
        size_t bytes = 0;
        for (auto & p : pairs) {
            bytes += GGML_PAD(ggml_nbytes(p.t[rank]), 256);
        }
        buffers[rank] = ggml_backend_alloc_buffer(backends[rank], bytes + 256);
        require(buffers[rank] != nullptr, "tensor buffer");
        uint8_t * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffers[rank]));
        size_t offset = 0;
        for (auto & p : pairs) {
            require(ggml_backend_tensor_alloc(buffers[rank], p.t[rank], base + offset) == GGML_STATUS_SUCCESS, "tensor allocation");
            offset += GGML_PAD(ggml_nbytes(p.t[rank]), 256);
        }
    }

    size_t mismatches = 0;
    run(reduce, comm_ref, backends, pairs, true, mismatches);
    const int passes = 3;
    for (int pass = 0; pass < passes; ++pass) {
        run(reduce, comm_ker, backends, pairs, false, mismatches);
    }
    if (mismatches != 0) {
        fprintf(stderr, "FAIL: %zu tensors differ from the P2P staging AllReduce\n", mismatches);
        return 1;
    }

    free_comm(comm_ker);
    free_comm(comm_ref);
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(ctx);
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
    printf("PASS: P2P kernel AllReduce bitwise == P2P staging, %zu cases x %d queued passes, ranks equal\n",
           pairs.size(), passes);
    return 0;
}
