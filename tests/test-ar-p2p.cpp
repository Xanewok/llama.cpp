#include "ggml-backend-impl.h"
#include "hash/hash.h"
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <thread>

static void require(bool ok, const char * message) {
    if (!ok) { fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
}
#define CUDA_CHECK_TEST(call) require((call) == cudaSuccess, #call)
using steady_clock = std::chrono::steady_clock;
static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now().time_since_epoch()).count();
}

struct watchdog {
    struct call {
        std::atomic<int64_t> start{0};
        std::atomic<unsigned> finished{0};
    } pending[8];
    std::atomic<bool> done{false};
    std::thread worker;
    watchdog() : worker([this] {
        while (!done.load()) {
            for (auto & c : pending) {
                const int64_t t = c.start.load();
                if (t && now_ms() - t > 10000 && c.start.load() == t) {
                    fprintf(stderr, "FAIL AllReduce watchdog: call/readback exceeded 10 s\n");
                    std::_Exit(3);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }) {}
    ~watchdog() { done = true; worker.join(); }
};

static void CUDART_CB finish_call(void * data) {
    auto * c = static_cast<watchdog::call *>(data);
    if (c->finished.fetch_add(1) == 1) { c->start = 0; }
}

static __global__ void spin(unsigned long long cycles) {
    const unsigned long long begin = clock64();
    while (clock64() - begin < cycles) { asm volatile(""); }
}

struct host_io {
    uint8_t * memory = nullptr;
    uint8_t * input[2];
    uint8_t * output[2];
    void alloc(size_t bytes) {
        CUDA_CHECK_TEST(cudaMallocHost(reinterpret_cast<void **>(&memory), 4 * bytes));
        for (int r = 0; r < 2; ++r) {
            input[r] = memory + r * bytes;
            output[r] = memory + (r + 2) * bytes;
        }
    }
    ~host_io() { if (memory) { CUDA_CHECK_TEST(cudaFreeHost(memory)); } }
};

static void fill(uint8_t * data, ggml_type type, size_t n, int rank, unsigned iteration) {
    uint32_t state = 0x1234567u ^ (rank + 1) * 0x9e3779b9u ^ iteration * 0x85ebca6bu ^ uint32_t(n);
    static const unsigned exponent[] = {0, 1, 80, 120, 127, 140, 150};
    static const unsigned exponent_half[] = {0, 1, 5, 10, 15, 20, 26};
    for (size_t i = 0; i < n; ++i) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        // Raw finite encodings retain subnormals without host floating-point conversion.
        const unsigned e = (i + iteration) % 7;
        if (type == GGML_TYPE_F32) {
            reinterpret_cast<uint32_t *>(data)[i] = (state & 0x807fffffu) | (exponent[e] << 23);
        } else {
            reinterpret_cast<uint16_t *>(data)[i] = type == GGML_TYPE_F16 ?
                uint16_t((state & 0x83ffu) | (exponent_half[e] << 10)) :
                uint16_t((state & 0x807fu) | (exponent[e] << 7));
        }
    }
}

static std::string env_value(const char * name, const char * fallback) {
    const char * value = std::getenv(name);
    return value ? value : fallback;
}

int main() {
    require(env_value("GGML_CUDA_ALLREDUCE", "") == "internal", "set GGML_CUDA_ALLREDUCE=internal");
    const std::string config = "copy=" + env_value("GGML_CUDA_AR_COPY_THRESHOLD", "default") +
        ",bf16=" + env_value("GGML_CUDA_AR_BF16_THRESHOLD", "default") +
        ",chunk=" + env_value("GGML_CUDA_AR_COPY_CHUNK_BYTES", "auto");
    const uint64_t bf16_threshold = std::strtoull(env_value("GGML_CUDA_AR_BF16_THRESHOLD", "1").c_str(), nullptr, 10);
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    require(reg && ggml_backend_reg_dev_count(reg) >= 2, "two CUDA devices required");
    auto init = reinterpret_cast<ggml_backend_comm_init_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init"));
    auto reduce = reinterpret_cast<ggml_backend_comm_allreduce_tensor_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor"));
    auto free_comm = reinterpret_cast<ggml_backend_comm_free_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free"));
    require(init && reduce && free_comm, "CUDA comm API missing");
    ggml_backend_t backend[2];
    ggml_backend_buffer_t buffer[2];
    ggml_backend_event_t delay_event[2], completed_event[2];
    cudaStream_t delay_stream[2];
    int device[2];
    const size_t max_n = (1u << 24) + 8;
    for (int r = 0; r < 2; ++r) {
        backend[r] = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, r), nullptr);
        require(backend[r] != nullptr, "CUDA backend init");
        buffer[r] = ggml_backend_alloc_buffer(backend[r], max_n * 4 + 256);
        require(buffer[r] != nullptr, "device allocation");
        cudaPointerAttributes attr{};
        CUDA_CHECK_TEST(cudaPointerGetAttributes(&attr, ggml_backend_buffer_get_base(buffer[r])));
        device[r] = attr.device;
        CUDA_CHECK_TEST(cudaSetDevice(device[r]));
        CUDA_CHECK_TEST(cudaStreamCreateWithFlags(&delay_stream[r], cudaStreamNonBlocking));
        delay_event[r] = ggml_backend_event_new(ggml_backend_get_device(backend[r]));
        completed_event[r] = ggml_backend_event_new(ggml_backend_get_device(backend[r]));
        require(delay_event[r] && completed_event[r], "CUDA events required");
    }
    require(device[0] != device[1], "two distinct physical CUDA devices required");
    void * comm = init(backend, 2);
    require(comm != nullptr, "comm init");
    watchdog guard;
    bool all_equal = true;
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_F16}) {
        const char * dtype = type == GGML_TYPE_F32 ? "F32" : type == GGML_TYPE_BF16 ? "BF16" : "F16";
        const size_t bytes_per_element = ggml_type_size(type), pack = 16 / bytes_per_element;
        const size_t wire = type == GGML_TYPE_F32 && bf16_threshold ? 2 : bytes_per_element;
        ggml_tensor tensor[2]{};
        ggml_tensor * tensors[] = {&tensor[0], &tensor[1]};
        auto enqueue = [&](host_io & io, size_t n, unsigned iteration, unsigned slot, bool delayed, bool inactive) {
            guard.pending[slot].finished = 0;
            guard.pending[slot].start = now_ms();
            const size_t bytes = n * bytes_per_element;
            for (int k = 0; k < 2; ++k) {
                const int r = (k + iteration) % 2;
                tensor[r] = {};
                tensor[r].type = type;
                for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                    tensor[r].ne[d] = d == 0 ? int64_t(n) : 1;
                    tensor[r].nb[d] = d == 0 ? bytes_per_element : bytes;
                }
                tensor[r].flags = inactive && r == int((iteration / 7) % 2) ? 0 : GGML_TENSOR_FLAG_COMPUTE;
                require(ggml_backend_tensor_alloc(buffer[r], &tensor[r], ggml_backend_buffer_get_base(buffer[r])) == GGML_STATUS_SUCCESS, "tensor allocation");
                CUDA_CHECK_TEST(cudaSetDevice(device[r]));
                ggml_backend_tensor_set_async(backend[r], &tensor[r], io.input[r], 0, bytes);
                if (delayed && r == int((iteration / 11) % 2)) {
                    // Bridge the spin stream into the backend stream without accessing its CUDA context.
                    spin<<<1, 1, 0, delay_stream[r]>>>(2000000);
                    CUDA_CHECK_TEST(cudaGetLastError());
                    CUDA_CHECK_TEST(cudaEventRecord(static_cast<cudaEvent_t>(delay_event[r]->context), delay_stream[r]));
                    ggml_backend_event_wait(backend[r], delay_event[r]);
                }
            }
            require(reduce(comm, tensors), "internal AllReduce declined input");
            for (int r = 0; r < 2; ++r) {
                CUDA_CHECK_TEST(cudaSetDevice(device[r]));
                ggml_backend_tensor_get_async(backend[r], &tensor[r], io.output[r], 0, bytes);
                ggml_backend_event_record(completed_event[r], backend[r]);
                CUDA_CHECK_TEST(cudaStreamWaitEvent(delay_stream[r], static_cast<cudaEvent_t>(completed_event[r]->context), 0));
                CUDA_CHECK_TEST(cudaLaunchHostFunc(delay_stream[r], finish_call, &guard.pending[slot]));
            }
        };
        auto synchronize = [&] {
            for (int r = 0; r < 2; ++r) {
                CUDA_CHECK_TEST(cudaSetDevice(device[r]));
                ggml_backend_synchronize(backend[r]);
                CUDA_CHECK_TEST(cudaStreamSynchronize(delay_stream[r]));
            }
        };
        std::set<size_t> sizes;
        for (size_t n = 1; n <= 64; ++n) { sizes.insert(n); }
        for (size_t n = 1; n <= (1u << 24); n *= 2) {
            for (int delta : {-1, 0, 1}) { if (n + delta > 0) { sizes.insert(n + delta); } }
        }
        for (size_t boundary : {size_t(1u << 20), size_t(32u << 20)}) {
            const size_t n = boundary / wire;
            for (int delta : {-int(pack), -1, 0, 1, int(pack)}) { sizes.insert(n + delta); }
        }
        {
            host_io io;
            io.alloc(max_n * bytes_per_element);
            for (size_t n : sizes) {
                if (n % pack) {
                    printf("skip %s %zu %s requires-16-byte-count\n", dtype, n, config.c_str());
                    continue;
                }
                for (int r = 0; r < 2; ++r) { fill(io.input[r], type, n, r, 0); }
                enqueue(io, n, 0, 0, false, false);
                synchronize();
                const bool equal = std::memcmp(io.output[0], io.output[1], n * bytes_per_element) == 0;
                all_equal &= equal;
                printf("case %s %zu %s sha256 %s %s ranks_equal=%d\n", dtype, n, config.c_str(),
                    hash_sha256_hex(io.output[0], n * bytes_per_element).c_str(),
                    hash_sha256_hex(io.output[1], n * bytes_per_element).c_str(), int(equal));
            }
        }
        constexpr unsigned depth = 8, iterations = 5000;
        const size_t large = (1u << 20) / wire + pack;
        host_io io[depth];
        for (auto & slot : io) { slot.alloc(large * bytes_per_element); }
        uint64_t digest_xor[2] = {}, digest_sum[2] = {};
        std::string final_hash[2];
        bool stress_equal = true;
        size_t counts[depth]{};
        const int64_t begin = now_ms();
        for (unsigned i = 0; i < iterations + depth; ++i) {
            const unsigned slot = i % depth;
            if (i >= depth) {
                // Reuse host storage only after both readbacks; keep the other calls in flight.
                while (guard.pending[slot].start.load()) { std::this_thread::yield(); }
                const size_t bytes = counts[slot] * bytes_per_element;
                stress_equal &= std::memcmp(io[slot].output[0], io[slot].output[1], bytes) == 0;
                for (int r = 0; r < 2; ++r) {
                    final_hash[r] = hash_sha256_hex(io[slot].output[r], bytes);
                    for (size_t k = 0; k < 64; k += 16) {
                        const uint64_t word = std::strtoull(final_hash[r].substr(k, 16).c_str(), nullptr, 16);
                        digest_xor[r] ^= word;
                        digest_sum[r] += word;
                    }
                }
            }
            if (i < iterations) {
                counts[slot] = i % 2 ? large : pack * (1 + i % 8);
                for (int r = 0; r < 2; ++r) { fill(io[slot].input[r], type, counts[slot], r, i + 1); }
                enqueue(io[slot], counts[slot], i + 1, slot, i % 11 == 0, i % 7 == 0);
            }
        }
        synchronize();
        all_equal &= stress_equal;
        printf("stress %s %u %s sha256 %s %s xor %016llx %016llx sum %016llx %016llx ranks_equal=%d\n",
            dtype, iterations, config.c_str(), final_hash[0].c_str(), final_hash[1].c_str(),
            (unsigned long long) digest_xor[0], (unsigned long long) digest_xor[1],
            (unsigned long long) digest_sum[0], (unsigned long long) digest_sum[1], int(stress_equal));
        printf("time %s stress_ms=%lld\n", dtype, (long long) (now_ms() - begin));
    }
    free_comm(comm);
    for (int r = 0; r < 2; ++r) {
        CUDA_CHECK_TEST(cudaSetDevice(device[r]));
        ggml_backend_event_free(delay_event[r]);
        ggml_backend_event_free(completed_event[r]);
        CUDA_CHECK_TEST(cudaStreamDestroy(delay_stream[r]));
        ggml_backend_buffer_free(buffer[r]);
        ggml_backend_free(backend[r]);
    }
    return all_equal ? 0 : 1;
}
