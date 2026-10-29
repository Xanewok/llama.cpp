#include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include <nccl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
        std::exit(1); \
    } \
} while (0)
#define NCCL_CHECK(call) do { \
    ncclResult_t status = (call); \
    if (status != ncclSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, ncclGetErrorString(status)); \
        std::exit(1); \
    } \
} while (0)

static int pattern(size_t index, int rank) {
    return static_cast<int>((index * (rank ? 37 : 19) + rank * 101) % 1009) - 504;
}

template<typename T>
static void check_floating(ncclDataType_t dtype, const char *name, ncclComm_t *comms,
                           cudaStream_t *streams, int **send, int **recv) {
    auto synchronize = [&]() {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(rank));
            CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
        }
    };
    for (size_t bytes : {size_t(4), size_t(1024), size_t(16384), size_t(1048576), size_t(16777216)}) {
        const size_t elements = bytes / sizeof(T);
        std::vector<T> host(elements);
        for (int rank = 0; rank < 2; ++rank) {
            // Inputs and sums are exactly representable in BF16, so any mismatch is an error.
            for (size_t index = 0; index < elements; ++index) host[index] = T(float(pattern(index, rank) % 64));
            CUDA_CHECK(cudaSetDevice(rank));
            CUDA_CHECK(cudaMemcpy(send[rank], host.data(), bytes, cudaMemcpyHostToDevice));
        }
        auto reduce = [&]() {
            NCCL_CHECK(ncclGroupStart());
            for (int rank = 0; rank < 2; ++rank) {
                CUDA_CHECK(cudaSetDevice(rank));
                NCCL_CHECK(ncclAllReduce(send[rank], recv[rank], elements, dtype,
                                         ncclSum, comms[rank], streams[rank]));
            }
            NCCL_CHECK(ncclGroupEnd());
        };
        for (int warmup = 0; warmup < 5; ++warmup) reduce();
        synchronize();
        constexpr int iterations = 20;
        auto started = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) reduce();
        synchronize();
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        size_t errors = 0;
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(rank));
            CUDA_CHECK(cudaMemcpy(host.data(), recv[rank], bytes, cudaMemcpyDeviceToHost));
            for (size_t index = 0; index < elements; ++index) {
                errors += float(host[index]) != float(pattern(index, 0) % 64 + pattern(index, 1) % 64);
            }
        }
        std::printf("{\"dtype\":\"%s\",\"bytes_per_rank\":%zu,\"iterations\":%d,\"host_batch_us_per_collective\":%.3f,\"errors\":%zu}\n",
                    name, bytes, iterations, elapsed * 1e6 / iterations, errors);
        std::fflush(stdout);
        if (errors) std::exit(1);
    }
}

int main() {
    static_assert(sizeof(int) == 4);
    int count = 0, version = 0, driver = 0, runtime = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count != 2) {
        std::fprintf(stderr, "Expose exactly two GPUs through CUDA_VISIBLE_DEVICES.\n");
        return 2;
    }
    NCCL_CHECK(ncclGetVersion(&version));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::printf("{\"nccl_version\":%d,\"cuda_driver\":%d,\"cuda_runtime\":%d}\n",
                version, driver, runtime);
    for (int rank = 0; rank < 2; ++rank) {
        int peer = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&peer, rank, 1 - rank));
        std::printf("{\"device\":%d,\"peer\":%d,\"can_access_peer\":%d}\n",
                    rank, 1 - rank, peer);
    }
    std::fflush(stdout);

    ncclComm_t comms[2];
    int devices[2] = {0, 1};
    NCCL_CHECK(ncclCommInitAll(comms, 2, devices));
    cudaStream_t streams[2];
    int *send[2], *recv[2];
    constexpr size_t maximum = 4 * 1024 * 1024;
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(rank));
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[rank], cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&send[rank]), maximum * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&recv[rank]), maximum * sizeof(int)));
    }
    auto reduce = [&](size_t elements) {
        NCCL_CHECK(ncclGroupStart());
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(rank));
            NCCL_CHECK(ncclAllReduce(send[rank], recv[rank], elements, ncclInt32,
                                     ncclSum, comms[rank], streams[rank]));
        }
        NCCL_CHECK(ncclGroupEnd());
    };
    auto synchronize = [&]() {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(rank));
            CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
        }
    };
    for (size_t elements : {size_t(1), size_t(256), size_t(4096), size_t(262144), maximum}) {
        std::vector<int> host(elements);
        for (int rank = 0; rank < 2; ++rank) {
            for (size_t index = 0; index < elements; ++index) host[index] = pattern(index, rank);
            CUDA_CHECK(cudaSetDevice(rank));
            CUDA_CHECK(cudaMemcpy(send[rank], host.data(), elements * sizeof(int), cudaMemcpyHostToDevice));
        }
        for (int warmup = 0; warmup < 5; ++warmup) reduce(elements);
        synchronize();
        constexpr int iterations = 20;
        auto started = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) reduce(elements);
        synchronize();
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        size_t errors = 0;
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(rank));
            CUDA_CHECK(cudaMemcpy(host.data(), recv[rank], elements * sizeof(int), cudaMemcpyDeviceToHost));
            for (size_t index = 0; index < elements; ++index) {
                errors += host[index] != pattern(index, 0) + pattern(index, 1);
            }
        }
        // Includes host submission and final synchronization; not raw device/link latency.
        std::printf("{\"dtype\":\"int32\",\"bytes_per_rank\":%zu,\"iterations\":%d,\"host_batch_us_per_collective\":%.3f,\"errors\":%zu}\n",
                    elements * sizeof(int), iterations, elapsed * 1e6 / iterations, errors);
        std::fflush(stdout);
        if (errors) return 1;
    }
    check_floating<float>(ncclFloat, "float32", comms, streams, send, recv);
    check_floating<__nv_bfloat16>(ncclBfloat16, "bfloat16", comms, streams, send, recv);
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(rank));
        CUDA_CHECK(cudaFree(send[rank]));
        CUDA_CHECK(cudaFree(recv[rank]));
        CUDA_CHECK(cudaStreamDestroy(streams[rank]));
        NCCL_CHECK(ncclCommDestroy(comms[rank]));
    }
    return 0;
}
