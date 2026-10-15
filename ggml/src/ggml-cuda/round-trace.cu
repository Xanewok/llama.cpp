#include "round-trace.cuh"

#include <chrono>
#include <thread>

struct trace_cuda_pair {
    cudaEvent_t begin, end;
};

struct trace_cuda_pool {
    int device;
    cudaStream_t reference_stream;
    trace_cuda_pair reference;
    std::vector<trace_cuda_pair> events;
    std::vector<size_t> free;
    std::mutex mutex;

    explicit trace_cuda_pool(int device) : device(device) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&reference_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&reference.begin));
        CUDA_CHECK(cudaEventCreate(&reference.end));
        // A fixed boot-time pool must fail rather than omit intervals or stall submission.
        events.resize(16384);
        for (size_t i = 0; i < events.size(); ++i) {
            CUDA_CHECK(cudaEventCreate(&events[i].begin));
            CUDA_CHECK(cudaEventCreate(&events[i].end));
            free.push_back(i);
        }
    }

    ~trace_cuda_pool() {
        ggml_cuda_set_device(device);
        for (auto & e : events) {
            CUDA_CHECK(cudaEventDestroy(e.begin));
            CUDA_CHECK(cudaEventDestroy(e.end));
        }
        CUDA_CHECK(cudaEventDestroy(reference.begin));
        CUDA_CHECK(cudaEventDestroy(reference.end));
        CUDA_CHECK(cudaStreamDestroy(reference_stream));
    }
};

static void trace_cuda_ready(cudaEvent_t event) {
    cudaError_t status;
    while ((status = cudaEventQuery(event)) == cudaErrorNotReady) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    CUDA_CHECK(status);
}

struct ggml_cuda_trace_batch : ggml_round_trace_gpu_source {
    struct entry {
        const char * kind;
        size_t phase, slot;
        cudaStream_t stream;
    };
    std::shared_ptr<trace_cuda_pool> pool;
    std::vector<entry> entries;

    explicit ggml_cuda_trace_batch(std::shared_ptr<trace_cuda_pool> pool) : pool(std::move(pool)) {}

    void read(std::vector<ggml_round_trace_gpu_interval> & out) override {
        ggml_cuda_set_device(pool->device);
        for (const auto & e : entries) { trace_cuda_ready(pool->events[e.slot].end); }
        int64_t lo = ggml_round_trace_now();
        CUDA_CHECK(cudaEventRecord(pool->reference.begin, pool->reference_stream));
        CUDA_CHECK(cudaEventRecord(pool->reference.end, pool->reference_stream));
        trace_cuda_ready(pool->reference.end);
        const int64_t hi = ggml_round_trace_now();
        float reference_ms;
        CUDA_CHECK(cudaEventElapsedTime(&reference_ms, pool->reference.begin, pool->reference.end));
        // The final reference cannot precede the first reference plus their measured separation.
        lo += std::max<int64_t>(0, std::llround(double(reference_ms) * 1e6) - 500);
        GGML_ASSERT(lo <= hi);
        for (const auto & e : entries) {
            const auto & pair = pool->events[e.slot];
            float begin_ms, end_ms;
            CUDA_CHECK(cudaEventElapsedTime(&begin_ms, pair.begin, pool->reference.end));
            CUDA_CHECK(cudaEventElapsedTime(&end_ms, pair.end, pool->reference.end));
            out.push_back({e.kind, e.phase, pool->device, (uint64_t) (uintptr_t) e.stream,
                ggml_round_trace_gpu_map(lo, hi, begin_ms), ggml_round_trace_gpu_map(lo, hi, end_ms),
                ggml_round_trace_gpu_error(lo, hi, begin_ms)});
        }
        std::lock_guard<std::mutex> lock(pool->mutex);
        for (const auto & e : entries) { pool->free.push_back(e.slot); }
    }
};

struct trace_cuda_device {
    std::shared_ptr<trace_cuda_pool> pool;
    std::shared_ptr<ggml_cuda_trace_batch> batch;
    uint64_t round = 0;
};

// The batch retains its pool through thread exit and backend destruction.
static thread_local std::unordered_map<int, trace_cuda_device> trace_cuda_devices;

void ggml_cuda_trace_scope::begin(const char * kind, int device) {
    size_t phase;
    const auto round = ggml_round_trace_gpu_round(phase);
    if (!round) { return; }
    ggml_cuda_set_device(device);
    auto & d = trace_cuda_devices[device];
    if (!d.pool) { d.pool = std::make_shared<trace_cuda_pool>(device); }
    if (d.round != round) {
        d.round = round;
        d.batch = std::make_shared<ggml_cuda_trace_batch>(d.pool);
        ggml_round_trace_gpu_defer(d.batch);
    }
    batch = d.batch.get();
    size_t slot;
    {
        std::lock_guard<std::mutex> lock(d.pool->mutex);
        GGML_ASSERT(!d.pool->free.empty() && "LLAMA_ROUND_TRACE_GPU event pool exhausted");
        slot = d.pool->free.back();
        d.pool->free.pop_back();
    }
    index = batch->entries.size();
    batch->entries.push_back({kind, phase, slot, stream});
    CUDA_CHECK(cudaEventRecord(d.pool->events[slot].begin, stream));
}

void ggml_cuda_trace_scope::end() {
    if (batch) {
        ggml_cuda_set_device(batch->pool->device);
        CUDA_CHECK(cudaEventRecord(batch->pool->events[batch->entries[index].slot].end, stream));
    }
}
