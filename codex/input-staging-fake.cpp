// CPU delayed queues exercise the production CUDA upload function without CUDA runtime calls.
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#define GGML_ASSERT assert
#define GGML_UNUSED(x) (void) (x)
#define CUDART_VERSION 13000
struct cudaMemcpyAttributes { int srcAccessOrder; };
constexpr int cudaMemcpySrcAccessOrderStream = 0;
#define CUDA_CHECK(expr) do { if ((expr) != 0) { std::cerr << #expr << " failed\n"; std::abort(); } } while (0)
struct queue { std::deque<std::function<void()>> jobs; size_t submitted = 0, completed = 0; };
using cudaStream_t = queue *;
struct event { queue * stream = nullptr; size_t target = 0; };
using cudaEvent_t = event *;
using cudaError_t = int;
constexpr int cudaErrorNotReady = 1;
constexpr int cudaSuccess = 0, cudaHostAllocDefault = 0, cudaEventDisableTiming = 0, cudaStreamNonBlocking = 0, cudaMemcpyHostToDevice = 0;
static bool fail_alloc = false;
static size_t allocation_calls = 0;
static void drain(queue * q, size_t target) {
    while (q->completed < target) {
        assert(!q->jobs.empty());
        auto job = std::move(q->jobs.front()); q->jobs.pop_front();
        ++q->completed;
        job();
    }
}
static void enqueue(queue * q, std::function<void()> job) { q->jobs.push_back(std::move(job)); ++q->submitted; }
static int cudaEventQuery(event * e) { return e->stream->completed >= e->target ? cudaSuccess : cudaErrorNotReady; }
static int cudaEventSynchronize(event * e) { drain(e->stream, e->target); return 0; }
static int cudaHostAlloc(void ** p, size_t n, int) { ++allocation_calls; if (fail_alloc) { return 1; } *p = malloc(n); return *p ? 0 : 1; }
static int cudaGetLastError() { return 0; }
static int cudaFreeHost(void * p) { free(p); return 0; }
static int cudaEventCreateWithFlags(event ** e, int) { *e = new event; return 0; }
static int cudaEventDestroy(event * e) { delete e; return 0; }
static int cudaStreamCreateWithFlags(queue ** q, int) { *q = new queue; return 0; }
static int cudaStreamDestroy(queue * q) { drain(q, q->submitted); delete q; return 0; }
static int cudaEventRecord(event * e, queue * q) { *e = {q, q->submitted}; return 0; }
static int cudaStreamWaitEvent(queue * q, event * e, int) {
    event snapshot = *e;
    enqueue(q, [snapshot] { drain(snapshot.stream, snapshot.target); });
    return 0;
}
static int cudaMemcpyAsync(void * dst, const void * src, size_t n, int, queue * q) {
    enqueue(q, [=] { memcpy(dst, src, n); }); return 0;
}
static int cudaMemcpyBatchAsync(void ** dst, const void ** src, size_t * sizes, size_t n,
        cudaMemcpyAttributes *, size_t *, size_t, queue * q) {
    // Reverse the copies to expose illegal use of unordered batches for overlaps.
    for (size_t i = n; i > 0; --i) {
        cudaMemcpyAsync(dst[i-1], src[i-1], sizes[i-1], cudaMemcpyHostToDevice, q);
    }
    return 0;
}
struct ggml_backend_buffer { void * buft; };
using ggml_backend_buffer_t = ggml_backend_buffer *;
struct ggml_tensor { ggml_tensor * view_src; ggml_backend_buffer_t buffer; void * data; size_t bytes; };
struct ggml_backend_tensor_transfer { ggml_tensor * tensor; const void * data; size_t offset, size; };
static size_t ggml_nbytes(const ggml_tensor * t) { return t->bytes; }
static void * ggml_backend_cuda_buffer_type(int) { return (void *) 1; }
static void ggml_cuda_set_device(int) {}
struct ggml_round_trace_scope { ggml_round_trace_scope(const char *, int, size_t, size_t) {} };
static void ggml_round_trace_count(const char *, int) {}
struct ggml_backend_cuda_context {
    int device = 0;
    queue compute;
    // PRODUCTION_FIELDS
    queue * stream() { return &compute; }
    ~ggml_backend_cuda_context() {
        // PRODUCTION_CLEANUP
        drain(&compute, compute.submitted);
    }
};
struct ggml_backend { void * context; };
using ggml_backend_t = ggml_backend *;
// PRODUCTION_UPLOAD
static void require(bool ok, const char * message) { if (!ok) { throw std::runtime_error(message); } }
int main() {
    try {
        ggml_backend_cuda_context ctx;
        ggml_backend backend{&ctx};
        ggml_backend_buffer buffer{(void *) 1};
        int dst = -1;
        ggml_tensor t{nullptr, &buffer, &dst, sizeof(dst)};
        require(ggml_backend_cuda_set_tensor_batch_owned(&backend, nullptr, 0), "capability query failed");
        require(ctx.input_stream == nullptr, "query allocated resources");
        std::vector<int> observed;
        for (int i = 0; i < 10; ++i) {
            int src = i + 100;
            ggml_backend_tensor_transfer transfer{&t, &src, 0, sizeof(src)};
            require(ggml_backend_cuda_set_tensor_batch_owned(&backend, &transfer, 1), "upload refused");
            src = -99;
            enqueue(ctx.stream(), [&] { observed.push_back(dst); });
        }
        // Advancing uploads first exposes a missing compute-to-upload dependency.
        drain(ctx.input_stream, ctx.input_stream->submitted);
        drain(ctx.stream(), ctx.compute.submitted);
        for (int i = 0; i < 10; ++i) {
            require(observed.at(i) == i + 100, "stale staging or missing event wait");
        }
        const int first = 111, last = 222;
        ggml_backend_tensor_transfer overlap[] = {{&t, &first, 0, sizeof(first)}, {&t, &last, 0, sizeof(last)}};
        require(ggml_backend_cuda_set_tensor_batch_owned(&backend, overlap, 2), "overlap refused");
        drain(ctx.stream(), ctx.compute.submitted);
        require(dst == last, "overlap order lost");
        const size_t submitted = ctx.compute.submitted;
        buffer.buft = (void *) 2;
        require(!ggml_backend_cuda_set_tensor_batch_owned(&backend, overlap, 2), "foreign buffer accepted");
        require(ctx.compute.submitted == submitted, "refusal queued work");
        buffer.buft = (void *) 1;
        // A fresh context must refuse allocation failure without touching destinations.
        {
            ggml_backend_cuda_context fresh;
            ggml_backend fresh_backend{&fresh};
            fail_alloc = true;
            require(!ggml_backend_cuda_set_tensor_batch_owned(&fresh_backend, overlap, 2), "allocation failure accepted");
            fail_alloc = false;
            require(dst == last && fresh.input_stream == nullptr, "allocation fallback had side effects");
        }
        // Destruction must finish the last slot's upload even without explicit synchronization.
        {
            ggml_backend_cuda_context fresh;
            ggml_backend fresh_backend{&fresh};
            require(ggml_backend_cuda_set_tensor_batch_owned(&fresh_backend, overlap, 2), "teardown upload refused");
        }
        require(dst == last, "teardown lost upload");
        {
            ggml_backend_cuda_context growing;
            ggml_backend growing_backend { &growing };
            std::vector<char> source(256), destination(256);
            ggml_tensor target { nullptr, &buffer, destination.data(), destination.size() };
            const size_t before = allocation_calls;
            for (size_t size = 1; size <= source.size(); ++size) {
                std::fill(source.begin(), source.end(), char(size));
                ggml_backend_tensor_transfer transfer { &target, source.data(), 0, size };
                require(ggml_backend_cuda_set_tensor_batch_owned(&growing_backend, &transfer, 1), "growth upload refused");
                drain(growing.stream(), growing.stream()->submitted);
                require(std::memcmp(source.data(), destination.data(), size) == 0, "growth changed bytes");
            }
            require(allocation_calls - before <= 18, "linear staging allocation growth");
            std::cout << "growth allocation calls M: " << allocation_calls - before << " for 256 uploads\n";
        }
        std::cout << "owned bytes, ring reuse, bidirectional event order, overlap, refusal, allocation fallback, teardown: pass\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
