#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

struct comm_test {
    bool gpu = false;
    bool deferred = false;
    bool reject = false;
    bool fail = false;
    int starts = 0;
    int finishes = 0;
    int active = 0;
    int high_water = 0;
    int independent = 0;
    std::map<ggml_backend_t, ggml_backend_t> simple_backends;
};

static comm_test state;
static ggml_backend_reg_t cuda_reg;
static void * cuda_comm;
static ggml_backend_comm_init_t cuda_init;
static ggml_backend_comm_free_t cuda_free;
static ggml_backend_comm_allreduce_tensor_t cuda_reduce;
static ggml_backend_comm_allreduce_start_t cuda_start;
static ggml_backend_comm_allreduce_finish_t cuda_finish;


struct test_pending {
    ggml_tensor * tensors[2];
    int sequence;
};

static bool reduce(void *, ggml_tensor ** tensors) {
    if (state.gpu) { return cuda_reduce(cuda_comm, tensors); }
    std::vector<float> a(ggml_nelements(tensors[0])), b(a.size());
    ggml_backend_tensor_get(tensors[0], a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_get(tensors[1], b.data(), 0, b.size() * sizeof(float));
    for (size_t i = 0; i < a.size(); i++) {
        a[i] = a[i] + b[i];
    }
    for (int r = 0; r < 2; r++) {
        ggml_backend_tensor_set(tensors[r], a.data(), 0, a.size() * sizeof(float));
    }
    return true;
}

static bool start(void *, ggml_tensor ** tensors, void ** pending) {
    *pending = nullptr;
    if (state.reject && state.starts > 0) {
        return false;
    }
    if (state.gpu) {
        if (!cuda_start(cuda_comm, tensors, pending)) { return false; }
        if (!*pending) { return true; }
        state.starts++;
    } else {
        *pending = new test_pending{{tensors[0], tensors[1]}, ++state.starts};
    }
    state.high_water = std::max(state.high_water, ++state.active);
    require(state.active <= 1, "pending limit");
    return true;
}

static void finish(void *, void * pending) {
    if (state.gpu) {
        cuda_finish(cuda_comm, pending);
    } else {
        auto * reduction = static_cast<test_pending *>(pending);
        require(reduction->sequence == state.finishes + 1, "FIFO completion");
        reduce(nullptr, reduction->tensors);
        delete reduction;
    }
    state.finishes++;
    state.active--;
}

static void * comm_init(ggml_backend_t * backends, size_t n) {
    if (state.gpu) {
        ggml_backend_t simple[2] = {state.simple_backends.at(backends[0]), state.simple_backends.at(backends[1])};
        cuda_comm = cuda_init(simple, n);
        require(cuda_comm, "CUDA communicator init");
    }
    return &state;
}
static void comm_free(void *) {
    require(!state.active, "free with pending work");
    if (state.gpu) { cuda_free(cuda_comm); }
}

static void * get_proc(ggml_backend_reg_t, const char * name) {
    if (!strcmp(name, "ggml_backend_comm_init"))             { return (void *) comm_init; }
    if (!strcmp(name, "ggml_backend_comm_free"))             { return (void *) comm_free; }
    if (!strcmp(name, "ggml_backend_comm_allreduce_tensor")) { return (void *) reduce; }
    if (state.deferred) {
        if (!strcmp(name, "ggml_backend_comm_allreduce_start"))  { return (void *) start; }
        if (!strcmp(name, "ggml_backend_comm_allreduce_finish")) { return (void *) finish; }
    }
    return nullptr;
}

static ggml_status compute(ggml_backend_t backend, ggml_cgraph * graph) {
    if (state.active) {
        state.independent++;
        if (state.fail) {
            return GGML_STATUS_ABORTED;
        }
    }
    return ggml_backend_graph_compute(state.simple_backends.at(backend), graph);
}

static void free_backend(ggml_backend_t backend) {
    ggml_backend_free(state.simple_backends.at(backend));
    state.simple_backends.erase(backend);
    delete backend;
}

static ggml_backend_t init_backend(ggml_backend_dev_t dev, const char *) {
    ggml_backend_dev_t simple_dev = nullptr;
    if (state.gpu) {
        for (size_t i = 0; i < 2; i++) {
            auto * candidate = ggml_backend_reg_dev_get(cuda_reg, i);
            if (candidate->context == dev->context) { simple_dev = candidate; }
        }
        require(simple_dev, "CUDA wrapper device");
    }
    ggml_backend_t cpu = state.gpu ? ggml_backend_dev_init(simple_dev, nullptr) : ggml_backend_cpu_init();
    if (!state.gpu) { ggml_backend_cpu_set_n_threads(cpu, 1); }
    auto * backend = new ggml_backend(*cpu);
    backend->device = dev;
    backend->iface.free = free_backend;
    backend->iface.graph_compute = compute;
    state.simple_backends[backend] = cpu;
    return backend;
}

static ggml_backend_meta_split_state split(const ggml_tensor * tensor, void *) {
    ggml_backend_meta_split_state result{};
    result.axis = GGML_BACKEND_SPLIT_AXIS_0;
    result.ne[0] = result.ne[1] = tensor->ne[0] / 2;
    result.nr[0] = result.n_segments = 1;
    return result;
}

static std::vector<float> cpu_case(bool deferred, bool reject, bool fail, bool views, bool unused, bool reuse = false, bool gpu = false) {
    state = {};
    state.gpu = gpu;
    state.deferred = deferred;
    state.reject = reject;
    state.fail = fail;
    ggml_backend_t cpu = ggml_backend_cpu_init();
    static ggml_backend_reg reg{};
    static ggml_backend_device devices[2];
    reg.iface.get_proc_address = get_proc;
    ggml_backend_dev_t devs[2] = {&devices[0], &devices[1]};
    for (auto & dev : devices) {
        dev = gpu ? *ggml_backend_reg_dev_get(cuda_reg, &dev - devices) : *ggml_backend_get_device(cpu);
        dev.reg = &reg;
        dev.iface.init_backend = init_backend;
    }
    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_meta_device(devs, 2, split, nullptr), nullptr);
    ggml_context * weights = ggml_init({4096, nullptr, true});
    const int64_t k = gpu ? 1024 : 4, m = gpu ? 5120 : 4, n = gpu ? 512 : 2;
    ggml_tensor * w = ggml_new_tensor_2d(weights, gpu ? GGML_TYPE_F16 : GGML_TYPE_F32, k, m);
    ggml_tensor * x = ggml_new_tensor_2d(weights, GGML_TYPE_F32, k, n);
    auto * buffer_w = ggml_backend_alloc_ctx_tensors(weights, backend);
    std::vector<float> values(std::max(k * m, k * n));
    for (size_t i = 0; i < values.size(); i++) { values[i] = float(int(i % 17) - 7) / 8; }
    if (gpu) {
        std::vector<ggml_fp16_t> half(k * m);
        ggml_fp32_to_fp16_row(values.data(), half.data(), half.size());
        ggml_backend_tensor_set(w, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(w, values.data(), 0, k * m * sizeof(float));
    }
    ggml_backend_tensor_set(x, values.data(), 0, k * n * sizeof(float));

    ggml_context * ctx = ggml_init({32768, nullptr, true});
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    // Independent producers exercise completion before the next start.
    ggml_tensor * p[4];
    ggml_tensor * early = nullptr;
    for (int i = 0; i < 4; i++) {
        p[i] = ggml_mul_mat(ctx, w, x);
        if (!reuse) { ggml_set_output(p[i]); }
        ggml_build_forward_expand(graph, p[i]);
        if (views && !unused && i == 1) {
            early = ggml_sqr(ctx, ggml_view_1d(ctx, ggml_view_1d(ctx, p[0], 8, 0), 8, 0));
            ggml_build_forward_expand(graph, early);
        }
    }
    ggml_tensor * output = nullptr;
    if (unused) {
        output = ggml_sqr(ctx, x);
        ggml_build_forward_expand(graph, output);
    } else {
        for (int i = 0; i < 4; i++) {
            auto * tensor = p[i];
            if (views) {
                tensor = ggml_view_1d(ctx, ggml_view_1d(ctx, tensor, 8, 0), 8, 0);
            }
            auto * squared = early && i == 0 ? early : ggml_sqr(ctx, tensor);
            output = output ? ggml_add(ctx, output, squared) : squared;
            ggml_build_forward_expand(graph, output);
        }
    }
    ggml_backend_buffer_t arena = nullptr;
    if (reuse) {
        arena = ggml_backend_alloc_buffer(backend, 128);
        ggml_backend_buffer_set_usage(arena, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        // These dead results share storage without a view or source edge between them.
        for (auto * tensor : p) {
            require(ggml_backend_tensor_alloc(arena, tensor, ggml_backend_buffer_get_base(arena)) == GGML_STATUS_SUCCESS, "shared storage allocation");
        }
        require(ggml_backend_tensor_alloc(arena, output, ggml_backend_buffer_get_base(arena)) == GGML_STATUS_SUCCESS, "shared output allocation");
    }
    auto * alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(ggml_gallocr_alloc_graph(alloc, graph), "graph allocation");
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    require(status == (fail ? GGML_STATUS_ABORTED : GGML_STATUS_SUCCESS), "compute status");
    std::vector<float> result(ggml_nelements(output));
    if (!fail) {
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        // Repeat to exercise communicator slot reuse.
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "repeat compute");
        std::vector<float> repeated(result.size());
        ggml_backend_tensor_get(output, repeated.data(), 0, repeated.size() * sizeof(float));
        require(!memcmp(result.data(), repeated.data(), result.size() * sizeof(float)), "slot reuse identity");
    }
    require(!state.active && state.starts == state.finishes, "drain every handle");
    if (deferred && !reject && !reuse) {
        require(state.starts > 0 && state.independent > 0, "deferred path engagement");
        if (!fail) { require(state.high_water == 1, "one pending reduction"); }
    }
    if (reuse) { require(state.starts == 0, "arena overwrite must finish at the next subgraph"); }
    printf("meta gpu=%d deferred=%d reject=%d fail=%d views=%d unused=%d reuse=%d starts=%d finishes=%d high_water=%d independent_rank_submissions=%d\n",
           gpu, deferred, reject, fail, views, unused, reuse, state.starts, state.finishes, state.high_water, state.independent);
    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(arena);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer_w);
    ggml_free(weights);
    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    return result;
}

static void fill_input(std::vector<unsigned char> & input, ggml_type type, int rank, int iteration) {
    uint32_t state = 0x1234567u ^ (rank + 1) * 0x9e3779b9u ^ iteration * 0x85ebca6bu;
    const unsigned exponent[] = {0, 1, 80, 120, 127, 140, 150};
    const unsigned exponent_half[] = {0, 1, 5, 10, 15, 20, 26};
    const size_t width = type == GGML_TYPE_F32 ? 4 : 2;
    for (size_t i = 0; i < input.size() / width; i++) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        // Raw finite encodings preserve subnormals without host floating-point conversion.
        if (type == GGML_TYPE_F32) {
            const uint32_t bits = (state & 0x807fffffu) | (exponent[i % 7] << 23);
            memcpy(input.data() + i * width, &bits, width);
        } else {
            const uint16_t bits = type == GGML_TYPE_F16 ?
                uint16_t((state & 0x83ffu) | (exponent_half[i % 7] << 10)) :
                uint16_t((state & 0x807fu) | (exponent[i % 7] << 7));
            memcpy(input.data() + i * width, &bits, width);
        }
    }
}

static void gpu_cases(const char * provider, bool synchronous, bool broken = false) {
    require(!strcmp(provider, "internal") || !strcmp(provider, "nccl"), "provider argument");
    require(std::getenv("GGML_CUDA_ALLREDUCE") && !strcmp(provider, std::getenv("GGML_CUDA_ALLREDUCE")), "provider environment");
    int deferred_count = 0;
    bool negative_checked = false;
    ggml_backend_load_all();
    auto * reg = ggml_backend_reg_by_name("CUDA");
    require(reg && ggml_backend_reg_dev_count(reg) >= 2, "two CUDA devices required");
    auto init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    auto sync = (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");
    auto begin = (ggml_backend_comm_allreduce_start_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_start");
    auto end = (ggml_backend_comm_allreduce_finish_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_finish");
    auto destroy = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
    require(init && sync && destroy && (synchronous || (begin && end)), "communicator API");
    ggml_backend_t backends[2];
    for (int r = 0; r < 2; r++) { backends[r] = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, r), nullptr); }
    void * comm = init(backends, 2);
    require(comm != nullptr, "communicator init");
    ggml_context * work_ctx[2];
    ggml_tensor * work_out[2];
    ggml_cgraph * work_graph[2];
    ggml_backend_buffer_t work_buffer[2];
    std::vector<float> work_values(4096, 0.25f);
    for (int r = 0; r < 2; r++) {
        work_ctx[r] = ggml_init({16384, nullptr, true});
        auto * input = ggml_new_tensor_2d(work_ctx[r], GGML_TYPE_F32, 64, 64);
        work_out[r] = ggml_mul_mat(work_ctx[r], input, input);
        work_graph[r] = ggml_new_graph_custom(work_ctx[r], 8, false);
        ggml_build_forward_expand(work_graph[r], work_out[r]);
        work_buffer[r] = ggml_backend_alloc_ctx_tensors(work_ctx[r], backends[r]);
        ggml_backend_tensor_set(input, work_values.data(), 0, work_values.size() * sizeof(float));
    }
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        if (!strcmp(provider, "nccl") && type != GGML_TYPE_F32) { continue; }
        for (int64_t ne : {int64_t(0), int64_t(1024), int64_t(2621440), int64_t(2621456), int64_t(17825792)}) {
            ggml_context * ctx = ggml_init({16384, nullptr, true});
            ggml_tensor * tensors[3][2];
            ggml_backend_buffer_t buffers[2];
            for (int r = 0; r < 2; r++) {
                for (int k = 0; k < 3; k++) {
                    tensors[k][r] = ggml_new_tensor_1d(ctx, type, ne);
                    tensors[k][r]->flags |= GGML_TENSOR_FLAG_COMPUTE;
                }
            }
            for (int r = 0; r < 2; r++) {
                buffers[r] = ggml_backend_alloc_buffer(backends[r], 3 * ggml_nbytes(tensors[0][r]) + 256);
                char * base = (char *) ggml_backend_buffer_get_base(buffers[r]);
                for (int k = 0; k < 3; k++) {
                    require(ggml_backend_tensor_alloc(buffers[r], tensors[k][r], base + k * ggml_nbytes(tensors[k][r])) == GGML_STATUS_SUCCESS, "tensor allocation");
                }
            }
            const size_t bytes = ggml_nbytes(tensors[0][0]);
            std::vector<unsigned char> input(bytes), expected[2], actual(bytes);
            for (int iteration = 0; iteration < 6; iteration++) {
                for (int r = 0; r < 2; r++) {
                    fill_input(input, type, r, iteration);
                    for (int k = 0; k < 3; k++) {
                        tensors[k][r]->flags = (iteration % 4 == r || iteration % 4 == 3) ? 0 : GGML_TENSOR_FLAG_COMPUTE;
                        ggml_backend_tensor_set(tensors[k][r], input.data(), 0, bytes);
                    }
                }
                require(sync(comm, tensors[0]), "synchronous reduction");
                for (int r = 0; r < 2; r++) {
                    expected[r].resize(bytes);
                    ggml_backend_tensor_get(tensors[0][r], expected[r].data(), 0, bytes);
                }
                for (int k = 1; k < 3; k++) {
                    void * pending = nullptr;
                    if (synchronous || !begin(comm, tensors[k], &pending)) {
                        require(sync(comm, tensors[k]), "unsupported start fallback");
                    }
                    deferred_count += pending != nullptr;
                    for (int r = 0; r < 2; r++) {
                        require(ggml_backend_graph_compute_async(backends[r], work_graph[r]) == GGML_STATUS_SUCCESS, "independent GPU compute");
                    }
                    if (broken && pending && !negative_checked) {
                        ggml_context * check_ctx = ggml_init({16384, nullptr, true});
                        ggml_tensor * snapshot[2];
                        ggml_backend_buffer_t snapshot_buf[2];
                        for (int r = 0; r < 2; r++) {
                            snapshot[r] = ggml_dup(check_ctx, tensors[k][r]);
                            snapshot_buf[r] = ggml_backend_alloc_buffer(backends[r], bytes);
                            require(ggml_backend_tensor_alloc(snapshot_buf[r], snapshot[r], ggml_backend_buffer_get_base(snapshot_buf[r])) == GGML_STATUS_SUCCESS, "negative allocation");
                            auto * graph = ggml_new_graph_custom(check_ctx, 8, false);
                            ggml_build_forward_expand(graph, snapshot[r]);
                            require(ggml_backend_graph_compute_async(backends[r], graph) == GGML_STATUS_SUCCESS, "consumer before finish");
                        }
                        end(comm, pending);
                        for (int r = 0; r < 2; r++) {
                            ggml_backend_tensor_get(snapshot[r], actual.data(), 0, bytes);
                            require(actual == expected[r], "bitwise synchronous/deferred identity (consumer before finish)");
                            ggml_backend_buffer_free(snapshot_buf[r]);
                        }
                        ggml_free(check_ctx);
                        negative_checked = true;
                    } else if (pending) {
                        end(comm, pending);
                    }
                }
                for (int r = 0; r < 2; r++) {
                    ggml_backend_tensor_get(work_out[r], work_values.data(), 0, work_values.size() * sizeof(float));
                    require(std::all_of(work_values.begin(), work_values.end(), [](float v) { return v == 4.0f; }), "independent compute result");
                }
                uint64_t hash = 14695981039346656037ull;
                for (int r = 0; r < 2; r++) {
                    for (int k = 1; k < 3; k++) {
                        ggml_backend_tensor_get(tensors[k][r], actual.data(), 0, bytes);
                        require(actual == expected[r], "bitwise synchronous/deferred identity");
                    }
                    for (auto byte : expected[r]) { hash = (hash ^ byte) * 1099511628211ull; }
                }
                printf("identity type=%d ne=%lld iteration=%d hash=%016llx\n", (int) type, (long long) ne, iteration, (unsigned long long) hash);
            }
            for (auto * buffer : buffers) { ggml_backend_buffer_free(buffer); }
            ggml_free(ctx);
        }
    }
    const char * threshold = std::getenv("GGML_CUDA_AR_COPY_THRESHOLD");
    const bool copy_disabled = !strcmp(provider, "internal") && threshold && !strcmp(threshold, "0");
    require(synchronous || !strcmp(provider, "nccl") || copy_disabled || deferred_count > 0, "GPU deferred engagement");
    printf("deferred_handles=%d\n", deferred_count);
    destroy(comm);
    for (int r = 0; r < 2; r++) {
        ggml_backend_buffer_free(work_buffer[r]);
        ggml_free(work_ctx[r]);
        ggml_backend_free(backends[r]);
    }
}

static void gpu_meta_cases() {
    require(std::getenv("GGML_CUDA_ALLREDUCE") && !strcmp(std::getenv("GGML_CUDA_ALLREDUCE"), "internal"), "internal provider required");
    ggml_backend_load_all();
    cuda_reg = ggml_backend_reg_by_name("CUDA");
    require(cuda_reg && ggml_backend_reg_dev_count(cuda_reg) >= 2, "two CUDA devices required");
    cuda_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(cuda_reg, "ggml_backend_comm_init");
    cuda_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(cuda_reg, "ggml_backend_comm_free");
    cuda_reduce = (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(cuda_reg, "ggml_backend_comm_allreduce_tensor");
    cuda_start = (ggml_backend_comm_allreduce_start_t) ggml_backend_reg_get_proc_address(cuda_reg, "ggml_backend_comm_allreduce_start");
    cuda_finish = (ggml_backend_comm_allreduce_finish_t) ggml_backend_reg_get_proc_address(cuda_reg, "ggml_backend_comm_allreduce_finish");
    require(cuda_init && cuda_free && cuda_reduce && cuda_start && cuda_finish, "CUDA communicator API");
    auto expected = cpu_case(false, false, false, false, false, false, true);
    auto actual = cpu_case(true, false, false, false, false, false, true);
    require(actual.size() == expected.size() && !memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), "GPU Meta bitwise legacy/deferred identity");
}

int main(int argc, char ** argv) {
    if (argc == 2 && !strcmp(argv[1], "--gpu-meta")) {
        gpu_meta_cases();
    } else if (argc == 3 && (!strcmp(argv[1], "--gpu") || !strcmp(argv[1], "--gpu-sync") || !strcmp(argv[1], "--gpu-broken"))) {
        gpu_cases(argv[2], !strcmp(argv[1], "--gpu-sync"), !strcmp(argv[1], "--gpu-broken"));
    } else {
        require(argc == 1, "usage: test-backend-comm [--gpu-meta | --gpu|--gpu-sync|--gpu-broken internal|nccl]");
        for (bool views : {false, true}) {
            for (bool unused : {false, true}) {
                if (views && unused) { continue; }
                auto expected = cpu_case(false, false, false, views, unused);
                for (bool reject : {false, true}) {
                    auto actual = cpu_case(true, reject, false, views, unused);
                    require(actual.size() == expected.size() && !memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), "CPU bitwise identity");
                }
            }
        }
        cpu_case(true, false, true, false, false);
        cpu_case(true, false, false, false, true, true);
    }
    puts("PASS backend communicator");
}
