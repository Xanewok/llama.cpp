// K11a state I/O under a stream-like CPU backend: graph computes queue until the backend's synchronize(), as on CUDA, while
// state save/restore copy through the buffer API at once. Correct results need a real sync in front of the copy.
//   save:    decode the first half of the prompt, save the sequence state (hash compared across arms by the caller)
//   restore: decode the second half, restore the saved state over that in-flight decode, let the queued work finish
//            (as the device would), save again: must equal the first save
// usage: k11a-state MODEL OUT.json
#include "llama.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static std::unordered_map<ggml_backend_t, std::deque<std::function<void()>>> g_queues;
static ggml_backend_t (*cpu_init)(ggml_backend_dev_t, const char *);
static ggml_status (*cpu_compute)(ggml_backend_t, ggml_cgraph *);
static void (*cpu_free)(ggml_backend_t);
static size_t g_busy;

static void drain(std::deque<std::function<void()>> & q) {
    g_busy += !q.empty();
    while (!q.empty()) {
        auto job = std::move(q.front());
        q.pop_front();
        job();
    }
}

static ggml_status deferred_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    g_queues[backend].push_back([=] { require(cpu_compute(backend, graph) == GGML_STATUS_SUCCESS, "deferred graph failed"); });
    return GGML_STATUS_SUCCESS;
}

static void deferred_get(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    g_queues[backend].push_back([=] { ggml_backend_tensor_get(tensor, data, offset, size); });
}

static void deferred_sync(ggml_backend_t backend) {
    drain(g_queues[backend]);
}

static void deferred_free(ggml_backend_t backend) {
    deferred_sync(backend);
    g_queues.erase(backend);
    cpu_free(backend);
}

static ggml_backend_t init_deferred(ggml_backend_dev_t dev, const char * params) {
    auto * backend = cpu_init(dev, params);
    cpu_compute = backend->iface.graph_compute;
    cpu_free = backend->iface.free;
    backend->iface.graph_compute = deferred_compute;
    backend->iface.get_tensor_async = deferred_get;
    backend->iface.synchronize = deferred_sync;
    backend->iface.free = deferred_free;
    return backend;
}

static uint64_t fnv1a(const std::vector<uint8_t> & v) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t b : v) {
        h = (h ^ b) * 1099511628211ull;
    }
    return h;
}

static std::vector<uint8_t> save(llama_context * ctx) {
    std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, 0));
    require(!buf.empty() && llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0) == buf.size(), "state save failed");
    return buf;
}

int main(int argc, char ** argv) {
    require(argc == 3, "usage: k11a-state MODEL OUT.json");
    require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    llama_backend_init();
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    cpu_init = cpu_dev->iface.init_backend;
    cpu_dev->iface.init_backend = init_deferred;

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    require(model, "model load failed");
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_batch = cparams.n_ubatch = 64;
    cparams.n_seq_max = 1;
    cparams.n_threads = cparams.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cparams);
    require(ctx, "context init failed");

    const std::string text = "The lowercase letters, in order: a b c d e f g h i j k l m n o p q r s t u v w x y z. "
                             "The digits, in order: 0 1 2 3 4 5 6 7 8 9. The end.";
    std::vector<llama_token> tok(256);
    const int n = llama_tokenize(llama_model_get_vocab(model), text.c_str(), (int32_t) text.size(), tok.data(), (int32_t) tok.size(), true, false);
    require(n >= 32, "prompt too short");
    const int half = n / 2;

    require(llama_decode(ctx, llama_batch_get_one(tok.data(), half)) == 0, "decode 1 failed");
    const auto saved = save(ctx);

    require(llama_decode(ctx, llama_batch_get_one(tok.data() + half, n - half)) == 0, "decode 2 failed");
    require(llama_state_seq_set_data(ctx, saved.data(), saved.size(), 0) == saved.size(), "state restore failed");
    for (auto & [backend, q] : g_queues) {
        drain(q);
    }
    const auto again = save(ctx);

    std::ofstream out(argv[2]);
    out << "{\"tokens\": " << n << ", \"state_bytes\": " << saved.size() << ", \"save_fnv\": \"" << std::hex << fnv1a(saved) << std::dec
        << "\", \"restore_holds\": " << (again == saved ? "true" : "false") << "}\n";
    require(bool(out), "output write failed");
    std::fprintf(stderr, "defer: busy=%zu\n", g_busy);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
