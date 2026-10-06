#include "server-context.h"
#include "llama.h"
#include "log.h"
#include "ggml-round-trace.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <thread>

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static ggml_backend_t (*cpu_init)(ggml_backend_dev_t, const char *);

static bool owned_cpu(ggml_backend_t, const ggml_backend_tensor_transfer * transfers, size_t count) {
    if (count) {
        const auto stamp = ggml_round_trace_now();
        ggml_round_trace_event("input_cpu_proxy", stamp, stamp, -1, count);
        ggml_backend_tensor_set_batch(transfers, count);
    }
    return true;
}

static ggml_backend_t init_owned_cpu(ggml_backend_dev_t dev, const char * params) {
    auto * backend = cpu_init(dev, params);
    backend->iface.set_tensor_batch_owned = owned_cpu;
    return backend;
}

int main(int argc, char ** argv) {
    require(argc == 7, "usage: host-identity MODEL OUTPUT META NP SEED TOKENS");
    require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    const bool meta = std::atoi(argv[3]);
    const int np = std::atoi(argv[4]);
    require(np == 1 || np == 2, "bad parallel count");
    llama_backend_init();
    common_log_set_verbosity_thold(5);
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (std::getenv("HOST_INPUT_COUNT")) {
        cpu_init = cpu_dev->iface.init_backend;
        cpu_dev->iface.init_backend = init_owned_cpu;
    }
    common_params params;
    params.model.path = argv[1];
    params.n_ctx = 4096;
    params.n_parallel = np;
    params.n_batch = params.n_ubatch = 128;
    params.n_gpu_layers = 0;
    params.cpuparams.n_threads = params.cpuparams_batch.n_threads = meta ? 4 : 8;
    params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    params.speculative.draft.n_max = 3;
    params.speculative.draft.n_min = 0;
    params.speculative.draft.p_min = 0;
    params.speculative.draft.probabilistic = true;
    params.speculative.draft.backend_sampling = true;
    params.speculative.draft.cpuparams = params.cpuparams;
    params.speculative.draft.cpuparams_batch = params.cpuparams_batch;
    for (int id = 248319; id >= 0; id -= 17) {
        params.speculative.draft.vocab_ids.push_back(id);
    }
    if (meta) {
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        params.devices = {cpu, cpu};
        params.speculative.draft.devices = params.devices;
        params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        params.n_gpu_layers = -1;
    }
    server_context server;
    require(server.load_model(params), "model load failed");
    auto reader = server.get_response_reader();
    std::vector<server_task> tasks;
    for (int slot = 0; slot < np; ++slot) {
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id = reader.get_new_id();
        task.id_slot = slot;
        task.cli = true;
        task.cli_prompt = slot == 0
            ? "List the positive integers starting with 1, 2, 3, 4. Continue for fifty entries:"
            : "List the lowercase letters repeatedly, starting with a, b, c, d. Continue for fifty entries:";
        task.params.n_predict = std::atoi(argv[6]);
        task.params.return_tokens = true;
        task.params.cache_prompt = false;
        task.params.stream = true;
        task.params.sampling.seed = std::strtoul(argv[5], nullptr, 10) + slot;
        task.params.sampling.temp = 1;
        task.params.sampling.top_k = 20;
        task.params.sampling.top_p = 0.95f;
        task.params.sampling.min_p = 0.05f;
        task.params.sampling.backend_sampling = true;
        tasks.push_back(std::move(task));
    }
    // Post both slots before the decode thread runs, so batch composition is deterministic.
    reader.post_tasks(std::move(tasks));
    std::thread inference([&] { server.start_loop(); });
    std::vector<llama_tokens> tokens(np);
    json stats = json::array();
    try {
        while (reader.has_next()) {
            auto result = reader.next([] { return false; });
            require(result && !result->is_error(), "completion failed");
            if (auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get())) {
                auto & ids = tokens.at(partial->index);
                ids.insert(ids.end(), partial->tokens.begin(), partial->tokens.end());
            }
            if (auto * final = dynamic_cast<server_task_result_cmpl_final *>(result.get())) {
                require(final->stats.n_draft_tokens > 0 && final->stats.n_draft_verif_steps > 0, "MTP did not engage");
                stats.push_back(json{{"slot", final->index}, {"draft", final->stats.n_draft_tokens},
                                     {"accepted", final->stats.n_draft_accepted}, {"rounds", final->stats.n_draft_verif_steps}});
            }
        }
    } catch (...) {
        server.terminate();
        inference.join();
        throw;
    }
    server.terminate();
    inference.join();
    require(stats.size() == (size_t) np, "missing final results");
    for (const auto & ids : tokens) {
        require(ids.size() == (size_t) std::atoi(argv[6]), "short completion");
    }
    std::ofstream out(argv[2]);
    out << json{{"tokens", tokens}, {"stats", stats}}.dump(2) << '\n';
    require(bool(out), "output write failed");
    ggml_round_trace_flush();
}
