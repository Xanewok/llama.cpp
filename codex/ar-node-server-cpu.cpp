#include "server-context.h"
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

static void require(bool ok, const char * why) {
    if (!ok) { throw std::runtime_error(why); }
}

// Token streams of CPU_NP concurrent requests (one batch posted atomically, so the slot interleaving is the same every
// run) followed by the final target logits. usage: ar-node-server-cpu MODEL OUTPUT META K SEED TOKENS
int main(int argc, char ** argv) {
    require(argc == 7, "usage: ar-node-server-cpu MODEL OUTPUT META K SEED TOKENS");
    require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    const int np = std::getenv("CPU_NP") ? std::atoi(std::getenv("CPU_NP")) : 1;
    require(np >= 1 && np <= 4, "bad CPU_NP");
    const auto begin = std::chrono::steady_clock::now();
    llama_backend_init();
    common_params params;
    params.model.path = argv[1];
    params.n_gpu_layers = 0;
    params.n_ctx = 4096;
    params.n_batch = 128;
    params.n_ubatch = 32;
    params.n_parallel = np;
    params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 8;
    params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    params.speculative.draft.n_max = std::atoi(argv[4]);
    for (int id = 248319; id >= 0; id -= 17) { params.speculative.draft.vocab_ids.push_back(id); }
    params.speculative.draft.probabilistic = true;
    params.speculative.draft.backend_sampling = true;
    params.speculative.draft.cpuparams.n_threads = params.speculative.draft.cpuparams_batch.n_threads = 8;
    const bool meta = std::atoi(argv[3]) != 0;
    if (meta) {
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        params.devices = {cpu, cpu};
        params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        params.n_gpu_layers = -1;
        params.speculative.draft.devices = {cpu, cpu};
        params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 4;
        params.speculative.draft.cpuparams.n_threads = params.speculative.draft.cpuparams_batch.n_threads = 4;
    }
    std::ofstream out(argv[2], std::ios::binary);
    require(bool(out), "output open failed");
    {
        server_context server;
        require(server.load_model(params), "server model load failed");
        std::thread inference([&] { server.start_loop(); });
        try {
            auto reader = server.get_response_reader();
            std::vector<server_task> tasks;
            for (int k = 0; k < np; ++k) {
                server_task task(SERVER_TASK_TYPE_COMPLETION);
                task.id = reader.get_new_id();
                task.index = k;
                task.cli = true;
                task.cli_prompt = k == 0 ? "The sequence of positive integers begins with 1, 2, 3, 4, 5. Continue the sequence:"
                                         : "Count upward from one and write the numbers separated by commas:";
                task.params.n_predict = std::atoi(argv[6]);
                task.params.return_tokens = true;
                task.params.cache_prompt = false;
                task.params.stream = true;
                task.params.sampling.seed = std::strtoul(argv[5], nullptr, 10) + k;
                task.params.sampling.temp = std::getenv("CPU_TEMP") ? std::atof(std::getenv("CPU_TEMP")) : 1;
                task.params.sampling.top_k = 20;
                task.params.sampling.top_p = 0.95f;
                task.params.sampling.min_p = 0.05f;
                task.params.sampling.backend_sampling = true;
                tasks.push_back(std::move(task));
            }
            reader.post_tasks(std::move(tasks));
            std::vector<llama_tokens> tokens(np);
            std::vector<bool> finished(np, false);
            while (reader.has_next()) {
                auto result = reader.next([] { return false; });
                require(result && !result->is_error(), "completion failed");
                if (auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get())) {
                    require((int) partial->index < np, "bad result index");
                    tokens[partial->index].insert(tokens[partial->index].end(), partial->tokens.begin(), partial->tokens.end());
                }
                if (auto * final = dynamic_cast<server_task_result_cmpl_final *>(result.get())) {
                    require((int) final->index < np, "bad result index");
                    finished[final->index] = true;
                }
            }
            for (int k = 0; k < np; ++k) {
                require(finished[k] && !tokens[k].empty(), "missing completion");
                const uint64_t count = tokens[k].size();
                out.write(reinterpret_cast<const char *>(&count), sizeof(count));
                out.write(reinterpret_cast<const char *>(tokens[k].data()), count * sizeof(llama_token));
            }
        } catch (...) {
            server.terminate();
            inference.join();
            throw;
        }
        server.terminate();
        inference.join();
        auto * ctx = server.get_llama_context();
        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        const float * logits = llama_get_logits_ith(ctx, 0);
        require(logits, "missing server logits");
        out.write(reinterpret_cast<const char *>(logits), nv * sizeof(float));
        std::printf("M: server np=%d, final target logit bytes=%zu\n", np, nv * sizeof(float));
    }
    out.close();
    require(bool(out), "output write failed");
    std::printf("M: CPU wall seconds=%.6f\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
}
