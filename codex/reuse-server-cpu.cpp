#include "server-context.h"
#include "llama.h"
#include "ggml-round-trace.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <thread>

static void require(bool ok, const char * why) {
    if (!ok) { throw std::runtime_error(why); }
}

int main(int argc, char ** argv) {
    require(argc == 7 || argc == 9, "usage: reuse-server-cpu MODEL OUTPUT META K SEED TOKENS [REQUESTS GRAMMAR]");
    const int requests = argc == 9 ? std::atoi(argv[7]) : 1;
    const bool grammar = argc == 9 && std::atoi(argv[8]);
    require(requests > 0, "bad request count");
    json records = json::array();
    require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    const auto begin = std::chrono::steady_clock::now();
    llama_backend_init();
    common_params params;
    params.model.path = argv[1];
    params.n_gpu_layers = 0;
    const int prefill_words = std::getenv("CPU_PREFILL_WORDS") ? std::atoi(std::getenv("CPU_PREFILL_WORDS")) : 0;
    params.n_ctx = prefill_words > 0 ? prefill_words * 2 + 4096 : (requests > 1 ? 4096 : 512);
    params.n_batch = 128;
    params.n_ubatch = std::getenv("CPU_UBATCH") ? std::atoi(std::getenv("CPU_UBATCH")) : 32;
    params.n_parallel = 1;
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
        std::string prompt = "The sequence of positive integers begins with 1, 2, 3, 4, 5. Continue the sequence:";
        for (int i = 0; i < prefill_words; ++i) { prompt += " sequence"; }
        try {
            for (int request = 0; request < requests; ++request) {
                auto reader = server.get_response_reader();
                server_task task(SERVER_TASK_TYPE_COMPLETION);
                task.id = reader.get_new_id();
                task.cli = true;
                task.cli_prompt = prompt;
                task.params.n_predict = std::atoi(argv[6]);
                task.params.return_tokens = true;
                task.params.cache_prompt = requests > 1 && (!std::getenv("CPU_CACHE_PROMPT") || std::atoi(std::getenv("CPU_CACHE_PROMPT")));
                task.params.stream = true;
                task.params.timings_per_token = true;
                task.params.sampling.seed = std::strtoul(argv[5], nullptr, 10) + request;
                task.params.sampling.temp = 1;
                task.params.sampling.top_k = 20;
                task.params.sampling.top_p = 0.95f;
                task.params.sampling.min_p = 0.05f;
                task.params.sampling.backend_sampling = true;
                if (grammar) {
                    task.params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, "root ::= [0-9, ]+"};
                    task.params.sampling.ignore_eos = true;
                }
                json chunks = json::array();
                llama_tokens tokens;
                std::string content;
                const auto request_begin = ggml_round_trace_now();
                reader.post_task(std::move(task));
                bool finished = false;
                while (reader.has_next()) {
                    auto result = reader.next([] { return false; });
                    require(result && !result->is_error(), "completion failed");
                    if (auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get())) {
                        tokens.insert(tokens.end(), partial->tokens.begin(), partial->tokens.end());
                        content += partial->content;
                        if (partial->stats.is_set()) { chunks.push_back(json{{"timings", partial->stats.to_json()}}); }
                    }
                    if (auto * final = dynamic_cast<server_task_result_cmpl_final *>(result.get())) {
                        require(!tokens.empty(), "missing completion token IDs");
                        const uint64_t count = tokens.size();
                        out.write(reinterpret_cast<const char *>(&count), sizeof(count));
                        out.write(reinterpret_cast<const char *>(tokens.data()), count * sizeof(llama_token));
                        const auto timings = final->stats.to_json();
                        chunks.push_back(json{{"timings", timings}});
                        records.push_back(json{{"case", "cpu-followup"}, {"max_tokens", std::atoi(argv[6])}, {"request", request},
                            {"begin_ns", request_begin}, {"end_ns", ggml_round_trace_now()},
                            {"response", {{"timings", timings}}}, {"chunks", chunks},
                            {"metrics", {{"drafts", final->stats.n_draft_verif_steps},
                                         {"draft_tokens", final->stats.n_draft_tokens}, {"accepted_tokens", final->stats.n_draft_accepted}}}});
                        prompt += content + "\nContinue the sequence: ";
                        finished = true;
                    }
                }
                require(finished, "missing final completion");
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
        std::printf("M: server requests=%d, final target logit bytes=%zu\n", requests, nv * sizeof(float));
    }
    std::ofstream stats(std::string(argv[2]) + ".stats.json");
    stats << records.dump(2) << "\n";
    require(bool(stats), "stats write failed");
    out.close();
    require(bool(out), "output write failed");
    std::printf("M: CPU wall seconds=%.6f\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
}
