// In-process server fixture for the MTP chain identity test. Takes the server's own command line (so the flags mirror
// serving) and runs the cases in CHAIN_CASES (JSON: [{"name","prompt","temp","seed","n_predict"}]) one after another on
// one loaded model. CHAIN_META=1 puts the target and the draft on a Meta(CPU,CPU) tensor split. CPU only.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "server-context.h"
#include "llama.h"
#include "ggml-round-trace.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

static void require(bool ok, const char * why) {
    if (!ok) { throw std::runtime_error(why); }
}

int main(int argc, char ** argv) {
    require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    require(std::getenv("CHAIN_CASES") && std::getenv("CHAIN_OUT"), "CHAIN_CASES and CHAIN_OUT are required");

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }
    common_init();
    llama_backend_init();

    if (std::getenv("CHAIN_META") && std::atoi(std::getenv("CHAIN_META"))) {
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        params.devices = {cpu, cpu, nullptr};
        params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        params.n_gpu_layers = -1;
        params.speculative.draft.devices = {cpu, cpu, nullptr};
    }

    std::ifstream cases_file(std::getenv("CHAIN_CASES"));
    require(bool(cases_file), "cannot open CHAIN_CASES");
    std::stringstream cases_text;
    cases_text << cases_file.rdbuf();
    const json cases = json::parse(cases_text.str());

    json out = json::array();
    {
        server_context server;
        require(server.load_model(params), "server model load failed");
        std::thread inference([&] { server.start_loop(); });
        try {
            for (const auto & c : cases) {
                auto reader = server.get_response_reader();
                server_task task(SERVER_TASK_TYPE_COMPLETION);
                task.id = reader.get_new_id();
                task.cli = true;
                task.cli_prompt = c.at("prompt").get<std::string>();
                task.params.n_predict = c.at("n_predict").get<int>();
                task.params.return_tokens = true;
                task.params.cache_prompt = false;
                task.params.stream = true;
                task.params.sampling.seed = c.at("seed").get<uint32_t>();
                task.params.sampling.temp = c.at("temp").get<float>();
                task.params.sampling.top_k = 20;
                task.params.sampling.top_p = 0.95f;
                task.params.sampling.min_p = 0.05f;
                task.params.sampling.backend_sampling = true;
                llama_tokens tokens;
                reader.post_task(std::move(task));
                bool finished = false;
                while (reader.has_next()) {
                    auto result = reader.next([] { return false; });
                    require(result && !result->is_error(), "completion failed");
                    if (auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get())) {
                        tokens.insert(tokens.end(), partial->tokens.begin(), partial->tokens.end());
                    }
                    if (auto * final = dynamic_cast<server_task_result_cmpl_final *>(result.get())) {
                        out.push_back(json{
                            {"name", c.at("name")}, {"tokens", tokens},
                            {"prompt_n", final->n_prompt_tokens},
                            {"drafts", final->stats.n_draft_verif_steps},
                            {"draft_tokens", final->stats.n_draft_tokens},
                            {"accepted_tokens", final->stats.n_draft_accepted}});
                        finished = true;
                    }
                }
                require(finished, "missing final completion");
                std::fprintf(stderr, "M: case %s done\n", c.at("name").get<std::string>().c_str());
            }
        } catch (...) {
            server.terminate();
            inference.join();
            throw;
        }
        server.terminate();
        inference.join();
    }
    ggml_round_trace_flush();
    std::ofstream f(std::getenv("CHAIN_OUT"));
    f << out.dump(1) << "\n";
    require(bool(f), "output write failed");
    return 0;
}
