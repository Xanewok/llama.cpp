// End-to-end CPU check of the banned-token candidate path through server_context: one slot, ignore_eos as the depth
// harness sends it, with the MTP draft (probabilistic verify) or without it (then post-sampling top-20 probabilities
// per token, i.e. each row's candidate set after the chain). Arms:
//   cand     ignore_eos only: the -inf EOG biases keep the backend candidate chain
//   host     ignore_eos, backend sampling off: host sampling of the full row
//   refused  ignore_eos plus a 0.0 bias on one token: a finite bias, so the backend chain starts with the logit bias
//            and a tensor-split context refuses it ("set_sampler ... using CPU"), which is the stack6 path
// usage: bw-cand-driver MODEL OUTPUT META SPEC TEMP ARM SEED TOKENS
#include "server-context.h"
#include "llama.h"
#include "log.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

int main(int argc, char ** argv) {
    require(argc == 9, "usage: bw-cand-driver MODEL OUTPUT META SPEC TEMP ARM SEED TOKENS");
    require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    const bool meta = std::atoi(argv[3]);
    const bool spec = std::atoi(argv[4]);
    const float temp = std::strtof(argv[5], nullptr);
    const std::string arm = argv[6];
    require(arm == "cand" || arm == "host" || arm == "refused", "bad arm");
    const int n_tokens = std::atoi(argv[8]);

    llama_backend_init();
    common_log_set_verbosity_thold(5);
    common_params params;
    params.model.path = argv[1];
    params.n_ctx = 4096;
    params.n_parallel = 1;
    params.n_batch = params.n_ubatch = 128;
    params.n_gpu_layers = 0;
    params.cpuparams.n_threads = params.cpuparams_batch.n_threads = meta ? 4 : 8;
    if (const char * cpu_only = std::getenv("BW_CPU_ONLY"); cpu_only && std::strcmp(cpu_only, "1") == 0) {
        params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 4;
    }
    if (spec) {
        params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        params.speculative.draft.n_max = 3;
        params.speculative.draft.n_min = 0;
        params.speculative.draft.p_min = 0;
        params.speculative.draft.probabilistic = true;
        params.speculative.draft.backend_sampling = true;
        params.speculative.draft.cpuparams = params.cpuparams;
        params.speculative.draft.cpuparams_batch = params.cpuparams_batch;
    }
    if (meta) {
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        params.devices = {cpu, cpu, nullptr}; // llama_model_params.devices is nullptr-terminated
        params.speculative.draft.devices = params.devices;
        params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        params.n_gpu_layers = -1;
    }
    server_context server;
    require(server.load_model(params), "model load failed");
    const auto meta_info = server.get_meta();
    require(!meta_info.logit_bias_eog.empty(), "no EOG tokens to ban");

    std::thread inference;
    json out;
    try {
        auto reader = server.get_response_reader();
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id = reader.get_new_id();
        task.id_slot = 0;
        task.cli = true;
        task.cli_prompt = "Write a short story about a lighthouse keeper who finds a message in a bottle. Begin:";
        task.params.n_predict = n_tokens;
        task.params.return_tokens = true;
        task.params.cache_prompt = false;
        task.params.stream = true;
        task.params.sampling.seed = std::strtoul(argv[7], nullptr, 10);
        task.params.sampling.temp = temp;
        task.params.sampling.top_k = 20;
        task.params.sampling.top_p = 0.95f;
        task.params.sampling.min_p = 0.05f;
        task.params.sampling.backend_sampling = arm != "host";
        // as server-schema.cpp does for "ignore_eos": true
        task.params.sampling.ignore_eos = true;
        auto & bias = task.params.sampling.logit_bias;
        bias.insert(bias.end(), meta_info.logit_bias_eog.begin(), meta_info.logit_bias_eog.end());
        if (arm == "refused") {
            bias.push_back({ 13, 0.0f });
        }
        if (!spec) {
            task.params.sampling.n_probs = 20;
            task.params.post_sampling_probs = true;
        }
        std::vector<server_task> tasks;
        tasks.push_back(std::move(task));
        reader.post_tasks(std::move(tasks));
        inference = std::thread([&] { server.start_loop(); });

        json tokens = json::array();
        json probs = json::array();
        json stats;
        while (reader.has_next()) {
            auto result = reader.next([] { return false; });
            require(result && !result->is_error(), "completion failed");
            if (auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get())) {
                for (llama_token t : partial->tokens) {
                    tokens.push_back(t);
                }
                for (const auto & o : partial->probs_output) {
                    json row = json::array();
                    for (const auto & p : o.probs) {
                        row.push_back(json::array({p.tok, p.prob}));
                    }
                    probs.push_back(json{{"tok", o.tok}, {"prob", o.prob}, {"top", row}});
                }
            }
            if (auto * final = dynamic_cast<server_task_result_cmpl_final *>(result.get())) {
                if (spec) {
                    require(final->stats.n_draft_tokens > 0 && final->stats.n_draft_verif_steps > 0, "MTP did not engage");
                }
                stats = json{{"draft", final->stats.n_draft_tokens}, {"accepted", final->stats.n_draft_accepted},
                             {"rounds", final->stats.n_draft_verif_steps}};
            }
        }
        require(tokens.size() == (size_t) n_tokens, "short completion");
        require(spec || probs.size() == (size_t) n_tokens, "missing per-token probabilities");
        out = json{{"tokens", tokens}, {"probs", probs}, {"stats", stats}};
    } catch (...) {
        server.terminate();
        if (inference.joinable()) {
            inference.join();
        }
        throw;
    }
    server.terminate();
    inference.join();
    std::ofstream file(argv[2]);
    file << out.dump(1) << '\n';
    require(bool(file), "output write failed");
}
