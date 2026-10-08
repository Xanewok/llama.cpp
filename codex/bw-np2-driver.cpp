#include "server-context.h"
#include "llama.h"
#include "log.h"
#include "ggml-round-trace.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <algorithm>
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
    require(argc == 7, "usage: bw-np2-driver MODEL OUTPUT META NP SEED TOKENS");
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
    if (std::getenv("HOST_KV_Q8")) {
        params.cache_type_k = params.cache_type_v = GGML_TYPE_Q8_0;
    }
    params.cpuparams.n_threads = params.cpuparams_batch.n_threads = std::getenv("NP2_THREADS") ? std::atoi(std::getenv("NP2_THREADS")) : meta ? 4 : 8;
    params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    params.speculative.draft.n_max = std::getenv("NP2_N_MAX") ? std::atoi(std::getenv("NP2_N_MAX")) : 3;
    params.kv_unified = std::getenv("NP2_KV_UNIFIED") && std::atoi(std::getenv("NP2_KV_UNIFIED"));
    params.speculative.draft.n_min = 0;
    params.speculative.draft.p_min = 0;
    params.speculative.draft.probabilistic = true;
    params.speculative.draft.backend_sampling = true;
    params.speculative.draft.mtp_chain = std::getenv("LLAMA_MTP_CHAIN") && std::atoi(std::getenv("LLAMA_MTP_CHAIN"));
    params.speculative.draft.cpuparams = params.cpuparams;
    params.speculative.draft.cpuparams_batch = params.cpuparams_batch;
    // sparse high ids keep the reduced-head path; the dense low block holds the byte tokens and common merges the
    // prompts produce, so drafts get accepted and the bulk verify rows are exercised (stride-only gave ~2 % acceptance)
    for (int id = 248319; id >= 12288; id -= 17) {
        params.speculative.draft.vocab_ids.push_back(id);
    }
    for (int id = 12287; id >= 0; --id) {
        params.speculative.draft.vocab_ids.push_back(id);
    }
    if (meta) {
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        params.devices = {cpu, cpu, nullptr}; // common_params device lists end in nullptr
        params.speculative.draft.devices = params.devices;
        params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        params.n_gpu_layers = -1;
    }
    server_context server;
    require(server.load_model(params), "model load failed");

    // slot 0: short prompt, slot 1: long prompt, so slot 1 prefills in several batches while slot 0 already drafts
    // (mixed prompt/decode batches). Phase 1 starts cold; phase 2 repeats the prompts and so takes the prefix-reuse
    // path (checkpoint restore, short re-prefill) that the GPU depth harness measures.
    std::string letters;
    for (int rep = 0; rep < 12; ++rep) {
        letters += "The lowercase letters, in order: a b c d e f g h i j k l m n o p q r s t u v w x y z. ";
    }
    const std::string prompts[2] = {
        "List the positive integers starting with 1, 2, 3, 4. Continue for fifty entries:",
        letters + "List the lowercase letters repeatedly, starting with a, b, c, d. Continue for fifty entries:",
    };

    std::thread inference;
    json phases = json::array();
    try {
        for (int phase = 0; phase < 2; ++phase) {
            auto reader = server.get_response_reader();
            std::vector<server_task> tasks;
            for (int slot = 0; slot < np; ++slot) {
                server_task task(SERVER_TASK_TYPE_COMPLETION);
                task.id = reader.get_new_id();
                task.id_slot = slot;
                task.cli = true;
                task.cli_prompt = prompts[slot];
                task.params.n_predict = std::atoi(argv[6]) + 16 * slot;
                task.params.return_tokens = true;
                if (slot == 1 && std::getenv("NP2_SLOT1_N_MAX")) {
                    task.params.speculative = params.speculative;
                    task.params.speculative.draft.n_max = std::atoi(std::getenv("NP2_SLOT1_N_MAX"));
                }
                task.params.cache_prompt = true;
                task.params.stream = true;
                task.params.sampling.seed = std::strtoul(argv[5], nullptr, 10) + slot;
                task.params.sampling.temp = 1;
                task.params.sampling.top_k = 20;
                task.params.sampling.top_p = 0.95f;
                task.params.sampling.min_p = 0.05f;
                task.params.sampling.backend_sampling = true;
                tasks.push_back(std::move(task));
            }
            // Both slots are posted in one call; phase 1 before the decode thread runs, so batch composition is deterministic.
            reader.post_tasks(std::move(tasks));
            if (phase == 0) {
                inference = std::thread([&] { server.start_loop(); });
            }
            std::vector<llama_tokens> tokens(np);
            std::vector<json> by_slot(np);
            std::vector<bool> seen(np, false);
            while (reader.has_next()) {
                auto result = reader.next([] { return false; });
                require(result && !result->is_error(), "completion failed");
                if (auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get())) {
                    auto & ids = tokens.at(partial->index);
                    ids.insert(ids.end(), partial->tokens.begin(), partial->tokens.end());
                }
                if (auto * final = dynamic_cast<server_task_result_cmpl_final *>(result.get())) {
                    require((final->index == 1 && std::getenv("NP2_SLOT1_N_MAX")) || (final->stats.n_draft_tokens > 0 && final->stats.n_draft_verif_steps > 0), "MTP did not engage");
                    require(final->index >= 0 && final->index < np && !seen[final->index], "bad or duplicate result index");
                    seen[final->index] = true;
                    by_slot[final->index] = json{{"slot", final->index}, {"draft", final->stats.n_draft_tokens},
                                         {"accepted", final->stats.n_draft_accepted}, {"rounds", final->stats.n_draft_verif_steps},
                                         {"prompt", final->n_prompt_tokens}, {"cached", final->n_prompt_tokens_cache}};
                }
            }
            // common_json has no ordering and no random-access iterators, so results are placed by slot instead of sorted.
            json stats = json::array();
            for (int slot = 0; slot < np; ++slot) {
                require(seen[slot], "missing final results");
                stats.push_back(by_slot[slot]);
            }
            json token_rows = json::array();
            for (int slot = 0; slot < np; ++slot) {
                require(tokens[slot].size() == (size_t) (std::atoi(argv[6]) + 16 * slot), "short completion");
                json row = json::array();
                for (llama_token token : tokens[slot]) {
                    row.push_back(token);
                }
                token_rows.push_back(std::move(row));
            }
            phases.push_back(json{{"phase", phase}, {"tokens", token_rows}, {"stats", stats}});
        }
    } catch (...) {
        server.terminate();
        if (inference.joinable()) {
            inference.join();
        }
        throw;
    }
    server.terminate();
    inference.join();
    std::ofstream out(argv[2]);
    out << json{{"phases", phases}}.dump(2) << '\n';
    require(bool(out), "output write failed");
    ggml_round_trace_flush();
}
