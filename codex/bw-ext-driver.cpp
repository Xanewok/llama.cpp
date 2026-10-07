// Server-level CPU driver for the draft vocabulary extension: copy-task completions through server_context with MTP.
// Usage: bw-ext-driver MODEL OUTPUT META NP MODE N_PREDICT   (MODE greedy|prob; env DRIVER_VOCAB_EXT=N, default 0)
// Built twice: with -DVOCAB_EXT against this tree, and without it against the stack6 base source (the off reference).
#include "server-context.h"
#include "llama.h"
#include "log.h"
#ifdef VOCAB_EXT
#include "speculative.h"
#endif

#include <algorithm>
#include <cstdlib>
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
    require(argc == 7, "usage: bw-ext-driver MODEL OUTPUT META NP MODE N_PREDICT");
    require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    const bool meta = std::atoi(argv[3]);
    const int np = std::atoi(argv[4]);
    const std::string mode = argv[5];
    const int n_predict = std::atoi(argv[6]);
    require(np == 1 || np == 2, "bad parallel count");
    require(mode == "greedy" || mode == "prob", "bad mode");
    const char * ext_env = std::getenv("DRIVER_VOCAB_EXT");
    const int vocab_ext = ext_env ? std::atoi(ext_env) : 0;
#ifndef VOCAB_EXT
    require(vocab_ext == 0, "the reference build has no extension");
#endif
    llama_backend_init();
    common_log_set_verbosity_thold(5);
    common_params params;
    params.model.path = argv[1];
    params.n_ctx = 4096;
    params.n_parallel = np;
    params.n_batch = params.n_ubatch = 128;
    params.n_gpu_layers = 0;
    params.no_extra_bufts = true; // the extension cannot read a repacked CPU head; every arm runs the same kernels
    params.cpuparams.n_threads = params.cpuparams_batch.n_threads = meta ? 4 : 8;
    params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    params.speculative.draft.n_max = 5;
    params.speculative.draft.n_min = 0;
    params.speculative.draft.p_min = 0;
    params.speculative.draft.probabilistic = mode == "prob";
    params.speculative.draft.backend_sampling = true;
    params.speculative.draft.cpuparams = params.cpuparams;
    params.speculative.draft.cpuparams_batch = params.cpuparams_batch;
    // 1/17 of the vocabulary: most copied tokens are outside the reduced head
    for (int id = 248319; id >= 0; id -= 17) {
        params.speculative.draft.vocab_ids.push_back(id);
    }
#ifdef VOCAB_EXT
    params.speculative.draft.vocab_ext = (uint32_t) vocab_ext;
#endif
    if (meta) {
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        params.devices = {cpu, cpu};
        params.speculative.draft.devices = params.devices;
        params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        params.n_gpu_layers = -1;
    }
    server_context server;
    require(server.load_model(params), "model load failed");

    // Copy tasks: the target repeats rare words of its prompt, which the 1/17 head mostly lacks.
    const std::string paragraphs[2] = {
        "Quetzalcoatl's obsidian zygomorphic xylophones juxtaposed Brobdingnagian kumquats, phlegmatic sphinxes, "
        "chrysanthemum-scented bivouacs and the quixotic Zanzibar philatelist's hieroglyphic marginalia.",
        "Kafkaesque cartographers in Ouagadougou triangulated syzygies between ptarmigan nests, mnemonic "
        "rhododendron gazebos, and Llanfairpwllgwyngyll's onomatopoeic xenon-lit haberdasheries.",
    };
    std::string prompts[2];
    for (int s = 0; s < 2; ++s) {
        prompts[s] = "Copy the following line exactly, ten times, one copy per line, with no other text.\nLine: " +
                paragraphs[s] + "\nCopies:\n";
    }

    std::thread inference;
    json phases = json::array();
    try {
        // phase 1 cold; phase 2 repeats the prompts (prefix reuse: the extension sees the same ids again)
        for (int phase = 0; phase < 2; ++phase) {
            auto reader = server.get_response_reader();
            std::vector<server_task> tasks;
            for (int slot = 0; slot < np; ++slot) {
                server_task task(SERVER_TASK_TYPE_COMPLETION);
                task.id = reader.get_new_id();
                task.id_slot = slot;
                task.cli = true;
                task.cli_prompt = prompts[slot];
                task.params.n_predict = n_predict;
                task.params.return_tokens = true;
                task.params.cache_prompt = true;
                task.params.stream = true;
                task.params.sampling.seed = 1729 + slot;
                if (mode == "greedy") {
                    task.params.sampling.temp = 0;
                    task.params.sampling.top_k = 1;
                } else {
                    task.params.sampling.temp = 1;
                    task.params.sampling.top_k = 20;
                    task.params.sampling.top_p = 0.95f;
                    task.params.sampling.min_p = 0.05f;
                }
                task.params.sampling.backend_sampling = true;
                tasks.push_back(std::move(task));
            }
            reader.post_tasks(std::move(tasks));
            if (phase == 0) {
                inference = std::thread([&] { server.start_loop(); });
            }
            std::vector<llama_tokens> tokens(np);
            std::vector<json> stats;
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
                                         {"accepted", final->stats.n_draft_accepted}, {"rounds", final->stats.n_draft_verif_steps},
                                         {"prompt", final->n_prompt_tokens}, {"cached", final->n_prompt_tokens_cache}});
                }
            }
            require(stats.size() == (size_t) np, "missing final results");
            std::sort(stats.begin(), stats.end(), [](const json & a, const json & b) { return a["slot"].get<int>() < b["slot"].get<int>(); });
            json token_rows = json::array();
            for (int slot = 0; slot < np; ++slot) {
                json row = json::array();
                for (llama_token token : tokens[slot]) {
                    row.push_back(token);
                }
                token_rows.push_back(std::move(row));
            }
            phases.push_back(json{{"phase", phase}, {"tokens", token_rows}, {"stats", json(stats)}});
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
    json out = {{"phases", phases}, {"vocab_ext", vocab_ext}, {"mode", mode}, {"meta", meta}, {"np", np}};
#ifdef VOCAB_EXT
    const auto ext = common_speculative_get_vocab_ext_stats();
    out["ext_stats"] = json{{"requests", ext.n_requests}, {"prompt_oov", ext.n_prompt_oov}, {"placed", ext.n_placed},
                            {"drafted_oov", ext.n_drafted_oov}, {"accepted_oov", ext.n_accepted_oov}};
#endif
    std::ofstream file(argv[2]);
    file << out.dump(2) << '\n';
    require(bool(file), "output write failed");
}
