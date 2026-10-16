// Model-free contract tests of common_speculative: the draft prompt is built only for drafters that read it.

#include "common.h"
#include "arg.h"
#include "json.h"
#include "speculative.h"
#include "sampling.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "../ggml/src/ggml-syncfree.h"
#include "../src/llama-ext.h"
#include "../src/llama-sampler.h"
#include "../src/llama-syncfree.h"
#include "../tools/server/server-output.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#define CHECK(cond)                                                           \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

static llama_tokens periodic_prompt(size_t n, size_t period) {
    llama_tokens res(n);
    for (size_t i = 0; i < n; ++i) {
        res[i] = 100 + (llama_token) (i % period);
    }
    return res;
}

// mimics tools/server: pass the prompt only if the drafters read it
static llama_tokens draft_once(common_speculative * spec, const llama_tokens & prompt, llama_token id_last) {
    llama_tokens result;
    llama_tokens spec_prompt;

    const bool uses_prompt = common_speculative_uses_prompt(spec);
    if (uses_prompt) {
        spec_prompt = prompt;
    }

    common_speculative_begin(spec, 0, prompt);

    common_speculative_get_draft_params(spec, 0) = {
        /* .drafting = */ true,
        /* .n_max    = */ 8,
        /* .pos0     = */ (llama_pos) prompt.size(),
        /* .id_last  = */ id_last,
        /* .prompt   = */ uses_prompt ? &spec_prompt : nullptr,
        /* .result   = */ &result,
    };

    common_speculative_draft(spec);

    return result;
}

static void test_ngram(common_speculative_type type, bool expect_draft) {
    common_params_speculative params;
    params.types = { type };
    params.ngram_simple.size_n = 4;
    params.ngram_simple.size_m = 6;
    params.ngram_map_k.size_n  = 4;
    params.ngram_map_k.size_m  = 6;

    common_speculative * spec = common_speculative_init(params, 1);
    CHECK(spec != nullptr);

    // every n-gram drafter reads the prompt: leaving it null would crash
    CHECK(common_speculative_uses_prompt(spec));

    const size_t period = 20;
    const llama_tokens prompt = periodic_prompt(200, period);
    const llama_token id_last = 100 + (llama_token) (prompt.size() % period);

    const llama_tokens draft = draft_once(spec, prompt, id_last);

    if (expect_draft) {
        CHECK(!draft.empty());
        for (size_t i = 0; i < draft.size(); ++i) {
            CHECK(draft[i] == 100 + (llama_token) ((prompt.size() + 1 + i) % period));
        }
    }

    printf("%s: uses_prompt, draft size %zu: OK\n", common_speculative_type_to_str(type).c_str(), draft.size());

    common_speculative_free(spec);
}

static void test_syncfree_greedy() {
    constexpr int k = 5;
    constexpr int ns_max = 2;
    float packet[ns_max*(2 + 23*k)] = {};
    for (int seq = 0; seq < ns_max; ++seq) {
        auto * data = packet + seq*(2 + 23*k);
        data[0] = 8192 + seq*4096;
        data[1] = 1;
        for (int i = 0; i < k; ++i) {
            data[2 + i] = 100 + seq*1000 + i;
        }
    }
    for (int ns : {1, 2}) {
        for (int np : {1, 4}) {
            for (int p0 = 0; p0 <= k; ++p0) {
                for (int p1 = 0; p1 <= (ns == 1 ? 0 : k); ++p1) {
                    const int prefixes[] = {p0, p1};
                    int32_t sampled[ns_max*(k + 1)];
                    for (int seq = 0; seq < ns; ++seq) {
                        for (int i = 0; i <= k; ++i) {
                            sampled[seq*(k + 1) + i] = (i < prefixes[seq] ? 100 : 200) + seq*1000 + i;
                        }
                    }
                    std::vector<int32_t> out(ns*(4 + k + np*k));
                    ggml_syncfree_accept_batch(nullptr, nullptr, sampled, packet, out.data(), k, np, 1, ns);
                    for (int seq = 0; seq < ns; ++seq) {
                        const int prefix = prefixes[seq];
                        const auto * row = out.data() + seq*(4 + k + np*k);
                        const int base = 8192 + seq*4096;
                        CHECK(row[0] == prefix);
                        CHECK(row[1] == 200 + seq*1000 + prefix);
                        CHECK(row[2] == base + prefix + 1);
                        for (int i = 0; i <= prefix; ++i) {
                            CHECK(row[3 + i] == (i < prefix ? 100 + seq*1000 + i : row[1]));
                        }
                        for (int axis = 0; axis < np; ++axis) {
                            for (int step = 0; step < k; ++step) {
                                CHECK(row[4 + k + axis*k + step] == (axis < 3 ? row[2] + step : 0));
                            }
                        }
                        for (int pos = base - 1; pos <= base + 2*k + 1; ++pos) {
                            const bool rejected = pos >= base + prefix + 1 && pos <= base + k;
                            CHECK(ggml_syncfree_mask_value(0.0f, pos, row, packet + seq*(2 + 23*k), k) == (rejected ? -INFINITY : 0.0f));
                            CHECK(ggml_syncfree_mask_value(-INFINITY, pos, row, packet + seq*(2 + 23*k), k) == -INFINITY);
                        }
                    }
                }
            }
        }
    }
    printf("syncfree: np1/np2 greedy prefix pairs, 1/4 position axes and per-sequence masks: OK\n");
}

static llama_model * sampler_test_model() {
    auto * meta = gguf_init_empty();
    gguf_set_val_str(meta, "general.architecture", "llama");
    gguf_set_val_str(meta, "tokenizer.ggml.model", "none");
    gguf_set_val_u32(meta, "llama.vocab_size", 64);
    std::vector<uint8_t> bytes(gguf_get_meta_size(meta));
    gguf_get_meta_data(meta, bytes.data());
    gguf_free(meta);
    FILE * file = std::tmpfile();
    CHECK(file);
    CHECK(std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size());
    std::rewind(file);
    auto params = llama_model_default_params();
    ggml_backend_dev_t devices[] = {nullptr};
    params.devices = devices;
    params.vocab_only = true;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file_ptr(file, params);
    std::fclose(file);
    CHECK(model);
    CHECK(llama_vocab_n_tokens(llama_model_get_vocab(model)) == 64);
    return model;
}

static common_params parse_served_params(int argc, char ** argv) {
    std::vector<std::string> command;
    common_json env;
    std::filesystem::path dir;
    if (argc == 3) {
        auto read_json = [](const char * path) {
            std::ifstream file(path);
            CHECK(file.good());
            return common_json::parse(std::string(std::istreambuf_iterator<char>(file), {}));
        };
        for (const auto & arg : read_json(argv[1])) {
            command.push_back(arg.get<std::string>());
        }
        env = read_json(argv[2]);
    } else {
        CHECK(argc == 1);
        dir = std::filesystem::temp_directory_path() / ("test-syncfree-" + std::to_string(std::random_device{}()));
        CHECK(std::filesystem::create_directory(dir));
        std::ofstream(dir / "chat.jinja") << "{{ messages }}";
        std::ofstream(dir / "vocab.json") << "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]";
        command = {"server", "-m", "synthetic.gguf", "-c", "102400", "-np", "1", "-ctk", "q8_0", "-ctv", "q8_0",
            "-ngl", "99", "-fa", "on", "--spec-type", "draft-mtp", "--spec-draft-n-max", "5", "--spec-draft-n-min", "0",
            "--spec-draft-p-min", "0", "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16", "-sm", "tensor",
            "--tensor-split", "50,50", "-b", "2048", "-ub", "512", "--jinja", "--chat-template-file", (dir / "chat.jinja").string(),
            "--temp", "1", "--top-k", "20", "--top-p", "0.95", "--min-p", "0.05", "--host", "127.0.0.1", "--port", "18100",
            "--metrics", "--spec-draft-sampling", "probabilistic", "--spec-draft-backend-sampling", "--backend-sampling",
            "-t", "8", "-tb", "8", "--spec-draft-vocab", (dir / "vocab.json").string(), "--cache-ram", "0"};
        env = common_json::parse(R"({"LLAMA_MTP_CHAIN":"1","LLAMA_MTP_FOLD_C":"1","LLAMA_ARG_KV_PAD":"4096","LLAMA_ARG_LOG_VERBOSITY":"5"})");
    }
    // Replay configuration variables without redirecting CPU logs/traces into the canary's result directory.
    std::vector<std::pair<std::string, std::string>> saved;
    for (const char * name : {"LLAMA_MTP_CHAIN", "LLAMA_MTP_FOLD_C", "LLAMA_ARG_KV_PAD", "LLAMA_ARG_LOG_VERBOSITY"}) {
        const char * old = std::getenv(name);
        saved.emplace_back(name, old ? old : "");
        common_set_env(name, !env.contains(name) || env[name].is_null() ? "" : env[name].get<std::string>());
    }
    std::vector<char *> args;
    for (auto & arg : command) {
        args.push_back(arg.data());
    }
    common_params params;
    CHECK(common_params_parse(args.size(), args.data(), params, LLAMA_EXAMPLE_SERVER));
    CHECK(std::find(params.speculative.types.begin(), params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end());
    CHECK(params.speculative.draft.mtp_chain && params.speculative.draft.probabilistic && params.speculative.draft.backend_sampling);
    CHECK(params.speculative.draft.n_max == 5 && params.speculative.draft.p_min == 0.0f && !params.speculative.draft.vocab_ids.empty());
    CHECK(params.sampling.backend_sampling && params.sampling.temp == 1.0f && params.sampling.top_k == 20);
    CHECK(params.sampling.top_p == 0.95f && params.sampling.min_p == 0.05f);
    CHECK(params.n_parallel == 1 || params.n_parallel == 2);
    CHECK(params.split_mode == LLAMA_SPLIT_MODE_TENSOR && params.tensor_split[0] == 50.0f && params.tensor_split[1] == 50.0f);
    const auto limits = common_speculative_get_output_limits(params.n_batch, params.n_parallel, params.speculative.draft.n_max);
    params.n_outputs_max = limits.total;
    params.n_outputs_max_per_seq = limits.per_seq;
    const auto target = common_context_params_to_llama(params);
    const auto draft = common_context_params_to_llama(common_base_params_to_speculative(params));
    CHECK(target.n_outputs_max_per_seq == 6 && target.n_rs_seq == 5);
    CHECK(draft.n_outputs_max_per_seq == (params.n_parallel == 1 || llama_syncfree_enabled() ? 5u : 1u));
    for (const auto & [name, value] : saved) {
        common_set_env(name, value);
    }
    if (!dir.empty()) {
        std::filesystem::remove_all(dir);
    }
    printf("syncfree: parsed served server args/env, np=%d, target rows=%u, draft rows=%u: OK\n", params.n_parallel, target.n_outputs_max_per_seq, draft.n_outputs_max_per_seq);
    return params;
}

static void test_syncfree_served_positions(const llama_model * model) {
    for (int ns : {1, 2}) {
        llama_batch_ext batch(12, 1, 1, ns, nullptr, 64, 4);
        for (int seq = 0; seq < ns; ++seq) {
            for (int step = 0; step <= 5; ++step) {
                const int row = llama_batch_ext_add_token(&batch, seq, step);
                const llama_pos pos[] = {8192 + seq*4096 + step, 0, 0, 0};
                CHECK(llama_batch_ext_set_pos(&batch, row, pos));
                CHECK(llama_batch_ext_set_output_logits(&batch, row, true));
            }
        }
        llama_batch_allocr alloc(4);
        CHECK(alloc.init(batch, *llama_model_get_vocab(model), true));
        auto verify = alloc.split_simple(12);
        CHECK(verify.n_tokens == (uint32_t) ns*6 && verify.n_pos == 4);
        for (int seq = 0; seq < ns; ++seq) {
            CHECK(llama_syncfree_text_positions(verify, seq*6, 6));
            CHECK(verify.pos[3*verify.n_tokens + seq*6] == 0);
            // The old guard required this value to equal base_pos, so every served text round failed.
            CHECK(verify.pos[3*verify.n_tokens + seq*6] != verify.pos[seq*6]);
        }
        verify.pos[3*verify.n_tokens] = verify.pos[0];
        CHECK(!llama_syncfree_text_positions(verify, 0, 6));
    }
    printf("syncfree: served batch allocator text positions [p,p,p,0], np1/np2 k=5: OK\n");
}

static void test_syncfree_served_samplers(const common_params & served) {
    auto * model = sampler_test_model();
    test_syncfree_served_positions(model);
    auto * backend = ggml_backend_cpu_init();
    CHECK(backend);
    auto * buft = ggml_backend_get_default_buffer_type(backend);
    auto * draft = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(draft, llama_sampler_init_top_k_sorted(10));
    llama_sampler_chain_add(draft, llama_sampler_init_temp_ext(1.0f, 0.0f, 1.0f));
    llama_sampler_chain_add(draft, llama_sampler_init_dist(1729 ^ 0x85ebca6bu));
    CHECK(draft->iface->backend_init(draft, buft, 5));

    common_params_sampling params = served.sampling;
    params.seed = 1729;
    // The server merges ignore_eos and model suppression into the same banning bias.
    params.logit_bias = {{0, -INFINITY}, {7, -INFINITY}, {25, -INFINITY}};

    auto * unsupported = common_sampler_init(model, params);
    auto * candidate_only = common_sampler_get_backend(unsupported);
    CHECK(!llama_syncfree_samplers_supported(candidate_only, draft));
    common_sampler_free(unsupported);
    for (bool typical : {true, false}) {
        params.typ_p = typical ? 0.8f : 1.0f;
        params.min_keep = typical ? 0 : 2;
        unsupported = common_sampler_init(model, params);
        candidate_only = common_sampler_get_backend(unsupported);
        CHECK(candidate_only->iface->backend_init(candidate_only, buft, 6) == !llama_syncfree_enabled());
        CHECK(!llama_syncfree_samplers_supported(candidate_only, draft));
        common_sampler_free(unsupported);
    }
    params.typ_p = 1.0f;
    params.min_keep = 0;

    auto logits_for = [](size_t row) {
        std::vector<float> logits(64);
        for (int id = 0; id < 64; ++id) {
            logits[id] = row % 3 == 0 ? -0.17f*((id*19 + int(row)*7) % 64) :
                row % 3 == 1 ? -3.2f - 0.001f*id : -10.0f - 0.01f*id;
        }
        if (row % 3 != 0) {
            logits[1] = 0.0f;
        }
        if (row % 3 == 2) {
            logits[2] = std::log(0.05f);
            logits[3] = std::log(0.04f);
        }
        return logits;
    };

    for (int ns : {1, 2}) {
        for (float temp : {1.0f, 0.7f}) {
            params.temp = temp;
            std::vector<common_sampler *> owners;
            std::vector<llama_sampler *> chains;
            for (int seq = 0; seq < ns; ++seq) {
                auto * owner = common_sampler_init(model, params);
                auto * chain = common_sampler_get_backend(owner);
                CHECK(chain->iface->backend_init(chain, buft, 6));
                CHECK(llama_syncfree_samplers_supported(chain, draft) == llama_syncfree_enabled());
                owners.push_back(owner);
                chains.push_back(chain);
            }
            constexpr int max_nodes = 4096;
            auto * ctx = ggml_init({max_nodes*ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false), nullptr, true});
            CHECK(ctx);
            auto * graph = ggml_new_graph_custom(ctx, max_nodes, false);
            std::vector<llama_sampler_data> rows;
            std::vector<ggml_tensor *> inputs;
            std::vector<ggml_tensor *> candidate_inputs;
            for (int seq = 0; seq < ns; ++seq) {
                for (int step = 0; step <= 5; ++step) {
                    const bool split = step % 2 == 0;
                    auto * logits = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, split ? 48 : 64);
                    auto * ids = split ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 48) : nullptr;
                    ggml_set_input(logits);
                    if (ids) {
                        ggml_set_input(ids);
                    }
                    inputs.push_back(logits);
                    candidate_inputs.push_back(ids);
                    llama_sampler_data row = {logits, nullptr, nullptr, ids};
                    chains[seq]->iface->backend_apply(chains[seq], ctx, graph, &row);
                    CHECK(row.candidates);
                    CHECK(bool(row.probs) == llama_syncfree_enabled());
                    CHECK(bool(row.sampled) == llama_syncfree_enabled());
                    CHECK(ggml_nelements(row.candidates) == (llama_syncfree_enabled() ? 20 : 24));
                    for (auto * output : {row.logits, row.probs, row.sampled, row.candidates}) {
                        if (output) {
                            ggml_build_forward_expand(graph, output);
                        }
                    }
                    rows.push_back(row);
                }
            }
            auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            CHECK(buffer);
            for (auto * chain : chains) {
                chain->iface->backend_set_input(chain);
            }
            for (size_t i = 0; i < rows.size(); ++i) {
                const auto logits = logits_for(i);
                if (candidate_inputs[i]) {
                    // Tensor splitting gathers the top 24 from each of two vocabulary shards.
                    std::vector<llama_token> ids;
                    std::vector<float> compact;
                    for (int device = 0; device < 2; ++device) {
                        std::vector<llama_token> shard(32);
                        for (int id = 0; id < 32; ++id) {
                            shard[id] = device*32 + id;
                        }
                        std::sort(shard.begin(), shard.end(), [&](llama_token a, llama_token b) { return logits[a] > logits[b]; });
                        for (int j = 0; j < 24; ++j) {
                            ids.push_back(shard[j]);
                            compact.push_back(logits[shard[j]]);
                        }
                    }
                    ggml_backend_tensor_set(inputs[i], compact.data(), 0, compact.size()*sizeof(float));
                    ggml_backend_tensor_set(candidate_inputs[i], ids.data(), 0, ids.size()*sizeof(llama_token));
                } else {
                    ggml_backend_tensor_set(inputs[i], logits.data(), 0, logits.size()*sizeof(float));
                }
            }
            CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            for (size_t i = 0; i < rows.size(); ++i) {
                const auto logits = logits_for(i);
                std::vector<llama_token_data> tokens(64);
                for (int id = 0; id < 64; ++id) {
                    tokens[id] = {id, logits[id], 0.0f};
                }
                auto * host = common_sampler_init(model, params);
                llama_token_data_array expected = {tokens.data(), tokens.size(), -1, false};
                llama_sampler_apply(common_sampler_get(host), &expected);
                if (i % 3 != 0) {
                    CHECK(expected.size == (i % 3 == 1 ? 1 : 2));
                }
                if (!llama_syncfree_enabled()) {
                    float compact[24];
                    llama_token ids[24];
                    ggml_backend_tensor_get(rows[i].logits, compact, 0, sizeof(compact));
                    ggml_backend_tensor_get(rows[i].candidates, ids, 0, sizeof(ids));
                    auto * owner = owners[i/6];
                    common_sampler_reset(owner);
                    int full_reads = 0;
                    common_sampler_row row;
                    row.n_vocab = 64;
                    row.cand_ids = ids;
                    row.cand_logits = compact;
                    row.n_cand = 24;
                    row.logits = [&] { ++full_reads; return logits.data(); };
                    CHECK(common_sampler_sample_row(owner, row) == expected.data[expected.selected].id);
                    CHECK(full_reads == 0);
                    const auto * actual = common_sampler_get_candidates(owner, true);
                    CHECK(actual->size == expected.size);
                    for (size_t j = 0; j < expected.size; ++j) {
                        CHECK(actual->data[j].id == expected.data[j].id);
                        CHECK(actual->data[j].p == expected.data[j].p);
                    }
                    common_sampler_free(host);
                    continue;
                }
                float probs[20];
                llama_token ids[20], sampled;
                ggml_backend_tensor_get(rows[i].probs, probs, 0, sizeof(probs));
                ggml_backend_tensor_get(rows[i].candidates, ids, 0, sizeof(ids));
                ggml_backend_tensor_get(rows[i].sampled, &sampled, 0, sizeof(sampled));
                float sum = 0.0f;
                bool found = false;
                for (int j = 0; j < 20; ++j) {
                    float p = 0.0f;
                    for (size_t k = 0; k < expected.size; ++k) {
                        if (expected.data[k].id == ids[j]) {
                            p = expected.data[k].p;
                        }
                    }
                    CHECK(std::fabs(probs[j] - p) < 2e-6f);
                    CHECK(ids[j] != 0 && ids[j] != 7 && ids[j] != 25);
                    found = found || (sampled == ids[j] && probs[j] > 0.0f);
                    sum += probs[j];
                }
                CHECK(found && std::fabs(sum - 1.0f) < 2e-6f);
                common_sampler_free(host);
            }
            if (ns == 1 && llama_syncfree_enabled()) {
                auto * uniform = ggml_get_tensor(ctx, "uniform_0");
                CHECK(uniform);
                std::mt19937 rng(params.seed);
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                float draws[4];
                for (auto & draw : draws) {
                    draw = dist(rng);
                }
                auto check_draw = [&](float expected) {
                    float actual;
                    ggml_backend_tensor_get(uniform, &actual, 0, sizeof(actual));
                    CHECK(actual == expected);
                };
                auto begin = [&] {
                    llama_sampler_backend_begin(chains[0]);
                    chains[0]->iface->backend_set_input(chains[0]);
                };
                check_draw(draws[0]);
                common_sampler_accept(owners[0], 1, true);
                common_sampler_accept(owners[0], 2, true);
                begin();
                check_draw(draws[2]);
                common_sampler_reset(owners[0]);
                begin();
                check_draw(draws[0]);
                auto * saved = common_sampler_clone(owners[0]);
                for (int i = 0; i < 3; ++i) {
                    common_sampler_accept(owners[0], 1, true);
                }
                begin();
                check_draw(draws[3]);
                common_sampler_copy(saved, owners[0]);
                begin();
                check_draw(draws[0]);
                common_sampler_free(saved);
            }
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            for (auto * owner : owners) {
                common_sampler_free(owner);
            }
        }
    }
    llama_sampler_free(draft);
    ggml_backend_free(backend);
    llama_model_free(model);
    printf("syncfree: served common sampler arming, unsupported chain rejection, np1/np2 k=5 probabilities and RNG accept/reset/copy: OK (enabled=%d)\n", llama_syncfree_enabled());
}

static void test_syncfree_rejection() {
    constexpr int draws = 100000;
    constexpr int ns = 2;
    std::mt19937 rng(1729);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    const float p[] = {0.2f, 0.3f, 0.5f, 0.2f, 0.3f, 0.5f, 0.6f, 0.1f, 0.3f, 0.6f, 0.1f, 0.3f};
    const int32_t ids[] = {0, 1, 2, 0, 1, 2, 10, 11, 12, 10, 11, 12};
    const float q[ns][2] = {{0.75f, 0.25f}, {0.1f, 0.9f}};
    float packet[ns*25] = {};
    for (int seq = 0; seq < ns; ++seq) {
        auto * data = packet + seq*25;
        data[0] = 8192 + seq*4096;
        for (int i = 0; i < 10; ++i) {
            data[3 + i] = -1;
        }
        data[3] = seq*10;
        data[4] = seq*10 + 1;
        data[13] = q[seq][0];
        data[14] = q[seq][1];
    }
    int counts[ns][3] = {};
    int accepted[ns] = {};
    for (int i = 0; i < draws; ++i) {
        int32_t sampled[ns*2];
        for (int seq = 0; seq < ns; ++seq) {
            auto * data = packet + seq*25;
            data[2] = seq*10 + (uniform(rng) < q[seq][0] ? 0 : 1);
            data[23] = uniform(rng);
            data[24] = uniform(rng);
            for (int j = 0; j < 2; ++j) {
                const float u = uniform(rng);
                sampled[seq*2 + j] = seq*10 + (u < p[seq*6] ? 0 : (u < p[seq*6] + p[seq*6 + 1] ? 1 : 2));
            }
        }
        int32_t out[ns*9];
        ggml_syncfree_accept_batch(p, ids, sampled, packet, out, 1, 4, 3, ns);
        for (int seq = 0; seq < ns; ++seq) {
            const auto * row = out + seq*9;
            CHECK(row[0] >= 0 && row[0] <= 1);
            CHECK(row[3] >= seq*10 && row[3] < seq*10 + 3);
            CHECK(row[2] == (int) packet[seq*25] + row[0] + 1);
            ++counts[seq][row[3] - seq*10];
            accepted[seq] += row[0];
        }
    }
    const float expected[] = {0.45f, 0.2f};
    for (int seq = 0; seq < ns; ++seq) {
        CHECK(std::abs((double) accepted[seq]/draws - expected[seq]) < 0.01);
        for (int i = 0; i < 3; ++i) {
            CHECK(std::abs((double) counts[seq][i]/draws - p[seq*6 + i]) < 0.01);
        }
        printf("syncfree: np2 seq=%d rejection %d draws, acceptance %.5f, output %.5f %.5f %.5f: OK\n",
            seq, draws, (double) accepted[seq]/draws, (double) counts[seq][0]/draws, (double) counts[seq][1]/draws, (double) counts[seq][2]/draws);
    }
}

static void test_syncfree_output_limits() {
    for (int n_seq : {1, 2}) {
        for (bool chain : {false, true}) {
            common_params params;
            params.n_parallel = n_seq;
            params.speculative.draft.mtp_chain = chain;
            params.speculative.draft.backend_sampling = true;
            params.speculative.draft.n_max = 5;
            const auto draft = common_base_params_to_speculative(params);
            const int per_seq = chain && (n_seq == 1 || llama_syncfree_enabled()) ? 5 : 1;
            CHECK(draft.n_outputs_max_per_seq == per_seq);
            CHECK(draft.n_outputs_max == n_seq*per_seq);
        }
    }
    printf("syncfree: per-sequence sampler output limits, gate=%d: OK\n", llama_syncfree_enabled());
}

struct output_case {
    std::vector<std::vector<llama_tokens>> rounds;
    std::vector<int> limits;
    std::vector<std::string> stops;
    std::vector<int> cancel_round;
};

static output_case make_output_case(int n_seq) {
    output_case c;
    c.rounds.resize(6, std::vector<llama_tokens>(n_seq));
    c.limits.assign(n_seq, -1);
    c.stops.resize(n_seq);
    c.cancel_round.assign(n_seq, -1);
    for (size_t r = 0; r < c.rounds.size(); ++r) {
        for (int seq = 0; seq < n_seq; ++seq) {
            for (int i = 0; i < 6; ++i) {
                c.rounds[r][seq].push_back('a' + (seq*7 + r*6 + i) % 26);
            }
        }
    }
    return c;
}

static std::vector<llama_tokens> run_output_case(const output_case & c, bool overlap, int & n_overlap) {
    const size_t ns = c.limits.size();
    server_output_queue queue;
    std::vector<llama_tokens> emitted(ns);
    std::vector<std::string> text(ns);
    std::vector<bool> active(ns, true);
    std::vector<int> pending(ns, 0);
    bool computing = false;
    n_overlap = 0;
    auto retire = [&]() {
        for (const auto & record : queue.take()) {
            const int seq = record.id_slot;
            CHECK(record.id_task == 100 + seq);
            CHECK(active[seq]);
            CHECK(pending[seq] == int(record.tokens.size()));
            pending[seq] = 0;
            n_overlap += computing;
            for (const auto token : record.tokens) {
                emitted[seq].push_back(token);
                if (token != 0) {
                    text[seq].push_back(char(token));
                }
                if (token == 0 || (c.limits[seq] >= 0 && int(emitted[seq].size()) >= c.limits[seq]) ||
                        (!c.stops[seq].empty() && text[seq].find(c.stops[seq]) != std::string::npos)) {
                    active[seq] = false;
                    break;
                }
            }
        }
    };
    for (size_t r = 0; r < c.rounds.size(); ++r) {
        bool eligible = overlap;
        for (size_t seq = 0; seq < ns; ++seq) {
            if (active[seq] && !c.stops[seq].empty()) {
                eligible = false;
            }
        }
        if (!eligible) {
            retire();
        }
        std::vector<server_output_record> current;
        for (size_t seq = 0; seq < ns; ++seq) {
            if (!active[seq]) {
                continue;
            }
            auto tokens = c.rounds[r][seq];
            if (c.limits[seq] >= 0) {
                const int remaining = server_output_remaining(c.limits[seq], emitted[seq].size(), pending[seq]);
                CHECK(remaining > 0);
                tokens.resize(std::min(tokens.size(), size_t(remaining)));
            }
            current.push_back({int(seq), 100 + int(seq), std::move(tokens)});
        }
        // A cancellation arriving here is declined during the queue yield and handled after this decode.
        computing = true;
        retire();
        computing = false;
        bool defer = eligible;
        for (const auto & record : current) {
            const int seq = record.id_slot;
            const int remaining = c.limits[seq] < 0 ? -1 : c.limits[seq] - int(emitted[seq].size());
            CHECK(remaining == -1 || record.tokens.size() <= size_t(remaining));
            const bool terminal = std::find(record.tokens.begin(), record.tokens.end(), 0) != record.tokens.end();
            defer = server_output_can_defer(defer, terminal, remaining, record.tokens.size());
            pending[seq] = record.tokens.size();
        }
        CHECK(queue.submit(std::move(current)));
        if (!defer) {
            retire();
        }
        for (size_t seq = 0; seq < ns; ++seq) {
            if (c.cancel_round[seq] == int(r)) {
                retire();
                active[seq] = false;
            }
        }
    }
    retire();
    CHECK(queue.empty());
    return emitted;
}

static void test_output_overlap() {
    server_output_queue queue;
    llama_tokens source = {1, 2, 3};
    CHECK(queue.submit({{0, 42, source}}));
    source[0] = 99;
    CHECK(!queue.submit({{0, 43, {4}}}));
    auto saved = queue.take();
    CHECK(saved.size() == 1 && saved[0].id_task == 42 && saved[0].tokens == llama_tokens({1, 2, 3}));
    CHECK(queue.empty() && queue.submit({{0, 43, {4}}}));
    CHECK(queue.take()[0].id_task == 43);
    for (const int64_t remaining : {-2, 0, 1, 6}) {
        CHECK(!server_output_can_defer(true, false, remaining, 6));
    }
    CHECK(server_output_can_defer(true, false, 7, 6));
    CHECK(server_output_can_defer(true, false, -1, 6));
    CHECK(!server_output_can_defer(false, false, -1, 6));
    CHECK(!server_output_can_defer(true, true, -1, 6));
    int n_cases = 0;
    auto compare = [&](const output_case & c, bool expect_overlap) {
        int n_sync, n_async;
        const auto sync = run_output_case(c, false, n_sync);
        const auto async = run_output_case(c, true, n_async);
        CHECK(sync == async && n_sync == 0);
        CHECK(!expect_overlap || n_async > 0);
        ++n_cases;
        return async;
    };
    for (int ns : {1, 2}) {
        compare(make_output_case(ns), true);
        for (int eos = 0; eos < 6; ++eos) {
            auto c = make_output_case(ns);
            for (int seq = 0; seq < ns; ++seq) {
                c.rounds[1 + 2*seq][seq][eos] = 0;
            }
            const auto emitted = compare(c, true);
            for (int seq = 0; seq < ns; ++seq) {
                CHECK(emitted[seq].size() == size_t((1 + 2*seq)*6 + eos + 1));
                CHECK(emitted[seq].back() == 0);
            }
        }
        for (int limit = 1; limit <= 18; ++limit) {
            auto c = make_output_case(ns);
            for (int seq = 0; seq < ns; ++seq) {
                c.limits[seq] = limit + 3*seq;
            }
            const auto emitted = compare(c, limit > 6);
            for (int seq = 0; seq < ns; ++seq) {
                CHECK(emitted[seq].size() == size_t(limit + 3*seq));
            }
        }
        auto stop = make_output_case(ns);
        stop.stops[0] = "hij";
        CHECK(compare(stop, false)[0].size() == 10);
        stop.stops[0] = "efg";
        CHECK(compare(stop, false)[0].size() == 7);
        auto cancel = make_output_case(ns);
        for (int seq = 0; seq < ns; ++seq) {
            cancel.cancel_round[seq] = 1 + 2*seq;
        }
        const auto emitted = compare(cancel, true);
        for (int seq = 0; seq < ns; ++seq) {
            CHECK(emitted[seq].size() == size_t((2 + 2*seq)*6));
        }
    }
    printf("syncfree: output overlap %d greedy streams, EOS/stop/budget/cancel, np1/np2, owned records and backpressure: OK\n", n_cases);
}

int main(int argc, char ** argv) {
    test_syncfree_served_samplers(parse_served_params(argc, argv));
    test_syncfree_greedy();
    test_syncfree_rejection();
    test_syncfree_output_limits();
    test_output_overlap();
    CHECK(!common_speculative_uses_prompt(nullptr));

    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, true);
    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,  true);
    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, false);
    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_MOD,    false);
    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,  false);

    return 0;
}
