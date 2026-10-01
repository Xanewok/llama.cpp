#include "common.h"
#include "chat.h"
#include "sampling.h"
#include "llama-cpp.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include "json.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <set>
#include <stdexcept>

// Use the frozen server's Responses conversion without starting a server.
using json = nlohmann::ordered_json;
common_json server_chat_convert_responses_to_chatcmpl(const common_json &);

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static json read_json(const std::string & path) {
    std::ifstream in(path);
    require(in.good(), "input missing");
    json value;
    in >> value;
    return value;
}

static void write_json(const std::string & path, const json & value) {
    std::ofstream out(path + ".tmp");
    out << value.dump(2) << '\n';
    require(out.good(), "JSON write failed");
    out.close();
    std::filesystem::rename(path + ".tmp", path);
}

static llama_context_params context_params(int n_ctx, ggml_type kv, int rs, bool mtp) {
    auto p = llama_context_default_params();
    p.n_ctx = n_ctx;
    p.n_batch = 2048;
    p.n_ubatch = 128;
    p.n_outputs_max_per_seq = 6;
    p.n_rs_seq = mtp ? 0 : rs;
    p.n_threads = p.n_threads_batch = 4;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    p.type_k = p.type_v = mtp ? GGML_TYPE_F16 : kv;
    p.ctx_type = mtp ? LLAMA_CONTEXT_TYPE_MTP : LLAMA_CONTEXT_TYPE_DEFAULT;
    return p;
}

static common_params_sampling sampling_params(bool draft) {
    common_params_sampling p;
    p.seed = 1729;
    p.no_perf = true;
    p.backend_sampling = !draft;
    p.temp = 1;
    p.top_k = draft ? 10 : 20;
    p.top_p = 0.95f;
    p.min_p = 0.05f;
    if (draft) p.samplers = {COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE};
    return p;
}

static void target_decode(llama_context * ctx, const llama_tokens & tokens, int pos, int count, bool outputs) {
    auto batch = llama_batch_init(count, 0, 1);
    for (int i = 0; i < count; ++i) common_batch_add(batch, tokens.at(pos + i), pos + i, {0}, outputs);
    const int status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    require(status == 0, "target decode failed");
}

static void draft_decode(llama_context * ctx, const llama_tokens & tokens, int pos, int count,
                         const std::vector<float> & shifted_hidden, int ne, bool outputs) {
    auto * batch = llama_batch_ext_init(ctx);
    for (int i = 0; i < count; ++i) {
        const llama_pos at = pos + i;
        const int idx = llama_batch_ext_add_token(batch, 0, tokens.at(at));
        require(idx >= 0, "draft token add failed");
        require(llama_batch_ext_set_pos(batch, idx, &at), "draft position failed");
        require(llama_batch_ext_set_embd_token(batch, idx, {shifted_hidden.data() + size_t(i)*ne, 1, size_t(ne)}), "draft hidden failed");
        require(llama_batch_ext_set_output_logits(batch, idx, outputs), "draft output flag failed");
    }
    const int status = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    llama_batch_ext_free(batch);
    require(status == 0, "draft decode failed");
}

static std::vector<float> hidden(llama_context * ctx, int row, int ne) {
    const auto * h = llama_get_embeddings_nextn_ith(ctx, row);
    require(h, "hidden row missing");
    return {h, h + ne};
}

struct checkpoint {
    std::vector<uint8_t> bytes;
    int pos;
    checkpoint(llama_context * ctx, int pos) : pos(pos) {
        const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        bytes.resize(llama_state_seq_get_size_ext(ctx, 0, flags));
        require(!bytes.empty(), "empty recurrent checkpoint");
        require(llama_state_seq_get_data_ext(ctx, bytes.data(), bytes.size(), 0, flags) == bytes.size(), "checkpoint save failed");
    }
    void restore(llama_context * ctx) const {
        require(llama_state_seq_set_data_ext(ctx, bytes.data(), bytes.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == bytes.size(), "checkpoint restore failed");
        require(llama_memory_seq_rm(llama_get_memory(ctx), 0, pos, -1), "checkpoint suffix trim failed");
    }
};

static void save_row(llama_model * model, llama_context * ctx, int row, bool draft,
                     const std::string & path, llama_token next, const std::set<llama_token> & loop) {
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * ptr = llama_get_logits_ith(ctx, row);
    require(ptr, "logit row missing");
    std::vector<float> logits(ptr, ptr + nv);
    for (float value : logits) require(std::isfinite(value) || value == -INFINITY, "invalid logit");
    std::ofstream out(path + ".f32.tmp", std::ios::binary);
    out.write(reinterpret_cast<const char *>(logits.data()), nv*sizeof(float));
    require(out.good(), "logit write failed");
    out.close();

    auto params = sampling_params(draft);
    params.backend_sampling = false;
    common_sampler_ptr sampler(common_sampler_init(model, params));
    require(sampler != nullptr, "sampler init failed");
    common_sampler_row input;
    input.n_vocab = nv;
    input.logits = [&] { return logits.data(); };
    const auto sampled = common_sampler_sample_row(sampler.get(), input);
    const auto * candidates = common_sampler_get_candidates(sampler.get(), true);
    std::vector<llama_token_data> law(candidates->data, candidates->data + candidates->size);
    auto actual_params = sampling_params(draft);
    common_sampler_ptr actual(common_sampler_init(model, actual_params));
    require(common_sampler_sample(actual.get(), ctx, row) == sampled, "backend/host seed mismatch");
    const auto * actual_law = common_sampler_get_candidates(actual.get(), true);
    require(actual_law->size == law.size(), "backend/host support mismatch");
    for (size_t i = 0; i < law.size(); ++i) {
        require(actual_law->data[i].id == law[i].id && actual_law->data[i].p == law[i].p, "backend/host probability mismatch");
    }
    json dist = json::array();
    double mass = 0, next_p = 0, loop_p = 0;
    for (auto e : law) {
        require(std::isfinite(e.p) && e.p >= 0, "invalid probability");
        if (e.p == 0) continue;
        dist.push_back({e.id, e.p});
        mass += e.p;
        if (e.id == next) next_p = e.p;
        if (loop.count(e.id)) loop_p += e.p;
    }
    require(std::fabs(mass - 1) < 1e-5, "probability mass not normalized");

    llama_sampler * draw = llama_sampler_init_dist(1729);
    llama_token_data_array array = {law.data(), law.size(), -1, true};
    int next_hits = 0, loop_hits = 0;
    json samples = json::array();
    for (int i = 0; i < 4096; ++i) {
        llama_sampler_apply(draw, &array);
        require(array.selected >= 0 && size_t(array.selected) < array.size, "sample missing");
        const llama_token id = array.data[array.selected].id;
        next_hits += id == next;
        loop_hits += loop.count(id) != 0;
        if (i < 16) samples.push_back(id);
    }
    llama_sampler_free(draw);
    std::vector<int> ids(nv);
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + 20, ids.end(), [&](int a, int b) {
        return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
    });
    ids.resize(20);
    write_json(path + ".json", {{"n_vocab", nv}, {"law", dist}, {"top20", ids},
        {"next_token", next}, {"next_probability_D", next_p}, {"loop_set_probability_D", loop_p},
        {"sample_count_M", 4096}, {"next_hits_M", next_hits}, {"loop_hits_M", loop_hits}, {"samples_M", samples}});
    std::filesystem::rename(path + ".f32.tmp", path + ".f32");
}

static void prepare(llama_model * model, const std::string & history, const std::string & output) {
    const auto data = read_json(history);
    auto tmpls = common_chat_templates_init(model, data.at("template"));
    const auto chat = server_chat_convert_responses_to_chatcmpl(common_json::parse(data.at("request").dump()));
    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(chat.at("messages"));
    inputs.enable_thinking = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    inputs.chat_template_kwargs["reasoning_effort"] = "\"xhigh\"";
    if (chat.contains("tools")) inputs.tools = common_chat_tools_parse_oaicompat(chat.at("tools"));
    const auto prompt = common_chat_templates_apply(tmpls.get(), inputs).prompt;
    const auto generation = data.at("generation").get<std::string>();
    const auto onset_byte = data.at("onset_utf8_bytes").get<size_t>();
    require(onset_byte <= generation.size(), "onset byte outside generation");
    const auto * vocab = llama_model_get_vocab(model);
    auto tokens = common_tokenize(vocab, prompt + generation, true, true);
    const auto before = common_tokenize(vocab, prompt + generation.substr(0, onset_byte), true, true);
    size_t onset = 0;
    while (onset < before.size() && onset < tokens.size() && before[onset] == tokens[onset]) ++onset;
    // The full stream determines token boundaries. A marker can fall inside its final prefix token.
    std::set<int> points;
    if (onset > 1 && onset + 6 < tokens.size()) points.insert(onset);
    for (int depth : {20000, 50000, 90000}) if (depth + 6 < int(tokens.size())) points.insert(depth);
    require(!points.empty(), "history has no usable point");
    write_json(output, {{"name", data.at("name")}, {"tokens", tokens}, {"points", points},
        {"onset_token_D", onset}, {"total_tokens_D", tokens.size()},
        {"prompt_tokens_D", common_tokenize(vocab, prompt, true, true).size()},
        {"configured_rs", data.at("configured_rs")}, {"context_size", data.at("context_size")},
        {"provenance", data.at("provenance")}, {"fidelity", data.at("fidelity")}});
    std::printf("D: %s prompt=%zu onset=%zu total=%zu\n", data.at("name").get<std::string>().c_str(),
                common_tokenize(vocab, prompt, true, true).size(), onset, tokens.size());
}

static void run(llama_model * model, const json & data, const std::string & output, ggml_type kv, int rs) {
    const auto tokens = data.at("tokens").get<llama_tokens>();
    const auto points = data.at("points").get<std::vector<int>>();
    require(!points.empty() && std::is_sorted(points.begin(), points.end()), "points not sorted");
    const int ne = llama_model_n_embd_out(model);
    if (rs == 0) rs = data.at("configured_rs").get<int>();
    const int n_ctx = data.value("context_size", points.back() + 128);
    require(n_ctx > points.back() + 6, "context too small");
    llama_context_ptr target(llama_init_from_model(model, context_params(n_ctx, kv, rs, false)));
    auto draft_params = context_params(n_ctx, kv, rs, true);
    draft_params.ctx_other = target.get();
    llama_context_ptr draft(llama_init_from_model(model, draft_params));
    require(target && draft, "context init failed");
    require(llama_model_n_layer_nextn(model) == 1, "probe requires one MTP head");
    llama_set_embeddings_nextn(target.get(), true, false);
    llama_set_embeddings_nextn(draft.get(), true, true);
    auto backend_params = sampling_params(false);
    common_sampler_ptr backend(common_sampler_init(model, backend_params));
    const bool installed = llama_set_sampler(target.get(), 0, common_sampler_get_backend(backend.get()));
    llama_sampler * draft_backend = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(draft_backend, llama_sampler_init_top_k(10));
    const bool draft_installed = llama_set_sampler(draft.get(), 0, draft_backend);
    std::filesystem::create_directories(output);
    write_json(output + "/config.json", {{"rs", rs}, {"target_backend", installed}, {"draft_backend", draft_installed},
        {"points", points}, {"kv", ggml_type_name(kv)}, {"n_ctx", n_ctx}, {"draft_rs", 0}, {"name", data.at("name")}});

    int pos = 0;
    std::vector<float> carry(ne, 0);
    const auto started = std::chrono::steady_clock::now();
    for (int depth : points) {
        require(depth > 1 && depth + 6 < int(tokens.size()), "point outside history");
        const int anchor = depth - 1;
        while (pos < anchor) {
            const int count = std::min(2048, anchor - pos);
            target_decode(target.get(), tokens, pos, count, false);
            std::vector<float> shifted(size_t(count)*ne);
            std::copy(carry.begin(), carry.end(), shifted.begin());
            for (int i = 1; i < count; ++i) {
                const auto h = hidden(target.get(), i - 1, ne);
                std::copy(h.begin(), h.end(), shifted.begin() + size_t(i)*ne);
            }
            carry = hidden(target.get(), count - 1, ne);
            draft_decode(draft.get(), tokens, pos, count, shifted, ne, false);
            pos += count;
        }
        const checkpoint base_target(target.get(), anchor), base_draft(draft.get(), anchor);
        const std::string point = output + "/d" + std::to_string(depth);
        std::filesystem::create_directories(point);
        const std::set<llama_token> loop(tokens.begin() + depth, tokens.begin() + std::min(int(tokens.size()), depth + 64));
        write_json(point + "/point.json", {{"depth_D", depth}, {"anchor_D", anchor}, {"loop_ids", loop},
            {"onset", depth == data.value("onset_token_D", -1)}});
        if (std::getenv("VERIFY_INDEPENDENT")) {
            for (int width : {1, 4, 5, 6}) {
                llama_context_ptr fresh(llama_init_from_model(model, context_params(n_ctx, kv, rs, false)));
                require(fresh != nullptr, "independent context init failed");
                llama_set_embeddings_nextn(fresh.get(), true, false);
                auto fresh_params = sampling_params(false);
                common_sampler_ptr fresh_backend(common_sampler_init(model, fresh_params));
                llama_set_sampler(fresh.get(), 0, common_sampler_get_backend(fresh_backend.get()));
                for (int at = 0; at < anchor;) {
                    const int count = std::min(2048, anchor - at);
                    target_decode(fresh.get(), tokens, at, count, false);
                    at += count;
                }
                if (width == 1) {
                    for (int row = 0; row < 6; ++row) {
                        target_decode(fresh.get(), tokens, anchor + row, 1, true);
                        save_row(model, fresh.get(), 0, false, point + "/independent-single-r" + std::to_string(row), tokens[depth + row], loop);
                    }
                } else {
                    target_decode(fresh.get(), tokens, anchor, width, true);
                    for (int row = 0; row < width; ++row) {
                        save_row(model, fresh.get(), row, false, point + "/independent-verify-k" + std::to_string(width - 1) + "-r" + std::to_string(row), tokens[depth + row], loop);
                    }
                }
                llama_set_sampler(fresh.get(), 0, nullptr);
            }
        }
        for (int row = 0; row < 6; ++row) {
            target_decode(target.get(), tokens, anchor + row, 1, true);
            save_row(model, target.get(), 0, false, point + "/single-r" + std::to_string(row), tokens[depth + row], loop);
        }
        // Restore and repeat n=1 to expose checkpoint or graph-order effects before testing batch width.
        base_target.restore(target.get());
        for (int row = 0; row < 6; ++row) {
            target_decode(target.get(), tokens, anchor + row, 1, true);
            save_row(model, target.get(), 0, false, point + "/repeat-r" + std::to_string(row), tokens[depth + row], loop);
        }
        for (int k : {3, 4, 5}) {
            base_target.restore(target.get());
            target_decode(target.get(), tokens, anchor, k + 1, true);
            for (int row = 0; row <= k; ++row) {
                const int source_row = std::getenv("VERIFY_SWAP_ROWS") && row == 0 ? 1 : row;
                save_row(model, target.get(), source_row, false, point + "/verify-k" + std::to_string(k) + "-r" + std::to_string(row), tokens[depth + row], loop);
            }
        }
        std::vector<float> h = carry;
        for (int row = 0; row < 5; ++row) {
            draft_decode(draft.get(), tokens, anchor + row, 1, h, ne, true);
            save_row(model, draft.get(), 0, true, point + "/draft-r" + std::to_string(row), tokens[depth + row], loop);
            h = hidden(draft.get(), 0, ne);
        }
        base_target.restore(target.get());
        base_draft.restore(draft.get());
        std::printf("M: completed %s depth=%d elapsed_s=%.3f\n", data.at("name").get<std::string>().c_str(), depth,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        std::fflush(stdout);
    }
    llama_set_sampler(target.get(), 0, nullptr);
    llama_set_sampler(draft.get(), 0, nullptr);
    llama_sampler_free(draft_backend);
}

int main(int argc, char ** argv) {
    try {
        require(argc >= 5, "usage: probe prepare MODEL HISTORY OUTPUT | probe run MODEL TOKENS OUT cpu|tensor|layer q8_0|f16 SUBSET|- [RS]");
        const std::string mode = argv[1];
        const bool gpu = mode == "run" && argc >= 8 && std::string(argv[5]) != "cpu";
#ifndef VERIFY_PARENT_GPU
        require(!gpu, "CPU binary refuses GPU mode");
#endif
        const char * visible = std::getenv("CUDA_VISIBLE_DEVICES");
        require(visible && (gpu ? *visible != '\0' : *visible == '\0'), "device visibility guard failed");
        if (gpu) require(std::getenv("RUN_PARENT_GPU") && std::string(std::getenv("RUN_PARENT_GPU")) == "1", "parent flag missing");
        llama_log_set([](ggml_log_level level, const char * message, void *) {
            if (level >= GGML_LOG_LEVEL_WARN) std::fputs(message, stderr);
        }, nullptr);
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.vocab_only = mode == "prepare";
        mp.load_mtp = !mp.vocab_only;
        mp.n_gpu_layers = gpu ? -1 : 0;
        std::vector<llama_token> subset;
        if (argc >= 8 && std::string(argv[7]) != "-") {
            subset = read_json(argv[7]).get<llama_tokens>();
            mp.draft_vocab_ids = subset.data();
            mp.n_draft_vocab = subset.size();
        }
        ggml_backend_dev_t devices[3] = {nullptr, nullptr, nullptr};
        float split[2] = {50, 50};
        if (gpu) {
            int found = 0;
            for (size_t i = 0; i < ggml_backend_dev_count() && found < 2; ++i) {
                auto dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) devices[found++] = dev;
            }
            require(found == 2, "need two visible GPUs");
            require(std::string(argv[5]) == "tensor" || std::string(argv[5]) == "layer", "invalid split mode");
            mp.devices = devices;
            mp.tensor_split = split;
            mp.split_mode = std::string(argv[5]) == "tensor" ? LLAMA_SPLIT_MODE_TENSOR : LLAMA_SPLIT_MODE_LAYER;
        }
        llama_model_ptr model(llama_model_load_from_file(argv[2], mp));
        require(model != nullptr, "model load failed");
        if (mode == "prepare") {
            prepare(model.get(), argv[3], argv[4]);
        } else {
            require(mode == "run" && argc >= 8, "invalid run arguments");
            const std::string kv = argv[6];
            require(kv == "q8_0" || kv == "f16", "invalid KV type");
            const int rs = argc > 8 && std::string(argv[8]) != "campaign" ? std::stoi(argv[8]) : 0;
            require(rs == 0 || (rs >= 3 && rs <= 5), "invalid recurrent snapshot count");
            const auto input = read_json(argv[3]);
            if (input.is_array()) {
                for (const auto & data : input) {
                    run(model.get(), data, std::string(argv[4]) + "/" + data.at("name").get<std::string>(),
                        kv == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16, rs);
                }
            } else {
                run(model.get(), input, argv[4], kv == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16, rs);
            }
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
