#include "arg.h"
#include "common.h"
#include "json.h"
#include "llama-cpp.h"
#include "llama-ext.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace fs = std::filesystem;
using json = common_json;

static json read_json(const fs::path & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path.string());
    }
    return json::parse(std::string(std::istreambuf_iterator<char>(in), {}));
}

static void write_json(const fs::path & path, const json & value) {
    std::ofstream out(path);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << value.dump(2) << '\n';
}

static std::ofstream output(const fs::path & path) {
    if (fs::exists(path)) {
        throw std::runtime_error("output already exists: " + path.string());
    }
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    return out;
}

template <typename T>
static void write_row(std::ofstream & out, const std::vector<T> & row) {
    out.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(T));
}

static llama_tokens read_tokens(const json & ids, int32_t n_vocab) {
    if (!ids.is_array()) {
        throw std::runtime_error("token ids must be a JSON array");
    }
    llama_tokens result;
    for (const auto & id : ids) {
        if (!id.is_number_integer()) {
            throw std::runtime_error("token id must be an integer");
        }
        const auto val = id.get<int64_t>();
        if (val < 0 || val >= n_vocab) {
            throw std::runtime_error("token id outside vocabulary");
        }
        result.push_back(val);
    }
    return result;
}

static void decode(llama_context * ctx, common_batch & batch) {
    const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    if (rc != 0) {
        throw std::runtime_error("decode failed: " + std::to_string(rc));
    }
    llama_synchronize(ctx);
}

static std::vector<llama_token> top_ids(const float * logits, int32_t n_vocab, int width) {
    std::vector<llama_token> ids(n_vocab);
    std::iota(ids.begin(), ids.end(), 0);
    width = std::min(width, n_vocab);
    std::partial_sort(ids.begin(), ids.begin() + width, ids.end(), [logits](llama_token a, llama_token b) {
        return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
    });
    ids.resize(width);
    return ids;
}

static void write_candidates(common_sampler * sampler, int width, std::ofstream & ids_out, std::ofstream & probs_out) {
    const auto * row = common_sampler_get_candidates(sampler, false);
    if (row->size > (size_t) width) {
        throw std::runtime_error("candidate row exceeds output width");
    }
    std::vector<llama_token> ids(width, -1);
    std::vector<float> probs(width, 0);
    for (size_t i = 0; i < row->size; ++i) {
        ids[i] = row->data[i].id;
        probs[i] = row->data[i].p;
    }
    write_row(ids_out, ids);
    write_row(probs_out, probs);
}

static void stage_a(common_params params, const fs::path & streams_path, const fs::path & out_dir) {
    // Stage A needs full target logits even when the served sampler uses backend candidates.
    params.sampling.backend_sampling = false;
    params.n_outputs_max = params.n_ubatch;
    params.n_outputs_max_per_seq = params.n_ubatch;
    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx = init->context();
    if (!model || !ctx || llama_model_n_layer_nextn(model) == 0) {
        throw std::runtime_error("target must have an MTP head");
    }
    const int n_embd = llama_model_n_embd_out(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int pk = params.sampling.top_k;
    if (pk <= 0 || pk > n_vocab) {
        throw std::runtime_error("Stage A requires a positive --top-k no larger than the vocabulary");
    }
    if (!params.sampling.grammar.empty() || params.sampling.mirostat || params.sampling.xtc_probability > 0 ||
        params.sampling.adaptive_target >= 0 || !params.sampling.reasoning_budget_start.empty()) {
        throw std::runtime_error("stateful stochastic samplers and grammars are not supported");
    }
    llama_set_embeddings_nextn(ctx, true, false);
    const auto streams = read_json(streams_path);
    if (!streams.is_array() || streams.empty()) {
        throw std::runtime_error("streams file must be a nonempty JSON list of {prompt, continuation}");
    }
    json header {
        {"format", "spec-tf-v1"}, {"stage", "A"}, {"byte_order", "little"}, {"hidden_dtype", "f32"},
        {"model", params.model.path}, {"n_embd", n_embd}, {"n_vocab", n_vocab}, {"p_width", pk},
        {"sampling", params.sampling.print()}, {"temperature", params.sampling.temp}, {"streams", json::array()},
    };
    for (size_t s = 0; s < streams.size(); ++s) {
        const auto & stream = streams.at(s);
        auto tokens = read_tokens(stream.at("prompt"), n_vocab);
        const int prompt = tokens.size();
        const auto continuation = read_tokens(stream.at("continuation"), n_vocab);
        tokens.insert(tokens.end(), continuation.begin(), continuation.end());
        if (prompt < 1 || tokens.size() > llama_n_ctx(ctx) || continuation.size() < 2) {
            throw std::runtime_error("stream needs a prompt, continuation, and sufficient explicit context");
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        common_sampler_ptr sampler(common_sampler_init(model, params.sampling));
        const fs::path dir = out_dir / std::to_string(s);
        fs::create_directories(dir);
        auto tok_out = output(dir / "tokens.bin");
        write_row(tok_out, tokens);
        auto hidden = output(dir / "hidden.bin");
        auto p_ids = output(dir / "p_ids.bin");
        auto p_probs = output(dir / "p_probs.bin");
        auto raw_ids = output(dir / "p_raw_ids.bin");
        auto raw_logits = output(dir / "p_raw_logits.bin");
        auto lse = output(dir / "p_logsumexp.bin");
        common_batch batch(ctx);
        const int chunk = std::min(llama_n_batch(ctx), llama_n_ubatch(ctx));
        for (int off = 0; off < (int) tokens.size(); off += chunk) {
            batch.clear();
            const int count = std::min(chunk, (int) tokens.size() - off);
            for (int i = 0; i < count; ++i) {
                batch.add(tokens[off + i], off + i, 0, off + i >= prompt - 1);
            }
            decode(ctx, batch);
            for (int i = 0; i < count; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx, i);
                if (!h) {
                    throw std::runtime_error("missing target hidden row");
                }
                hidden.write(reinterpret_cast<const char *>(h), n_embd * sizeof(float));
                common_sampler_accept(sampler.get(), tokens[off + i], off + i >= prompt);
                std::vector<llama_token> ids_raw(24, -1);
                std::vector<float> logits_raw(24, -INFINITY);
                float logsumexp = -INFINITY;
                if (off + i >= prompt - 1) {
                    const float * logits = llama_get_logits_ith(ctx, i);
                    const auto top = top_ids(logits, n_vocab, 24);
                    for (size_t j = 0; j < top.size(); ++j) {
                        ids_raw[j] = top[j];
                        logits_raw[j] = logits[top[j]];
                    }
                    double sum = 0;
                    for (int id = 0; id < n_vocab; ++id) {
                        sum += std::exp((double) logits[id] - logits_raw[0]);
                    }
                    logsumexp = logits_raw[0] + std::log(sum);
                    common_sampler_sample(sampler.get(), ctx, i);
                    write_candidates(sampler.get(), pk, p_ids, p_probs);
                } else {
                    write_row(p_ids, std::vector<llama_token>(pk, -1));
                    write_row(p_probs, std::vector<float>(pk, 0));
                }
                write_row(raw_ids, ids_raw);
                write_row(raw_logits, logits_raw);
                lse.write(reinterpret_cast<const char *>(&logsumexp), sizeof(float));
            }
        }
        header["streams"].push_back(json {
            {"id", stream.value("id", std::to_string(s))}, {"n_tokens", tokens.size()},
            {"n_prompt", prompt}, {"n_continuation", continuation.size()},
        });
        LOG_INF("stage A stream %zu complete\n", s);
    }
    write_json(out_dir / "header.json", header);
}

static void stage_b(common_params params, const fs::path & target_dir, const fs::path & out_dir, int hidden_offset, bool permute) {
    const auto header = read_json(target_dir / "header.json");
    if (header.at("format").get<std::string>() != "spec-tf-v1" || header.at("hidden_dtype").get<std::string>() != "f32") {
        throw std::runtime_error("unsupported target format");
    }
    const int n_embd = header.at("n_embd").get<int>();
    const int n_vocab = header.at("n_vocab").get<int>();
    if (n_embd <= 0 || n_vocab <= 0 || n_embd > 65536) {
        throw std::runtime_error("invalid target dimensions");
    }
    auto dparams = common_base_params_to_speculative(params);
    if (params.speculative.has_dft()) {
        params.model = dparams.model;
    }
    auto mparams = common_model_params_to_llama(dparams);
    mparams.draft_vocab_ids = params.speculative.draft.vocab_ids.data();
    mparams.n_draft_vocab = params.speculative.draft.vocab_ids.size();
    llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mparams));
    if (!model || llama_model_n_embd_out(model.get()) != n_embd ||
        llama_vocab_n_tokens(llama_model_get_vocab(model.get())) != n_vocab || llama_model_n_layer_nextn(model.get()) == 0) {
        throw std::runtime_error("draft model dimensions or MTP head do not match target");
    }
    auto cparams = common_context_params_to_llama(dparams);
    cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cparams.n_rs_seq = 0;
    cparams.ctx_other = nullptr;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
    if (!ctx || llama_get_ctx_other(ctx.get())) {
        throw std::runtime_error("Stage B requires an independent MTP context");
    }
    llama_set_embeddings_nextn(ctx.get(), true, true);
    const int heads = llama_model_n_layer_nextn(model.get());
    const int k = heads > 1 ? std::min(params.speculative.draft.n_max, heads) : params.speculative.draft.n_max;
    if (k < 1 || k > 64 || params.sampling.temp <= 0 || params.sampling.temp != header.at("temperature").get<float>()) {
        throw std::runtime_error("invalid draft count or target/draft temperature mismatch");
    }
    common_params_sampling sampling;
    sampling.top_k = 10;
    sampling.temp = params.sampling.temp;
    sampling.seed = 1729;
    sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };
    common_sampler_ptr sampler(common_sampler_init(model.get(), sampling));
    int32_t n_suppress = 0;
    const auto * suppress = llama_vocab_get_suppress_tokens(llama_model_get_vocab(model.get()), &n_suppress);
    std::vector<llama_token> suppress_ids;
    for (int i = 0; i < n_suppress; ++i) {
        suppress_ids.push_back(suppress[i]);
    }
    json result {
        {"suppress_ids", suppress_ids}, {"format", "spec-tf-v1"}, {"stage", "B"}, {"target_header", header}, {"model", params.model.path},
        {"k", k}, {"q_width", 128}, {"temperature", sampling.temp}, {"draft_vocab", params.speculative.draft.vocab_ids},
        {"hidden_offset", hidden_offset}, {"permuted", permute},
    };
    for (size_t s = 0; s < header.at("streams").size(); ++s) {
        const auto & info = header.at("streams").at(s);
        const int64_t n64 = info.at("n_tokens").get<int64_t>();
        const int64_t prompt64 = info.at("n_prompt").get<int64_t>();
        if (n64 > llama_n_ctx(ctx.get()) || prompt64 < 1 || prompt64 >= n64 - k) {
            throw std::runtime_error("stream exceeds context or has too few continuation tokens");
        }
        const int n = n64;
        const int prompt = prompt64;
        const fs::path in = target_dir / std::to_string(s);
        const fs::path dir = out_dir / std::to_string(s);
        fs::create_directories(dir);
        std::vector<llama_token> tokens(n);
        if (fs::file_size(in / "tokens.bin") != (size_t) n * sizeof(llama_token) ||
            fs::file_size(in / "hidden.bin") != (size_t) n * n_embd * sizeof(float)) {
            throw std::runtime_error("target binary size mismatch");
        }
        std::ifstream tok_in(in / "tokens.bin", std::ios::binary);
        tok_in.exceptions(std::ios::failbit | std::ios::badbit);
        tok_in.read(reinterpret_cast<char *>(tokens.data()), n * sizeof(llama_token));
        for (const auto id : tokens) {
            if (id < 0 || id >= n_vocab) {
                throw std::runtime_error("saved token id outside vocabulary");
            }
        }
        if (permute) {
            std::reverse(tokens.begin() + prompt, tokens.end());
        }
        std::ifstream hidden(in / "hidden.bin", std::ios::binary);
        hidden.exceptions(std::ios::failbit | std::ios::badbit);
        auto h_at = [&](int pos, float * dst) {
            const int row = pos + hidden_offset;
            if (row < 0) {
                std::fill_n(dst, n_embd, 0);
            } else {
                hidden.seekg((size_t) row * n_embd * sizeof(float));
                hidden.read(reinterpret_cast<char *>(dst), n_embd * sizeof(float));
            }
        };
        auto starts = output(dir / "starts.bin");
        auto q_ids = output(dir / "q_ids.bin");
        auto q_logits = output(dir / "q_logits.bin");
        auto sample_ids = output(dir / "q_sample_ids.bin");
        auto probs = output(dir / "q_probs.bin");
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        common_batch batch(ctx.get());
        const int chunk = std::min(llama_n_batch(ctx.get()), llama_n_ubatch(ctx.get()));
        std::vector<float> catch_h((size_t) chunk * n_embd);
        auto catch_up = [&](int off, int count) {
            batch.clear();
            for (int i = 0; i < count; ++i) {
                h_at(off + i, catch_h.data() + (size_t) i * n_embd);
                const int idx = batch.add(tokens[off + i], off + i, 0, false);
                batch.set_embd(idx, {catch_h.data() + (size_t) i * n_embd, 1, (size_t) n_embd});
            }
            for (int head = 0; head < heads; ++head) {
                if (heads > 1 && !llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, off, -1)) {
                    throw std::runtime_error("MTP context cannot remove catch-up rows");
                }
                llama_set_nextn_layer_offset(ctx.get(), head);
                decode(ctx.get(), batch);
            }
            llama_set_nextn_layer_offset(ctx.get(), 0);
        };
        for (int off = 0; off < prompt; off += chunk) {
            catch_up(off, std::min(chunk, prompt - off));
        }
        std::vector<float> chain_h((size_t) (k + 1) * n_embd);
        for (int t = prompt; t < n - 1; ++t) {
            write_row(starts, std::vector<int32_t>{t});
            h_at(t, chain_h.data());
            for (int j = 0; j < k; ++j) {
                if (t + j >= n - 1) {
                    write_row(q_ids, std::vector<llama_token>(128, -1));
                    write_row(q_logits, std::vector<float>(128, -INFINITY));
                    write_row(sample_ids, std::vector<llama_token>(10, -1));
                    write_row(probs, std::vector<float>(10, 0));
                    continue;
                }
                batch.clear();
                if (heads > 1) {
                    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, t, -1)) {
                        throw std::runtime_error("MTP context cannot remove draft rows");
                    }
                    llama_set_nextn_layer_offset(ctx.get(), j);
                }
                const int begin = heads > 1 ? 0 : j;
                for (int i = begin; i <= j; ++i) {
                    const int idx = batch.add(tokens[t + i], t + i, 0, i == j);
                    batch.set_embd(idx, {chain_h.data() + (size_t) i * n_embd, 1, (size_t) n_embd});
                }
                decode(ctx.get(), batch);
                const int last = batch.size() - 1;
                const float * logits = llama_get_logits_ith(ctx.get(), last);
                const auto top = top_ids(logits, n_vocab, 128);
                std::vector<llama_token> ids(128, -1);
                std::vector<float> vals(128, -INFINITY);
                for (size_t i = 0; i < top.size(); ++i) {
                    ids[i] = top[i];
                    vals[i] = logits[top[i]];
                }
                write_row(q_ids, ids);
                write_row(q_logits, vals);
                common_sampler_sample(sampler.get(), ctx.get(), last, true);
                write_candidates(sampler.get(), 10, sample_ids, probs);
                const float * h = llama_get_embeddings_nextn_ith(ctx.get(), last);
                if (!h) {
                    throw std::runtime_error("missing draft hidden row");
                }
                std::copy_n(h, n_embd, chain_h.data() + (size_t) (j + 1) * n_embd);
            }
            if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, t + 1, -1)) {
                throw std::runtime_error("MTP context cannot remove draft rows");
            }
            if (heads > 1) {
                llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, t, -1);
                catch_up(t, 1);
            }
        }
        LOG_INF("stage B stream %zu complete\n", s);
    }
    write_json(out_dir / "header.json", result);
}

int main(int argc, char ** argv) {
    try {
        std::string stage, streams, target, out;
        std::vector<std::string> controls;
        int hidden_offset = -1;
        bool permute = false;
        std::vector<char *> args {argv[0]};
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--tf-permute") {
                permute = true;
            } else if (arg == "--tf-stage" || arg == "--tf-streams" || arg == "--tf-target" || arg == "--tf-out" || arg == "--tf-hidden-offset" || arg == "--tf-control-vocab") {
                if (++i == argc) {
                    throw std::runtime_error("missing value for " + arg);
                }
                if (arg == "--tf-stage") { stage = argv[i]; }
                if (arg == "--tf-streams") { streams = argv[i]; }
                if (arg == "--tf-target") { target = argv[i]; }
                if (arg == "--tf-out") { out = argv[i]; }
                if (arg == "--tf-hidden-offset") { hidden_offset = std::stoi(argv[i]); }
                if (arg == "--tf-control-vocab") { controls.emplace_back(argv[i]); }
            } else {
                args.push_back(argv[i]);
            }
        }
        common_init();
        common_params params;
        if (!common_params_parse(args.size(), args.data(), params, LLAMA_EXAMPLE_SERVER)) {
            return 1;
        }
        const uint32_t endian = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian) != 1 || params.n_ctx <= 0 || params.n_parallel != 1 ||
            out.empty() || (stage != "A" && stage != "B" && stage != "AB") || hidden_offset > -1 || hidden_offset < -2) {
            throw std::runtime_error("use --tf-stage A|B|AB --tf-out DIR, explicit -c and -np 1 (little-endian only)");
        }
        fs::create_directories(out);
        if (fs::exists(fs::path(out) / "header.json")) {
            throw std::runtime_error("output is already complete");
        }
        llama_backend_init();
        llama_numa_init(params.numa);
        if (stage == "A") {
            stage_a(params, streams, out);
        } else if (stage == "B") {
            stage_b(params, target, out, hidden_offset, permute);
        } else {
            const fs::path dir(out);
            stage_a(params, streams, dir / "target");
            stage_b(params, dir / "target", dir / "full", hidden_offset, permute);
            for (size_t i = 0; i < controls.size(); ++i) {
                params.speculative.draft.vocab_ids = read_tokens(read_json(controls[i]),
                        read_json(dir / "target/header.json").at("n_vocab").get<int>());
                stage_b(params, dir / "target", dir / ("control-" + std::to_string(i)), hidden_offset, permute);
            }
        }
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        LOG_ERR("spec-tf: %s\n", e.what());
        return 1;
    }
}
