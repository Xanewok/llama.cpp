#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        fprintf(stderr, "s10-kld: %s\n", message);
        std::exit(8);
    }
}

static int32_t read_i32(std::ifstream & input) {
    int32_t value;
    input.read((char *) &value, sizeof(value));
    require(bool(input), "truncated input");
    return value;
}

static void add_token(llama_batch & batch, llama_token token, llama_pos pos, bool logits) {
    const int32_t i = batch.n_tokens++;
    batch.token[i] = token;
    batch.pos[i] = pos;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = logits;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 8;
    }
    const char * input_path = std::getenv("S10_KLD_INPUT");
    const char * output_path = std::getenv("S10_KLD_OUT");
    require(input_path && output_path, "input and output are required");
    require(params.n_ctx > 0 && params.n_ctx <= 102400, "invalid explicit context");
    require(params.n_parallel == 1 && params.n_batch >= 1 && params.n_batch <= 2048, "invalid batch shape");
    require(params.n_ubatch >= 1 && params.n_ubatch <= 512, "invalid prefill ubatch");
    require(params.cpuparams.n_threads > 0 && params.cpuparams.n_threads <= 8, "invalid thread count");
    std::ifstream input(input_path, std::ios::binary);
    char magic[4];
    input.read(magic, 4);
    require(bool(input) && std::memcmp(magic, "KLDW", 4) == 0, "bad input magic");
    const int32_t count = read_i32(input);
    require(count >= 1 && count <= 3, "invalid window count");
    struct window {
        int32_t depth;
        int32_t rows;
        std::vector<llama_token> tokens;
    };
    std::vector<window> windows;
    for (int i = 0; i < count; ++i) {
        window w;
        w.depth = read_i32(input);
        w.rows = read_i32(input);
        require(w.depth >= 1 && w.rows >= 1 && w.rows <= 64 && w.depth <= params.n_ctx - w.rows, "invalid window bounds");
        w.tokens.resize(w.depth + w.rows);
        input.read((char *) w.tokens.data(), w.tokens.size()*sizeof(llama_token));
        require(bool(input), "truncated tokens");
        windows.push_back(std::move(w));
    }
    require(input.peek() == EOF, "trailing input bytes");

    params.load_mtp = true;
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max = 5;
    params.speculative.draft.n_min = 0;
    params.speculative.draft.p_min = 0;
    llama_backend_init();
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    require(model && ctx, "model/context init failed");
    require(llama_n_ctx(ctx) == uint32_t(params.n_ctx) && llama_n_ubatch(ctx) == uint32_t(params.n_ubatch), "effective context or ubatch mismatch");
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    require(n_vocab > 0 && n_vocab <= 300000, "invalid vocabulary size");
    llama_set_embeddings_nextn(ctx, true, false);
    llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
    for (const auto & w : windows) {
        for (llama_token token : w.tokens) {
            require(token >= 0 && token < n_vocab, "invalid token id");
        }
        // Clear metadata and bytes to exclude request-history state.
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int32_t start = 0; start < w.depth;) {
            const int32_t end = std::min<int32_t>(w.depth, start + params.n_ubatch);
            batch.n_tokens = 0;
            for (int32_t pos = start; pos < end; ++pos) {
                add_token(batch, w.tokens[pos], pos, false);
            }
            require(llama_decode(ctx, batch) == 0, "prefill decode failed");
            start = end;
        }
        const std::string path = std::string(output_path) + "/depth-" + std::to_string(w.depth) + ".logits";
        std::ofstream output(path + ".tmp", std::ios::binary);
        require(bool(output), "cannot open output");
        const int32_t header[] = { w.depth, w.rows, n_vocab };
        output.write("KLDL", 4);
        output.write((const char *) header, sizeof(header));
        for (int32_t pos = w.depth; pos < w.depth + w.rows; ++pos) {
            batch.n_tokens = 0;
            // Each scored decode has one input token, regardless of the prefill ubatch size.
            add_token(batch, w.tokens[pos], pos, true);
            require(llama_decode(ctx, batch) == 0, "one-row decode failed");
            const float * logits = llama_get_logits_ith(ctx, 0);
            require(logits != nullptr, "missing logits");
            for (int32_t j = 0; j < n_vocab; ++j) {
                require(std::isfinite(logits[j]), "non-finite logits");
            }
            output.write((const char *) logits, size_t(n_vocab)*sizeof(float));
        }
        output.close();
        require(bool(output), "output write failed");
        require(std::rename((path + ".tmp").c_str(), path.c_str()) == 0, "output rename failed");
        fprintf(stderr, "s10-kld: depth=%d rows=%d vocab=%d ctx=%u ub=%u scored_batch=1\n", w.depth, w.rows, n_vocab, llama_n_ctx(ctx), llama_n_ubatch(ctx));
    }
    llama_batch_free(batch);
    return 0;
}
