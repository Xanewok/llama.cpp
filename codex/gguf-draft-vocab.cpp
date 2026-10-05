#include "gguf.h"
#include "llama.h"
#include "llama-model.h"
#include "json.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

int main(int argc, char ** argv) {
    require(argc == 5, "usage: gguf-draft-vocab INPUT OUTPUT IDS.json {embed,test}");
    require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    require(std::string(argv[1]) != argv[2], "output must differ from input");
    require(!std::ifstream(argv[2]).good(), "output already exists");
    const bool test = std::string(argv[4]) == "test";
    require(test || std::string(argv[4]) == "embed", "invalid mode");
    std::ifstream json_file(argv[3]);
    const auto json = nlohmann::json::parse(json_file);
    require(json.is_array() && !json.empty(), "expected nonempty JSON array");
    std::vector<int32_t> ids;
    for (const auto & id : json) {
        require(id.is_number_integer() && id.get<int64_t>() >= 0 && id.get<int64_t>() <= INT32_MAX, "invalid token id");
        ids.push_back(id.get<int32_t>());
    }
    auto * meta = gguf_init_from_file(argv[1], {true, nullptr});
    require(meta, "cannot read GGUF");
    const size_t old_offset = gguf_get_data_offset(meta);
    std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
    const size_t payload = size_t(input.tellg()) - old_offset;
    const char * key = "general.draft_vocab_ids";
    auto write_model = [&]() {
        require(gguf_write_to_file(meta, argv[2], true), "cannot write GGUF header");
        std::ofstream output(argv[2], std::ios::binary | std::ios::app);
        if (test) {
            // The loader checks file bounds even with no_alloc; sparse zero weights suffice.
            output.close();
            FILE * file = std::fopen(argv[2], "rb+");
            require(file, "cannot reopen fixture");
            require(fseeko(file, gguf_get_meta_size(meta) + payload - 1, SEEK_SET) == 0 && std::fputc(0, file) != EOF, "cannot size fixture");
            require(std::fclose(file) == 0, "cannot close fixture");
        } else {
            input.seekg(old_offset);
            std::vector<char> buffer(4 * 1024 * 1024);
            while (input) {
                input.read(buffer.data(), buffer.size());
                output.write(buffer.data(), input.gcount());
            }
            require(input.eof() && output.good(), "GGUF payload copy failed");
        }
    };
    gguf_set_arr_data(meta, key, GGUF_TYPE_INT32, ids.data(), ids.size());
    write_model();
    if (test) {
        llama_log_set([](ggml_log_level level, const char * message, void *) {
            if (level >= GGML_LOG_LEVEL_WARN) { std::fputs(message, stderr); }
        }, nullptr);
        llama_backend_init();
        auto params = llama_model_default_params();
        params.load_mtp = true;
        params.no_alloc = true;
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        params.n_gpu_layers = 0;
        params.progress_callback = [](float, void *) { return true; };
        auto check = [&](bool valid) {
            auto * model = llama_model_load_from_file(argv[2], params);
            require(bool(model) == valid, "metadata loader validation failed");
            if (model) { llama_model_free(model); }
        };
        auto * model = llama_model_load_from_file(argv[2], params);
        require(model && model->draft_vocab_ids == ids && model->draft_output->ne[1] == int64_t(ids.size()), "GGUF mapping not loaded");
        const int32_t nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
        llama_model_free(model);
        for (const auto & bad : std::vector<std::vector<int32_t>>{{}, {1, 1}, {-1}, {nv}, std::vector<int32_t>(nv + 1, 0)}) {
            gguf_set_arr_data(meta, key, GGUF_TYPE_INT32, bad.data(), bad.size());
            write_model();
            check(false);
        }
        const uint8_t byte = 1;
        gguf_set_arr_data(meta, key, GGUF_TYPE_UINT8, &byte, 1);
        write_model();
        check(false);
        gguf_set_val_i32(meta, key, 1);
        write_model();
        check(false);
        params.load_mtp = false;
        check(true);
        params.load_mtp = true;
        params.draft_vocab_ids = ids.data();
        params.n_draft_vocab = ids.size();
        check(true);
        llama_backend_free();
        std::puts("M: GGUF mapping load, malformed type/count/id rejection, target-only ignore and explicit override PASS");
    }
    gguf_free(meta);
}
