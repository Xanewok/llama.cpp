#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include <algorithm>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdio>

int main(int argc, char ** argv) {
    try {
        if (argc != 3) { throw std::runtime_error("usage: prepare-cpu-target SOURCE OUTPUT"); }
        ggml_context * tensors = nullptr;
        gguf_context_ptr metadata(gguf_init_from_file(argv[1], {true, &tensors}));
        ggml_context_ptr context(tensors);
        if (!metadata || gguf_find_tensor(metadata.get(), "output.weight") >= 0) { throw std::runtime_error("fixture must have a tied head"); }
        const int64_t embed_index = gguf_find_tensor(metadata.get(), "token_embd.weight");
        auto * embed = ggml_get_tensor(context.get(), "token_embd.weight");
        if (embed_index < 0 || !embed) { throw std::runtime_error("missing embedding tensor"); }
        const size_t input_offset = gguf_get_data_offset(metadata.get());
        const int64_t input_count = gguf_get_n_tensors(metadata.get());
        ggml_tensor head = *embed;
        ggml_set_name(&head, "output.weight");
        gguf_add_tensor(metadata.get(), &head);
        if (!gguf_write_to_file(metadata.get(), argv[2], true)) { throw std::runtime_error("metadata write failed"); }
        const size_t output_offset = gguf_get_meta_size(metadata.get());
        std::ifstream in(argv[1], std::ios::binary);
        std::fstream out(argv[2], std::ios::in | std::ios::out | std::ios::binary);
        std::vector<char> block(8 * 1024 * 1024);
        for (int64_t i = 0; i <= input_count; ++i) {
            const int64_t src = i == input_count ? embed_index : i;
            auto * tensor = ggml_get_tensor(context.get(), gguf_get_tensor_name(metadata.get(), src));
            size_t left = ggml_nbytes(tensor);
            in.seekg(input_offset + gguf_get_tensor_offset(metadata.get(), src));
            out.seekp(output_offset + gguf_get_tensor_offset(metadata.get(), i));
            while (left) {
                const size_t count = std::min(left, block.size());
                in.read(block.data(), count);
                out.write(block.data(), count);
                if (!in || !out) { throw std::runtime_error("payload copy failed"); }
                left -= count;
            }
        }
        puts("M: copied original quantized tensor bytes; output.weight duplicates token_embd.weight for the TP test fixture");
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
