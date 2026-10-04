#include "server-common.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

int main(int argc, char ** argv) {
    {
        server_tokens text({1, 2, 3}, false);
        llama_tokens storage = {99};
        const auto & borrowed = text.get_text_tokens(storage);
        CHECK(&borrowed == &text.get_tokens());
        CHECK(borrowed == text.get_text_tokens());
        CHECK(storage == llama_tokens{99});
        text.push_back(4);
        CHECK(borrowed.size() == 4);
        CHECK(text.get_text_tokens(storage) == text.get_text_tokens());

        server_tokens media({1, LLAMA_TOKEN_NULL, 2, LLAMA_TOKEN_NULL, 3}, true);
        CHECK(&media.get_text_tokens(storage) == &storage);
        CHECK(storage == media.get_text_tokens());
        media.keep_first(2);
        CHECK(media.get_text_tokens(storage) == llama_tokens{1});
        media.keep_first(0);
        CHECK(media.get_text_tokens(storage).empty());
    }

    if (argc > 1 && std::string(argv[1]) == "--bench") {
        for (size_t depth : {8192, 90112}) {
            server_tokens text(llama_tokens(depth, 42), false);
            llama_tokens storage;
            volatile size_t count = 0;
            for (bool borrow : {false, true}) {
                const auto start = std::chrono::steady_clock::now();
                for (int repeat = 0; repeat < 1000; ++repeat) {
                    count = borrow ? text.get_text_tokens(storage).size() : text.get_text_tokens().size();
                }
                const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
                CHECK(count == depth);
                printf("depth %zu, %s ns/call M: %.1f\n", depth, borrow ? "borrow" : "copy", ns / 1000.0);
            }
        }
    }

    puts("text prompt borrowing and media filtering pass");
}
