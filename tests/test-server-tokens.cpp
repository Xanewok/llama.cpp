// server_tokens::get_common_prefix_resync: a span that spells the same text with different tokens
// (non-canonical split) keeps matching the cache; anything else keeps the plain common prefix.

#include "llama.h"
#include "common.h"
#include "server-common.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

static llama_tokens tok(const llama_vocab * vocab, const std::string & text, bool parse_special = false) {
    return common_tokenize(vocab, text, false, parse_special);
}

static llama_tokens cat(std::initializer_list<llama_tokens> parts) {
    llama_tokens res;
    for (const auto & p : parts) {
        res.insert(res.end(), p.begin(), p.end());
    }
    return res;
}

// split `word` in two so that tokenizing the halves separately differs from tokenizing it whole
static llama_tokens non_canonical(const llama_vocab * vocab, const std::string & word) {
    const auto canon = tok(vocab, word);
    for (size_t i = 1; i < word.size(); ++i) {
        const auto split = cat({tok(vocab, word.substr(0, i)), tok(vocab, word.substr(i))});
        if (split != canon) {
            return split;
        }
    }
    return {};
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s vocab-file.gguf\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    CHECK(model);
    const llama_vocab * vocab = llama_model_get_vocab(model);

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

    if (argc > 2 && std::string(argv[2]) == "--bench") {
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

    const auto head = tok(vocab, "The quick brown fox");
    const auto tail = tok(vocab, " is here.");
    const auto more = tok(vocab, " And then some more text follows");

    const std::string word = " basket";
    const auto canon = tok(vocab, word);
    const auto split = non_canonical(vocab, word);
    CHECK(!split.empty());
    CHECK(split.size() > canon.size());
    printf("'%s': canonical %zu token(s), non-canonical %zu token(s)\n", word.c_str(), canon.size(), split.size());

    // cache holds what was sampled, the new prompt holds what re-tokenizing the same text gives
    const server_tokens cached(cat({head, split, tail}), false);
    const auto expect = cat({head, split, tail, more});

    {
        server_tokens fresh(cat({head, canon, tail, more}), false);
        CHECK(cached.get_common_prefix(fresh) == head.size());

        server_tokens out(llama_tokens{}, false);
        CHECK(cached.get_common_prefix_resync(fresh, vocab, &out).first == cached.size());
        CHECK(out.get_tokens() == expect);
        CHECK(fresh.get_tokens() == cat({head, canon, tail, more})); // the input is left alone
        CHECK(cached.get_common_prefix(out) == cached.size());

        // b_out may alias b
        CHECK(cached.get_common_prefix_resync(fresh, vocab, &fresh).first == cached.size());
        CHECK(fresh.get_tokens() == expect);

        // no b_out: same answer, nothing to rewrite
        server_tokens fresh2(cat({head, canon, tail, more}), false);
        CHECK(cached.get_common_prefix_resync(fresh2, vocab).first == cached.size());
    }

    // shifts tell where positions of b move to
    {
        server_tokens fresh(cat({head, canon, tail, more}), false);
        std::vector<std::pair<size_t, int64_t>> shifts;
        const auto n = cached.get_common_prefix_resync(fresh, vocab, nullptr, &shifts);
        CHECK(n.first == cached.size());
        CHECK(n.second == cached.size() - split.size() + canon.size()); // the same prefix in b's own tokens
        CHECK(shifts.size() == 1);
        CHECK(shifts[0].first == head.size() + canon.size());
        CHECK(shifts[0].second == int64_t(split.size()) - int64_t(canon.size()));

        const server_tokens cached2(cat({head, split, tail, split, tail}), false);
        server_tokens fresh2(cat({head, canon, tail, canon, tail}), false);
        shifts.clear();
        cached2.get_common_prefix_resync(fresh2, vocab, nullptr, &shifts).first;
        CHECK(shifts.size() == 2);
        CHECK(shifts[1].second == 2*shifts[0].second);
    }

    // the other way around: the cache holds the canonical form
    {
        const server_tokens cached2(cat({head, canon, tail}), false);
        server_tokens fresh(cat({head, split, tail, more}), false);
        CHECK(cached2.get_common_prefix_resync(fresh, vocab, &fresh).first == cached2.size());
        CHECK(fresh.get_tokens() == cat({head, canon, tail, more}));
    }

    // two separate spans
    {
        const server_tokens cached2(cat({head, split, tail, split, tail}), false);
        server_tokens fresh(cat({head, canon, tail, canon, tail, more}), false);
        CHECK(cached2.get_common_prefix_resync(fresh, vocab, &fresh).first == cached2.size());
        CHECK(fresh.get_tokens() == cat({head, split, tail, split, tail, more}));
    }

    // negative control: different text keeps the plain common prefix and leaves b alone
    {
        const auto other = tok(vocab, " baskex");
        server_tokens fresh(cat({head, other, tail, more}), false);
        const auto before = fresh.get_tokens();
        server_tokens out(llama_tokens{}, false);
        CHECK(cached.get_common_prefix_resync(fresh, vocab, &out).first == head.size());
        CHECK(out.empty());
        CHECK(fresh.get_tokens() == before);
    }

    // negative control: text that diverges after the span resyncs up to the divergence only
    {
        server_tokens fresh(cat({head, canon, tok(vocab, " is there."), more}), false);
        const size_t n = cached.get_common_prefix_resync(fresh, vocab, &fresh).first;
        CHECK(n >= head.size() + split.size());
        CHECK(n < cached.size());
        CHECK(cached.get_common_prefix(fresh) == n);
    }

    // negative control: a span longer than the window is not resynced
    {
        const std::string long_word = " internationalization";
        llama_tokens chars;
        for (char c : long_word) {
            const auto t = tok(vocab, std::string(1, c));
            CHECK(t.size() == 1);
            chars.push_back(t[0]);
        }
        const auto canon_long = tok(vocab, long_word);
        CHECK(canon_long != chars);
        const server_tokens cached2(cat({head, chars, tail}), false);
        server_tokens fresh(cat({head, canon_long, tail}), false);
        CHECK(cached2.get_common_prefix_resync(fresh, vocab, &fresh).first == cached2.get_common_prefix(fresh));
        CHECK(fresh.get_tokens() == cat({head, canon_long, tail}));
    }

    // negative control: a control token is not interchangeable with its spelling
    {
        const std::string spelled = "<|im_end|>";
        const auto ctrl   = tok(vocab, spelled, true);
        const auto literal = tok(vocab, spelled, false);
        CHECK(ctrl.size() == 1 && literal.size() > 1);
        const server_tokens cached2(cat({head, ctrl, tail}), false);
        server_tokens fresh(cat({head, literal, tail}), false);
        CHECK(cached2.get_common_prefix_resync(fresh, vocab, &fresh).first == head.size());
        CHECK(fresh.get_tokens() == cat({head, literal, tail}));
    }

    // ids outside the vocab (unvalidated user input) are just a mismatch
    for (llama_token bad : {llama_token(-7), llama_token(llama_vocab_n_tokens(vocab)), llama_token(0x7fffffff)}) {
        server_tokens fresh(cat({head, {bad}, tail, more}), false);
        CHECK(cached.get_common_prefix_resync(fresh, vocab, &fresh).first == head.size());
        const server_tokens cached2(cat({head, {bad}, tail}), false);
        server_tokens fresh2(cat({head, canon, tail}), false);
        CHECK(cached2.get_common_prefix_resync(fresh2, vocab, &fresh2).first == head.size());
    }

    // no vocab: plain behaviour
    {
        server_tokens fresh(cat({head, canon, tail, more}), false);
        CHECK(cached.get_common_prefix_resync(fresh, nullptr, &fresh).first == head.size());
    }

    llama_model_free(model);
    llama_backend_free();
    printf("OK\n");
    return 0;
}
