// Model-free contract tests of common_speculative: the draft prompt is built only for drafters that read it.

#include "common.h"
#include "speculative.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

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

int main() {
    CHECK(!common_speculative_uses_prompt(nullptr));

    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, true);
    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,  true);
    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, false);
    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_MOD,    false);
    test_ngram(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,  false);

    return 0;
}
