// common_sampler_clone/copy must give the destination its own candidates buffer

#include "llama.h"
#include "common.h"
#include "sampling.h"

#include <cstdio>
#include <vector>

static bool same_candidates(const llama_token_data_array * a, const llama_token_data_array * b) {
    if (a->size != b->size || a->selected != b->selected || a->sorted != b->sorted) {
        return false;
    }
    for (size_t i = 0; i < a->size; ++i) {
        if (a->data[i].id != b->data[i].id || a->data[i].logit != b->data[i].logit || a->data[i].p != b->data[i].p) {
            return false;
        }
    }
    return true;
}

static bool test_sampler(llama_model * model, llama_context * ctx, common_params_sampling sparams) {
    common_sampler * src = common_sampler_init(model, sparams);
    common_sampler * dup = nullptr;
    common_sampler * cpy = common_sampler_init(model, sparams);
    if (!src || !cpy) {
        fprintf(stderr, "failed to create sampler\n");
        return false;
    }

    common_sampler_sample(src, ctx, 0);

    dup = common_sampler_clone(src);
    common_sampler_copy(src, cpy);

    const auto * cand_src = common_sampler_get_candidates(src, false);
    const auto * cand_dup = common_sampler_get_candidates(dup, false);
    const auto * cand_cpy = common_sampler_get_candidates(cpy, false);

    bool ok = true;

    if (cand_dup->data == cand_src->data || cand_cpy->data == cand_src->data) {
        fprintf(stderr, "candidates alias the source's buffer\n");
        ok = false;
    } else {
        // a stale pointer into the freed source buffer would be caught here under ASAN
        std::vector<llama_token_data> expected(cand_src->data, cand_src->data + cand_src->size);
        llama_token_data_array ref = *cand_src;
        ref.data = expected.data();

        common_sampler_free(src);
        src = nullptr;

        if (!same_candidates(cand_dup, &ref)) {
            fprintf(stderr, "clone's candidates differ from the source's\n");
            ok = false;
        }
        if (!same_candidates(cand_cpy, &ref)) {
            fprintf(stderr, "copy's candidates differ from the source's\n");
            ok = false;
        }
    }

    common_sampler_free(src);
    common_sampler_free(dup);
    common_sampler_free(cpy);

    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model_ptr model { llama_model_load_from_file(argv[1], mparams) };
    if (!model) {
        fprintf(stderr, "failed to load model '%s'\n", argv[1]);
        return 1;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 64;

    llama_context_ptr ctx { llama_init_from_model(model.get(), cparams) };
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    llama_token bos = llama_vocab_bos(llama_model_get_vocab(model.get()));
    if (llama_decode(ctx.get(), llama_batch_get_one(&bos, 1)) != 0) {
        fprintf(stderr, "failed to decode\n");
        return 1;
    }

    common_params_sampling sparams;
    sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
    sparams.top_k    = 8;

    bool ok = test_sampler(model.get(), ctx.get(), sparams);

    // XTC advances cur_p.data past the start of the buffer
    sparams.samplers        = { COMMON_SAMPLER_TYPE_XTC };
    sparams.xtc_probability = 1.0f;
    sparams.xtc_threshold   = 0.0f;
    sparams.min_keep        = 1;

    ok = test_sampler(model.get(), ctx.get(), sparams) && ok;

    return ok ? 0 : 1;
}
