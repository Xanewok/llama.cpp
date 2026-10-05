#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>

static bool census(ggml_tensor * t, bool ask, void *) {
    if (!ask) { return true; }
    std::printf("node\t%p\t%s\t%s\t%p\t%p\t%s\t%lld\t%lld\t%lld\t%lld\n", (void *) t, t->name, ggml_op_name(t->op),
                (void *) t->src[0], (void *) t->src[1], t->src[0] ? ggml_type_name(t->src[0]->type) : "-",
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    return false;
}

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) { return 2; }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mtp = true;
    mp.use_extra_bufts = false;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { return 3; }
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = cp.n_ubatch = 8;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.cb_eval = census;
    if (argc == 4) { cp.abort_callback = [](void *) { return true; }; }
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) { return 4; }
    const int width = std::atoi(argv[2]);
    auto batch = llama_batch_init(width, 0, 1);
    batch.n_tokens = width;
    for (int i = 0; i < width; ++i) {
        batch.token[i] = 100 + i;
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return argc == 4 && rc == 2 ? 0 : rc;
}
