#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool decode(llama_context * ctx, int n_seq, int count, int pos) {
    llama_batch batch = llama_batch_init(count * n_seq, 0, 1);
    for (int seq = 0; seq < n_seq; ++seq) {
        for (int i = 0; i < count; ++i) {
            const int idx = batch.n_tokens++;
            batch.token[idx] = 100 + pos + i + seq;
            batch.pos[idx] = pos + i;
            batch.n_seq_id[idx] = 1;
            batch.seq_id[idx][0] = seq;
            batch.logits[idx] = i == count - 1;
        }
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 2) return 2;
    const bool meta = argc < 3 || atoi(argv[2]) != 0;
    llama_backend_init();
    std::vector<ggml_backend_dev_t> devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) { devs.push_back(d); devs.push_back(d); break; }
    }
    devs.push_back(nullptr);
    auto mp = llama_model_default_params();
    if (meta) {
        mp.devices = devs.data();
        mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        mp.n_gpu_layers = 999;
    } else {
        mp.n_gpu_layers = 0;
    }
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    for (int n_seq : {1, 2}) {
        auto cp = llama_context_default_params();
        cp.n_ctx = 4096; cp.n_batch = 512; cp.n_ubatch = 512;
        cp.n_seq_max = n_seq; cp.n_rs_seq = 5;
        cp.n_threads = 4; cp.n_threads_batch = 4;
        auto * ctx = llama_init_from_model(model, cp);
        if (!ctx) return 4;
        fprintf(stderr, "PROBE np=%d prefill=32\n", n_seq);
        if (!decode(ctx, n_seq, 32, 0)) return 5;
        int pos = 32;
        for (int n_tok : {1, 4, 6}) {
            fprintf(stderr, "PROBE np=%d tokens=%d\n", n_seq, n_tok);
            if (!decode(ctx, n_seq, n_tok, pos)) return 6;
            llama_synchronize(ctx);
            pos += n_tok;
        }
        llama_free(ctx);
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
