#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        return 2;
    }
    llama_backend_init();
    auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    const bool tp = atoi(argv[2]) != 0;
    ggml_backend_dev_t devices[] = { cpu, tp ? cpu : nullptr, nullptr };
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.split_mode = tp ? LLAMA_SPLIT_MODE_TENSOR : LLAMA_SPLIT_MODE_LAYER;
    mp.n_gpu_layers = tp ? 99 : 0;
    mp.load_mtp = true;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return 3;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 32;
    cp.n_ubatch = 8;
    cp.n_threads = tp ? 4 : 8;
    cp.n_threads_batch = cp.n_threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto * target = llama_init_from_model(model, cp);
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    auto * draft = llama_init_from_model(model, cp);
    if (!target || !draft) {
        return 4;
    }
    llama_set_embeddings_nextn(target, true, false);
    llama_set_embeddings_nextn(draft, true, true);
    FILE * out = fopen(argv[3], "wb");
    if (!out) {
        return 5;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_embd = llama_model_n_embd_out(model);
    auto batch = llama_batch_init(32, 0, 1);
    auto dbatch = llama_batch_init(32, n_embd, 1);
    std::vector<llama_token> dtokens(32);
    auto fill = [&](llama_batch & b, int pos, int n, bool output) {
        b.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            if (b.token) {
                b.token[i] = 100 + (pos + i) % 80 + (argc == 5 && pos + i == 4);
            }
            b.pos[i] = pos + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = output;
        }
    };
    auto save = [&](llama_context * ctx, int n) {
        for (int i = 0; i < n; ++i) {
            const float * logits = llama_get_logits_ith(ctx, i);
            if (!logits || fwrite(logits, sizeof(float), n_vocab, out) != size_t(n_vocab)) {
                exit(6);
            }
        }
    };
    int pos = 0;
    for (int round = 0; round < 4; ++round) {
        fill(batch, pos, 4, true);
        if (llama_decode(target, batch)) {
            return 7;
        }
        save(target, 4);
        fill(dbatch, pos, 4, false);
        for (int i = 0; i < 4; ++i) {
            dtokens[i] = batch.token[i];
            const float * h = llama_get_embeddings_nextn_ith(target, i);
            for (int j = 0; j < n_embd; ++j) {
                dbatch.embd[i*n_embd + j] = h[j];
            }
        }
        dbatch.token = dtokens.data();
        if (llama_decode(draft, dbatch)) {
            return 8;
        }
        pos += 4;
        for (int step = 0; step < 3; ++step) {
            fill(dbatch, pos + step, 1, true);
            dtokens[0] = 150 + round*3 + step;
            if (llama_decode(draft, dbatch)) {
                return 9;
            }
            save(draft, 1);
        }
        llama_memory_seq_rm(llama_get_memory(draft), 0, pos, -1);
    }
    dbatch.token = nullptr;
    llama_batch_free(dbatch);
    llama_batch_free(batch);
    fclose(out);
    llama_free(draft);
    llama_free(target);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
