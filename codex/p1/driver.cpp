// Greedy decode of two sequences in one batch stream, then one raw-embedding batch; prints the tokens. argv: model meta(0|1) n_gen
// (copy of patches/upstream/llama.cpp/89-embd-rows-on-device/evidence/mac/driver.cpp)
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { return 2; }
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    std::vector<ggml_backend_dev_t> devs;
    if (atoi(argv[2])) {
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        devs = { cpu, cpu, nullptr };
        mp.devices = devs.data();
        mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        mp.n_gpu_layers = -1;
    }
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 3; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 256; cp.n_ubatch = 256; cp.n_seq_max = 2;
    cp.n_threads = cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 4; }

    const char * prompts[2] = { "The quick brown fox jumps over the lazy dog. Then", "Write one sentence about the sea:" };
    std::vector<llama_token> toks[2];
    for (int s = 0; s < 2; ++s) {
        std::string p = prompts[s];
        toks[s].resize(p.size() + 8);
        int n = llama_tokenize(vocab, p.data(), (int) p.size(), toks[s].data(), (int) toks[s].size(), true, true);
        if (n < 0) { return 5; }
        toks[s].resize(n);
    }

    llama_batch b = llama_batch_init(256, 0, 1);
    int pos[2] = { 0, 0 };
    auto add = [&](llama_token t, int s, bool out) {
        int i = b.n_tokens++;
        b.token[i] = t; b.pos[i] = pos[s]++; b.n_seq_id[i] = 1; b.seq_id[i][0] = s; b.logits[i] = out;
    };
    int last_idx[2] = { -1, -1 };
    for (int s = 0; s < 2; ++s) {
        for (size_t i = 0; i < toks[s].size(); ++i) {
            const bool out = i + 1 == toks[s].size();
            if (out) { last_idx[s] = b.n_tokens; }
            add(toks[s][i], s, out);
        }
    }
    const int n_gen = atoi(argv[3]);
    std::vector<llama_token> gen[2];
    for (int step = 0; step <= n_gen; ++step) {
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 6; }
        llama_token next[2];
        for (int s = 0; s < 2; ++s) {
            const float * lg = llama_get_logits_ith(ctx, last_idx[s]);
            next[s] = (llama_token) (std::max_element(lg, lg + llama_vocab_n_tokens(vocab)) - lg);
            gen[s].push_back(next[s]);
        }
        b.n_tokens = 0;
        for (int s = 0; s < 2; ++s) { last_idx[s] = b.n_tokens; add(next[s], s, true); }
    }
    for (int s = 0; s < 2; ++s) {
        printf("seq%d:", s);
        for (auto t : gen[s]) { printf(" %d", t); }
        printf("\n");
    }
    llama_batch_free(b);

    // raw-embedding batch (ubatch.token == nullptr): the other select branch; the gathered-rows input stays unused
    llama_memory_clear(llama_get_memory(ctx), true);
    const int n_embd = llama_model_n_embd_inp(model), n_e = 6;
    llama_batch be = llama_batch_init(n_e, n_embd, 1);
    uint32_t lcg = 12345;
    for (int i = 0; i < n_e*n_embd; ++i) {
        lcg = lcg*1664525u + 1013904223u;
        be.embd[i] = ((lcg >> 8) & 0xffff)/65536.0f - 0.5f;
    }
    for (int i = 0; i < n_e; ++i) {
        be.pos[i] = i; be.n_seq_id[i] = 1; be.seq_id[i][0] = 0; be.logits[i] = i + 1 == n_e;
    }
    be.n_tokens = n_e;
    if (llama_decode(ctx, be) != 0) { fprintf(stderr, "embd decode failed\n"); return 7; }
    const float * lg = llama_get_logits_ith(ctx, n_e - 1);
    printf("embd: %d\n", (int) (std::max_element(lg, lg + llama_vocab_n_tokens(vocab)) - lg));
    llama_batch_free(be);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
