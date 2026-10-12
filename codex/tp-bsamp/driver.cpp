// Two sequences decoded in one batch stream; prints per step each sequence's token and its top-20 candidate ids (sorted).
// argv: model mode n_gen. mode: cpu (plain CPU, host argmax), meta (Meta(CPU,CPU) tensor split, host argmax),
// meta-bs (Meta(CPU,CPU), backend chain top-k(20) + greedy; token and candidates come from the backend sampler).
// Host modes print the host top-20 of the full logits row, so all three modes must print the same lines.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

static const int K = 20;

int main(int argc, char ** argv) {
    if (argc < 4) { return 2; }
    const std::string mode = argv[2];
    const bool meta = mode != "cpu";
    const bool bs   = mode == "meta-bs";
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    std::vector<ggml_backend_dev_t> devs;
    if (meta) {
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        devs = { cpu, cpu, nullptr };
        mp.devices = devs.data();
        mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        mp.n_gpu_layers = -1;
    }
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 3; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 256; cp.n_ubatch = 256; cp.n_seq_max = 2;
    cp.n_threads = cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 4; }

    llama_sampler * chains[2] = { nullptr, nullptr };
    if (bs) {
        for (int s = 0; s < 2; ++s) {
            chains[s] = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(chains[s], llama_sampler_init_top_k(K));
            llama_sampler_chain_add(chains[s], llama_sampler_init_greedy());
            if (!llama_set_sampler(ctx, s, chains[s])) {
                printf("backend sampler refused\n");
                return 7;
            }
        }
        printf("backend sampler accepted\n");
    }

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
    std::vector<int> order(n_vocab);
    for (int step = 0; step <= n_gen; ++step) {
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 6; }
        llama_token next[2];
        for (int s = 0; s < 2; ++s) {
            std::vector<llama_token> cand;
            if (bs) {
                next[s] = llama_get_sampled_token_ith(ctx, last_idx[s]);
                const uint32_t n = llama_get_sampled_candidates_count_ith(ctx, last_idx[s]);
                const llama_token * ids = llama_get_sampled_candidates_ith(ctx, last_idx[s]);
                if (n != (uint32_t) K || ids == nullptr) { fprintf(stderr, "candidates: %u\n", n); return 8; }
                cand.assign(ids, ids + n);
            } else {
                const float * lg = llama_get_logits_ith(ctx, last_idx[s]);
                next[s] = (llama_token) (std::max_element(lg, lg + n_vocab) - lg);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + K, order.end(), [&](int x, int y) { return lg[x] > lg[y]; });
                cand.assign(order.begin(), order.begin() + K);
            }
            std::sort(cand.begin(), cand.end());
            printf("step%d seq%d tok %d cand", step, s, next[s]);
            for (auto t : cand) { printf(" %d", t); }
            printf("\n");
        }
        b.n_tokens = 0;
        for (int s = 0; s < 2; ++s) { last_idx[s] = b.n_tokens; add(next[s], s, true); }
    }
    llama_batch_free(b);
    llama_free(ctx);
    for (auto * c : chains) { if (c) { llama_sampler_free(c); } }
    llama_model_free(model);
    return 0;
}
