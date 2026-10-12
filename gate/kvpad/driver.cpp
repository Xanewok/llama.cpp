// CPU pad-invariance driver. argv: model meta(0|1) pad n_ctx n_gen n_threads [variant]
// Prints n_ctx as created, the prompt/generated token ids and an FNV-1a hash over the bytes of every captured logit row
// (every 16th prompt row, then every decode step; prefill_fnv covers the prompt rows only). Greedy; variant != 0 perturbs the prompt (negative control).
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint64_t fnv(uint64_t h, const void * p, size_t n) {
    const unsigned char * b = (const unsigned char *) p;
    for (size_t i = 0; i < n; ++i) { h = (h ^ b[i]) * 1099511628211ull; }
    return h;
}

int main(int argc, char ** argv) {
    if (argc < 7) { return 2; }
    const int meta = atoi(argv[2]), pad = atoi(argv[3]), n_ctx = atoi(argv[4]), n_gen = atoi(argv[5]), n_thr = atoi(argv[6]);
    const int variant = argc > 7 ? atoi(argv[7]) : 0;
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

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = 1024; cp.n_ubatch = 128; cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = n_thr;
    cp.n_kv_pad = pad;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 4; }
    printf("n_ctx %u\n", llama_n_ctx(ctx));

    // ~720 prompt tokens (ubatches of 128 cross two 256-cell n_kv steps); decode must cross the next one at 768 (asserted by the gate script)
    std::string text;
    for (int i = 0; i < 36; ++i) {
        text += "Item " + std::to_string(i * 7 + variant) + ": the quick brown fox jumps over the lazy dog near the river bank. ";
    }
    std::vector<llama_token> toks(text.size() + 8);
    int n = llama_tokenize(vocab, text.data(), (int) text.size(), toks.data(), (int) toks.size(), true, true);
    if (n < 0) { return 5; }
    toks.resize(n);
    printf("n_prompt %d\n", n);

    const int n_vocab = llama_vocab_n_tokens(vocab);
    uint64_t h = 1469598103934665603ull;
    llama_batch b = llama_batch_init(n, 0, 1);
    std::vector<int> rows;
    for (int i = 0; i < n; ++i) {
        b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
        b.logits[i] = (i % 16 == 15) || i + 1 == n;
        if (b.logits[i]) { rows.push_back(i); }
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill failed\n"); return 6; }
    std::vector<float> top1;  // top-1 logit and margin of every hashed row, in capture order
    auto note_row = [&](const float * l) {
        float a = -1e30f, b2 = -1e30f;
        for (int i = 0; i < n_vocab; ++i) {
            if (l[i] > a) { b2 = a; a = l[i]; } else if (l[i] > b2) { b2 = l[i]; }
        }
        top1.push_back(a); top1.push_back(a - b2);
    };
    for (int i : rows) { h = fnv(h, llama_get_logits_ith(ctx, i), sizeof(float) * n_vocab); note_row(llama_get_logits_ith(ctx, i)); }
    const uint64_t h_prefill = h;
    const float * lg = llama_get_logits_ith(ctx, n - 1);
    llama_token next = (llama_token) (std::max_element(lg, lg + n_vocab) - lg);
    std::vector<llama_token> gen = { next };
    std::vector<float> margin;  // top-1 minus top-2 logit of each greedy pick: a flip is only legitimate at a near tie
    auto push_margin = [&](const float * l) {
        float a = -1e30f, b2 = -1e30f;
        for (int i = 0; i < n_vocab; ++i) {
            if (l[i] > a) { b2 = a; a = l[i]; } else if (l[i] > b2) { b2 = l[i]; }
        }
        margin.push_back(a - b2);
    };
    push_margin(lg);
    llama_batch d = llama_batch_init(1, 0, 1);
    for (int step = 0; step < n_gen; ++step) {
        d.n_tokens = 1; d.token[0] = next; d.pos[0] = n + step; d.n_seq_id[0] = 1; d.seq_id[0][0] = 0; d.logits[0] = true;
        if (llama_decode(ctx, d) != 0) { fprintf(stderr, "decode failed\n"); return 7; }
        lg = llama_get_logits_ith(ctx, 0);
        h = fnv(h, lg, sizeof(float) * n_vocab);
        note_row(lg);
        next = (llama_token) (std::max_element(lg, lg + n_vocab) - lg);
        gen.push_back(next);
        push_margin(lg);
    }
    printf("gen:");
    for (auto t : gen) { printf(" %d", t); }
    printf("\nmargin:");
    for (auto m : margin) { printf(" %.4f", m); }
    printf("\ntop1:");
    for (auto v : top1) { printf(" %.5f", v); }
    printf("\nprefill_fnv %016llx\nlogits_fnv %016llx rows %zu steps %d\n", (unsigned long long) h_prefill, (unsigned long long) h, rows.size(), n_gen);
    llama_batch_free(b); llama_batch_free(d);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
