// Target logits for a fixed accepted sequence, with the MTP draft context windowed or not. The schedule is fixed: a prompt in
// 32-token chunks (the fold engages from the second chunk), then rounds that draft k tokens through common_speculative,
// discard the draft rows like the server does, and verify k + 1 tokens of the fixed sequence, all accepted. The drafts
// never reach the target, so its logits must not depend on the window; the drafted ids and probabilities must.
// Usage: swa-target MODEL OUT_PREFIX META W S  ->  OUT_PREFIX.tgt.txt (FNV-1a per logits row), OUT_PREFIX.dft.txt
#include "common.h"
#include "speculative.h"
#include "llama.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

// compiles against a tree without the window parameters too, so the same fixture produces the base reference
template <typename T>
static auto set_window(T & draft, int w, int s, int) -> decltype(draft.swa_window, void()) {
    draft.swa_window = w;
    draft.swa_sinks  = s;
}

template <typename T>
static void set_window(T &, int w, int s, long) {
    require(w == 0 && s == 0, "this tree has no draft window");
}

static uint64_t fnv1a(const void * data, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ ((const uint8_t *) data)[i])*1099511628211ull;
    }
    return h;
}

int main(int argc, char ** argv) {
    try {
        require(argc == 6, "usage: swa-target MODEL OUT_PREFIX META W S");
        require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
        const std::string out = argv[2];
        const bool meta = std::atoi(argv[3]);

        llama_backend_init();
        common_params params;
        params.model.path = argv[1];
        params.n_ctx = 2048;
        params.n_batch = params.n_ubatch = 32;
        params.n_gpu_layers = 0;
        params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 4;
        params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        params.speculative.draft.n_max = 5;
        params.speculative.draft.n_min = 0;
        params.speculative.draft.p_min = 0;
        params.speculative.draft.probabilistic = true;
        params.speculative.draft.backend_sampling = true;
        params.speculative.draft.mtp_chain = std::getenv("LLAMA_MTP_CHAIN") && std::atoi(std::getenv("LLAMA_MTP_CHAIN"));
        params.speculative.draft.cpuparams = params.cpuparams;
        params.speculative.draft.cpuparams_batch = params.cpuparams_batch;
        for (int id = 248319; id >= 12288; id -= 17) {
            params.speculative.draft.vocab_ids.push_back(id);
        }
        for (int id = 12287; id >= 0; --id) {
            params.speculative.draft.vocab_ids.push_back(id);
        }
        set_window(params.speculative.draft, std::atoi(argv[4]), std::atoi(argv[5]), 0);
        if (meta) {
            auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            params.devices = {cpu, cpu, nullptr};
            params.speculative.draft.devices = params.devices;
            params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
            params.n_gpu_layers = -1;
        }

        auto init = common_init_from_params(params);
        llama_context * ctx_tgt = init->context();
        require(ctx_tgt, "target context");
        common_params params_dft = common_base_params_to_speculative(params);
        auto init_dft = common_speculative_init_from_params(params_dft, init->model(), ctx_tgt);
        llama_context * ctx_dft = init_dft->context();
        require(ctx_dft, "draft context");
        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = ctx_dft;
        common_speculative_ptr spec(common_speculative_init(params.speculative, 1));
        require(bool(spec), "speculative init");

        std::string text;
        for (int i = 0; i < 40; ++i) {
            text += "Item " + std::to_string(i) + ": the quick brown fox number " + std::to_string(i) + " jumps over lazy dog " +
                    std::to_string(i*7) + ". ";
        }
        const llama_tokens seq = common_tokenize(ctx_tgt, text, true, true);
        require(seq.size() > 600, "fixed sequence too short");
        const llama_pos n_prompt = 200;
        const size_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(init->model()));

        FILE * f_tgt = std::fopen((out + ".tgt.txt").c_str(), "w");
        FILE * f_dft = std::fopen((out + ".dft.txt").c_str(), "w");
        require(f_tgt && f_dft, "output open");

        common_batch batch(ctx_tgt);
        auto decode = [&](llama_pos p0, llama_pos n) {
            batch.clear();
            for (llama_pos p = p0; p < p0 + n; ++p) {
                batch.add(seq[p], p, 0, true);
            }
            require(llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "target decode");
            for (llama_pos i = 0; i < n; ++i) {
                const float * logits = llama_get_logits_ith(ctx_tgt, i);
                std::fprintf(f_tgt, "%d %016" PRIx64 "\n", p0 + i, fnv1a(logits, n_vocab*sizeof(float)));
            }
            require(common_speculative_process(spec.get(), batch), "speculative process");
        };

        for (llama_pos p = 0; p < n_prompt; p += 32) {
            decode(p, std::min<llama_pos>(32, n_prompt - p));
        }
        common_speculative_begin(spec.get(), 0, llama_tokens(seq.begin(), seq.begin() + n_prompt));

        llama_pos n = n_prompt;
        llama_tokens prompt(seq.begin(), seq.begin() + n_prompt);
        llama_tokens draft;
        std::vector<std::vector<llama_token_data>> draft_q;
        for (int round = 0; n + 6 < (llama_pos) seq.size(); ++round) {
            const int k = 1 + round % 5;
            draft.clear();
            draft_q.clear();
            common_speculative_get_draft_params(spec.get(), 0) = {
                /* .drafting = */ true,
                /* .n_max    = */ k,
                /* .pos0     = */ n,
                /* .id_last  = */ seq[n],
                /* .prompt   = */ &prompt,
                /* .result   = */ &draft,
                /* .result_q = */ &draft_q,
                /* .temp     = */ 1.0f,
                /* .seed     = */ 1729,
            };
            common_speculative_draft(spec.get());
            std::fprintf(f_dft, "round %d pos %d:", round, n);
            for (size_t i = 0; i < draft.size(); ++i) {
                std::fprintf(f_dft, " %d", draft[i]);
                if (i < draft_q.size() && !draft_q[i].empty()) {
                    std::fprintf(f_dft, "/%.9g", draft_q[i][0].p);
                }
            }
            std::fprintf(f_dft, "\n");
            require(llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, n, -1), "draft rows removal");

            decode(n, k + 1);
            common_speculative_accept(spec.get(), 0, k);
            prompt.insert(prompt.end(), seq.begin() + n, seq.begin() + n + k + 1);
            n += k + 1;
        }
        std::fprintf(f_dft, "draft pos_min %d pos_max %d\n", llama_memory_seq_pos_min(llama_get_memory(ctx_dft), 0),
                     llama_memory_seq_pos_max(llama_get_memory(ctx_dft), 0));
        std::fclose(f_tgt);
        std::fclose(f_dft);
        std::fprintf(stderr, "swa-target: %d positions, done\n", n);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "swa-target: FAIL %s\n", e.what());
        return 1;
    }
}
