#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

struct probe_data {
    std::string directory;
    std::unordered_set<std::string> seen;
};

static bool observe(ggml_tensor * t, bool ask, void * data) {
    const bool value = strncmp(t->name, "Vcur-", 5) == 0 && strchr(t->name, ' ') == nullptr;
    const bool probs = strncmp(t->name, "kq_soft_max-", 12) == 0;
    if (ask) {
        return value || probs;
    }
    auto & probe = *static_cast<probe_data *>(data);
    if (!probe.seen.insert(t->name).second) {
        return true;
    }
    GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
    std::vector<float> x(ggml_nelements(t));
    ggml_backend_tensor_get(t, x.data(), 0, x.size()*sizeof(float));
    const std::string path = probe.directory + "/" + t->name + ".f32";
    FILE * f = fopen(path.c_str(), "wb");
    GGML_ASSERT(f && fwrite(x.data(), sizeof(float), x.size(), f) == x.size());
    fclose(f);
    if (value) {
        double sum2 = 0.0;
        float peak = 0.0f;
        for (float v : x) {
            GGML_ASSERT(std::isfinite(v));
            peak = std::max(peak, std::abs(v));
            sum2 += double(v)*v;
        }
        printf("V,%s,%lld,%.9g,%.9g\n", t->name, (long long) x.size(), peak, sqrt(sum2/x.size()));
    } else {
        double hsum = 0.0, flat_sum = 0.0, flat_max = 0.0;
        const int64_t rows = ggml_nrows(t);
        for (int64_t row = 0; row < rows; ++row) {
            double entropy = 0.0;
            float pmax = 0.0f;
            for (int64_t k = 0; k < t->ne[0]; ++k) {
                const float p = x[row*t->ne[0] + k];
                if (p > 0) {
                    entropy -= p*log(double(p));
                }
                pmax = std::max(pmax, p);
            }
            hsum += entropy;
            flat_sum += 1.0/pmax;
            flat_max = std::max(flat_max, 1.0/pmax);
        }
        printf("P,%s,%lld,%.9g,%.9g,%.9g\n", t->name, (long long) rows, hsum/rows, flat_sum/rows, flat_max);
    }
    fflush(stdout);
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: swift-v-probe MODEL OUTPUT_DIRECTORY\n");
        return 2;
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return 3;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.no_perf = false;
    probe_data probe{argv[2], {}};
    cp.cb_eval = observe;
    cp.cb_eval_user_data = &probe;
    auto * ctx = llama_init_from_model(model, cp);
    GGML_ASSERT(ctx);
    const std::string prompt =
        "Explain how to review a numerical kernel. Check shapes, strides, precision, synchronization, and boundary conditions. "
        "For a long attention sequence, the online softmax subtracts a running maximum. The numerator and denominator must use the same scale. "
        "A test should cover flat logits, concentrated logits, signed values, constant values, and masked tails. "
        "Write a concise Python function that computes a stable softmax, and explain why overflow can occur in a half precision running sum. "
        "Consider a tensor with twenty four query heads, four key heads, and a head dimension of two hundred and fifty six. "
        "The matrix multiplication is partitioned across blocks. Each partial result must be rescaled before combining. "
        "Which measurements distinguish an accumulator range problem from a key value quantization error?";
    const auto * vocab = llama_model_get_vocab(model);
    int n = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    GGML_ASSERT(n > 0 && n <= 512);
    std::vector<llama_token> tokens(n);
    GGML_ASSERT(llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), n, true, true) == n);
    printf("TOKENS,%d\n", n);
    const auto start = std::chrono::steady_clock::now();
    const int rc = llama_decode(ctx, llama_batch_get_one(tokens.data(), n));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    printf("DECODE_MS,%.3f\n", ms);
    llama_perf_context_print(ctx);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
