#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "sampling.h"
#include "speculative.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static bool arch_matches(const std::string & filter, llm_arch arch) {
    if (filter.empty()) {
        return true;
    }
    return std::regex_search(llama_arch_name(arch), std::regex(filter));
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

struct tensor_data_params {
    size_t seed;
    float  stdev;
};

// fills the values of one tensor, the name carries whatever the property needs
typedef void (*tensor_filler)(std::vector<float> & dst, const char * name, const tensor_data_params & params);

// a normal draw keyed on a name, so that tensors which share a draw ask for the same key
static void fill_normal(std::vector<float> & dst, const char * key, const tensor_data_params & params) {
    std::mt19937 gen(params.seed ^ std::hash<std::string>{}(key));
    std::normal_distribution<float> dis(0.0f, params.stdev);
    for (float & v : dst) {
        v = dis(gen);
    }
}

// Mamba A must be negative (state decay)
static void fill_ssm_a(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = -fabsf(v);
    }
}

// a norm preserving residual gate holds alpha^2 + beta^2 == 1, and the conversion resolves both from one raw weight
// the alpha and the beta of a block share that weight, so the key drops the suffix and keeps the block
static void fill_resid_gate(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    const char * suffix = strstr(name, "_resid_");
    const bool is_beta = strstr(suffix, "_resid_beta") != nullptr;
    fill_normal(dst, std::string(name, suffix - name).c_str(), params);
    for (float & v : dst) {
        const float beta = 1.0f / (1.0f + expf(-v));
        v = is_beta ? beta : sqrtf(1.0f - beta*beta);
    }
}

// a random router bias can tie within the noise between the backends, a ramp keeps the top-k selection the same
static void fill_exp_probs_b(std::vector<float> & dst, const char *, const tensor_data_params &) {
    for (size_t i = 0; i < dst.size(); i++) {
        dst[i] = 2.0f*i;
    }
}

// a tensor whose values hold a property of their own, matched on a substring of its name, first match wins
static const struct { const char * match; tensor_filler fill; } TENSOR_FILLERS[] = {
    { "ssm_a",       fill_ssm_a       },
    { "_resid_",     fill_resid_gate  },
    { "exp_probs_b", fill_exp_probs_b },
};

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    const tensor_data_params & params = *(const tensor_data_params *) userdata;

    tensor_filler fill = fill_normal;
    for (const auto & filler : TENSOR_FILLERS) {
        if (strstr(tensor->name, filler.match) != nullptr) {
            fill = filler.fill;
            break;
        }
    }

    std::vector<float> tmp(ggml_nelements(tensor));
    fill(tmp, tensor->name, params);

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp16(tmp.size());
        for (size_t i = 0; i < tmp.size(); i++) {
            tmp16[i] = ggml_fp32_to_fp16(tmp[i]);
        }
        ggml_backend_tensor_set(tensor, tmp16.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    LOG("Usage: %s [options]\n\n", argv[0]);
    LOG("Options:\n");
    LOG("  -a, --arch <arch|regex>  Run only matching LLM architectures (default: all supported)\n");
    LOG("  -s, --seed <seed>        Set the random seed for tensor initialization and token generation\n");
    LOG("  -d, --stdev <stdev>      Set the standard deviation of the tensor initialization distribution (default: 0.1f)\n");
    LOG("  -o, --out <dir>          Save generated test models to <dir> instead of running backend tests\n");
    LOG("  -v <N>                   Set log verbosity level\n");
    LOG("  -b, --backend <backend>  Run only on the given backend device\n");
    LOG("  --tp-sampling-rebuild    Alternate two backend samplers (requires TP sampling support)\n");
    LOG("  --tp-sampling-rebuild-single  Single-sampler control\n");
    LOG("      --tp-sampling        Check backend sampling under tensor split on two CPU devices, then exit\n");
    LOG("      --tp-sampling-gpu    The same on the first two GPU devices\n");
    LOG("  -h, --help               Show this help message\n\n");
    LOG("Examples:\n");
    LOG("  %s\n", argv[0]);
    LOG("  %s -a qwen35moe\n", argv[0]);
    LOG("  %s -a deepseek4 -o tests/test-models/\n", argv[0]);
    LOG("  %s -a cohere2moe -v 5\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

// MoE archs that are also tested with the experts in host memory
static bool host_experts_test(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_DEEPSEEK2:
            return true;
        default:
            return false;
    }
}

// writes arch-resolved GGUF keys into the fixture context, the test-side equivalent of llama_model_saver
struct kv_writer {
    struct gguf_context * gguf_ctx;
    const llm_arch        arch;

    kv_writer(enum llm_arch arch, struct gguf_context * gguf_ctx) : gguf_ctx(gguf_ctx), arch(arch) {}

    void add_kv(enum llm_kv key, uint32_t     value) { gguf_set_val_u32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, int32_t      value) { gguf_set_val_i32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, uint64_t     value) { gguf_set_val_u64 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, float        value) { gguf_set_val_f32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, bool         value) { gguf_set_val_bool(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, const char * value) { gguf_set_val_str (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }

    template <typename T>
    void add_kv(enum llm_kv key, const std::vector<T> & value) {
        if (value.empty()) {
            return;
        }
        if (std::is_same<T, uint32_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_UINT32, value.data(), value.size());
        } else if (std::is_same<T, int32_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_INT32, value.data(), value.size());
        } else if (std::is_same<T, uint64_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_UINT64, value.data(), value.size());
        } else if (std::is_same<T, float>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_FLOAT32, value.data(), value.size());
        } else {
            GGML_ABORT("fatal error");
        }
    }

    void add_kv(enum llm_kv key, const std::vector<std::string> & value) {
        std::vector<const char *> tmp(value.size());
        for (size_t i = 0; i < value.size(); ++i) {
            tmp[i] = value[i].c_str();
        }
        gguf_set_arr_str(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), tmp.data(), tmp.size());
    }
};

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe, const uint32_t n_vocab_override = 0) {
    gguf_context_ptr ret(gguf_init_empty());
    kv_writer ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_MAION_CODER) {
        n_head = 4;
        n_layer = 4; // dense and MoE layers, each with local and full attention
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK41) {
        // window-only, Engram, ratio-2 and ratio-1 sources, top-k reuse and a candidate-masked index layer
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 6;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_GLM5_NEXT
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_HY_V4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        //n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
        n_vocab = 3072; // TODO: should be 4096, but user code cannot get `n_vocab_out` yet [TAG_LLAMA_N_VOCAB_OUT]
    } else if (arch == LLM_ARCH_HRM_TEXT) {
        n_layer = 6; // 1 layer per stack x 2 h-cycles x (2 l-cycles + 1) cache slots
    }

    if (n_vocab_override > 0) {
        n_vocab = n_vocab_override;
    }

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_MUSE_SPARK || arch == LLM_ARCH_AFMOE || arch == LLM_ARCH_MAION_CODER ||
            arch == LLM_ARCH_KOLIBRI1) {
        n_head_kv = 2; // GQA coverage
    }
    if (arch == LLM_ARCH_BEAM) {
        n_head    = 4;
        n_head_kv = 2;
        n_layer   = 4;
    }
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llama_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_K2_HORIZON) {
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_FAST, 128.0f);
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   4.0f);
    }

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR ||
            arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3 || arch == LLM_ARCH_GLM5_NEXT) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        // GLM5 next KDA heads come from the uniform head count, only head_count_kv is per layer.
        if (arch == LLM_ARCH_GLM5_NEXT) {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        } else {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_GLM5_NEXT
            || arch == LLM_ARCH_HY_V4
            || arch == LLM_ARCH_MISTRAL4) {
        // GLM5 next MLA is nope only, the cache row is the compressed latent alone.
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(512) : uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(0) : uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        if (arch == LLM_ARCH_DOTS3NOTE) {
            // SWA layers reuse the same MLA geometry as the full layers in this fixture
            ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,       uint32_t(576));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   uint32_t(192));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, uint32_t(128));
            ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,             10000.0f);
            // indexer on the full-attention layers (inverse of the swa pattern)
            std::vector<uint32_t> indexer_types;
            indexer_types.reserve(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) {
                indexer_types.push_back(il % 2 ? 0 : 1);
            }
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
        }
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    if (arch == LLM_ARCH_SENSENOVA_U1) {
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SPATIAL, 10000.0f);
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) ? uint32_t(64) : uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_MAION_CODER) {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, std::vector<uint32_t>{1, 1, 0, 0});
    } else if (arch == LLM_ARCH_BEAM) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_SPARK2_5 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_MUSE_SPARK || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE ||
            arch == LLM_ARCH_KOLIBRI1 || arch == LLM_ARCH_MAPLE || arch == LLM_ARCH_FLINT) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
        if (arch == LLM_ARCH_FLINT) {
            ms.add_kv(LLM_KV_ATTENTION_ROPE_PATTERN, pattern);
        }
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    if (arch == LLM_ARCH_MUSE_SPARK) {
        ms.add_kv(LLM_KV_POST_NORM_EPS,           1e-5f);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,      10000.0f);
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING, 30.0f);
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE,         n_embd/2);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,    1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,     true);
    }

    if (arch == LLM_ARCH_KOLIBRI1) {
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM, false);
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_GLM5_NEXT) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,  1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));

        // has_cell_ext() needs ple_n_heads here: the indexer cache serializes no ext without it
        const uint32_t ple_ngram_size      = 3;
        const uint32_t ple_heads_per_ngram = 2;
        const uint32_t ple_n_heads         = (ple_ngram_size - 1)*ple_heads_per_ngram;
        GGML_ASSERT(n_embd % ple_n_heads == 0);
        const uint32_t ple_head_dim = n_embd/ple_n_heads;

        std::vector<uint64_t> ple_head_offsets(ple_n_heads);
        std::vector<uint64_t> ple_head_vocab_sizes(ple_n_heads, n_vocab);
        for (uint32_t h = 0; h < ple_n_heads; h++) {
            ple_head_offsets[h] = uint64_t(h)*n_vocab;
        }

        // the PLE history lives in the recurrent cache, so it must sit on a linear attention layer
        ms.add_kv(LLM_KV_PLE_LAYERS,                  std::vector<uint32_t>({ 0 }));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,              ple_ngram_size);
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,         ple_heads_per_ngram);
        ms.add_kv(LLM_KV_PLE_CONV_KERNEL,             uint32_t(4));
        ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,            uint32_t(0));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  ple_head_dim);
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,       std::vector<uint64_t>({ 1, 3, 5 }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,            ple_head_offsets);
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,        ple_head_vocab_sizes);
    }

    // minimax-m3 keeps one indexer head per GQA head; the rest use a fixed 64 to match the fused
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 ? n_head : uint32_t(64));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : uint32_t(128));

    // note: using a realistic top-k here makes the results unstable and hard to match between CPU and GPU
    //       a large value makes things deterministic since all data is selected by the indexer
    //ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(131072));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,        uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL, true);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    // mrope sections count rope pairs; Ling 3.0 VL files carry [t, h, w] sections
    // summing to n_rot / 2 (n_rot is 64 in this fixture)
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({8, 12, 12, 0}));
    } else {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    }

    if (arch == LLM_ARCH_HY_V4) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,     uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,   1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_MAGNITUDE, 2.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,           10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,        true);
        // layer 0 must own an indexer, the odd layers share it
        std::vector<uint32_t> indexer_types;
        indexer_types.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            indexer_types.push_back(il % 2 ? 0 : 1);
        }
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
    }

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,          uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,            uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,             std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,     160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,              1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    }

    if (arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,          uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,            uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,             std::vector<uint32_t>({0, 0, 2, 2, 1, 1}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,     160000.0f);
        ms.add_kv(LLM_KV_ATTENTION_KV_SOURCE_LAYERS,            std::vector<uint32_t>({2, 4}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_SOURCE_LAYERS,       std::vector<uint32_t>({2, 4, 5}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, int32_t(4));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS, uint32_t(4096));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE,   uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,              1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
        std::vector<int32_t> token_map(n_vocab);
        for (uint32_t i = 0; i < n_vocab; ++i) {
            token_map[i] = (int32_t) (i % 61);
        }
        ms.add_kv(LLM_KV_PLE_LAYERS,                   std::vector<uint32_t>({1, 3}));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,               uint32_t(3));
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,          uint32_t(2));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,   uint32_t(64));
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,        std::vector<uint64_t>({ 76632096046245ull, 4839876093313ull, 35959672319349ull,
                                                                               67716810739261ull, 51510806800915ull, 30921347202721ull }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,             std::vector<uint64_t>({0, 101, 204, 311, 420, 533, 660, 791}));
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,         std::vector<uint64_t>({101, 103, 107, 109, 113, 127, 131, 137}));
        ms.add_kv(LLM_KV_ENGRAM_PAD_TOKEN_ID,          uint32_t(2));
        ms.add_kv(LLM_KV_ENGRAM_TOKEN_MAP,             token_map);
    }

    if (arch == LLM_ARCH_HRM_TEXT) {
        // 6 cache slots alias 2 physical blocks: 1 low-stack layer + 1 high-stack layer
        ms.add_kv(LLM_KV_HRM_LAYERS_PER_STACK, uint32_t(1));
        ms.add_kv(LLM_KV_HRM_H_CYCLES,         uint32_t(2));
        ms.add_kv(LLM_KV_HRM_L_CYCLES,         uint32_t(2));
    }

    if (arch == LLM_ARCH_MAPLE) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP, 7.0f);
    }

    // dummy tokenizer: token ids are derived from fixed-size chunks and detokenized as hex ids
    {
        std::vector<std::string> tokenizer_list(n_vocab);
        std::vector<float>       tokenizer_scores(n_vocab, 0.0f);

        ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "test");
        for (uint32_t i = 0; i < n_vocab; i++) {
            tokenizer_list[i] = "tok_" + std::to_string(i);
        }
        ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokenizer_list);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES, tokenizer_scores);
    }

    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        // with more experts than a ubatch uses, the copy of the used experts in host memory skips some of them
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(host_experts_test(arch) ? 64 : 2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));

        uint32_t gating_func = 2;
        switch (arch) {
            case LLM_ARCH_MELLUM:
                gating_func = 1;
                break;
            case LLM_ARCH_DEEPSEEK4:
            case LLM_ARCH_DEEPSEEK41:
                gating_func = 4;
                break;
            default:
                break;
        }

        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         gating_func); // sigmoid : softmax : sqrtsoftplus
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
        if (arch == LLM_ARCH_FLINT) {
            ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM, true);
        }
        if (arch == LLM_ARCH_K2_HORIZON) {
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_COUNT,      uint32_t(2));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_USED_COUNT, uint32_t(2));
        }
    }

    if (arch == LLM_ARCH_BEAM) {
        ms.add_kv(LLM_KV_EXPERT_COUNT,            uint32_t(8));
        ms.add_kv(LLM_KV_EXPERT_GROUP_COUNT,      uint32_t(4));
        ms.add_kv(LLM_KV_EXPERT_GROUP_USED_COUNT, uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,    2.5f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,     true);
    }

    if (arch == LLM_ARCH_MAION_CODER) {
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE, n_embd / 2);
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, std::vector<uint32_t>{n_ff, n_ff / 2, n_ff, n_ff});
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,             true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   std::vector<float>({0.0f, 4.0f}));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, std::vector<float>({0.0f, 5.0f}));
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,               n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,           uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,              3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,         uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,        4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,        -5.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::string get_tmp_gguf_path() {
    static size_t id = 0;
    return (std::filesystem::temp_directory_path() / ("test-llama-archs-" + std::to_string(++id) + ".gguf")).string();
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr, const size_t moe_cache_size = 0) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;
    model_params.tensor_buft_overrides = tensor_buft_overrides;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }
    if (moe_cache_size > 0) {
        // the MoE cache is only used for small ubatches
        ctx_params.moe_cache_size = moe_cache_size;
        ctx_params.n_ubatch = 2;
    }

    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tensor_params, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    common_batch batch(lctx);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        batch.add(tokens[pos], pos, 0, true);
    }
    if (encode) {
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_ENCODE, batch.get())) {
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    return ret;
}

// entries [n/4, n/2) are embd rows, decoded either as token/embd/token chunks or as one mixed batch
// returns the llama_process() error code
static int32_t get_logits_mixed(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, const std::vector<float> & embd, bool mixed,
        std::vector<float> & ret) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_embd   = llama_model_n_embd_inp(model);
    const uint32_t n_tokens = tokens.size();
    const uint32_t i_embd_0 = n_tokens/4;
    const uint32_t i_embd_1 = n_tokens/2;

    const std::vector<uint32_t> bounds = mixed
        ? std::vector<uint32_t>{0, n_tokens}
        : std::vector<uint32_t>{0, i_embd_0, i_embd_1, n_tokens};

    llama_memory_clear(llama_get_memory(lctx), true);
    llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));

    ret.clear();
    ret.reserve(n_tokens*n_vocab);
    for (size_t c = 0; c + 1 < bounds.size(); c++) {
        llama_batch_ext_clear(batch.get());
        for (uint32_t i = bounds[c]; i < bounds[c + 1]; i++) {
            const bool is_embd = i >= i_embd_0 && i < i_embd_1;
            const int32_t idx = is_embd
                ? llama_batch_ext_add_embd(batch.get(), 0, { embd.data() + (size_t) (i - i_embd_0)*n_embd, 1, n_embd })
                : llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
            GGML_ASSERT(idx >= 0);
            const llama_pos pos[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, 0 };
            llama_batch_ext_set_pos(batch.get(), idx, pos);
            llama_batch_ext_set_output_logits(batch.get(), idx, true);
        }
        const int32_t err = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (err != 0) {
            return err;
        }
        for (uint32_t i = 0; i < bounds[c + 1] - bounds[c]; i++) {
            const float * logits_ith = llama_get_logits_ith(lctx, i);
            ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
        }
    }
    return 0;
}

static bool check_causal_attn_toggle(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_past   = tokens.size();
    const uint32_t n_ubatch = llama_n_ubatch(lctx);

    GGML_ASSERT(n_past + n_ubatch/2 + n_ubatch <= llama_n_ctx(lctx));

    llama_set_causal_attn(lctx, false);

    common_batch batch(lctx);

    bool ok = true;
    uint32_t pos = n_past;
    for (const uint32_t n_tokens : { n_ubatch/2, n_ubatch }) {
        batch.clear();
        for (uint32_t i = 0; i < n_tokens; i++) {
            batch.add(tokens[i], pos++, 0, true);
        }

        const int32_t rc = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) {
            LOG_ERR("%s: n_tokens=%u: llama_process returned %d\n", __func__, n_tokens, rc);
            ok = false;
            break;
        }

        const float * logits = llama_get_logits_ith(lctx, n_tokens - 1);
        if (logits == nullptr) {
            LOG_ERR("%s: n_tokens=%u: no logits\n", __func__, n_tokens);
            ok = false;
            break;
        }
        for (uint32_t j = 0; j < n_vocab; j++) {
            if (std::isnan(logits[j])) {
                LOG_ERR("%s: n_tokens=%u: nan logit\n", __func__, n_tokens);
                ok = false;
                break;
            }
        }
    }

    return ok;
}

static bool check_mtp_chain_capacity() {
    bool ok = true;
    for (int slots : { 1, 2 }) {
        for (int k : { 1, 2, LLAMA_MTP_CHAIN_MAX, LLAMA_MTP_CHAIN_MAX + 1 }) {
            common_params params;
            params.n_parallel = slots;
            params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
            params.speculative.draft.probabilistic = true;
            params.speculative.draft.backend_sampling = true;
            params.speculative.draft.p_min = 0.0f;
            params.speculative.draft.n_max = k;
            const auto draft = common_base_params_to_speculative(params);
            const int expected = slots == 1 && k >= 2 && k <= LLAMA_MTP_CHAIN_MAX ? k : 1;
            ok &= draft.n_outputs_max_per_seq == expected;
            ok &= draft.n_outputs_max >= slots*expected;
        }
    }
    LOG("MTP chain output capacity: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

struct gdn_probe {
    int     n_indexed       = 0;
    int     n_plain         = 0;
    int     n_indexed_multi = 0;
    int     n_plain_single  = 0;
    int64_t n_gathered      = 0;
};

static bool gdn_probe_cb(ggml_tensor * t, bool ask, void * user_data) {
    if (!ask) {
        return false;
    }
    auto * probe = (gdn_probe *) user_data;
    if (t->op == GGML_OP_GATED_DELTA_NET) {
        (t->src[6] ? probe->n_indexed : probe->n_plain)++;
        if (t->src[6] && t->src[2]->ne[3] > 1) {
            ++probe->n_indexed_multi;
        }
        if (!t->src[6] && t->src[2]->ne[3] == 1) {
            ++probe->n_plain_single;
        }
    }
    const ggml_tensor * src = t->op == GGML_OP_GET_ROWS ? t->src[0] : nullptr;
    if (src && src->view_src) {
        src = src->view_src;
    }
    if (src && strncmp(src->name, "cache_s_l", 9) == 0) {
        probe->n_gathered += t->ne[1];
    }
    return false;
}

// Compare getter and gather builds on identical batches; single-sequence batches can use different arithmetic.
static bool check_gdn_getter(llama_model * model) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    constexpr uint32_t n_ubatch = 16;

    gdn_probe probe;
    const auto make_ctx_gate = [&](uint32_t n_seq) {
        auto cparams = llama_context_default_params();
        cparams.n_threads = 4;
        cparams.n_threads_batch = 4;
        cparams.n_seq_max  = n_seq;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256*n_seq;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        cparams.cb_eval           = gdn_probe_cb;
        cparams.cb_eval_user_data = &probe;
        return llama_context_ptr(llama_init_from_model(model, cparams));
    };

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((11*(uint32_t) pos + 29*seq + 3) % (uint32_t) n_vocab);
    };

    enum class form { BY_ID, GATHERED };
    // the streams share one batch in order, tokens from out_from on request logits
    struct phase {
        const char *          name;
        std::vector<uint32_t> streams;
        llama_pos             p0, p1, out_from;
        form                  expect;
    };
    struct scenario {
        const char *       name;
        uint32_t           n_seq;
        std::vector<phase> phases;
    };
    const std::vector<scenario> scenarios = {
        { "two slots", 2, {
            { "prefill seq 0",             { 0 },    0, 19, 18, form::BY_ID }, // two ubatches of one sequence
            { "prefill seq 1",             { 1 },    0, 19, 18, form::BY_ID },
            { "verify batch, both seqs",   { 0, 1 }, 19, 25, 19, form::BY_ID }, // n_seqs = n_rs = 2, several tokens each
            { "decode, both seqs",         { 0, 1 }, 25, 26, 25, form::BY_ID }, // one token each
            { "decode, both seqs again",   { 0, 1 }, 26, 27, 26, form::BY_ID },
            { "seq 1 beside an idle slot", { 1 },    27, 28, 27, form::BY_ID },
        } },
        // seq 1 takes the first cell, so the joint batch swaps the cells: each sequence reads the cell the other writes
        { "swapped cells", 2, {
            { "prefill seq 1",             { 1 },    0, 19, 18, form::BY_ID },
            { "prefill seq 0",             { 0 },    0, 19, 18, form::BY_ID },
            { "verify batch, both seqs",   { 0, 1 }, 19, 25, 19, form::BY_ID },
            { "decode, both seqs",         { 0, 1 }, 25, 26, 25, form::BY_ID },
        } },
        // both start from the shared zero state in one batch
        { "joint start", 2, {
            { "prefill both seqs",         { 0, 1 }, 0, 12, 11, form::BY_ID },
            { "decode, both seqs",         { 0, 1 }, 12, 13, 12, form::BY_ID },
        } },
        // seq 2 takes the cell between seqs 0 and 1, so their joint batch holds its state as an extra state the gather
        // must move (without a unified KV cache a ubatch joins only consecutive sequence ids)
        { "extra state", 3, {
            { "prefill seq 0",              { 0 },    0, 19, 18, form::BY_ID },
            { "prefill seq 2",              { 2 },    0, 19, 18, form::BY_ID },
            { "prefill seq 1",              { 1 },    0, 19, 18, form::BY_ID },
            { "verify batch, seqs 0 and 1", { 0, 1 }, 19, 25, 19, form::GATHERED },
            { "seq 2 after the move",       { 2 },    19, 25, 19, form::BY_ID },
            { "decode, seqs 0 and 1",       { 0, 1 }, 25, 26, 25, form::BY_ID },
        } },
    };

    int n_gdn = 0;
    for (const auto & sc : scenarios) {
        llama_context_ptr ctx_gate = make_ctx_gate(sc.n_seq);
        if (!ctx_gate) {
            LOG_ERR("%s: %s: failed to init context\n", __func__, sc.name);
            return false;
        }
        if (llama_n_rs_seq(ctx_gate.get()) == 0) {
            LOG_INF("%s: skipping because n_rs_seq is disabled\n", __func__);
            return false;
        }

        for (const auto & ph : sc.phases) {
            probe = {};
            common_batch batch(ctx_gate.get());
            for (uint32_t s : ph.streams) {
                for (llama_pos pos = ph.p0; pos < ph.p1; ++pos) {
                    batch.add(tok(s, pos), pos, (llama_seq_id) s, pos >= ph.out_from);
                }
            }
            if (llama_process(ctx_gate.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                LOG_ERR("%s: %s: %s: decode failed\n", __func__, sc.name, ph.name);
                return false;
            }
            llama_synchronize(ctx_gate.get());
            LOG_INF("%s: %s: %s: gdn nodes %d by id, %d gathered; state rows gathered %" PRId64 "\n", __func__,
                    sc.name, ph.name, probe.n_indexed, probe.n_plain, probe.n_gathered);

            for (size_t j = 0; j < ph.streams.size(); ++j) {
                const uint32_t s = ph.streams[j];
                const int32_t first = (int32_t) j*(ph.p1 - ph.p0) - ph.p0; // batch index of pos is first + pos
                for (llama_pos pos = ph.out_from; pos < ph.p1; ++pos) {
                    const float * l_gate = llama_get_logits_ith(ctx_gate.get(), first + pos);
                    if (!l_gate || !std::all_of(l_gate, l_gate + n_vocab, [](float v) { return std::isfinite(v); })) {
                        LOG_ERR("%s: %s: %s: seq %u pos %d missing or non-finite logits\n", __func__, sc.name, ph.name, s, (int) pos);
                        return false;
                    }
                }
            }

            n_gdn += probe.n_indexed + probe.n_plain;
            if (probe.n_indexed + probe.n_plain == 0) {
                continue;
            }
            if (ph.expect == form::BY_ID && (probe.n_indexed == 0 || probe.n_plain != 0 || probe.n_gathered != 0)) {
                LOG_ERR("%s: %s: %s: ubatches without extra states must read the state by id and gather none\n",
                        __func__, sc.name, ph.name);
                return false;
            }
            if (ph.expect == form::GATHERED && (probe.n_plain == 0 || probe.n_indexed != 0)) {
                LOG_ERR("%s: %s: %s: ubatches with extra states must keep the gathered state\n", __func__, sc.name, ph.name);
                return false;
            }
        }
    }

    if (n_gdn == 0) {
        LOG_INF("%s: skipping because the model has no gated delta net\n", __func__);
        return false;
    }
    LOG_INF("%s: all scenarios pass\n", __func__);
    return true;
}

static bool check_gdn_getter_arch(const size_t seed, const float stdev) {
    auto gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN35, false);
    auto pair = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev,
                                 { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) });
    return check_gdn_getter(pair.first.get());
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_FLINT:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_MAION_CODER:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_DEEPSEEK41:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_BEAM:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_HY_V4:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_GLM5_NEXT:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
        case LLM_ARCH_MAPLE:
        case LLM_ARCH_MUSE_SPARK:
        case LLM_ARCH_KOLIBRI1:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
        case LLM_ARCH_K2_HORIZON:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        return false; // FIXME @ngxson
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_GEMMA_EMBEDDING2 || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_CLEF || arch == LLM_ARCH_VLLM_DECISION2) {
        return false; // TODO decision head tensors
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP ||
            arch == LLM_ARCH_HY_V4) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

static int save_models(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_arch_supports_save(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llama_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, {});
            const std::string path = dir + "/" + llama_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: Saving %s model (%s) to %s...\n", __func__, llama_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model_and_ctx.first.get(), path.c_str());
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return 0;
}

// the CPU and the GPU of an Apple SoC carry the same description, the name tells them apart (CPU, MTL0)
static std::string get_device_label(ggml_backend_dev_t dev) {
    const char * desc = ggml_backend_dev_description(dev);
    const char * name = ggml_backend_dev_name(dev);

    if (strcmp(desc, name) == 0) {
        return desc;
    }

    return std::string(desc) + " (" + name + ")";
}

static int test_backends(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const char * target_backend) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;
        bool                            host_experts; // keep the experts in host memory, see host_experts_test
        size_t                          moe_cache_size;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode, bool host_experts = false, size_t moe_cache_size = 0)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode), host_experts(host_experts), moe_cache_size(moe_cache_size) {}
    };

    const llama_model_tensor_buft_override host_experts_overrides[] = {
        { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (target_backend != nullptr && strcmp(target_backend, ggml_backend_dev_name(dev)) != 0) {
                    continue;
                }
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, get_device_label(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        if (target_backend == nullptr) {
            dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
        }

        // the ops that use the experts are offloaded to the first device and the scheduler copies the used experts
        if (!devices_meta.empty()) {
            dev_configs.emplace_back(devices_meta, "Host experts", LLAMA_SPLIT_MODE_LAYER, true);
            max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
        }

        // the ops that use the host experts run on a GPU and read the experts from a cache
        // the cache has only a few slots (4 for 288 KiB experts), so the experts are evicted and uploaded again
        if (!devices_meta.empty()) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(devices_meta[0]);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{devices_meta[0]}, "MoE cache", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                // each GPU caches the layers assigned to it
                if (devices_meta.size() > 1) {
                    dev_configs.emplace_back(devices_meta, "MoE cache, layer split", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                    max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                }
            }
        }
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llama_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llama_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|%15s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|%15s %10s|\n";

    bool all_ok = true;
    size_t n_tests = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip", "Mixed batch");
    LOG("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        LOG("-");
    }
    LOG("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        LOG("-");
    }
    LOG("|------|---------------|---------|---------------|\n");
    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                if (dc.host_experts && (!moe || !host_experts_test(arch))) {
                    continue;
                }
                const llama_model_tensor_buft_override * overrides = dc.host_experts ? host_experts_overrides : nullptr;

                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                LOG(template_row_cfg.c_str(), llama_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                std::string status_mixed     = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                char mixed_str[12] = {0};

                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                bool test_executed = false;
                bool test_ok = true;
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llama_arch_supports_sm_tensor(arch)) {
                        test_executed = true;
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }

                        // chunked decode matches a single batch only with causal attention over a memory
                        llama_context * lctx_dev = model_and_ctx_dev.second.get();
                        if (!encode && llama_get_memory(lctx_dev) != nullptr) {
                            std::vector<float> embd_mixed((size_t) llama_model_n_embd_inp(model_and_ctx_dev.first.get())*tokens.size()/4);
                            std::mt19937 gen(seed);
                            std::normal_distribution<float> dis(0.0f, stdev);
                            for (float & v : embd_mixed) {
                                v = dis(gen);
                            }
                            std::vector<float> logits_mixed;
                            std::vector<float> logits_chunks;
                            if (llama_arch_supports_mixed_batch(arch)) {
                                if (get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, false, logits_chunks) != 0 ||
                                    get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, true,  logits_mixed)  != 0) {
                                    throw std::runtime_error("failed to decode mixed batch");
                                }
                                const double nmse_mixed = nmse(logits_chunks, logits_mixed);
                                snprintf(mixed_str, sizeof(mixed_str), "(%.2e)", nmse_mixed);
                                status_mixed = "\033[1;32mOK\033[0m";
                                if (nmse_mixed > 1e-4) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                            } else {
                                // must be rejected as an invalid batch, mute the expected error log
                                ud.verbosity = LOG_LEVEL_OUTPUT;
                                const int32_t err = get_logits_mixed(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, embd_mixed, true, logits_mixed);
                                ud.verbosity = verbosity;
                                if (err != -1) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                            }
                        }

                        // runs after the mixed batch check, as it leaves the context with non-causal attention
                        if (!encode && !check_causal_attn_toggle(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens)) {
                            if (test_ok) {
                                status_nmse = "\033[1;31mFAIL\033[0m (toggle)";
                            }
                            test_ok = false;
                        }
                    }

                    if (llama_arch_supports_save(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        test_executed = true;
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                        //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                        const std::string path = get_tmp_gguf_path();
                        llama_model_save_to_file(model_and_ctx_dev.first.get(), path.c_str());
                        FILE * file = fopen(path.c_str(), "rb");
                        GGML_ASSERT(file != nullptr);

                        {
                            auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size);
                            const std::vector<float> logits_roundtrip = get_logits(
                                model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                            status_roundtrip = "\033[1;32mOK\033[0m";
                            GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                            for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                                if (logits_roundtrip[i] != logits_dev[i]) {
                                    test_ok = false;
                                    status_roundtrip = "\033[1;31mFAIL\033[0m";
                                    break;
                                }
                            }
                        }

                        fclose(file);
                        std::filesystem::remove(path);
                    }
                }

                if (test_executed) {
                    n_tests++;
                    if (!test_ok) {
                        n_failed++;
                        all_ok = false;
                    }
                }

                // log the results for this test case
                LOG(template_row_res.c_str(), status_nmse.c_str(), nmse_str, status_roundtrip.c_str(), status_mixed.c_str(), mixed_str);
            }
        }
    }

    n_tests++;
    const bool capacity_ok = check_mtp_chain_capacity();
    n_failed += !capacity_ok;
    all_ok &= capacity_ok;

    if (arch_matches(arch_filter, LLM_ARCH_QWEN35)) {
        n_tests++;
        const bool ok = check_gdn_getter_arch(seed, stdev);
        n_failed += !ok;
        all_ok &= ok;
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return all_ok ? 0 : 1;
}

// 64 layers expose the cold ancestor walk after alternating per-sequence sampler branches.
static int test_tp_sampling_rebuild(bool single) {
    ggml_backend_load_all();
    auto gf = get_gguf_ctx(LLM_ARCH_QWEN35, false);
    gguf_set_val_u32(gf.get(), "qwen35.block_count", 64);
    gguf_set_val_u32(gf.get(), "qwen35.full_attention_interval", 4);
    tensor_data_params td = {1, 0.01f};
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_dev_t devs[] = {cpu, cpu, nullptr};
    auto mp = llama_model_default_params();
    mp.devices = devs;
    mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    mp.progress_callback = silent_model_load_progress;
    llama_model_ptr model(llama_model_init_from_user(gf.get(), set_tensor_data, &td, mp));
    GGML_ASSERT(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = cp.n_ubatch = 16;
    cp.n_seq_max = 2;
    cp.n_threads = cp.n_threads_batch = 4;
    llama_sampler_ptr chains[2];
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    GGML_ASSERT(ctx);
    for (int seq = 0; seq < (single ? 1 : 2); seq++) {
        chains[seq].reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(chains[seq].get(), llama_sampler_init_top_k(20));
        llama_sampler_chain_add(chains[seq].get(), llama_sampler_init_dist(1234 + seq));
        GGML_ASSERT(llama_set_sampler(ctx.get(), seq, chains[seq].get()));
    }
    common_batch batch(ctx.get());
    for (int step = 0; step < 4; step++) {
        const int seq = single ? 0 : step % 2;
        batch.clear();
        for (int i = 0; i < 4; i++) {
            batch.add(1 + i, (single ? step : step / 2)*4 + i, seq, i == 3);
        }
        fprintf(stderr, "rebuild step %d seq %d\n", step, seq);
        GGML_ASSERT(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
        const llama_token token = llama_get_sampled_token_ith(ctx.get(), 3);
        GGML_ASSERT(token >= 0 && token < llama_vocab_n_tokens(llama_model_get_vocab(model.get())));
    }
    return 0;
}

// Backend sampling under SPLIT_MODE_TENSOR against host sampling of the same logits. Each check runs a backend chain
// on a tensor-split context and compares it with the logits that a twin context on the same model (no backend
// sampler) copies to the host. The TP forward pass is identical in both, so any difference comes from the sampling
// graph: the per-device top-k, the vocab id mapping, the candidate gather, the row layout.
//
// Covered: ubatches without output rows (MTP catch-up, split prompt), several output rows (draft batches, the k+1
// verify), the draft chain (top-k only), greedy, the server's default chain (top-k, top-p < 1, min-p, temp, dist)
// and the distribution of dist.
//
// gpu = false runs on two CPU devices, which exchange the candidates through the meta backend's exact generic
// AllReduce. gpu = true runs on the first two GPUs, where the backend's own AllReduce is used; that is the only way
// to cover a lossy exchange (the CUDA internal AllReduce rounds F32 through BF16).
static bool test_banned_candidates(const llama_model * model) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    common_params_sampling params;
    params.top_k = 4;
    params.temp = 1.0f;
    params.seed = 42;
    params.logit_bias = {{0, -INFINITY}, {2, -INFINITY}};
    params.backend_sampling = true;
    common_sampler_ptr cand(common_sampler_init(model, params));
    params.backend_sampling = false;
    common_sampler_ptr host(common_sampler_init(model, params));
    std::vector<float> logits(n_vocab);
    for (int32_t i = 0; i < n_vocab; ++i) {
        logits[i] = -float(i);
    }
    std::vector<llama_token_data> backend_row(n_vocab);
    for (int32_t i = 0; i < n_vocab; ++i) {
        backend_row[i] = {i, logits[i], 0.0f};
    }
    llama_token_data_array backend = {backend_row.data(), backend_row.size(), -1, false};
    llama_sampler_apply(common_sampler_get_backend(cand.get()), &backend);
    if (backend.size != 7) {
        return false;
    }
    llama_token ids[] = {6, 5, 4, 3, 2, 1, 0};
    float vals[] = {-6, -5, -4, -3, -2, -1, 0};
    for (bool tied : {false, true}) {
        // The fourth and fifth unbanned logits tie only in the fallback case.
        logits[6] = vals[0] = tied ? -5.0f : -6.0f;
        int full_reads = 0;
        common_sampler_row row;
        row.n_vocab = n_vocab;
        row.cand_ids = ids;
        row.cand_logits = vals;
        row.n_cand = 7;
        row.logits = [&]() { ++full_reads; return logits.data(); };
        const llama_token got = common_sampler_sample_row(cand.get(), row);
        if (full_reads != int(tied)) {
            return false;
        }
        row.cand_ids = nullptr;
        row.cand_logits = nullptr;
        row.n_cand = 0;
        const llama_token want = common_sampler_sample_row(host.get(), row);
        const auto * a = common_sampler_get_candidates(cand.get(), false);
        const auto * b = common_sampler_get_candidates(host.get(), false);
        if (got != want || got == 0 || got == 2 || a->size != b->size || a->selected != b->selected) {
            return false;
        }
        for (size_t i = 0; i < a->size; ++i) {
            if (a->data[i].id != b->data[i].id || a->data[i].logit != b->data[i].logit || a->data[i].p != b->data[i].p) {
                return false;
            }
        }
    }
    return true;
}

static int test_tp_sampling(const size_t seed, const float stdev, const bool gpu) {
    std::vector<ggml_backend_dev_t> devs;
    for (size_t i = 0; i < ggml_backend_dev_count() && devs.size() < 2; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == (gpu ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU)) {
            devs.push_back(dev);
        }
    }
    if (!gpu && devs.size() == 1) {
        devs.push_back(devs[0]);
    }
    if (devs.size() < 2) {
        LOG_ERR("%s: need two %s devices, found %zu\n", __func__, gpu ? "GPU" : "CPU", devs.size());
        return 1;
    }
    LOG("%s: devices %s, %s\n", __func__, ggml_backend_dev_name(devs[0]), ggml_backend_dev_name(devs[1]));

    int n_reserves = 0;
    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        if (std::strstr(text, "sched_reserve: reserving ...")) {
            ++*static_cast<int *>(user_data);
        }
        if (level >= GGML_LOG_LEVEL_WARN) {
            fputs(text, stderr);
        }
    }, &n_reserves);

    // ids up to 40959 have up to 16 significant bits: a BF16 (8-bit) exchange of the candidate ids changes most of
    // them. With the default 128-token test vocab every id is exact in BF16 and such a bug goes unseen.
    const uint32_t n_vocab_test = 40960;
    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_LLAMA, false, n_vocab_test);
    tensor_data_params tensor_params = { seed, stdev };

    std::vector<ggml_backend_dev_t> devs_model = devs;
    devs_model.push_back(nullptr);
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.devices    = devs_model.data();
    model_params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_params, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    GGML_ASSERT(n_vocab == n_vocab_test);

    const std::vector<llama_token> tokens = get_tokens(48, n_vocab, seed);
    const int32_t n_rows = 4;                     // output rows at the end of the last ubatch
    const int32_t i_row0 = 32 - n_rows;           // batch index of the first output row in the second batch

    auto make_ctx = [&](llama_sampler * chain) {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx           = 256;
        ctx_params.n_batch         = 64;
        ctx_params.n_ubatch        = 16;
        ctx_params.n_threads       = 4;
        ctx_params.n_threads_batch = 4;
        ctx_params.n_outputs_max_per_seq = n_rows;
        llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
        if (!lctx) {
            throw std::runtime_error("failed to create llama context");
        }
        if (chain && !llama_set_sampler(lctx.get(), 0, chain)) {
            throw std::runtime_error("backend sampler refused under SPLIT_MODE_TENSOR");
        }
        return lctx;
    };


    // [0, 16): one ubatch, no outputs; [16, 48): a ubatch without outputs, then one ending in n_rows output rows
    auto decode = [&](llama_context * lctx, bool split_outputs = false) {
        common_batch batch(lctx);
        for (uint32_t pos = 0; pos < 16; pos++) {
            batch.add(tokens[pos], pos, 0, false);
        }
        int rc = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        batch.clear();
        for (uint32_t pos = 16; pos < tokens.size() && rc == 0; pos++) {
            const bool output = split_outputs ? pos == 30 || pos == 31 || pos == 46 || pos == 47 : pos + n_rows >= tokens.size();
            batch.add(tokens[pos], pos, 0, output);
        }
        rc = rc == 0 ? llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) : rc;
        if (rc != 0) {
            throw std::runtime_error("failed to decode batch");
        }
    };

    // host reference: the same TP forward pass, full logits copied out
    std::vector<std::vector<float>> ref(n_rows);
    {
        llama_context_ptr ctx_ref = make_ctx(nullptr);
        decode(ctx_ref.get());
        for (int32_t r = 0; r < n_rows; r++) {
            const float * logits = llama_get_logits_ith(ctx_ref.get(), i_row0 + r);
            ref[r].assign(logits, logits + n_vocab);
        }
    }
    // the k-th largest logit of a row
    auto kth = [&](const std::vector<float> & row, int32_t k) {
        std::vector<float> tmp = row;
        std::nth_element(tmp.begin(), tmp.begin() + (k - 1), tmp.end(), std::greater<float>());
        return tmp[k - 1];
    };
    // ids and logits must match the host logits; eps only admits ties at the k-th logit
    auto close = [](float a, float b) { return std::fabs(a - b) <= 1e-4f*(1.0f + std::fabs(b)); };
    auto valid_id = [&](llama_token id) { return id >= 0 && uint32_t(id) < n_vocab; };

    bool ok = true;
    auto fail = [&](const char * what, int32_t r, llama_token id, float got, float want) {
        LOG_ERR("%s: %s: row %d: id %d, logit %f, want %f\n", __func__, what, r, id, got, want);
        ok = false;
    };

    const int32_t k = 20;

    {
        common_params_sampling params;
        params.seed = seed;
        params.top_k = k;
        params.top_p = 0.95f;
        params.min_p = 0.05f;
        params.temp = 1.0f;
        params.backend_sampling = true;
        common_sampler_ptr backend(common_sampler_init(model.get(), params));
        llama_context_ptr ctx = make_ctx(common_sampler_get_backend(backend.get()));
        decode(ctx.get());
        const int n_reserves_before = n_reserves;
        llama_set_sampler(ctx.get(), 0, nullptr);
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        decode(ctx.get());
        if (n_reserves != n_reserves_before || llama_get_sampled_candidates_count_ith(ctx.get(), -1) != 0) {
            throw std::runtime_error("sampler detach rebuilt the scheduler or retained candidates");
        }
        params.top_k = k - 3;
        common_sampler_ptr replacement(common_sampler_init(model.get(), params));
        if (!llama_set_sampler(ctx.get(), 0, common_sampler_get_backend(replacement.get()))) {
            throw std::runtime_error("replacement backend sampler refused");
        }
        backend = std::move(replacement);
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        decode(ctx.get());
        if (n_reserves != n_reserves_before) {
            throw std::runtime_error("sampler detach and replacement rebuilt the scheduler");
        }
        LOG("%s: sampler detach and replacement retains the scheduler: OK\n", __func__);
        for (int32_t r = 0; r < n_rows; ++r) {
            const uint32_t count = llama_get_sampled_candidates_count_ith(ctx.get(), i_row0 + r);
            if (count != uint32_t(params.top_k + 1) || llama_get_sampled_token_ith(ctx.get(), i_row0 + r) != LLAMA_TOKEN_NULL) {
                throw std::runtime_error("verify candidate chain did not return k+1 logits without a draw");
            }
            for (uint32_t s = 0; s < 64; ++s) {
                params.seed = s;
                common_sampler_ptr candidate(common_sampler_init(model.get(), params));
                params.backend_sampling = false;
                common_sampler_ptr host(common_sampler_init(model.get(), params));
                params.backend_sampling = true;
                common_sampler_row row;
                row.n_vocab = n_vocab;
                row.logits = [&]() { return ref[r].data(); };
                const llama_token want = common_sampler_sample_row(host.get(), row);
                const llama_token got = common_sampler_sample(candidate.get(), ctx.get(), i_row0 + r);
                if (got != want) {
                    fail("verify candidate seed identity", r, got, got, want);
                }
            }
            const float * full = llama_get_logits_ith(ctx.get(), i_row0 + r);
            if (std::memcmp(full, ref[r].data(), n_vocab*sizeof(float)) != 0) {
                for (uint32_t t = 0; t < n_vocab; ++t) {
                    if (full[t] != ref[r][t]) {
                        LOG_ERR("deferred row %d id %u: got %.9g want %.9g\n", r, t, full[t], ref[r][t]);
                        break;
                    }
                }
                throw std::runtime_error("deferred TP logits differ from the full-row reference");
            }
        }
        LOG("%s: verify candidates and deferred logits, 64 seeds x %d rows: %s\n", __func__, n_rows, ok ? "OK" : "FAIL");
    }

    if (ok && !gpu) {
        ok = test_banned_candidates(model.get());
    }

    {
        const int32_t rows[] = { 14, 15, 30, 31 };
        llama_context_ptr host = make_ctx(nullptr);
        decode(host.get(), true);
        common_params_sampling params;
        params.seed = seed;
        params.top_k = k;
        params.backend_sampling = true;
        common_sampler_ptr candidates(common_sampler_init(model.get(), params));
        llama_context_ptr ctx = make_ctx(common_sampler_get_backend(candidates.get()));
        decode(ctx.get(), true);
        for (const auto row : rows) {
            const float * want = llama_get_logits_ith(host.get(), row);
            const float * got = llama_get_logits_ith(ctx.get(), row);
            if (std::memcmp(got, want, n_vocab*sizeof(float)) != 0) {
                throw std::runtime_error("fallback logits spanning two ubatches differ");
            }
        }
        LOG("%s: fallback logits across two ubatches: OK\n", __func__);
    }

    // the MTP draft chain: top-k on the backend, the host finishes from the candidates
    {
        llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(k));
        llama_context_ptr ctx = make_ctx(chain.get());
        decode(ctx.get());
        for (int32_t r = 0; r < n_rows && ok; r++) {
            const uint32_t      n_got  = llama_get_sampled_candidates_count_ith(ctx.get(), i_row0 + r);
            const llama_token * ids    = llama_get_sampled_candidates_ith(ctx.get(), i_row0 + r);
            const float       * values = llama_get_sampled_logits_ith(ctx.get(), i_row0 + r);
            if (n_got != uint32_t(k) || ids == nullptr || values == nullptr ||
                    llama_get_sampled_logits_count_ith(ctx.get(), i_row0 + r) != uint32_t(k)) {
                LOG_ERR("%s: top-k: row %d: expected %d candidates, got %u\n", __func__, r, k, n_got);
                ok = false;
                break;
            }
            const float thr = kth(ref[r], k);
            std::vector<llama_token> seen(ids, ids + k);
            std::sort(seen.begin(), seen.end());
            if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) {
                LOG_ERR("%s: top-k: row %d: duplicate candidate ids\n", __func__, r);
                ok = false;
            }
            for (int32_t i = 0; i < k && ok; i++) {
                if (!valid_id(ids[i])) {
                    fail("top-k: id out of range", r, ids[i], values[i], NAN);
                } else if (!close(values[i], ref[r][ids[i]])) {
                    fail("top-k: candidate logit differs from the host logit of its id", r, ids[i], values[i], ref[r][ids[i]]);
                } else if (ref[r][ids[i]] < thr && !close(ref[r][ids[i]], thr)) {
                    fail("top-k: candidate is not in the host top-k", r, ids[i], ref[r][ids[i]], thr);
                }
            }
        }
        LOG("%s: top-k candidates, %d rows: %s\n", __func__, n_rows, ok ? "OK" : "FAIL");
    }

    // greedy after top-k: the argmax indexes the candidates and must be mapped back to a vocab id
    if (ok) {
        llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(k));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
        llama_context_ptr ctx = make_ctx(chain.get());
        decode(ctx.get());
        for (int32_t r = 0; r < n_rows && ok; r++) {
            const llama_token id   = llama_get_sampled_token_ith(ctx.get(), i_row0 + r);
            const float       best = kth(ref[r], 1);
            if (!valid_id(id) || !close(ref[r][id], best)) {
                fail("greedy: sampled token is not the host argmax", r, id, valid_id(id) ? ref[r][id] : NAN, best);
            }
        }
        LOG("%s: top-k + greedy, %d rows: %s\n", __func__, n_rows, ok ? "OK" : "FAIL");
    }

    // the server's default chain; top-p < 1 used to abort the graph build on the candidate shape
    if (ok) {
        llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(k));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(1.0f));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(uint32_t(seed)));
        llama_context_ptr ctx = make_ctx(chain.get());
        decode(ctx.get());
        for (int32_t r = 0; r < n_rows && ok; r++) {
            const llama_token id  = llama_get_sampled_token_ith(ctx.get(), i_row0 + r);
            const float       thr = kth(ref[r], k);
            if (!valid_id(id) || (ref[r][id] < thr && !close(ref[r][id], thr))) {
                fail("top-k/top-p/min-p/temp/dist: sampled token is not in the host top-k", r, id, valid_id(id) ? ref[r][id] : NAN, thr);
            }
        }
        LOG("%s: top-k + top-p + min-p + temp + dist, %d rows: %s\n", __func__, n_rows, ok ? "OK" : "FAIL");
    }

    // distribution of dist over the candidates of one row, against the host softmax over its top-k
    if (ok) {
        const float temp    = 1.0f;
        const int   n_draws = 4000;

        llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(k));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(temp));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(uint32_t(seed)));
        llama_context_ptr ctx = make_ctx(chain.get());
        decode(ctx.get());

        const std::vector<float> & row = ref[n_rows - 1];
        const float thr  = kth(row, k);
        const float best = kth(row, 1);
        std::vector<double> want(n_vocab, 0.0);
        double sum = 0.0;
        for (uint32_t i = 0; i < n_vocab; i++) {
            if (row[i] >= thr) {
                want[i] = std::exp(double(row[i] - best)/temp);
                sum += want[i];
            }
        }

        std::vector<int> count(n_vocab, 0);
        const llama_pos pos_last = llama_pos(tokens.size() - 1);
        common_batch batch(ctx.get());
        for (int d = 0; d < n_draws && ok; d++) {
            if (d > 0) {
                // re-decode the last token: same logits, a fresh uniform draw
                llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, pos_last, -1);
                batch.clear();
                batch.add(tokens.back(), pos_last, 0, true);
                if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                    throw std::runtime_error("failed to decode batch");
                }
            }
            const llama_token id = llama_get_sampled_token_ith(ctx.get(), d == 0 ? i_row0 + n_rows - 1 : 0);
            if (!valid_id(id) || want[id] == 0.0) {
                fail("dist: sampled token is not in the host top-k", n_rows - 1, id, valid_id(id) ? row[id] : NAN, thr);
                break;
            }
            count[id]++;
            // commit the draw so the next decode draws a new uniform
            llama_sampler_accept(chain.get(), id);
        }

        if (ok) {
            double tv = 0.0;
            for (uint32_t i = 0; i < n_vocab; i++) {
                tv += std::fabs(double(count[i])/n_draws - want[i]/sum);
            }
            tv *= 0.5;
            // sampling noise alone gives about 0.03 for k = 20 and 4000 draws
            if (tv > 0.08) {
                LOG_ERR("%s: dist: total variation distance %.3f to the host distribution over %d draws\n", __func__, tv, n_draws);
                ok = false;
            }
            LOG("%s: top-k + temp + dist distribution, %d draws, TV %.3f: %s\n", __func__, n_draws, tv, ok ? "OK" : "FAIL");
        }
    }

    // a device whose vocab slice is smaller than top-k: the sampler is refused (host sampling), not an abort
    if (ok) {
        const float split[2] = { 1.0f, 0.0004f };
        llama_model_params mp = model_params;
        mp.tensor_split = split;
        llama_model_ptr model_uneven(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_params, mp));
        GGML_ASSERT(model_uneven);
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 256;
        llama_context_ptr lctx(llama_init_from_model(model_uneven.get(), ctx_params));
        GGML_ASSERT(lctx);
        llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(k));
        const bool accepted = llama_set_sampler(lctx.get(), 0, chain.get());
        LOG("%s: vocab slice smaller than top-k: sampler %s\n", __func__, accepted ? "accepted (FAIL)" : "refused, OK");
        ok = !accepted;
    }

    llama_log_set(nullptr, nullptr);
    LOG("%s: %s\n", __func__, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

// KQ masks seen by attention must not depend on how the mask input reaches the device. A cache whose devices
// cannot read host memory in place (here: a meta device over two CPUs) builds the visible mask prefix in the
// graph and uploads only the rest; its masks must match byte for byte those of a plain CPU context.
struct kq_mask_capture {
    std::vector<uint8_t> bytes;  // the first attention mask of every graph compute, concatenated
    size_t n_masks   = 0;
    size_t n_compact = 0;        // masks whose prefix came from the graph
    bool   seen      = false;    // the current compute already captured its mask
};

static bool kq_mask_capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = (kq_mask_capture *) user_data;
    const bool is_attn = t->op == GGML_OP_FLASH_ATTN_EXT || t->op == GGML_OP_SOFT_MAX;
    if (ask) {
        return is_attn && !cap->seen;
    }
    if (is_attn && !cap->seen) {
        const ggml_tensor * mask = t->op == GGML_OP_FLASH_ATTN_EXT ? t->src[3] : t->src[1];
        GGML_ASSERT(mask != nullptr && ggml_is_contiguous(mask));
        const size_t off = cap->bytes.size();
        cap->bytes.resize(off + ggml_nbytes(mask));
        ggml_backend_tensor_get(mask, cap->bytes.data() + off, 0, ggml_nbytes(mask));
        cap->n_masks++;
        cap->n_compact += mask->op == GGML_OP_CONCAT;
        cap->seen = true;
    }
    return true;
}

static bool test_kq_mask_prefix(const size_t seed, const float stdev) {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu != nullptr);

    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_LLAMA, false);

    struct run_result {
        kq_mask_capture   cap;
        std::vector<float> logits;
    };

    auto run = [&](const std::vector<ggml_backend_dev_t> & devs, llama_split_mode split_mode, bool flash_attn) {
        run_result r;

        llama_model_params mparams = llama_model_default_params();
        mparams.progress_callback = silent_model_load_progress;
        std::vector<ggml_backend_dev_t> devs_copy = devs;
        devs_copy.push_back(nullptr);
        mparams.devices    = devs_copy.data();
        mparams.split_mode = split_mode;

        tensor_data_params tparams = { seed, stdev };
        llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tparams, mparams));
        if (!model) {
            throw std::runtime_error("failed to create llama model");
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx             = 1024;
        cparams.n_ubatch          = 64;
        cparams.n_seq_max         = 2;
        cparams.kv_unified        = true;
        cparams.n_threads         = 4;
        cparams.n_threads_batch   = 4;
        cparams.flash_attn_type   = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cparams.cb_eval           = kq_mask_capture_cb;
        cparams.cb_eval_user_data = &r.cap;
        llama_context_ptr lctx(llama_init_from_model(model.get(), cparams));
        if (!lctx) {
            throw std::runtime_error("failed to create llama context");
        }

        llama_memory_t mem = llama_get_memory(lctx.get());
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        const std::vector<llama_token> tokens = get_tokens(1024, n_vocab, seed);

        llama_pos n_past = 0;
        auto decode = [&](uint32_t n) {
            common_batch batch(lctx.get());
            for (uint32_t i = 0; i < n; i++) {
                batch.add(tokens[n_past], n_past, 0, true);
                n_past++;
            }
            r.cap.seen = false;
            if (llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                throw std::runtime_error("failed to decode batch");
            }
            for (uint32_t i = 0; i < n; i++) {
                const float * logits = llama_get_logits_ith(lctx.get(), i);
                r.logits.insert(r.logits.end(), logits, logits + n_vocab);
            }
        };

        decode(64*9 + 24);          // prefill ubatches, the later ones past one padding block
        decode(1);
        decode(4);
        decode(4);                  // same shape: graph reuse with a grown prefix
        llama_memory_seq_rm(mem, 0, n_past - 2, -1);
        n_past -= 2;
        decode(4);                  // rewind as after rejected draft tokens
        llama_memory_seq_cp(mem, 0, 1, 100, 101);
        llama_memory_seq_rm(mem, 0, 100, 101);
        decode(4);                  // a cell of another sequence inside the prefix
        decode(1);
        llama_memory_seq_rm(mem, 1, -1, -1);
        decode(1);                  // fills the freed cell with a late position
        decode(4);
        llama_memory_seq_rm(mem, 0, 200, 210);
        decode(4);                  // holes inside the prefix
        decode(1);
        decode(6);
        decode(64*3);               // more prefill, until the holes are filled again
        decode(4);
        decode(1);
        return r;
    };

    // the tensor split requires flash attention
    const run_result ref = run({cpu}, LLAMA_SPLIT_MODE_LAYER, true);
    const run_result dut = run({cpu, cpu}, LLAMA_SPLIT_MODE_TENSOR, true);

    const bool same_masks = ref.cap.n_masks == dut.cap.n_masks && ref.cap.bytes == dut.cap.bytes;
    const double nmse_val = nmse(ref.logits, dut.logits);
    // LOCAL: the serving gates (prefix >= 32768) never compact at this context size
    const bool ok = same_masks && ref.cap.n_compact == 0 && nmse_val <= 1e-4;
    LOG("kq mask prefix: masks ref %zu dut %zu, %s, built in graph ref %zu dut %zu, logits NMSE %.2e: %s\n",
        ref.cap.n_masks, dut.cap.n_masks, same_masks ? "identical" : "DIFFERENT",
        ref.cap.n_compact, dut.cap.n_compact, nmse_val, ok ? "OK" : "FAIL");
    return ok;
}

int main(int argc, char ** argv) {
    // init the logger at max verbosity. filter with a custom callback respecting the user-configure verbosity
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    common_init();
    llama_backend_init();

    std::random_device rd;

    std::string arch_filter;
    size_t seed = rd();
    float stdev = 0.1f;
    std::string out;
    const char * target_backend = nullptr;
    bool tp_rebuild = false;
    bool tp_rebuild_single = false;
    bool tp_sampling = false;
    bool tp_sampling_gpu = false;

    int verbosity = LOG_LEVEL_ERROR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                if (llama_arch_from_string(arch_name) != LLM_ARCH_UNKNOWN) {
                    // exact architecture name
                    arch_filter = "^" + arch_name + "$";
                } else {
                    try {
                        std::regex re(arch_name);
                        arch_filter = arch_name;
                    } catch (const std::regex_error & err) {
                        LOG_ERR("%s: invalid architecture regex: %s (%s)\n", __func__, arch_name.c_str(), err.what());
                        return 1;
                    }
                }
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--tp-sampling-rebuild") == 0) {
            tp_rebuild = true;
        } else if (strcmp(argv[i], "--tp-sampling-rebuild-single") == 0) {
            tp_rebuild = tp_rebuild_single = true;
        } else if (strcmp(argv[i], "--tp-sampling") == 0) {
            tp_sampling = true;
        } else if (strcmp(argv[i], "--tp-sampling-gpu") == 0) {
            tp_sampling = true;
            tp_sampling_gpu = true;
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--stdev") == 0) {
            if (i + 1 < argc) {
                stdev = std::stof(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            if (i + 1 < argc) {
                verbosity = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--backend") == 0) {
            if (i + 1 < argc) {
                const char * backend_name = argv[++i];
                ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name);
                if (dev == nullptr) {
                    LOG_ERR("%s: unknown backend device: %s\n", __func__, backend_name);
                    return 1;
                }
                target_backend = ggml_backend_dev_name(dev);
            } else {
                usage(argv);
                return 1;
            }
        } else {
            LOG_ERR("%s: unknown argument: %s\n", __func__, argv[i]);
            usage(argv);
            return 1;
        }
    }
    if (stdev <= 0.0f) {
        LOG_ERR("%s: stdev must be > 0\n", __func__);
        return 1;
    }
    LOG_INF("%s: using seed %zu, stdev %f\n", __func__, seed, stdev);

    try {
        if (tp_rebuild) {
            return test_tp_sampling_rebuild(tp_rebuild_single);
        }
        if (tp_sampling) {
            return test_tp_sampling(seed, stdev, tp_sampling_gpu);
        }
        if (!out.empty()) {
            return save_models(arch_filter, seed, stdev, verbosity, out);
        }
        int ret = test_backends(arch_filter, seed, stdev, verbosity, target_backend);
        if (target_backend == nullptr && arch_matches(arch_filter, LLM_ARCH_LLAMA) && !test_kq_mask_prefix(seed, stdev)) {
            ret = 1;
        }
        return ret;
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
