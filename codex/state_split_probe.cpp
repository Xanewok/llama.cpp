#include "common.h"
#include "llama-cpp.h"
#include "llama-context.h"
#include "llama-memory-hybrid.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * msg) {
    if (!ok) throw std::runtime_error(msg);
}
static int setting(const char * key, int fallback) {
    const char * value = std::getenv(key);
    return value ? std::stoi(value) : fallback;
}
static void decode(llama_context * ctx, const llama_tokens & tokens, int pos, bool all = true) {
    llama_batch b = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) common_batch_add(b, tokens[i], pos + i, {0}, all || i + 1 == tokens.size());
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    check(rc == 0, "decode failed");
}
using floats = std::vector<float>;
static floats logits(llama_context * ctx, int row, int nv) {
    const float * p = llama_get_logits_ith(ctx, row);
    check(p != nullptr, "missing logits");
    return {p, p + nv};
}
static bool bit_equal(const floats & a, const floats & b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}
static double nmse(const floats & a, const floats & b) {
    check(a.size() == b.size(), "state shape mismatch");
    double err = 0, norm = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        check(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite state/logits");
        const double d = double(a[i]) - b[i];
        err += d*d;
        norm += double(b[i])*b[i];
    }
    return err / std::max(norm, 1e-300);
}
static llama_memory_recurrent * recurrent(llama_context * ctx) {
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(ctx->get_memory());
    check(hybrid != nullptr, "expected hybrid memory");
    return hybrid->get_mem_recr();
}
static floats state(llama_context * ctx, bool conv, int slot) {
    llama_synchronize(ctx);
    auto * mem = recurrent(ctx);
    const int tail = mem->cells[0].tail;
    check(tail >= 0, "missing recurrent tail");
    floats result;
    for (auto * tensor : conv ? mem->r_l : mem->s_l) {
        if (!tensor) continue;
        check(tensor->type == GGML_TYPE_F32, "expected f32 recurrent state");
        const size_t count = tensor->ne[0];
        const size_t old = result.size();
        result.resize(old + count);
        ggml_backend_tensor_get(tensor, result.data() + old, (slot * mem->size + tail) * tensor->nb[1], count * sizeof(float));
    }
    return result;
}
static std::vector<uint8_t> save(llama_context * ctx) {
    std::vector<uint8_t> bytes(llama_state_seq_get_size(ctx, 0));
    check(llama_state_seq_get_data(ctx, bytes.data(), bytes.size(), 0) == bytes.size(), "state save failed");
    return bytes;
}
static void restore(llama_context * ctx, const std::vector<uint8_t> & bytes) {
    check(llama_state_seq_set_data(ctx, bytes.data(), bytes.size(), 0) == bytes.size(), "state restore failed");
}
static void trim(llama_context * ctx, int end) {
    check(llama_memory_seq_rm(llama_get_memory(ctx), 0, end, -1), "rollback failed");
    check(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == end - 1, "wrong retained position");
}
static void artifact(const floats & row, const std::string & name) {
    const char * dir = std::getenv("E_LOGITS_DIR");
    if (!dir) return;
    std::ofstream out(std::string(dir) + "/" + name + ".f32", std::ios::binary);
    out.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(float));
    check(out.good(), "artifact write failed");
}
int main(int argc, char ** argv) {
    try {
        check(argc >= 2, "usage: state-split-probe MODEL [gpu]");
        bool gpu = false;
#ifdef E_PARENT_GPU
        gpu = argc > 2 && std::string(argv[2]) == "gpu";
#endif
        const char * visible = std::getenv("CUDA_VISIBLE_DEVICES");
        check(visible && (gpu ? *visible != '\0' : *visible == '\0'), "device visibility guard failed");
        llama_log_set([](ggml_log_level level, const char * msg, void *) {
            if (level >= GGML_LOG_LEVEL_WARN) std::fputs(msg, stderr);
        }, nullptr);
        check(setting("E_K", 0) >= 0 && setting("E_K", 0) <= 6, "invalid depth");
        check(setting("E_R", 0) >= 0 && setting("E_R", 0) <= 6, "invalid rollback distance");
        check(setting("E_THREADS", 8) >= 1 && setting("E_THREADS", 8) <= 8, "thread budget exceeded");
        check(setting("E_PREFIX", 0) >= 0 && setting("E_PREFIX", 0) <= 2048, "invalid prefix length");
        check(setting("E_FOLLOW", 0) >= 0 && setting("E_FOLLOW", 0) <= 200, "continuation budget exceeded");
        check(setting("E_UBATCH", 128) > 7, "ubatch must hold the snapshot tail");
        check(!bit_equal({0.0f}, {-0.0f}) && nmse({0.0f}, {-0.0f}) == 0, "signed-zero byte oracle failed");
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mtp = true;
        ggml_backend_dev_t devices[] = {nullptr, nullptr, nullptr};
        std::vector<float> split(llama_max_devices(), 0.0f);
        if (gpu) {
            int n = 0;
            for (size_t i = 0; i < ggml_backend_dev_count() && n < 2; ++i) {
                auto * d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) devices[n++] = d;
            }
            check(n == 2, "expected two GPUs");
            mp.devices = devices;
            mp.n_gpu_layers = 999;
            mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
            split[0] = split[1] = 0.5f;
            mp.tensor_split = split.data();
        }
        llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
        check(bool(model), "model load failed");
        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        const auto motif = common_tokenize(llama_model_get_vocab(model.get()), "A short CPU rollback fixture: solve SEND plus MORE equals MONEY.", true, false);
        std::vector<int> prefixes;
        if (setting("E_PREFIX", 0)) prefixes.push_back(setting("E_PREFIX", 0));
        else {
            for (int n = 120; n <= 140; ++n) prefixes.push_back(n);
            for (int n = 250; n <= 260; ++n) prefixes.push_back(n);
        }
        for (int k = 1; k <= 6; ++k) {
            if (setting("E_K", 0) && k != setting("E_K", 0)) continue;
            auto cp = llama_context_default_params();
            cp.n_ctx = std::max(1024, *std::max_element(prefixes.begin(), prefixes.end()) + 256);
            cp.n_batch = 2048;
            cp.n_ubatch = setting("E_UBATCH", 128);
            cp.n_rs_seq = k;
            cp.n_outputs_max_per_seq = 7;
            cp.n_threads = cp.n_threads_batch = gpu ? 4 : setting("E_THREADS", 8);
            cp.type_k = cp.type_v = gpu && !setting("E_F16", 0) ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
            cp.flash_attn_type = gpu || setting("E_FA", 0) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
            llama_context_ptr roll(llama_init_from_model(model.get(), cp));
            llama_context_ptr ref(llama_init_from_model(model.get(), cp));
            llama_context_ptr matched(llama_init_from_model(model.get(), cp));
            check(roll && ref && matched, "context load failed");
            for (int prefix : prefixes) {
                llama_memory_clear(llama_get_memory(roll.get()), true);
                llama_tokens prompt(prefix);
                for (int i = 0; i < prefix; ++i) prompt[i] = motif[i % motif.size()];
                decode(roll.get(), prompt, 0, false);
                const auto base = save(roll.get());
                for (int r = 1; r <= k; ++r) {
                    if (setting("E_R", 0) && r != setting("E_R", 0)) continue;
                    const int keep = k + 1 - r;
                    restore(roll.get(), base);
                    restore(ref.get(), base);
                    restore(matched.get(), base);
                    decode(roll.get(), llama_tokens(k + 1, 100), prefix);
                    const auto before = logits(roll.get(), keep - 1, nv);
                    trim(roll.get(), prefix + keep);
                    check(recurrent(roll.get())->rs_idx[0] == unsigned(r), "wrong snapshot index");
                    if (setting("E_BROKEN_SLOT", 0)) recurrent(roll.get())->rs_idx[0] = r - 1;
                    decode(ref.get(), llama_tokens(keep, 100), prefix);
                    const auto before_ref = logits(ref.get(), keep - 1, nv);
                    const auto conv_roll = state(roll.get(), true, recurrent(roll.get())->rs_idx[0]);
                    const auto gdn_roll = state(roll.get(), false, recurrent(roll.get())->rs_idx[0]);
                    const auto conv_ref = state(ref.get(), true, 0), gdn_ref = state(ref.get(), false, 0);
                    // This shape control uses distance one, independently of the tested distance.
                    decode(matched.get(), llama_tokens(keep + 1, 100), prefix);
                    const auto before_matched = logits(matched.get(), keep - 1, nv);
                    trim(matched.get(), prefix + keep);
                    const auto conv_matched = state(matched.get(), true, 1), gdn_matched = state(matched.get(), false, 1);
                    const double gdn_error = nmse(gdn_roll, gdn_ref);
                    const double matched_gdn = nmse(gdn_roll, gdn_matched);
                    decode(roll.get(), {100}, prefix + keep);
                    decode(ref.get(), {100}, prefix + keep);
                    decode(matched.get(), {100}, prefix + keep);
                    check(recurrent(roll.get())->rs_idx[0] == 0, "rollback index not consumed");
                    auto got = logits(roll.get(), 0, nv), want = logits(ref.get(), 0, nv), same = logits(matched.get(), 0, nv);
                    const int serialized_equal = setting("E_SERIALIZED", 0) ? int(save(roll.get()) == save(ref.get())) : -1;
                    if (setting("E_EXACT", 0) && setting("E_SERIALIZED", 0)) check(serialized_equal == 1, "whole logical state mismatch");
                    const std::string id = "p" + std::to_string(prefix) + "-k" + std::to_string(k) + "-r" + std::to_string(r);
                    artifact(got, id + "-roll"); artifact(want, id + "-ref"); artifact(same, id + "-matched");
                    std::printf("M: case prefix=%d k=%d r=%d keep=%d pre_nmse=%.9g conv_nmse=%.9g gdn_nmse=%.9g matched_pre_nmse=%.9g matched_conv_nmse=%.9g matched_gdn_nmse=%.9g post_nmse=%.9g matched_post_nmse=%.9g bit_equal=%d state_equal=%d serialized_equal=%d\n", prefix, k, r, keep, nmse(before, before_ref), nmse(conv_roll, conv_ref), gdn_error, nmse(before, before_matched), nmse(conv_roll, conv_matched), matched_gdn, nmse(got, want), nmse(got, same), bit_equal(got, want), bit_equal(conv_roll, conv_ref) && bit_equal(gdn_roll, gdn_ref), serialized_equal);
                    std::fflush(stdout);
                    if (setting("E_EXACT", 0)) check(bit_equal(got, want) && bit_equal(conv_roll, conv_ref) && bit_equal(gdn_roll, gdn_ref), "deterministic rollback mismatch");
                    if (setting("E_BROKEN_SLOT", 0)) check(matched_gdn == 0 && bit_equal(got, same), "negative control: wrong snapshot detected");
                    const int follow = setting("E_FOLLOW", 0);
                    double max_error = 0, max_matched = 0;
                    for (int t = 1; t <= follow; ++t) {
                        const llama_token tok = motif[(t * 7) % motif.size()];
                        decode(roll.get(), {tok}, prefix + keep + t);
                        decode(ref.get(), {tok}, prefix + keep + t);
                        decode(matched.get(), {tok}, prefix + keep + t);
                        got = logits(roll.get(), 0, nv); want = logits(ref.get(), 0, nv); same = logits(matched.get(), 0, nv);
                        max_error = std::max(max_error, nmse(got, want));
                        max_matched = std::max(max_matched, nmse(got, same));
                        if (setting("E_EXACT", 0)) check(bit_equal(got, want), "deterministic continuation mismatch");
                        if (t == 1 || t == 50 || t == 100 || t == 200 || t == follow) {
                            std::printf("M: follow prefix=%d k=%d r=%d step=%d nmse=%.9g matched_nmse=%.9g max_nmse=%.9g max_matched_nmse=%.9g bit_equal=%d\n", prefix, k, r, t, nmse(got, want), nmse(got, same), max_error, max_matched, bit_equal(got, want));
                            std::fflush(stdout);
                            artifact(got, id + "-follow" + std::to_string(t) + "-roll");
                            artifact(want, id + "-follow" + std::to_string(t) + "-ref");
                        }
                    }
                }
            }
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
