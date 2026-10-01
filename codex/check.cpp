#include "common.h"
#include "sampling.h"
#include "llama-cpp.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static llama_token correction = 100;

static common_params_sampling params(unsigned seed, bool backend = true) {
    common_params_sampling p;
    p.seed = seed;
    p.backend_sampling = backend;
    p.top_k = 20;
    p.top_p = 0.95f;
    p.min_p = 0.05f;
    p.temp = 1.0f;
    p.no_perf = true;
    return p;
}

struct row {
    std::vector<float> logits, values;
    std::vector<llama_token> ids;
    void select() {
        ids.resize(logits.size());
        std::iota(ids.begin(), ids.end(), 0);
        std::partial_sort(ids.begin(), ids.begin() + 21, ids.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        ids.resize(21);
        values.clear();
        for (int id : ids) values.push_back(logits[id]);
    }
    common_sampler_row get(bool candidates = true) {
        common_sampler_row r;
        r.n_vocab = logits.size();
        r.logits = [&] { return logits.data(); };
        if (candidates) {
            r.cand_ids = ids.data();
            r.cand_logits = values.data();
            r.n_cand = ids.size();
        }
        return r;
    }
};

static common_sampler * make_sampler(llama_model * model, common_params_sampling p) {
    return common_sampler_init(model, p);
}

static std::vector<llama_token_data> distribution(llama_model * model, row & r) {
    common_sampler_ptr s(make_sampler(model, params(7)));
    common_sampler_sample_row(s.get(), r.get());
    auto * p = common_sampler_get_candidates(s.get(), true);
    return {p->data, p->data + p->size};
}

static double chi_square(const std::vector<int> & counts, const std::vector<llama_token_data> & law, int n) {
    double chi = 0;
    int covered = 0;
    for (const auto & e : law) {
        const double expected = n * double(e.p);
        if (expected <= 0) continue;
        const double d = counts[e.id] - expected;
        chi += d*d/expected;
        covered += counts[e.id];
    }
    require(covered == n, "draw escaped target support");
    return chi;
}

static void law_check(llama_model * model, row & r, int k, bool subset, int reject_at, bool broken) {
    const auto p = distribution(model, r);
    require(p.size() > 2, "law fixture needs several target tokens");
    auto q = p;
    if (subset) q.resize(p.size()/2);
    if (!subset) {
        for (size_t i = 0; i < q.size(); ++i) q[i].p *= 1 + i%3;
    }
    double sum = 0;
    for (auto & e : q) sum += e.p;
    std::vector<double> weights;
    for (auto & e : q) {
        e.p /= sum;
        weights.push_back(e.p);
    }
    std::discrete_distribution<size_t> draw(weights.begin(), weights.end());
    std::mt19937 rng(413 + k*31 + subset*7 + reject_at);
    common_sampler_ptr s(make_sampler(model, params(917 + k*17 + subset*3 + reject_at)));
    std::vector<int> counts(r.logits.size());
    const int n = 20000;
    int rejected = 0, outside = 0;
    for (int trial = 0; trial < n; ++trial) {
        llama_tokens draft(k);
        std::vector<std::vector<llama_token_data>> qs(k, p);
        for (int i = 0; i < k; ++i) draft[i] = p[0].id;
        // Earlier rows use q=p, so the requested row is always reached.
        qs[reject_at] = q;
        draft[reject_at] = q[draw(rng)].id;
        if (broken || std::getenv("S4_CORRUPT_Q")) for (auto & e : qs[reject_at]) e.p *= 0.1f;
        auto out = common_sampler_sample_and_accept_n_rejection_rows(s.get(), [&](size_t) { return r.get(); }, draft, qs);
        ++counts[out[reject_at]];
        rejected += out.size() == size_t(reject_at + 1);
        outside += std::none_of(q.begin(), q.end(), [&](const auto & e) { return e.id == out[reject_at]; });
    }
    const double chi = chi_square(counts, p, n);
    std::printf("M: law k=%d subset=%d row=%d broken=%d n=%d df=%zu chi2=%.9g rejected=%d outside_q=%d\n", k, subset, reject_at, broken, n, p.size()-1, chi, rejected, outside);
    // A deliberately loose upper bound avoids treating an ordinary tail draw as a bug.
    require(broken ? chi > 200 : chi < 100, "law oracle failed");
    if (!broken) require(rejected > 0, "rejection not exercised");
    if (subset && !broken) require(outside > 0, "residual outside subset not exercised");
}

static void identities(llama_model * model, row & r, int k, bool subset) {
    auto q = distribution(model, r);
    if (subset) q.resize(q.size()/2);
    float sum = 0;
    for (auto e : q) sum += e.p;
    for (auto & e : q) e.p /= sum;
    for (unsigned seed = 0; seed < 256; ++seed) {
        common_sampler_ptr a(make_sampler(model, params(seed, false)));
        common_sampler_ptr b(make_sampler(model, params(seed, true)));
        llama_tokens draft(k, q[seed % q.size()].id);
        std::vector<std::vector<llama_token_data>> qs(k, q);
        auto want = common_sampler_sample_and_accept_n_rejection_rows(a.get(), [&](size_t) { return r.get(false); }, draft, qs);
        auto got = common_sampler_sample_and_accept_n_rejection_rows(b.get(), [&](size_t) { return r.get(); }, draft, qs);
        require(want == got, "host/backend verify seed mismatch");
    }
    std::printf("M: identity k=%d subset=%d seeds=256\n", k, subset);
}

static void decode(llama_context * ctx, const llama_tokens & tokens, int pos, bool all = true) {
    auto b = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) common_batch_add(b, tokens[i], pos + i, {0}, all || i + 1 == tokens.size());
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    require(rc == 0, "decode failed");
}

static row read_row(llama_context * ctx, int i, int nv) {
    row r;
    const auto * logits = llama_get_logits_ith(ctx, i);
    require(logits, "logits missing");
    r.logits.assign(logits, logits + nv);
    r.select();
    return r;
}

static double nmse(const row & a, const row & b) {
    double error = 0, norm = 0;
    for (size_t i = 0; i < a.logits.size(); ++i) {
        double d = a.logits[i] - b.logits[i];
        error += d*d;
        norm += double(b.logits[i])*b.logits[i];
    }
    return error/norm;
}

static void save_logits(const row & r, const char * kind, int k, bool subset, int rollback) {
    const char * dir = std::getenv("S4_LOGITS_DIR");
    if (!dir) return;
    const std::string path = std::string(dir) + "/" + kind + "-k" + std::to_string(k) + "-s" + std::to_string(subset) + "-r" + std::to_string(rollback) + ".f32";
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(r.logits.data()), r.logits.size()*sizeof(float));
    require(out.good(), "logit artifact write failed");
}

static void replay_law(llama_model * model, int k) {
    row r;
    r.logits.assign(32, -INFINITY);
    r.logits[0] = r.logits[1] = 0;
    r.select();
    auto p = params(271);
    p.top_k = 2;
    p.top_p = 1;
    p.min_p = 0;
    p.backend_sampling = false;
    common_sampler_ptr s(common_sampler_init(model, p));
    int counts[2] = {}, preserved[2] = {};
    const int trials = 20000;
    for (int trial = 0; trial < trials; ++trial) {
        common_sampler_ptr saved(common_sampler_clone(s.get()));
        const llama_tokens draft(k, 0);
        const std::vector<std::vector<llama_token_data>> qs(k, {{0, 0, 1}});
        auto first = common_sampler_sample_and_accept_n_rejection_rows(s.get(), [&](size_t) { return r.get(false); }, draft, qs);
        if (first.size() != 1) continue;
        require(first[0] == 1, "first residual must be token 1");
        common_sampler_ptr correct(common_sampler_clone(s.get()));
        // Accepted history must not cause the next verification to reuse its acceptance draw.
        common_sampler_copy(saved.get(), s.get());
        common_sampler_sample_row(s.get(), r.get(false));
        common_sampler_accept(s.get(), first[0], true);
        auto bonus = common_sampler_sample_row(s.get(), r.get(false));
        common_sampler_accept(s.get(), bonus, true);
        common_sampler_sample_row(correct.get(), r.get(false));
        common_sampler_accept(correct.get(), bonus, true);
        auto next = common_sampler_sample_and_accept_n_rejection_rows(s.get(), [&](size_t) { return r.get(false); }, draft, qs);
        auto next_correct = common_sampler_sample_and_accept_n_rejection_rows(correct.get(), [&](size_t) { return r.get(false); }, draft, qs);
        ++counts[next[0]];
        ++preserved[next_correct[0]];
    }
    const int n = counts[0] + counts[1];
    const double chi = double(counts[0]-counts[1])*(counts[0]-counts[1])/n;
    const double chi_correct = double(preserved[0]-preserved[1])*(preserved[0]-preserved[1])/n;
    std::printf("M: checkpoint replay k=%d n=%d counts=%d,%d chi2=%.9g preserved_rng_counts=%d,%d chi2=%.9g df=1\n", k, n, counts[0], counts[1], chi, preserved[0], preserved[1], chi_correct);
    require(n > 9000 && chi > 1000 && chi_correct < 30, "checkpoint RNG repro failed");
}

static void rollback_checks(llama_model * model, int k, bool subset, bool tp) {
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = cp.n_ubatch = 32;
    cp.n_threads = cp.n_threads_batch = tp ? 4 : 8;
    cp.n_outputs_max_per_seq = 6;
    cp.n_rs_seq = k;
    cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    common_sampler_ptr backend(make_sampler(model, params(1)));
    llama_context_ptr roll(llama_init_from_model(model, cp)), ref(llama_init_from_model(model, cp));
    require(roll && ref, "context load failed");
    const bool installed = llama_set_sampler(roll.get(), 0, common_sampler_get_backend(backend.get()));
    require(installed || (tp && !std::getenv("S4_DIAGNOSTIC")), "backend installation failed");
    std::printf("M: target backend k=%d subset=%d tp=%d installed=%d\n", k, subset, tp, installed);
    llama_tokens prompt = common_tokenize(llama_model_get_vocab(model), "A short CPU rollback fixture: solve SEND plus MORE equals MONEY.", true, false);
    decode(roll.get(), prompt, 0, false);
    decode(ref.get(), prompt, 0, false);
    const int pos = prompt.size();
    common_prompt_checkpoint base_roll, base_ref;
    base_roll.update_tgt(roll.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    base_ref.update_tgt(ref.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    auto restore = [&](llama_context * ctx, const common_prompt_checkpoint & base) {
        base.load_tgt(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        require(llama_memory_seq_rm(llama_get_memory(ctx), 0, pos, -1), "checkpoint attention trim failed");
    };
    const llama_tokens block(k + 1, 100);
    for (int r = 1; r <= k; ++r) {
        restore(roll.get(), base_roll);
        restore(ref.get(), base_ref);
        decode(roll.get(), block, pos);
        // Read all verify rows before the next graph replaces the output buffers.
        for (int i = 0; i <= k; ++i) {
            auto full = read_row(roll.get(), i, nv);
            for (unsigned seed = 0; seed < 32; ++seed) {
                common_sampler_ptr a(make_sampler(model, params(seed, false)));
                common_sampler_ptr b(make_sampler(model, params(seed, true)));
                require(common_sampler_sample_row(a.get(), full.get(false)) == common_sampler_sample(b.get(), roll.get(), i), "real verify candidate identity failed");
            }
        }
        const int keep = k + 1 - r;
        auto before = read_row(roll.get(), keep - 1, nv);
        save_logits(before, "verify", k, subset, r);
        require(llama_memory_seq_rm(llama_get_memory(roll.get()), 0, pos + keep, -1), "partial rollback failed");
        decode(ref.get(), llama_tokens(keep, 100), pos);
        auto before_ref = read_row(ref.get(), keep - 1, nv);
        std::printf("M: prefix k=%d subset=%d r=%d nmse=%.9g\n", k, subset, r, nmse(before, before_ref));
        decode(roll.get(), {correction}, pos + keep);
        decode(ref.get(), {correction}, pos + keep);
        auto got = read_row(roll.get(), 0, nv), want = read_row(ref.get(), 0, nv);
        save_logits(got, "rollback", k, subset, r);
        save_logits(want, "reference", k, subset, r);
        double error = nmse(got, want);
        std::printf("M: rollback k=%d subset=%d r=%d nmse=%.9g bit_equal=%d\n", k, subset, r, error, got.logits == want.logits);
        const bool diagnostic = std::getenv("S4_DIAGNOSTIC") != nullptr;
        require(diagnostic || error < 1e-8, "rollback logits changed");
        if (r == k) {
            if (!diagnostic) law_check(model, got, k, subset, 0, false);
            // The negative control keeps the rejected suffix in recurrent memory.
            restore(roll.get(), base_roll);
            decode(roll.get(), block, pos);
            decode(roll.get(), {correction}, pos + k + 1);
            auto broken = read_row(roll.get(), 0, nv);
            const double wrong = nmse(broken, want);
            std::printf("M: rollback negative k=%d subset=%d nmse=%.9g\n", k, subset, wrong);
            require(wrong > 1e-8, "rollback negative control was insensitive");
        }
    }
    llama_set_sampler(roll.get(), 0, nullptr);
}

int main(int argc, char ** argv) {
    try {
        require(argc >= 2, "usage: check MODEL [decode|tp]");
        bool gpu = false;
#ifdef S4_PARENT_GPU
        gpu = argc > 2 && std::strcmp(argv[2], "gpu") == 0;
#endif
        require(std::getenv("CUDA_VISIBLE_DEVICES") && (gpu ? *std::getenv("CUDA_VISIBLE_DEVICES") != '\0' : !*std::getenv("CUDA_VISIBLE_DEVICES")), "device visibility guard failed");
        llama_log_set([](ggml_log_level level, const char * message, void *) { if (level >= GGML_LOG_LEVEL_WARN) std::fputs(message, stderr); }, nullptr);
        llama_backend_init();
        const bool real = argc > 2;
        const bool tp = gpu || (real && std::strcmp(argv[2], "tp") == 0);
        auto mp = llama_model_default_params();
        mp.vocab_only = !real;
        mp.n_gpu_layers = 0;
        mp.load_mtp = real;
        auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_dev_t devs[] = {cpu, cpu, nullptr};
        if (gpu) {
            size_t found = 0;
            for (size_t i = 0; i < ggml_backend_dev_count() && found < 2; ++i) {
                auto dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) devs[found++] = dev;
            }
            require(found == 2, "parent probe needs two GPUs");
        }
        if (tp) {
            mp.devices = devs;
            mp.n_gpu_layers = -1;
            mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        }
        llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
        require(model != nullptr, "model load failed");
        if (!real) {
            row r;
            r.logits.assign(llama_vocab_n_tokens(llama_model_get_vocab(model.get())), -INFINITY);
            for (int i = 0; i < 24; ++i) r.logits[i] = 3.0f - 0.035f*i;
            r.select();
            for (int k : {3, 4, 5}) for (bool sub : {false, true}) {
                identities(model.get(), r, k, sub);
                for (int i = 0; i < k; ++i) law_check(model.get(), r, k, sub, i, false);
                if (sub) law_check(model.get(), r, k, sub, 0, true);
            }
            for (int k : {3, 4, 5}) replay_law(model.get(), k);
        } else {
            for (int k : {3, 4, 5}) rollback_checks(model.get(), k, false, tp);
            const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
            model.reset();
            std::vector<llama_token> ids;
            if (const char * path = std::getenv("S4_SUBSET")) {
                std::ifstream input(path);
                nlohmann::json data;
                input >> data;
                ids = data.get<std::vector<llama_token>>();
            } else {
                for (int i = nv - 1; i >= 0; i -= 10) ids.push_back(i);
            }
            mp.draft_vocab_ids = ids.data();
            mp.n_draft_vocab = ids.size();
            while (std::find(ids.begin(), ids.end(), correction) != ids.end()) ++correction;
            require(correction < nv, "no correction token outside subset");
            model.reset(llama_model_load_from_file(argv[1], mp));
            require(model != nullptr, "subset model load failed");
            for (int k : {3, 4, 5}) rollback_checks(model.get(), k, true, tp);
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
