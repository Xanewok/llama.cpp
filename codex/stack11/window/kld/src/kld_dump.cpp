// Scores fixed token-id sequences with llama.cpp: per position, the log-prob of the next token, the entropy,
// and the top-K log-probs. Same record layout as the vLLM client (score_vllm.py) so both feed one analysis.
//
// Model/device flags are the stock llama.cpp common args (-m -c -b -ub -ngl -fa -sm -ts -ctk -ctv ...).
// Harness inputs come from the environment so they cannot collide with common args:
//   KLD_TOKENS  input  : "KLDT" i32 n_seq, per seq { i32 n, i32 ids[n] }
//   KLD_OUT     output : "KLDO" i32 K i32 n_vocab i32 n_seq, per seq { i32 n_pos, n_pos x rec }
//                        rec = f32 lp_next, f32 entropy, i32 ids[K], f32 lps[K]   (position j predicts ids[j+1])
//                        optional trailer "FULL" i32 seq i32 from i32 m f32 kld_exact[m]
//   KLD_TOPK    K (default 64); KLD_THREADS  CPU threads for the per-row reduction
//   KLD_DECODE_TAIL  the last N tokens of each sequence go through one-token batches (the serving decode path:
//                    FA vector kernel over the cached K/V) instead of the batched prefill path
//   KLD_DECODE_TAIL_BS  tail batch width (default 1); all rows retain their original positions
//   KLD_TAIL_WRITE  optional full-vocabulary f32 log-probs for every scored tail row (KLF1)
//   KLD_FULL_SEQ/KLD_FULL_FROM/KLD_FULL_N  exact-KLD validation window (positions of one sequence)
//   KLD_FULL_WRITE  write the window's full f32 log-prob rows (reference arm)
//   KLD_FULL_READ   read those rows and emit exact KL(ref || this) per window position
// Output is written to KLD_OUT.tmp and renamed on success: a present KLD_OUT means a complete arm.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "batch_compat.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

static const char * env_str(const char * k) {
    const char * v = std::getenv(k);
    return (v && *v) ? v : nullptr;
}

static long env_int(const char * k, long dflt) {
    const char * v = env_str(k);
    return v ? std::strtol(v, nullptr, 10) : dflt;
}

[[noreturn]] static void fail(const std::string & msg) {
    fprintf(stderr, "kld_dump: %s\n", msg.c_str());
    std::exit(2);
}

static std::vector<std::vector<llama_token>> read_tokens(const char * path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail(std::string("cannot open ") + path);
    char magic[4];
    int32_t n_seq = 0;
    f.read(magic, 4);
    f.read((char *) &n_seq, 4);
    if (!f || std::memcmp(magic, "KLDT", 4) != 0 || n_seq <= 0) fail("bad token file");
    std::vector<std::vector<llama_token>> seqs(n_seq);
    for (auto & s : seqs) {
        int32_t n = 0;
        f.read((char *) &n, 4);
        if (!f || n < 2) fail("bad sequence length");
        s.resize(n);
        f.read((char *) s.data(), size_t(n) * 4);
        if (!f) fail("truncated token file");
    }
    return seqs;
}

struct Rec {
    float lp_next;
    float entropy;
    std::vector<int32_t> ids;
    std::vector<float> lps;
};

// Exact log-softmax in double accumulation; top-K by partial sort of indices.
static void score_row(const float * logits, int n_vocab, int next, int K, Rec & r, std::vector<int32_t> & idx,
                      float * full_row_out) {
    float mx = logits[0];
    for (int i = 1; i < n_vocab; ++i) mx = std::max(mx, logits[i]);
    double se = 0.0;
    for (int i = 0; i < n_vocab; ++i) se += std::exp(double(logits[i] - mx));
    const double lse = double(mx) + std::log(se);
    double ent = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        const double lp = double(logits[i]) - lse;
        ent -= std::exp(lp) * lp;
        if (full_row_out) full_row_out[i] = float(lp);
    }
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [logits](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    r.lp_next = float(double(logits[next]) - lse);
    r.entropy = float(ent);
    r.ids.assign(idx.begin(), idx.begin() + K);
    r.lps.resize(K);
    for (int k = 0; k < K; ++k) r.lps[k] = float(double(logits[idx[k]]) - lse);
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) return 1;

    const char * tok_path = env_str("KLD_TOKENS");
    const char * out_path = env_str("KLD_OUT");
    if (!tok_path || !out_path) fail("KLD_TOKENS and KLD_OUT are required");
    const int K = int(env_int("KLD_TOPK", 64));
    const long full_seq = env_int("KLD_FULL_SEQ", -1);
    const long full_from = env_int("KLD_FULL_FROM", 0);
    const long full_n = env_int("KLD_FULL_N", 0);
    const char * full_write = env_str("KLD_FULL_WRITE");
    const char * full_read = env_str("KLD_FULL_READ");
    const long decode_tail = env_int("KLD_DECODE_TAIL", 0);
    const long tail_bs = env_int("KLD_DECODE_TAIL_BS", 1);
    const char * tail_write = env_str("KLD_TAIL_WRITE");
    if (tail_bs < 1 || tail_bs > params.n_batch || tail_bs > params.n_ubatch)
        fail("KLD_DECODE_TAIL_BS must fit both -b and -ub");
    if (tail_write && decode_tail <= 0) fail("KLD_TAIL_WRITE requires KLD_DECODE_TAIL > 0");

    auto seqs = read_tokens(tok_path);
    size_t max_len = 0;
    for (auto & s : seqs) max_len = std::max(max_len, s.size());
    if (size_t(params.n_ctx) < max_len) fail("-c must be >= longest sequence (" + std::to_string(max_len) + ")");
    params.n_parallel = 1;

    llama_backend_init();
    llama_numa_init(params.numa);
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) fail("model/context init failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_batch = std::min<int>(params.n_batch, llama_n_ctx(ctx));

    if (full_seq >= 0) {
        if (full_seq >= long(seqs.size()) || full_from < 0 || full_n <= 0 ||
            full_from + full_n > long(seqs[full_seq].size()) - 1)
            fail("validation window out of range");
    }
    std::ofstream fw;
    std::ifstream fr;
    if (full_seq >= 0 && full_write) {
        fw.open(std::string(full_write) + ".tmp", std::ios::binary);
        if (!fw) fail("cannot write full rows");
    }
    if (full_seq >= 0 && full_read) {
        fr.open(full_read, std::ios::binary);
        if (!fr) fail("cannot read full rows");
        fr.seekg(0, std::ios::end);
        if (size_t(fr.tellg()) != size_t(full_n) * n_vocab * sizeof(float)) fail("full-row file size mismatch");
        fr.seekg(0);
    }
    std::ofstream ft;
    if (tail_write) {
        ft.open(std::string(tail_write) + ".tmp", std::ios::binary);
        if (!ft) fail("cannot write tail rows");
        const int32_t h[2] = {n_vocab, int32_t(seqs.size())};
        ft.write("KLF1", 4);
        ft.write((const char *) h, sizeof(h));
    }
    std::vector<float> kld_exact;

    const std::string tmp_out = std::string(out_path) + ".tmp";
    std::ofstream out(tmp_out, std::ios::binary);
    if (!out) fail("cannot open output");
    const int32_t hdr[3] = {K, n_vocab, int32_t(seqs.size())};
    out.write("KLDO", 4);
    out.write((const char *) hdr, sizeof(hdr));

    const int n_threads = int(env_int("KLD_THREADS", std::max(1u, std::thread::hardware_concurrency() / 2)));
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<float> full_rows;

    for (size_t si = 0; si < seqs.size(); ++si) {
        const auto & toks = seqs[si];
        const int n = int(toks.size());
        const int32_t n_pos = n - 1;
        out.write((const char *) &n_pos, 4);
        llama_memory_clear(llama_get_memory(ctx), true);
        const auto t0 = ggml_time_us();
        const int tail_from = std::max(0, n - int(decode_tail));
        if (ft.is_open()) {
            const int32_t h[2] = {tail_from, std::max(0, n_pos - tail_from)};
            ft.write((const char *) h, sizeof(h));
        }
        for (int start = 0, bs = 0; start < n; start += bs) {
            bs = start >= tail_from ? int(std::min<long>(tail_bs, n - start)) : std::min(n_batch, tail_from - start);
            common_batch_clear(batch);
            for (int k = 0; k < bs; ++k) common_batch_add(batch, toks[start + k], start + k, {0}, true);
            if (llama_decode(ctx, batch) != 0) fail("llama_decode failed");

            const int n_rows = std::min(bs, n_pos - start);  // the last token of a sequence predicts nothing
            if (n_rows <= 0) continue;
            std::vector<Rec> recs(n_rows);
            // llama_get_logits_ith synchronizes the context: resolve row pointers before fanning out.
            std::vector<const float *> rows(n_rows);
            for (int r = 0; r < n_rows; ++r) rows[r] = llama_get_logits_ith(ctx, r);
            const bool keep_tail = ft.is_open() && start >= tail_from;
            std::vector<float> tail_rows;
            if (keep_tail) tail_rows.resize(size_t(n_rows) * n_vocab);
            const bool in_window = long(si) == full_seq;
            const long w0 = std::max<long>(full_from, start);
            const long w1 = std::min<long>(full_from + full_n, start + n_rows);
            const bool has_window = in_window && w1 > w0;
            if (has_window) full_rows.assign(size_t(w1 - w0) * n_vocab, 0.0f);
            std::atomic<int> next_row{0};
            auto work = [&]() {
                std::vector<int32_t> idx(n_vocab);
                for (int r; (r = next_row.fetch_add(1)) < n_rows;) {
                    const long pos = start + r;
                    const bool keep = has_window && pos >= w0 && pos < w1;
                    score_row(rows[r], n_vocab, toks[pos + 1], K, recs[r], idx,
                              keep_tail ? tail_rows.data() + size_t(r) * n_vocab :
                              keep ? full_rows.data() + size_t(pos - w0) * n_vocab : nullptr);
                    if (keep_tail && keep)
                        std::copy_n(tail_rows.data() + size_t(r) * n_vocab, n_vocab,
                                    full_rows.data() + size_t(pos - w0) * n_vocab);
                }
            };
            std::vector<std::thread> pool;
            for (int t = 1; t < n_threads; ++t) pool.emplace_back(work);
            work();
            for (auto & t : pool) t.join();

            for (int r = 0; r < n_rows; ++r) {
                out.write((const char *) &recs[r].lp_next, 4);
                out.write((const char *) &recs[r].entropy, 4);
                out.write((const char *) recs[r].ids.data(), size_t(K) * 4);
                out.write((const char *) recs[r].lps.data(), size_t(K) * 4);
            }
            if (keep_tail) ft.write((const char *) tail_rows.data(), tail_rows.size() * sizeof(float));
            if (has_window) {
                std::vector<float> ref(n_vocab);
                for (long pos = w0; pos < w1; ++pos) {
                    const float * row = full_rows.data() + size_t(pos - w0) * n_vocab;
                    if (fw.is_open()) fw.write((const char *) row, size_t(n_vocab) * sizeof(float));
                    if (fr.is_open()) {
                        fr.read((char *) ref.data(), size_t(n_vocab) * sizeof(float));
                        if (!fr) fail("short read of full rows");
                        double kl = 0.0;
                        for (int i = 0; i < n_vocab; ++i) {
                            const double lp = ref[i];
                            kl += std::exp(lp) * (lp - double(row[i]));
                        }
                        kld_exact.push_back(float(kl));
                    }
                }
            }
        }
        fprintf(stderr, "kld_dump: seq %zu n=%d %.1f s\n", si, n, (ggml_time_us() - t0) / 1e6);
    }
    if (fr.is_open()) {
        if (long(kld_exact.size()) != full_n) fail("validation window incomplete");
        const int32_t tr[3] = {int32_t(full_seq), int32_t(full_from), int32_t(full_n)};
        out.write("FULL", 4);
        out.write((const char *) tr, sizeof(tr));
        out.write((const char *) kld_exact.data(), kld_exact.size() * sizeof(float));
    }
    out.close();
    if (!out) fail("output write failed");
    if (fw.is_open()) {
        fw.close();
        if (!fw) fail("full-row write failed");
        std::rename((std::string(full_write) + ".tmp").c_str(), full_write);
    }
    if (ft.is_open()) {
        ft.close();
        if (!ft) fail("tail-row write failed");
        if (std::rename((std::string(tail_write) + ".tmp").c_str(), tail_write)) fail("tail-row rename failed");
    }
    std::rename(tmp_out.c_str(), out_path);
    llama_batch_free(batch);
    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
