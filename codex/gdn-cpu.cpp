#include "llama.h"
#include "llama-ext.h"
#include "llama-memory-hybrid.h"
#include "ggml-backend.h"

#include <algorithm>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * why) {
    if (!ok) { throw std::runtime_error(why); }
}

static uint64_t hash_bytes(const void * ptr, size_t n) {
    uint64_t h = 14695981039346656037ULL;
    const auto * p = static_cast<const unsigned char *>(ptr);
    for (size_t i = 0; i < n; ++i) { h = (h ^ p[i]) * 1099511628211ULL; }
    return h;
}

struct probe {
    int indexed = 0;
    int mutation = 0;
    int size = 0;
    bool mutated = false;
};

static bool eval(ggml_tensor * t, bool ask, void * user) {
    auto & p = *static_cast<probe *>(user);
    if (ask && std::string(ggml_op_name(t->op)) == "GATED_DELTA_NET_INDEXED") {
        ++p.indexed;
        if (p.mutation && !p.mutated) {
            std::vector<int32_t> rows(t->src[6]->ne[0]);
            ggml_backend_tensor_get(t->src[6], rows.data(), 0, rows.size() * sizeof(int32_t));
            for (auto & row : rows) {
                row = p.mutation == 1 ? row % p.size : 5 * p.size + row % p.size;
            }
            ggml_backend_tensor_set(t->src[6], rows.data(), 0, rows.size() * sizeof(int32_t));
            p.mutated = true;
        }
    }
    return false;
}

static void fingerprint(std::ofstream & out, llama_context * ctx, int np, const std::string & phase) {
    llama_synchronize(ctx);
    auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
    auto * mem = hybrid->get_mem_recr();
    for (int seq = 0; seq < np; ++seq) {
        const int tail = mem->cells[seq].tail;
        check(tail >= 0, "missing recurrent tail");
        const int row = mem->rs_idx[seq] * mem->size + mem->cells[tail].src;
        for (size_t il = 0; il < mem->s_l.size(); ++il) {
            for (auto * t : {mem->r_l[il], mem->s_l[il]}) {
                if (!t) { continue; }
                if (std::getenv("GDN_FIXTURE_TRACE")) {
                    std::fprintf(stderr, "fingerprint %s seq=%d layer=%zu row=%d %s\n", phase.c_str(), seq, il, row, t->name);
                }
                const size_t bytes = t->nb[1];
                std::vector<unsigned char> data(bytes);
                ggml_backend_tensor_get(t, data.data(), row * bytes, bytes);
                out << phase << " state " << seq << ' ' << il << ' ' << t->name << ' '
                    << hash_bytes(data.data(), data.size()) << '\n';
            }
        }
    }
}

static int run(int argc, char ** argv) {
    check(argc == 9, "usage: gdn-cpu MODEL META NP OUTPUT MUTANT -c 4096 ROUNDS");
    check(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "GPU visibility guard");
    check(std::string(argv[6]) == "-c" && std::atoi(argv[7]) == 4096, "explicit context required");
    const int meta = std::atoi(argv[2]), np = std::atoi(argv[3]), mutant = std::atoi(argv[5]);
    const int rounds = std::atoi(argv[8]);
    check((np == 1 || np == 2) && rounds >= 3, "bad fixture arguments");
    llama_log_set([](ggml_log_level level, const char * msg, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) { std::fputs(msg, stderr); }
    }, nullptr);
    llama_backend_init();
    auto mp = llama_model_default_params();
    auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devs[] = {cpu, cpu, nullptr};
    mp.n_gpu_layers = meta ? -1 : 0;
    mp.load_mtp = true;
    if (meta) { mp.devices = devs; mp.split_mode = LLAMA_SPLIT_MODE_TENSOR; }
    auto * model = llama_model_load_from_file(argv[1], mp);
    check(model, "model load failed");
    probe p;
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_seq_max = np;
    cp.n_batch = cp.n_ubatch = 32;
    cp.n_threads = cp.n_threads_batch = meta ? 4 : 8;
    cp.n_rs_seq = 5;
    cp.n_outputs_max = 32;
    cp.n_outputs_max_per_seq = 16;
    cp.cb_eval = eval;
    cp.cb_eval_user_data = &p;
    auto * ctx = llama_init_from_model(model, cp);
    check(ctx, "context load failed");
    llama_set_embeddings_nextn(ctx, true, false);
    auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
    p.size = hybrid->get_mem_recr()->size;
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_sampler *> samplers(np);
    for (int seq = 0; seq < np; ++seq) {
        auto * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(20));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(1));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(1729 + seq));
        samplers[seq] = smpl;
    }
    std::ofstream out(argv[4]);
    check(bool(out), "output open failed");
    std::vector<int> pos(np, 0);
    auto batch = llama_batch_init(32, 0, 1);
    const int64_t begin = ggml_time_us();
    auto decode = [&](int count, int step, const std::string & phase) {
        if (std::getenv("GDN_FIXTURE_TRACE")) { std::fprintf(stderr, "decode %s\n", phase.c_str()); }
        batch.n_tokens = 0;
        for (int s = 0; s < np; ++s) {
            const int seq = step % 2 ? np - 1 - s : s;
            for (int t = 0; t < count; ++t) {
                const int i = batch.n_tokens++;
                batch.token[i] = (100 + step * 13 + seq * 17 + t) % nv;
                batch.pos[i] = pos[seq]++;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = seq;
                batch.logits[i] = true;
            }
        }
        check(llama_decode(ctx, batch) == 0, "decode failed");
        for (int i = 0; i < batch.n_tokens; ++i) {
            const int seq = batch.seq_id[i][0];
            const float * logits = llama_get_logits_ith(ctx, i);
            const float * hidden = llama_get_embeddings_nextn_ith(ctx, i);
            check(hidden, "missing MTP target hidden state");
            out << phase << " hidden " << seq << ' ' << i << ' '
                << hash_bytes(hidden, llama_model_n_embd_out(model) * sizeof(float)) << '\n';
            out << phase << " logits " << seq << ' ' << i << ' ' << hash_bytes(logits, nv * sizeof(float)) << '\n';
            const llama_token token = llama_sampler_sample(samplers[seq], ctx, i);
            llama_sampler_accept(samplers[seq], token);
            out << phase << " token " << seq << ' ' << i << ' ' << token << '\n';
        }
        fingerprint(out, ctx, np, phase);
    };
    decode(8, 0, "prompt");
    for (int round = 0; round < rounds; ++round) {
        decode(6, round, "verify" + std::to_string(round));
        for (int seq = 0; seq < np; ++seq) {
            const int accepted = (round + seq) % 3 == 0 ? 0 : (round + seq) % 3 == 1 ? 2 : 5;
            const int remove = 5 - accepted;
            if (remove) {
                check(llama_memory_seq_rm(llama_get_memory(ctx), seq, pos[seq] - remove, -1), "rollback refused");
                pos[seq] -= remove;
                check(!llama_memory_seq_rm(llama_get_memory(ctx), seq, pos[seq] - 1, -1), "second pending rollback accepted");
            }
        }
        fingerprint(out, ctx, np, "rollback" + std::to_string(round));
        if (mutant == 1 && round == 0) { p.mutation = 1; }
        decode(1, round, "resume" + std::to_string(round));
        p.mutation = 0;
    }
    decode(6, 0, "before-checkpoint");
    for (int seq = 0; seq < np; ++seq) {
        check(llama_memory_seq_rm(llama_get_memory(ctx), seq, pos[seq] - 3, -1), "checkpoint rollback refused");
        pos[seq] -= 3;
        std::vector<uint8_t> saved(llama_state_seq_get_size(ctx, seq));
        check(llama_state_seq_get_data(ctx, saved.data(), saved.size(), seq) == saved.size(), "checkpoint save failed");
        out << "checkpoint " << seq << ' ' << hash_bytes(saved.data(), saved.size()) << '\n';
        check(llama_state_seq_set_data(ctx, saved.data(), saved.size(), seq) == saved.size(), "checkpoint restore failed");
    }
    fingerprint(out, ctx, np, "restored-checkpoint");
    decode(1, 1, "checkpoint-resume");
    // A clear while a rollback is pending must discard the snapshot index.
    decode(6, 1, "before-clear");
    for (int seq = 0; seq < np; ++seq) {
        check(llama_memory_seq_rm(llama_get_memory(ctx), seq, pos[seq] - 5, -1), "pending rollback refused");
        check(llama_memory_seq_rm(llama_get_memory(ctx), seq, 0, -1), "clear refused");
        pos[seq] = 0;
    }
    if (mutant == 2) { p.mutation = 2; p.mutated = false; }
    decode(6, 0, "after-clear");
    p.mutation = 0;
    if (np == 2 && (!meta || std::getenv("GDN_TEST_META_SEQ_CP"))) {
        check(llama_memory_seq_rm(llama_get_memory(ctx), 1, 0, -1), "copy destination clear refused");
        llama_memory_seq_cp(llama_get_memory(ctx), 0, 1, 0, -1);
        pos[1] = pos[0];
        decode(6, 1, "shared-tail");
        check(llama_memory_seq_rm(llama_get_memory(ctx), 0, pos[0] - 3, -1), "asymmetric rollback refused");
        pos[0] -= 3;
        decode(1, 0, "copy-resume");
    }
    std::printf("M: meta=%d np=%d rounds=%d indexed_calls=%d mutated=%d wall_ms=%.3f\n",
            meta, np, rounds, p.indexed, p.mutated, (ggml_time_us() - begin) / 1000.0);
    llama_batch_free(batch);
    for (auto * smpl : samplers) { llama_sampler_free(smpl); }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

int main(int argc, char ** argv) {
    if (std::getenv("GDN_FIXTURE_TRACE")) {
        std::signal(SIGSEGV, [](int) {
            void * frames[32];
            backtrace_symbols_fd(frames, backtrace(frames, 32), STDERR_FILENO);
            _exit(139);
        });
    }
    try { return run(argc, argv); }
    catch (const std::exception & e) { std::fprintf(stderr, "fixture: %s\n", e.what()); return 1; }
}
