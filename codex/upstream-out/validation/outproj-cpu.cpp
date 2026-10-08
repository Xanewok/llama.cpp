#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

struct outproj_probe {
    int nodes = 0;
    int batched = 0;
    int multi_seq = 0;
};

static bool probe_cb(ggml_tensor * t, bool ask, void * user) {
    if (ask && t->op == GGML_OP_GATED_DELTA_NET && t->src[2]->ne[3] == 2) {
        ++static_cast<outproj_probe *>(user)->multi_seq;
    }
    if (ask && t->op == GGML_OP_MUL_MAT && std::strstr(t->src[0]->name, "ssm_out.weight")) {
        auto & p = *static_cast<outproj_probe *>(user);
        ++p.nodes;
        p.batched += t->src[1]->ne[2] != 1 || t->src[1]->ne[3] != 1;
    }
    return false;
}

static void require(bool ok, const char * why) {
    if (!ok) {
        throw std::runtime_error(why);
    }
}

int main(int argc, char ** argv) {
    require(argc == 4, "usage: outproj-cpu MODEL OUT.bin EXPECT_2D");
    require(std::getenv("CUDA_VISIBLE_DEVICES") && !*std::getenv("CUDA_VISIBLE_DEVICES"), "CPU-only guard");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    require(model != nullptr, "model load failed");
    outproj_probe probe;
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = cp.n_ubatch = 128;
    cp.n_seq_max = 2;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.n_rs_seq = 8;
    cp.kv_unified = false;
    cp.cb_eval = probe_cb;
    cp.cb_eval_user_data = &probe;
    llama_context * ctx = llama_init_from_model(model, cp);
    require(ctx != nullptr, "context init failed");
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::ofstream out(argv[2], std::ios::binary);
    require(bool(out), "cannot open output");
    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    llama_token next[2] = { 17, 43 };
    int pos = 0;
    for (int step = 0; step < 17; ++step) {
        const int rows = step == 0 ? 12 : 1;
        llama_batch_ext_clear(batch);
        int n_tokens = 0;
        for (int seq = 0; seq < 2; ++seq) {
            for (int row = 0; row < rows; ++row) {
                const auto token = step == 0 ? (17 + 31 * seq + 7 * row) % nv : next[seq];
                const int i = llama_batch_ext_add_token(batch, seq, token);
                const llama_pos positions[4] = {pos + row, pos + row, pos + row, 0};
                require(i >= 0 && llama_batch_ext_set_pos(batch, i, positions), "batch position failed");
                require(llama_batch_ext_set_output_logits(batch, i, true), "batch output failed");
                ++n_tokens;
            }
        }
        require(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0, "np2 decode failed");
        for (int i = 0; i < n_tokens; ++i) {
            const float * logits = llama_get_logits_ith(ctx, i);
            require(logits != nullptr, "missing logits");
            out.write(reinterpret_cast<const char *>(logits), nv * sizeof(float));
            const auto token = static_cast<llama_token>(std::max_element(logits, logits + nv) - logits);
            out.write(reinterpret_cast<const char *>(&token), sizeof(token));
            next[i / rows] = token;
        }
        pos += rows;
    }
    out.close();
    require(bool(out), "output write failed");
    std::fprintf(stderr, "outproj_graph: nodes=%d batched=%d n_seqs=2\n", probe.nodes, probe.batched);
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    require(probe.nodes > 0 && probe.multi_seq > 0, "np2 out projection probe never engaged");
    return std::atoi(argv[3]) && probe.batched > 0 ? 42 : 0;
}
