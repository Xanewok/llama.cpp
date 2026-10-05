#include "llama.h"
#include "ggml-backend.h"
#include "ggml-ssm-conv.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static bool record_node(ggml_tensor * node, bool ask, void * data) {
    if (ask) {
        static_cast<std::vector<ggml_tensor *> *>(data)->push_back(node);
    }
    return false;
}

static bool decode(llama_context * ctx, int n_seq, int count, int pos) {
    llama_batch batch = llama_batch_init(count * n_seq, 0, 1);
    for (int seq = 0; seq < n_seq; ++seq) {
        for (int i = 0; i < count; ++i) {
            const int idx = batch.n_tokens++;
            batch.token[idx] = 100 + pos + i + seq;
            batch.pos[idx] = pos + i;
            batch.n_seq_id[idx] = 1;
            batch.seq_id[idx][0] = seq;
            batch.logits[idx] = i == count - 1;
        }
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        return 2;
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return 2;
    }
    for (int n_seq : {1, 2}) {
        for (int prefill : {32, 512}) {
            std::vector<ggml_tensor *> nodes;
            auto cp = llama_context_default_params();
            cp.n_ctx = 4096;
            cp.n_batch = 1024;
            cp.n_ubatch = 1024;
            cp.n_seq_max = n_seq;
            cp.n_rs_seq = 6;
            cp.n_threads = 4;
            cp.n_threads_batch = 4;
            cp.cb_eval = record_node;
            cp.cb_eval_user_data = &nodes;
            auto * ctx = llama_init_from_model(model, cp);
            if (!ctx || !decode(ctx, n_seq, prefill, 0)) {
                return 2;
            }
            int pos = prefill;
            for (int n_tok : {1, 4, 6}) {
                nodes.clear();
                if (!decode(ctx, n_seq, n_tok, pos)) {
                    return 2;
                }
                llama_synchronize(ctx);
                // Rebuild use counts over the scheduler's allocated tensors, without reallocating them.
                auto * gctx = ggml_init({ggml_graph_overhead_custom(16384, false), nullptr, true});
                auto * graph = ggml_new_graph_custom(gctx, 16384, false);
                for (auto * node : nodes) {
                    ggml_build_forward_expand(graph, node);
                }
                int candidates = 0, matched = 0, aliases = 0;
                for (int i = 0; i < graph->n_nodes; ++i) {
                    auto * node = graph->nodes[i];
                    if (node->op != GGML_OP_SSM_CONV) {
                        continue;
                    }
                    ++candidates;
                    bool found = false;
                    for (int start = i - 1; start >= 0 && i - start < 30; --start) {
                        ggml_ssm_conv_state_fusion match;
                        if (ggml_match_ssm_conv_state_fusion(graph, start, match) && match.conv == node) {
                            ++matched;
                            aliases += ggml_ssm_conv_overlaps(match.out, node->src[0]->src[1]);
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        std::fprintf(stderr, "unmatched real layer %s\n", node->name);
                    }
                }
                std::printf("MODEL_ENGAGEMENT np=%d prefill=%d tokens=%d matched=%d/%d dead_input_aliases=%d\n",
                            n_seq, prefill, n_tok, matched, candidates, aliases);
                ggml_free(gctx);
                if (candidates == 0 || matched != candidates) {
                    return 8;
                }
                pos += n_tok;
            }
            llama_free(ctx);
        }
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
