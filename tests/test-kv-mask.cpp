#include "../src/llama-kv-cache.h"
#include "../src/llama-model.h"
#include "ggml-cpp.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void mask_require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static void check_kv_mask(llama_kv_cache & cache, const std::vector<llama_seq_id> & streams,
                         const std::vector<int> & depths, int rows, bool causal, bool pos_2d) {
    const int n_stream = streams.size();
    const int n_tokens = rows*n_stream;
    const int n_kv = GGML_PAD(*std::max_element(depths.begin(), depths.end()) + 4, 256);
    std::vector<llama_pos> pos(n_tokens*(pos_2d ? 4 : 1));
    std::vector<int32_t> counts(n_tokens, 1);
    std::vector<llama_seq_id> seqs(n_tokens);
    auto unq = streams;
    std::vector<llama_seq_id *> ids(n_tokens);
    for (int s = 0; s < n_stream; ++s) {
        for (int r = 0; r < rows; ++r) {
            const int i = s*rows + r;
            seqs[i] = streams[s];
            ids[i] = &seqs[i];
            pos[i] = depths[streams[s]] - 4 + r;
            if (pos_2d) {
                pos[i + n_tokens] = pos[i + 2*n_tokens] = pos[i];
            }
        }
    }
    llama_ubatch ub = {};
    ub.n_tokens = n_tokens;
    ub.n_seq_tokens = rows;
    ub.n_seqs = ub.n_seqs_unq = n_stream;
    ub.n_pos = pos_2d ? 4 : 1;
    ub.b_equal_seqs = true;
    ub.pos = pos.data();
    ub.n_seq_id = counts.data();
    ub.seq_id = ids.data();
    ub.seq_id_unq = unq.data();
    const uint32_t certificate = cache.get_kq_mask_prefix(ub, n_kv, causal);
    const uint32_t prefix = certificate >= 32768 ? certificate : 0;

    for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        ggml_context_ptr ref_ctx(ggml_init({4*ggml_tensor_overhead(), nullptr, true}));
        auto * full = ggml_new_tensor_4d(ref_ctx.get(), type, n_kv, rows, 1, n_stream);
        ggml_backend_buffer_ptr ref_buffer(ggml_backend_alloc_ctx_tensors_from_buft(ref_ctx.get(), ggml_backend_cpu_buffer_type()));
        cache.set_input_kq_mask(full, &ub, causal);
        // The oracle must not share the fill path's stream indexing.
        for (int s = 0; s < n_stream; ++s) {
            const auto & cells = cache.get_cells(streams[s]);
            for (int r = 0; r < rows; ++r) {
                for (int j = 0; j < n_kv; ++j) {
                    const int i = s*rows + r;
                    bool keep = !cells.is_empty(j) && cells.seq_has(j, streams[s]);
                    if (keep && causal) {
                        keep = cells.pos_get(j) <= pos[i];
                        if (keep && pos_2d && cells.pos_get(j) == pos[i]) {
                            keep = !cells.ext_get(j).is_2d_gt(pos[i + 2*n_tokens], pos[i + n_tokens]);
                        }
                    }
                    const float expected = keep ? 0.0f : -INFINITY;
                    const size_t index = size_t(i)*n_kv + j;
                    if (type == GGML_TYPE_F16) {
                        mask_require(((ggml_fp16_t *) full->data)[index] == ggml_fp32_to_fp16(expected), "full mask differs from cell oracle");
                    } else {
                        mask_require(((float *) full->data)[index] == expected, "full mask differs from cell oracle");
                    }
                }
            }
        }
        ggml_context_ptr ctx(ggml_init({32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true}));
        auto * suffix = ggml_new_tensor_4d(ctx.get(), type, n_kv - prefix, rows, 1, n_stream);
        ggml_set_input(suffix);
        auto * result = suffix;
        if (prefix) {
            auto * zeros = ggml_fill(ctx.get(), ggml_new_tensor_4d(ctx.get(), type, prefix, rows, 1, n_stream), 0.0f);
            result = ggml_concat(ctx.get(), zeros, suffix, 0);
        } else {
            result = ggml_dup(ctx.get(), suffix);
        }
        ggml_set_output(result);
        auto * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, result);
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_dev_t devs[] = {cpu_dev, cpu_dev};
        auto * meta_dev = ggml_backend_meta_device(devs, 2, [](const ggml_tensor *, void *) {
            return ggml_backend_meta_split_state{GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }, nullptr);
        ggml_backend_ptr meta(ggml_backend_dev_init(meta_dev, nullptr));
        ggml_backend_ptr cpu(ggml_backend_cpu_init());
        ggml_backend_cpu_set_n_threads(cpu.get(), 4);
        ggml_backend_t backends[] = {meta.get(), cpu.get()};
        auto * sched = ggml_backend_sched_new(backends, nullptr, 2, 256, false, true);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_backend_sched_set_tensor_backend(sched, ggml_graph_node(graph, i), meta.get());
        }
        mask_require(ggml_backend_sched_alloc_graph(sched, graph), "compact mask allocation failed");
        mask_require(ggml_backend_sched_get_tensor_backend(sched, result) == meta.get(), "compact mask left the meta backend");
        if (prefix) {
            mask_require(ggml_backend_sched_get_tensor_backend(sched, result->src[0]->src[0]) == meta.get(), "zero prefix would require a host upload");
        }
        cache.set_input_kq_mask(suffix, &ub, causal, prefix);
        mask_require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "compact mask compute failed");
        std::vector<uint8_t> actual(ggml_nbytes(full));
        ggml_backend_tensor_get(result, actual.data(), 0, actual.size());
        mask_require(std::memcmp(full->data, actual.data(), actual.size()) == 0, "compact mask differs from full mask");
        ggml_backend_sched_free(sched);
    }
}

static int test_kv_mask() {
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model_ptr model(llama_model_create(LLM_ARCH_LLAMA, mp));
    mask_require(bool(model), "mask test model creation failed");
    auto hp = model->hparams;
    hp.no_alloc = true;
    int cases = 0;
    for (const std::vector<int> & depths : {std::vector<int>{8192, 33024}, {33280, 35072}, {90112, 89600}, {33280, 35072, 33792}}) {
        const int ns = depths.size();
        llama_kv_cache cache(*model, hp, GGML_TYPE_F16, GGML_TYPE_F16, false, false, false,
                             90624, ns, 256, 0, LLAMA_SWA_TYPE_NONE, nullptr,
                             [](uint32_t) { return false; }, nullptr, nullptr);
        auto restore = [&](int s) {
            auto & cells = const_cast<llama_kv_cells &>(cache.get_cells(s));
            for (int j = 0; j < depths[s] + 4; ++j) {
                if (!cells.is_empty(j)) {
                    cells.rm(j);
                }
                cells.pos_set(j, j);
                cells.seq_add(j, s);
            }
        };
        for (int s = 0; s < ns; ++s) {
            restore(s);
        }
        std::vector<llama_seq_id> streams;
        for (int s = ns - 1; s >= 0; --s) {
            streams.push_back(s);
        }
        for (int a = 0; a <= 3; ++a) {
            for (int b = 0; b <= 3; ++b) {
                for (int s = 0; s < ns; ++s) {
                    restore(s);
                    mask_require(cache.seq_rm(s, depths[s] + (s == 0 ? a : b), -1), "rejected tail removal failed");
                }
                for (int rows : {1, 4, 6, 16}) {
                    check_kv_mask(cache, streams, depths, rows, true, false);
                    ++cases;
                }
            }
        }
        for (int s = 0; s < ns; ++s) {
            restore(s);
        }
        check_kv_mask(cache, streams, depths, 4, false, false);
        ++cases;
        for (int s = 0; s < ns; ++s) {
            auto & cells = const_cast<llama_kv_cells &>(cache.get_cells(s));
            auto ext = cells.ext_get(depths[s] - 4);
            ext.y = depths[s];
            cells.ext_set(depths[s] - 4, ext);
            check_kv_mask(cache, streams, depths, 4, true, true);
            // A used cell belonging only to another sequence must invalidate the common prefix.
            cells.seq_add(17, (s + 1)%ns);
            cells.seq_rm(17, s);
            check_kv_mask(cache, streams, depths, 4, true, false);
            cells.seq_add(17, s);
            cells.seq_rm(17, (s + 1)%ns);
            mask_require(cache.seq_rm(s, 23, 24), "hole removal failed");
            check_kv_mask(cache, streams, depths, 4, true, false);
            cells.pos_set(23, 23);
            cells.seq_add(23, s);
            cells.pos_add(19, depths[s]);
            check_kv_mask(cache, streams, depths, 4, true, false);
            cells.pos_add(19, -depths[s]);
            cases += 4;
        }
    }
    printf("%s: %d multistream mask cases OK\n", __func__, cases);
    return 0;
}

int main() {
    try {
        ggml_backend_load_all();
        llama_backend_init();
        const int result = test_kv_mask();
        llama_backend_free();
        return result;
    } catch (const std::exception & error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
