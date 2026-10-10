#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-impl.h"

#include <openssl/evp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "meta-layout.h"

static bool complete_meta_node = false;
static uint64_t meta_completion_calls = 0;
static ggml_tensor * observed_node = nullptr;
static void (*capture_inputs_hook)(ggml_tensor *) = nullptr;

// A one-node callback graph must complete its AllReduce before the next callback graph assumes synchronized inputs.
extern "C" enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, ggml_cgraph * graph) {
    if (observed_node && graph->n_nodes == 1 && graph->nodes[0] == observed_node) {
        ggml_backend_synchronize(backend);
        auto * node = observed_node;
        observed_node = nullptr;
        capture_inputs_hook(node);
    }
    if (!complete_meta_node || !ggml_backend_is_meta(backend) || graph->n_nodes != 1) {
        return backend->iface.graph_compute(backend, graph);
    }
    static std::map<ggml_tensor *, ggml_tensor> observers;
    auto * node = graph->nodes[0];
    auto & observer = observers[node];
    observer = *node;
    observer.op = GGML_OP_VIEW;
    memset(observer.op_params, 0, sizeof(observer.op_params));
    std::fill(std::begin(observer.src), std::end(observer.src), nullptr);
    observer.src[0] = node;
    observer.view_src = node->view_src ? node->view_src : node;
    observer.view_offs = node->view_offs;
    observer.buffer = nullptr;
    observer.data = nullptr;
    snprintf(observer.name, sizeof(observer.name), "history_observer");
    if (ggml_backend_view_init(&observer) != GGML_STATUS_SUCCESS) {
        return GGML_STATUS_FAILED;
    }
    auto completed = *graph;
    ggml_tensor * nodes[] = {node, &observer};
    completed.nodes = nodes;
    completed.n_nodes = completed.size = 2;
    completed.uid = 0;
    ++meta_completion_calls;
    return backend->iface.graph_compute(backend, &completed);
}

static void require(bool ok, const char * message) {
    if (!ok) {
        fprintf(stderr, "history-root: %s\n", message);
        std::exit(8);
    }
}

static std::string quote(const std::string & text) {
    std::string out = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (c < 32) {
            char escaped[8];
            snprintf(escaped, sizeof(escaped), "\\u%04x", c);
            out += escaped;
        } else {
            out += c;
        }
    }
    return out + '"';
}

static std::string shape(const ggml_tensor * t, bool strides) {
    std::string out = "[";
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        out += (i ? "," : "") + std::to_string(strides ? t->nb[i] : t->ne[i]);
    }
    return out + "]";
}

static std::vector<ggml_tensor *> slices(const ggml_tensor * t) {
    if (!t->buffer) {
        return {};
    }
    if (!ggml_backend_buffer_is_meta(t->buffer)) {
        return {const_cast<ggml_tensor *>(t)};
    }
    // The generated definition must match the frozen library. No map entry is created or changed here.
    auto * context = (ggml_backend_meta_buffer_context *) t->buffer->context;
    auto & container = context->get_simple_tensor_container(t);
    auto found = container.simple_tensors.find(t);
    return found == container.simple_tensors.end() ? std::vector<ggml_tensor *>{} : found->second;
}

static std::string fingerprint(const ggml_tensor * t) {
    EVP_MD_CTX * hash = EVP_MD_CTX_new();
    require(hash && EVP_DigestInit_ex(hash, EVP_sha256(), nullptr) == 1, "SHA256 init failed");
    const size_t block = ggml_blck_size(t->type);
    const size_t unit = ggml_type_size(t->type);
    require(t->ne[0] % block == 0, "unaligned tensor blocks");
    const size_t units = t->ne[0] / block;
    const size_t row_size = units * unit;
    const size_t span = units ? (units - 1)*t->nb[0] + unit : 0;
    require(span <= 64*1024*1024, "tensor row span exceeds read buffer cap");
    uint64_t sum = 0, count = 0;
    unsigned absmax = 0;
    double sum_values = 0, absmax_values = 0, min_values = 0, max_values = 0;
    uint64_t finite_values = 0, nonfinite_values = 0;
    const bool scalar = t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_I32;
    auto update = [&](const uint8_t * bytes, size_t size) {
        require(EVP_DigestUpdate(hash, bytes, size) == 1, "SHA256 update failed");
        for (size_t i = 0; i < size; ++i) {
            sum += bytes[i];
            absmax = std::max<unsigned>(absmax, bytes[i]);
        }
        if (scalar) {
            for (size_t i = 0; i < size; i += unit) {
                double value;
                if (t->type == GGML_TYPE_F32) {
                    float x;
                    memcpy(&x, bytes + i, sizeof(x));
                    value = x;
                } else if (t->type == GGML_TYPE_I32) {
                    int32_t x;
                    memcpy(&x, bytes + i, sizeof(x));
                    value = x;
                } else if (t->type == GGML_TYPE_F16) {
                    ggml_fp16_t x;
                    memcpy(&x, bytes + i, sizeof(x));
                    value = ggml_fp16_to_fp32(x);
                } else {
                    ggml_bf16_t x;
                    memcpy(&x, bytes + i, sizeof(x));
                    value = ggml_bf16_to_fp32(x);
                }
                if (std::isfinite(value)) {
                    min_values = finite_values ? std::min(min_values, value) : value;
                    max_values = finite_values ? std::max(max_values, value) : value;
                    sum_values += value;
                    absmax_values = std::max(absmax_values, std::abs(value));
                    ++finite_values;
                } else {
                    ++nonfinite_values;
                }
            }
        }
        count += size;
    };
    const size_t nbytes = ggml_nbytes(t);
    if (ggml_is_contiguous(t)) {
        std::vector<uint8_t> data(std::min<size_t>(nbytes, 8*1024*1024));
        for (size_t offset = 0; offset < nbytes; offset += data.size()) {
            const size_t size = std::min(data.size(), nbytes - offset);
            ggml_backend_tensor_get(t, data.data(), offset, size);
            update(data.data(), size);
        }
    } else {
        const bool bulk = nbytes <= 128*1024*1024;
        std::vector<uint8_t> data(bulk ? nbytes : span);
        std::vector<uint8_t> packed(row_size);
        if (bulk && nbytes) {
            ggml_backend_tensor_get(t, data.data(), 0, nbytes);
        }
        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                    const size_t offset = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1];
                    if (!bulk && span) {
                        ggml_backend_tensor_get(t, data.data(), offset, span);
                    }
                    for (size_t i = 0; i < units; ++i) {
                        memcpy(packed.data() + i*unit, data.data() + (bulk ? offset : 0) + i*t->nb[0], unit);
                    }
                    update(packed.data(), row_size);
                }
            }
        }
    }
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned n = 0;
    require(EVP_DigestFinal_ex(hash, digest, &n) == 1 && n == 32, "SHA256 final failed");
    EVP_MD_CTX_free(hash);
    char hex[65];
    for (unsigned i = 0; i < n; ++i) {
        snprintf(hex + 2*i, 3, "%02x", digest[i]);
    }
    char stats[320];
    snprintf(stats, sizeof(stats), "{\"sum\":%.17g,\"absmax\":%.17g,\"min\":%.17g,\"max\":%.17g,\"finite\":%llu,\"nonfinite\":%llu}",
             sum_values, absmax_values, min_values, max_values, (unsigned long long) finite_values, (unsigned long long) nonfinite_values);
    return "{\"sha256\":" + quote(hex) + ",\"bytes\":" + std::to_string(count) +
           ",\"sum_bytes\":" + std::to_string(sum) + ",\"absmax_bytes\":" + std::to_string(absmax) +
           ",\"value_stats\":" + (scalar ? std::string(stats) : "null") + "}";
}

static std::string tensor_record(const ggml_tensor * t, bool source) {
    int split_axis = -1;
    bool planned = true;
    if (t->buffer && ggml_backend_buffer_is_meta(t->buffer)) {
        auto * context = (ggml_backend_meta_buffer_context *) t->buffer->context;
        auto found = context->split_state_cache.find({t, false});
        planned = found != context->split_state_cache.end() &&
                  memcmp(found->second.second, t, ggml_backend_meta_buffer_context::nbtc) == 0;
        if (!planned && source) {
            found = context->split_state_cache.find({t, true});
            planned = found != context->split_state_cache.end() &&
                      memcmp(found->second.second, t, ggml_backend_meta_buffer_context::nbtc) == 0;
        }
        split_axis = planned ? found->second.first.axis : GGML_BACKEND_SPLIT_AXIS_UNKNOWN;
    }
    std::string out = "{\"name\":" + quote(t->name) + ",\"op\":" + quote(ggml_op_name(t->op)) +
                      ",\"type\":" + quote(ggml_type_name(t->type)) + ",\"shape\":" + shape(t, false) +
                      ",\"strides\":" + shape(t, true) + ",\"split_axis\":" + std::to_string(split_axis) + ",\"op_params\":[";
    for (size_t i = 0; i < sizeof(t->op_params)/sizeof(t->op_params[0]); ++i) {
        out += (i ? "," : "") + std::to_string(t->op_params[i]);
    }
    out += "]";
    if (source && t->buffer && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return out + ",\"status\":\"frozen_weight\"}";
    }
    if (source && !planned) {
        return out + ",\"status\":\"unavailable_before_plan\"}";
    }
    const auto physical = slices(t);
    if (physical.empty()) {
        return out + ",\"status\":\"unavailable_before_plan\"}";
    }
    out += ",\"status\":\"ok\",\"slices\":[";
    for (size_t i = 0; i < physical.size(); ++i) {
        const auto * slice = physical[i];
        require(slice != nullptr && slice->buffer && slice->type == t->type, "invalid private tensor map");
        out += (i ? "," : "") + std::string("{\"rank\":") + std::to_string(i) +
               ",\"shape\":" + shape(slice, false) + ",\"strides\":" + shape(slice, true) +
               ",\"flags\":" + std::to_string(slice->flags) + ",\"data\":" + fingerprint(slice) + "}";
    }
    return out + "]}";
}

struct trace_state {
    std::ofstream output;
    std::string phase;
    std::string domain = "target";
    std::string inputs;
    int index = 0;
    int completed = 0;
};

static trace_state * observation = nullptr;

static void capture_inputs(ggml_tensor * t) {
    auto & trace = *observation;
    trace.inputs = "[";
    bool first = true;
    for (auto * src : t->src) {
        if (src) {
            trace.inputs += (first ? "" : ",") + tensor_record(src, true);
            first = false;
        }
    }
    trace.inputs += "]";
}

static bool callback(ggml_tensor * t, bool ask, void * user_data) {
    auto & trace = *(trace_state *) user_data;
    if (trace.phase.empty()) {
        complete_meta_node = false;
        observed_node = nullptr;
        return false;
    }
    if (ask) {
        complete_meta_node = true;
        observed_node = t;
        observation = &trace;
        capture_inputs_hook = capture_inputs;
        trace.inputs.clear();
        return true;
    }
    require(!trace.inputs.empty(), "input observation hook was not called");
    trace.output << "{\"kind\":\"node\",\"phase\":" << quote(trace.phase)
                 << ",\"domain\":" << quote(trace.domain) << ",\"index\":" << trace.index++
                 << ",\"inputs\":" << trace.inputs << ",\"output\":" << tensor_record(t, false) << "}\n";
    ++trace.completed;
    require(bool(trace.output), "trace write failed");
    complete_meta_node = false;
    return true;
}

static int32_t read_i32(std::ifstream & input) {
    int32_t value;
    input.read((char *) &value, sizeof(value));
    require(bool(input), "truncated input");
    return value;
}

static void add_token(llama_batch & batch, llama_token token, llama_pos pos, bool logits) {
    const int32_t i = batch.n_tokens++;
    batch.token[i] = token;
    batch.pos[i] = pos;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = logits;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 8;
    }
    const char * input_path = std::getenv("HISTORY_INPUT");
    const char * output_path = std::getenv("HISTORY_OUT");
    const char * clear_env = std::getenv("HISTORY_CLEAR");
    require(input_path && output_path && clear_env, "missing history input/output/clear mode");
    const std::string clear = clear_env;
    require(clear == "none" || clear == "data" || clear == "meta", "invalid clear mode");
    const bool use_draft = std::string(std::getenv("HISTORY_DRAFT") ? std::getenv("HISTORY_DRAFT") : "1") == "1";
    const bool traced = std::string(std::getenv("HISTORY_TRACE") ? std::getenv("HISTORY_TRACE") : "1") == "1";
    require(params.n_ctx > 0 && params.n_ctx <= 102400, "invalid explicit context");
    require(params.n_parallel == 1 && params.n_batch >= 512 && params.n_batch <= 2048, "invalid batch shape");
    require(params.n_ubatch >= 1 && params.n_ubatch <= 512, "invalid ubatch");
    require(params.cpuparams.n_threads > 0 && params.cpuparams.n_threads <= 8, "invalid thread count");
    require(params.cpuparams_batch.n_threads > 0 && params.cpuparams_batch.n_threads <= 8, "invalid batch thread count");
    std::ifstream input(input_path, std::ios::binary);
    char magic[4];
    input.read(magic, 4);
    require(bool(input) && std::memcmp(magic, "HRT1", 4) == 0, "bad input magic");
    const int count = read_i32(input);
    const int draft_steps = read_i32(input);
    require(draft_steps >= 1 && draft_steps <= 8, "invalid draft step count");
    require(count >= 2 && count <= 3, "invalid request count");
    struct request {
        int depth;
        int steps;
        std::vector<llama_token> tokens;
    };
    std::vector<request> requests;
    for (int i = 0; i < count; ++i) {
        request r;
        r.depth = read_i32(input);
        r.steps = read_i32(input);
        require(r.depth >= 1 && r.steps >= 1 && r.steps <= 8 && r.depth <= params.n_ctx - r.steps - draft_steps, "invalid request bounds");
        r.tokens.resize(r.depth + r.steps + draft_steps);
        input.read((char *) r.tokens.data(), r.tokens.size()*sizeof(llama_token));
        require(bool(input), "truncated request tokens");
        requests.push_back(std::move(r));
    }
    require(input.peek() == EOF, "trailing input bytes");
    trace_state trace;
    trace.output.open(std::string(output_path) + "/trace.jsonl");
    require(bool(trace.output), "cannot open trace");
    params.cb_eval = traced ? callback : nullptr;
    params.cb_eval_user_data = &trace;
    params.warmup = false;
    params.load_mtp = true;
    params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    params.speculative.draft.n_max = draft_steps;
    llama_backend_init();
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    require(model && ctx, "model/context init failed");
    require(llama_n_ctx(ctx) == uint32_t(params.n_ctx) && llama_n_ubatch(ctx) == uint32_t(params.n_ubatch), "effective context or ubatch mismatch");
    require(llama_n_rs_seq(ctx) == params.speculative.need_n_rs_seq(), "target rollback reservation mismatch");
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int32_t n_embd = llama_model_n_embd(model);
    require(n_vocab > 0 && n_vocab <= 300000 && n_embd > 0 && n_embd <= 16384, "invalid model dimensions");
    llama_set_embeddings_nextn(ctx, true, false);
    llama_context_ptr draft;
    if (use_draft) {
        auto cparams = common_context_params_to_llama(params);
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cparams.ctx_other = ctx;
        cparams.type_k = cparams.type_v = GGML_TYPE_F16;
        cparams.n_rs_seq = 0;
        draft.reset(llama_init_from_model(model, cparams));
        require(draft && llama_mtp_fold_init(ctx, draft.get()), "MTP fold attach failed");
        llama_set_embeddings_nextn(draft.get(), true, true);
    }
    llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
    auto * draft_batch = draft ? llama_batch_ext_init(draft.get()) : nullptr;
    std::vector<float> previous_h(n_embd, 0.0f);
    std::ofstream logits(std::string(output_path) + "/logits.bin", std::ios::binary);
    require(bool(logits), "cannot open logits");
    const int32_t header[] = {requests.back().depth, requests.back().steps, n_vocab};
    logits.write("KLDL", 4);
    logits.write((const char *) header, sizeof(header));
    for (size_t ri = 0; ri < requests.size(); ++ri) {
        const auto & r = requests[ri];
        const bool final = ri + 1 == requests.size();
        for (llama_token token : r.tokens) {
            require(token >= 0 && token < n_vocab, "invalid token id");
        }
        if (ri) {
            for (auto * context : {ctx, draft.get()}) {
                if (!context) {
                    continue;
                }
                auto mem = llama_get_memory(context);
                // A cache_n=0 request must remove the old sequence even in the no-clear arm.
                if (clear == "none") {
                    require(llama_memory_seq_rm(mem, 0, -1, -1), "request sequence removal failed");
                } else {
                    llama_memory_clear(mem, clear == "data");
                }
            }
        }
        for (int start = 0; start < r.depth + r.steps;) {
            const bool prefill = start < r.depth;
            const int end = prefill ? std::min(r.depth, start + params.n_ubatch) : start + 1;
            const bool capture = final && (end == r.depth || !prefill);
            trace.phase = capture && traced ? (prefill ? "prefill" : "decode-" + std::to_string(start - r.depth)) : "";
            trace.domain = "target";
            trace.index = 0;
            batch.n_tokens = 0;
            for (int pos = start; pos < end; ++pos) {
                add_token(batch, r.tokens[pos], pos, !prefill);
            }
            if (draft) {
                llama_mtp_fold_arm(ctx, previous_h.data(), 1);
            }
            require(llama_decode(ctx, batch) == 0, "target decode failed");
            if (final && !prefill) {
                const float * values = llama_get_logits_ith(ctx, 0);
                require(values != nullptr, "missing target logits");
                for (int j = 0; j < n_vocab; ++j) {
                    require(std::isfinite(values[j]), "non-finite target logits");
                }
                logits.write((const char *) values, size_t(n_vocab)*sizeof(float));
            }
            if (draft && llama_mtp_fold_n_last(ctx) != end - start) {
                trace.domain = "catchup";
                trace.index = 0;
                llama_batch_ext_clear(draft_batch);
                for (int pos = start; pos < end; ++pos) {
                    const int idx = llama_batch_ext_add_token(draft_batch, 0, r.tokens[pos]);
                    const float * hidden = pos == start ? previous_h.data() : llama_get_embeddings_nextn_ith(ctx, pos - start - 1);
                    require(hidden != nullptr, "missing catchup hidden row");
                    require(llama_batch_ext_set_pos(draft_batch, idx, &pos), "catchup position failed");
                    require(llama_batch_ext_set_embd_token(draft_batch, idx, {hidden, 1, size_t(n_embd)}), "catchup embedding failed");
                }
                require(llama_process(draft.get(), LLAMA_PROCESS_TYPE_DECODE, draft_batch) == 0, "draft catchup failed");
            }
            if (draft) {
                const float * hidden = llama_get_embeddings_nextn_ith(ctx, end - start - 1);
                require(hidden != nullptr, "missing target hidden row");
                std::copy(hidden, hidden + n_embd, previous_h.begin());
                if (!prefill) {
                    std::vector<float> draft_h = previous_h;
                    for (int di = 0; di < draft_steps; ++di) {
                        trace.domain = "draft-" + std::to_string(di);
                        trace.index = 0;
                        llama_batch_ext_clear(draft_batch);
                        const int pos = end + di;
                        const int idx = llama_batch_ext_add_token(draft_batch, 0, r.tokens[pos]);
                        require(llama_batch_ext_set_pos(draft_batch, idx, &pos), "draft position failed");
                        require(llama_batch_ext_set_embd_token(draft_batch, idx, {draft_h.data(), 1, size_t(n_embd)}), "draft embedding failed");
                        require(llama_batch_ext_set_output_logits(draft_batch, idx, true), "draft output flag failed");
                        require(llama_process(draft.get(), LLAMA_PROCESS_TYPE_DECODE, draft_batch) == 0, "forced draft step failed");
                        hidden = llama_get_embeddings_nextn_ith(draft.get(), 0);
                        require(hidden != nullptr, "missing draft hidden row");
                        std::copy(hidden, hidden + n_embd, draft_h.begin());
                    }
                    require(llama_memory_seq_rm(llama_get_memory(draft.get()), 0, end, -1), "draft tail removal failed");
                }
            }
            trace.phase.clear();
            trace.output.flush();
            fprintf(stderr, "history-root: request=%zu depth=%d end=%d capture=%d draft=%d\n", ri, r.depth, end, capture, use_draft);
            start = end;
        }
    }
    if (draft) {
        llama_mtp_fold_init(ctx, nullptr);
        llama_batch_ext_free(draft_batch);
    }
    llama_batch_free(batch);
    logits.close();
    trace.output.close();
    require(bool(logits) && bool(trace.output) && (!traced || trace.completed > 0), "incomplete output");
    fprintf(stderr, "history-root: complete nodes=%d traced=%d requests=%zu meta_completion_calls=%llu\n",
            trace.completed, traced, requests.size(), (unsigned long long) meta_completion_calls);
    return 0;
}
