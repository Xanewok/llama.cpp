// GET_ROWS through a Meta(CUDA, CUDA) backend, the weight in the Meta device-resident buffer (a) and in a CUDA_Host_Mapped buffer (b):
// both gathers must match the rows dequantized on the host, a and b bitwise, and b must run on the Meta backend with no CPU split and no
// copy of the weight. Only the weights are pre-allocated: a pre-allocated destination pins its op to that buffer's backend ("1.dst" in
// ggml_backend_sched_backend_id_from_cur) before the scheduler looks at the weight, which would make the placement checks vacuous.
// Run once with each GPU first in the Meta device: a mirrored tensor is read back from the first device, so the two orders show each GPU's rows.
// GGML_CUDA_NO_PINNED=1 (--nopin) is the control: the allocation falls back to a CPU buffer, which Meta does not read, so the scheduler
// gathers b on the CPU (2 splits) without copying the weight, and the rows stay equal.
// Public API only, built against the tree under test by the window (bw-stack11-s2.sh); prints "meta_rows: PASS" or "meta_rows: FAIL".
#include "ggml.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::mt19937 rng(42);

static ggml_backend_buffer_type_t mapped_host_buft(ggml_backend_dev_t dev) {
    auto * get_extra_bufts = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    if (!get_extra_bufts) {
        return nullptr;
    }
    for (ggml_backend_buffer_type_t * buft = get_extra_bufts(dev); buft && *buft; ++buft) {
        if (ggml_backend_buft_is_host(*buft) && ggml_backend_dev_supports_buft(dev, *buft)) {
            return *buft;
        }
    }
    return nullptr;
}

static ggml_backend_meta_split_state mirrored(const ggml_tensor *, void *) {
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

static std::vector<uint8_t> random_rows(ggml_type type, int64_t ne0, int64_t nrows) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> f(ne0*nrows);
    for (auto & x : f) {
        x = dist(rng);
    }
    std::vector<uint8_t> data(nrows*ggml_row_size(type, ne0));
    if (type == GGML_TYPE_F32) {
        memcpy(data.data(), f.data(), data.size());
    } else if (ggml_is_quantized(type) || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
        std::vector<float> imatrix(ne0, 1.0f);
        ggml_quantize_chunk(type, f.data(), data.data(), 0, nrows, ne0, ggml_quantize_requires_imatrix(type) ? imatrix.data() : nullptr);
    } else {
        for (auto & b : data) {
            b = (uint8_t) rng();
        }
    }
    return data;
}

// raw: n_batches x nr rows of type; ids: n_batches x n_ids. Empty when the type has no host conversion.
static std::vector<uint8_t> host_rows(ggml_type type, ggml_type out_type, int64_t ne0, int64_t nr, const std::vector<uint8_t> & raw,
                                      const std::vector<int32_t> & ids, int64_t n_ids, int64_t n_batches) {
    const size_t row_in  = ggml_row_size(type, ne0);
    const size_t row_out = ggml_row_size(out_type, ne0);
    const auto to_float  = ggml_get_type_traits(type)->to_float;
    if (out_type != type && type != GGML_TYPE_F32 && !to_float) {
        return {};
    }
    std::vector<uint8_t> out(n_batches*n_ids*row_out);
    for (int64_t j = 0; j < n_batches; ++j) {
        for (int64_t i = 0; i < n_ids; ++i) {
            const uint8_t * src = raw.data() + (j*nr + ids[j*n_ids + i])*row_in;
            uint8_t * dst = out.data() + (j*n_ids + i)*row_out;
            if (out_type == type || type == GGML_TYPE_F32) {
                memcpy(dst, src, row_out);
            } else {
                to_float(src, (float *) dst, ne0);
            }
        }
    }
    return out;
}

static size_t n_words_differing(const std::vector<uint8_t> & x, const std::vector<uint8_t> & y) {
    size_t n = 0;
    for (size_t i = 0; i + sizeof(float) <= x.size(); i += sizeof(float)) {
        n += memcmp(x.data() + i, y.data() + i, sizeof(float)) != 0;
    }
    return n;
}

// Elements of out off the host reference. Exact types (no dequantization) must match bitwise; dequantized types within a bound that
// absorbs FMA contraction and operation order on the device but not a wrong, stale or zero row.
static size_t n_off_reference(const std::vector<uint8_t> & out, const std::vector<uint8_t> & ref, bool exact, float & max_err) {
    if (exact) {
        return n_words_differing(out, ref);
    }
    const float * o = (const float *) out.data();
    const float * r = (const float *) ref.data();
    const size_t n = out.size()/sizeof(float);
    float max_ref = 1.0f;
    for (size_t i = 0; i < n; ++i) {
        max_ref = std::isfinite(r[i]) ? std::fmax(max_ref, std::fabs(r[i])) : max_ref;
    }
    size_t n_off = 0;
    for (size_t i = 0; i < n; ++i) {
        if (memcmp(o + i, r + i, sizeof(float)) == 0) {
            continue;
        }
        const float err = std::fabs(o[i] - r[i]);
        max_err = std::isfinite(err) ? std::fmax(max_err, err) : INFINITY;
        n_off += !(err <= 1e-3f*max_ref);
    }
    return n_off;
}

static ggml_context * new_ctx(size_t n_tensors) {
    ggml_init_params params = { ggml_tensor_overhead()*n_tensors + ggml_graph_overhead(), nullptr, true };
    return ggml_init(params);
}

struct copy_log {
    bool weight_copied = false;
};

static bool record_copy(ggml_backend_t, const ggml_tensor * src, ggml_tensor *, ggml_cgraph *, void * user_data) {
    if (strcmp(src->name, "weight_b") == 0) {
        ((copy_log *) user_data)->weight_copied = true;
    }
    return true;
}

// 0 pass, 1 fail, 2 skipped (no mapped buffer type on a device)
static int run_order(ggml_backend_dev_t first, ggml_backend_dev_t second, ggml_backend_t cpu, bool nopin) {
    ggml_backend_buffer_type_t mapped = mapped_host_buft(first);
    if (!mapped || !mapped_host_buft(second)) {
        printf("meta_rows: no mapped host buffer type, skipped\n");
        return 2;
    }
    ggml_backend_dev_t devs[2] = { first, second };
    char order[64];
    snprintf(order, sizeof(order), "%s,%s", ggml_backend_dev_name(first), ggml_backend_dev_name(second));
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2, mirrored, nullptr);
    ggml_backend_t backend = ggml_backend_dev_init(meta, nullptr);
    if (!backend || !ggml_backend_dev_supports_buft(meta, mapped)) {
        printf("meta_rows: order=%s the Meta device rejects the mapped host buffer type: FAIL\n", order);
        return 1;
    }

    bool ok = true;
    int n_types = 0;
    int n_rows_checks = 0;
    for (int it = 0; it < GGML_TYPE_COUNT; ++it) {
        const ggml_type type = (ggml_type) it;
        if (ggml_type_size(type) == 0 || ggml_get_type_traits(type)->blck_size == 0) {
            continue;
        }
        const int64_t blck = ggml_blck_size(type);
        const int64_t ne0  = ((256 + blck - 1)/blck)*blck;
        const int64_t nr   = 67;
        const int64_t shapes[][2] = { {1, 1}, {13, 1}, {512, 1}, {5, 2} };
        for (const auto & shape : shapes) {
            ggml_context * ctx_b = new_ctx(1);
            ggml_tensor * src_b = ggml_new_tensor_3d(ctx_b, type, ne0, nr, shape[1]);
            ggml_set_name(src_b, "weight_b");
            ggml_backend_buffer_t buf_b = ggml_backend_alloc_ctx_tensors_from_buft(ctx_b, mapped);
            if (!buf_b) {
                printf("meta_rows: order=%s type=%s mapped allocation failed: FAIL\n", order, ggml_type_name(type));
                return 1;
            }
            const bool fallback = ggml_backend_buffer_get_type(buf_b) != mapped;
            if (fallback != nopin) {
                printf("meta_rows: order=%s type=%s fallback=%d but nopin=%d: FAIL\n", order, ggml_type_name(type), fallback, nopin);
                ok = false;
            }
            ggml_backend_buffer_set_usage(buf_b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            ggml_context * ctx_a = new_ctx(1);
            ggml_tensor * src_a = ggml_new_tensor_3d(ctx_a, type, ne0, nr, shape[1]);
            ggml_backend_buffer_t buf_a = ggml_backend_alloc_ctx_tensors(ctx_a, backend);
            if (!buf_a) {
                printf("meta_rows: order=%s type=%s device allocation failed: FAIL\n", order, ggml_type_name(type));
                return 1;
            }
            ggml_backend_buffer_set_usage(buf_a, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            // ids and the gathers: placed and allocated by the scheduler
            ggml_context * ctx = new_ctx(8);
            ggml_tensor * ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, shape[0], shape[1]);
            ggml_set_input(ids);
            ggml_tensor * out_a = ggml_get_rows(ctx, src_a, ids);
            ggml_tensor * out_b = ggml_get_rows(ctx, src_b, ids);
            ggml_set_output(out_a);
            ggml_set_output(out_b);

            const bool sup_a = ggml_backend_dev_supports_op(meta, out_a);
            const bool sup_b = ggml_backend_dev_supports_op(meta, out_b);
            if (sup_a && sup_b) {
                std::vector<uint8_t> raw = random_rows(type, ne0, nr*shape[1]);
                ggml_backend_tensor_set(src_a, raw.data(), 0, raw.size());
                ggml_backend_tensor_set(src_b, raw.data(), 0, raw.size());
                std::vector<int32_t> id_data(ggml_nelements(ids));
                for (auto & id : id_data) {
                    id = (int32_t) (rng() % nr);
                }
                const std::vector<uint8_t> ref = host_rows(type, out_b->type, ne0, nr, raw, id_data, shape[0], shape[1]);

                ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, out_a);
                ggml_build_forward_expand(gf, out_b);
                ggml_backend_t backends[2] = { backend, cpu };
                ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
                copy_log copies;
                ggml_backend_sched_set_copy_callback(sched, record_copy, &copies);
                bool computed = ggml_backend_sched_alloc_graph(sched, gf);
                if (computed) {
                    ggml_backend_tensor_set(ids, id_data.data(), 0, ggml_nbytes(ids));
                    computed = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
                }
                if (!computed) {
                    printf("meta_rows: order=%s type=%s compute failed: FAIL\n", order, ggml_type_name(type));
                    ok = false;
                } else if (ref.size() != ggml_nbytes(out_b)) {
                    printf("meta_rows: order=%s type=%s no host reference: FAIL\n", order, ggml_type_name(type));
                    ok = false;
                } else {
                    std::vector<uint8_t> va(ggml_nbytes(out_a)), vb(ggml_nbytes(out_b));
                    ggml_backend_tensor_get(out_a, va.data(), 0, va.size());
                    ggml_backend_tensor_get(out_b, vb.data(), 0, vb.size());
                    const size_t n_diff = n_words_differing(va, vb);
                    const bool   exact  = !ggml_is_quantized(type);
                    float  max_err = 0.0f;
                    const size_t n_ref_bad = n_off_reference(va, ref, exact, max_err) + n_off_reference(vb, ref, exact, max_err);
                    const size_t n_ref_diff = n_words_differing(vb, ref);
                    const bool a_on_meta = ggml_backend_sched_get_tensor_backend(sched, out_a) == backend;
                    const bool b_on_meta = ggml_backend_sched_get_tensor_backend(sched, out_b) == backend;
                    const int  n_splits  = ggml_backend_sched_get_n_splits(sched);
                    printf("meta_rows: order=%s type=%s rows=%" PRId64 "x%" PRId64 " a_on_meta=%d b_on_meta=%d splits=%d weight_copied=%d differing=%zu/%" PRId64
                        " ref_bad=%zu ref_bitwise_differing=%zu ref_max_err=%g\n",
                        order, ggml_type_name(type), shape[0], shape[1], a_on_meta, b_on_meta, n_splits, copies.weight_copied, n_diff, ggml_nelements(out_a),
                        n_ref_bad, n_ref_diff, max_err);
                    ok = ok && n_diff == 0 && n_ref_bad == 0 && a_on_meta && !copies.weight_copied;
                    if (nopin) {
                        ok = ok && !b_on_meta && n_splits == 2;
                    } else {
                        ok = ok && b_on_meta && n_splits == 1;
                    }
                    n_types += shape[0] == 1;
                    n_rows_checks++;
                }
                ggml_backend_sched_free(sched);
            } else if (sup_a != sup_b && !nopin) {
                printf("meta_rows: order=%s type=%s supports_op device-resident=%d mapped=%d: FAIL\n", order, ggml_type_name(type), sup_a, sup_b);
                ok = false;
            }
            ggml_backend_buffer_free(buf_a);
            ggml_backend_buffer_free(buf_b);
            ggml_free(ctx);
            ggml_free(ctx_a);
            ggml_free(ctx_b);
        }
    }
    printf("meta_rows: order=%s types=%d checks=%d %s\n", order, n_types, n_rows_checks, ok && n_rows_checks >= 100 ? "bitwise OK, placement OK" : "FAIL");
    ggml_backend_free(backend);
    return ok && n_rows_checks >= 100 ? 0 : 1;
}

int main(int argc, char ** argv) {
    const bool nopin = argc > 1 && strcmp(argv[1], "--nopin") == 0;
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        printf("meta_rows: no CPU backend: FAIL\n");
        return 1;
    }
    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpus.push_back(dev);
        }
    }
    if (gpus.size() != 2) {
        printf("meta_rows: SKIP (%zu GPU devices, need 2)\n", gpus.size());
        return 0;
    }
    // both orders always run: a failure in the first must not hide the second GPU's rows
    const int rc_01 = run_order(gpus[0], gpus[1], cpu, nopin);
    const int rc_10 = run_order(gpus[1], gpus[0], cpu, nopin);
    ggml_backend_free(cpu);
    if (rc_01 == 1 || rc_10 == 1) {
        printf("meta_rows: FAIL%s (2 orders)\n", nopin ? " nopin" : "");
        return 1;
    }
    if (rc_01 == 2 || rc_10 == 2) {
        printf("meta_rows: SKIP\n");
        return 0;
    }
    printf("meta_rows: PASS%s (2 orders)\n", nopin ? " nopin" : "");
    return 0;
}
