// GET_ROWS from a host buffer type that a device reads in place (an extra buffer type of the device that is host memory,
// e.g. CUDA_Host_Mapped):
//   - the result is bitwise equal to GET_ROWS from the device's own buffer, for every src0 type the device supports
//   - the device rejects every other op on such a buffer
//   - the scheduler runs GET_ROWS on the device without copying the weight; MUL_MAT on the same weight stays on the CPU
// An allocation may fall back to a plain CPU buffer (e.g. GGML_CUDA_NO_PINNED=1): GET_ROWS must then run on the CPU.
// Devices without such a buffer type are skipped.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cinttypes>
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

static void fill(ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> f(n);
    for (auto & x : f) {
        x = dist(rng);
    }
    std::vector<uint8_t> data(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F32) {
        memcpy(data.data(), f.data(), data.size());
    } else if (ggml_is_quantized(t->type) || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16) {
        std::vector<float> imatrix(t->ne[0], 1.0f);
        const int64_t nrows = n / t->ne[0];
        ggml_quantize_chunk(t->type, f.data(), data.data(), 0, nrows, t->ne[0],
            ggml_quantize_requires_imatrix(t->type) ? imatrix.data() : nullptr);
    } else {
        for (auto & b : data) {
            b = (uint8_t) rng();
        }
    }
    ggml_backend_tensor_set(t, data.data(), 0, data.size());
}

static ggml_tensor * set_ids(ggml_tensor * ids, int64_t n_rows) {
    std::vector<int32_t> v(ggml_nelements(ids));
    for (auto & id : v) {
        id = (int32_t) (rng() % n_rows);
    }
    ggml_backend_tensor_set(ids, v.data(), 0, v.size()*sizeof(int32_t));
    return ids;
}

static std::vector<uint8_t> get(const ggml_tensor * t) {
    std::vector<uint8_t> v(ggml_nbytes(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size());
    return v;
}

static size_t n_differing(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, size_t elsize) {
    size_t n = 0;
    for (size_t i = 0; i + elsize <= a.size(); i += elsize) {
        n += memcmp(a.data() + i, b.data() + i, elsize) != 0;
    }
    return n;
}

static ggml_context * new_ctx(size_t n_tensors) {
    ggml_init_params params = { ggml_tensor_overhead()*n_tensors + ggml_graph_overhead(), nullptr, true };
    return ggml_init(params);
}

// every type: the same GET_ROWS on the device from its own buffer (a) and from the mapped host buffer (b)
static bool test_bitwise(ggml_backend_dev_t dev, ggml_backend_t backend, ggml_backend_buffer_type_t mapped) {
    bool ok = true;
    int n_types = 0;
    for (int it = 0; it < GGML_TYPE_COUNT; ++it) {
        const ggml_type type = (ggml_type) it;
        if (ggml_type_size(type) == 0 || ggml_get_type_traits(type)->blck_size == 0) {
            continue;
        }
        const int64_t blck = ggml_blck_size(type);
        const int64_t ne0  = ((256 + blck - 1)/blck)*blck;
        const int64_t nr   = 67;

        // shapes: { rows per batch, batches }
        const int64_t shapes[][2] = { {1, 1}, {13, 1}, {512, 1}, {5, 2} };
        for (const auto & shape : shapes) {
            ggml_context * ctx_b = new_ctx(1);
            ggml_tensor * src_b = ggml_new_tensor_3d(ctx_b, type, ne0, nr, shape[1]);
            ggml_backend_buffer_t buf_b = ggml_backend_alloc_ctx_tensors_from_buft(ctx_b, mapped);
            GGML_ASSERT(buf_b);
            if (ggml_backend_buffer_get_type(buf_b) != mapped) {
                // allocation fell back to a CPU buffer: the device must not read it, only the scheduler test applies
                ggml_backend_buffer_free(buf_b);
                ggml_free(ctx_b);
                printf("host_mapped: dev=%s fallback=1, bitwise test skipped\n", ggml_backend_dev_name(dev));
                return true;
            }

            ggml_context * ctx = new_ctx(8);
            ggml_tensor * src_a = ggml_new_tensor_3d(ctx, type, ne0, nr, shape[1]);
            ggml_tensor * ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, shape[0], shape[1]);
            ggml_tensor * out_a = ggml_get_rows(ctx, src_a, ids);
            ggml_tensor * out_b = ggml_get_rows(ctx, src_b, ids);
            ggml_backend_buffer_t buf_a = ggml_backend_alloc_ctx_tensors(ctx, backend);
            GGML_ASSERT(buf_a);

            const bool sup_a = ggml_backend_dev_supports_op(dev, out_a);
            const bool sup_b = ggml_backend_dev_supports_op(dev, out_b);
            if (sup_a != sup_b) {
                printf("host_mapped: dev=%s type=%s supports_op device=%d mapped=%d: FAIL\n",
                    ggml_backend_dev_name(dev), ggml_type_name(type), sup_a, sup_b);
                ok = false;
            }
            if (sup_a && sup_b) {
                fill(src_a);
                std::vector<uint8_t> raw = get(src_a);
                ggml_backend_tensor_set(src_b, raw.data(), 0, raw.size());
                set_ids(ids, nr);

                ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, out_a);
                ggml_build_forward_expand(gf, out_b);
                if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                    printf("host_mapped: dev=%s type=%s compute failed: FAIL\n", ggml_backend_dev_name(dev), ggml_type_name(type));
                    ok = false;
                } else {
                    const size_t n_diff = n_differing(get(out_a), get(out_b), ggml_type_size(out_a->type));
                    printf("host_mapped: dev=%s type=%s rows=%" PRId64 "x%" PRId64 " differing=%zu/%" PRId64 "\n",
                        ggml_backend_dev_name(dev), ggml_type_name(type), shape[0], shape[1], n_diff, ggml_nelements(out_a));
                    ok = ok && n_diff == 0;
                }
                n_types += shape[0] == 1;
            }
            ggml_backend_buffer_free(buf_a);
            ggml_backend_buffer_free(buf_b);
            ggml_free(ctx);
            ggml_free(ctx_b);
        }
    }

    // a mapped weight is only gathered: other ops must not read it in place
    {
        ggml_context * ctx = new_ctx(5);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 64);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
        ggml_tensor * mm  = ggml_mul_mat(ctx, w, x);
        ggml_tensor * add = ggml_add(ctx, x, b);
        ggml_backend_buffer_t dummy = ggml_backend_buft_alloc_buffer(mapped, 0);
        w->buffer = dummy;
        b->buffer = dummy;
        const bool sup_mm  = ggml_backend_dev_supports_op(dev, mm);
        const bool sup_add = ggml_backend_dev_supports_op(dev, add);
        w->buffer = nullptr;
        b->buffer = nullptr;
        ggml_backend_buffer_free(dummy);
        ggml_free(ctx);
        printf("host_mapped: dev=%s types=%d supports_op mul_mat=%d add=%d (want 0 0)\n", ggml_backend_dev_name(dev), n_types, sup_mm, sup_add);
        ok = ok && !sup_mm && !sup_add && n_types > 0;
    }
    return ok;
}

struct copy_log {
    std::vector<std::string> names;
};

static bool record_copy(ggml_backend_t, const ggml_tensor * src, ggml_tensor *, ggml_cgraph *, void * user_data) {
    ((copy_log *) user_data)->names.push_back(src->name);
    return false; // the scheduler copies the whole weight
}

struct sched_result {
    ggml_backend_t       rows_backend = nullptr;
    ggml_backend_t       mm_backend   = nullptr;
    int                  n_splits     = 0;
    bool                 weight_copied = false;
    std::vector<uint8_t> rows;
};

// [dev, CPU] scheduler over rows = get_rows(w, ids) and mm = mul_mat(w, x); w (F16, exact on every backend) lives in buft
static sched_result run_sched(ggml_backend_t dev_backend, ggml_backend_t cpu_backend, ggml_backend_buffer_type_t buft,
                              bool force_rows_on_dev, bool with_mul_mat) {
    const int64_t ne0 = 256, nr = 4096, n_ids = 13;
    sched_result res;

    ggml_context * ctx_w = new_ctx(1);
    ggml_tensor * w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F16, ne0, nr);
    ggml_set_name(w, "weight");
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    GGML_ASSERT(buf_w);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 saved = rng;
    rng.seed(7);
    fill(w);

    ggml_context * ctx = new_ctx(8);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_ids);
    ggml_set_input(ids);
    ggml_tensor * rows = ggml_get_rows(ctx, w, ids);
    ggml_set_output(rows);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, rows);
    // x only exists in the graph with the mul_mat: the scheduler allocates graph inputs only
    ggml_tensor * x  = nullptr;
    ggml_tensor * mm = nullptr;
    if (with_mul_mat) {
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, 2);
        ggml_set_input(x);
        mm = ggml_mul_mat(ctx, w, x);
        ggml_set_output(mm);
        ggml_build_forward_expand(gf, mm);
    }

    ggml_backend_t backends[2] = { dev_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 64, false, true);
    copy_log copies;
    ggml_backend_sched_set_copy_callback(sched, record_copy, &copies);
    if (force_rows_on_dev) {
        ggml_backend_sched_set_tensor_backend(sched, rows, dev_backend);
    }
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, gf));
    set_ids(ids, nr);
    if (x) {
        std::vector<float> xv(ggml_nelements(x), 0.5f);
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
    }
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
    rng = saved;

    res.rows_backend = ggml_backend_sched_get_tensor_backend(sched, rows);
    res.mm_backend   = mm ? ggml_backend_sched_get_tensor_backend(sched, mm) : nullptr;
    res.n_splits     = ggml_backend_sched_get_n_splits(sched);
    for (const auto & name : copies.names) {
        res.weight_copied = res.weight_copied || name == "weight";
    }
    res.rows = get(rows);

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    return res;
}

static const char * name_of(ggml_backend_t backend) {
    return backend ? ggml_backend_name(backend) : "NULL";
}

static bool test_sched(ggml_backend_dev_t dev, ggml_backend_t dev_backend, ggml_backend_t cpu_backend, ggml_backend_buffer_type_t mapped) {
    const char * dname = ggml_backend_dev_name(dev);
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_get_default_buffer_type(cpu_backend);

    // reference and control: the weight in a CPU buffer gathers on the CPU; forced onto the device it is copied, which
    // shows that the copy callback sees a weight copy
    const sched_result ref   = run_sched(dev_backend, cpu_backend, cpu_buft, false, false);
    const sched_result moved = run_sched(dev_backend, cpu_backend, cpu_buft, true,  false);
    const sched_result gr    = run_sched(dev_backend, cpu_backend, mapped,   false, false);
    const sched_result mm    = run_sched(dev_backend, cpu_backend, mapped,   false, true);

    ggml_context * ctx = new_ctx(1);
    ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 16);
    ggml_backend_buffer_t probe = ggml_backend_alloc_ctx_tensors_from_buft(ctx, mapped);
    const bool fallback = ggml_backend_buffer_get_type(probe) != mapped;
    ggml_backend_buffer_free(probe);
    ggml_free(ctx);

    printf("host_mapped: dev=%s control cpu-buffer get_rows=%s; forced to device: copied=%d\n",
        dname, name_of(ref.rows_backend), moved.weight_copied);
    printf("host_mapped: dev=%s fallback=%d get_rows=%s splits=%d weight_copied=%d equal=%d\n",
        dname, fallback, name_of(gr.rows_backend), gr.n_splits, gr.weight_copied, gr.rows == ref.rows);
    printf("host_mapped: dev=%s same weight: get_rows=%s mul_mat=%s weight_copied=%d\n",
        dname, name_of(mm.rows_backend), name_of(mm.mm_backend), mm.weight_copied);

    bool ok = ref.rows_backend == cpu_backend && moved.rows_backend == dev_backend && moved.weight_copied;
    ok = ok && moved.rows == ref.rows && gr.rows == ref.rows && mm.rows == ref.rows;
    ok = ok && !gr.weight_copied && !mm.weight_copied && mm.mm_backend == cpu_backend;
    if (fallback) {
        ok = ok && gr.rows_backend == cpu_backend && mm.rows_backend == cpu_backend;
    } else {
        ok = ok && gr.rows_backend == dev_backend && gr.n_splits == 1 && mm.rows_backend == dev_backend;
    }
    return ok;
}

int main() {
    ggml_backend_load_all();

    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu_backend);

    int n_tested = 0;
    bool ok = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        ggml_backend_buffer_type_t mapped = mapped_host_buft(dev);
        if (!mapped) {
            printf("host_mapped: dev=%s no mapped host buffer type, skipped\n", ggml_backend_dev_name(dev));
            continue;
        }
        printf("host_mapped: dev=%s buft=%s\n", ggml_backend_dev_name(dev), ggml_backend_buft_name(mapped));
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        GGML_ASSERT(backend);
        const bool ok_bitwise = test_bitwise(dev, backend, mapped);
        const bool ok_sched   = test_sched(dev, backend, cpu_backend, mapped);
        printf("host_mapped: dev=%s bitwise %s, sched %s\n", ggml_backend_dev_name(dev),
            ok_bitwise ? "OK" : "FAIL", ok_sched ? "OK" : "FAIL");
        ok = ok && ok_bitwise && ok_sched;
        ggml_backend_free(backend);
        n_tested++;
    }
    ggml_backend_free(cpu_backend);

    if (n_tested == 0) {
        printf("test-host-mapped: SKIP (no device with a mapped host buffer type)\n");
        return 0;
    }
    printf("test-host-mapped: %s (%d devices)\n", ok ? "PASS" : "FAIL", n_tested);
    return ok ? 0 : 1;
}
