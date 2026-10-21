#include "ggml-alloc.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml.h"

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#ifndef _WIN32
#include <pthread.h>
#endif

//
// dummy backend with configurable max_buffer_size, tracks allocations

uint8_t * const alloc_base = (uint8_t *) 16;

struct dummy_backend_context {
    size_t max_buffer_size = 64;
    size_t alignment       = 8;

    ggml_backend_buffer_i              buffer_interface;
    ggml_backend_device                device;
    ggml_backend                       backend;
    std::vector<ggml_backend_buffer_t> buffers;

    bool custom_alloc_buffer_n_called   = false;
    bool custom_get_alloc_size_n_called = false;

    size_t allocated_total() const {
        size_t n = 0;
        for (ggml_backend_buffer_t buf : buffers) {
            n += ggml_backend_buffer_get_size(buf);
        }
        return n;
    }
};

// ggml_backend_buffer_type interface

static const char * dummy_backend_buffer_type_get_name(ggml_backend_buffer_type_t /*buft*/) {
    return "dummy_buffer_type";
}

static ggml_backend_buffer_t dummy_backend_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    dummy_backend_context * ctx    = (dummy_backend_context *) buft->context;
    ggml_backend_buffer_t & buffer = ctx->buffers.emplace_back();
    buffer                         = ggml_backend_buffer_init(buft, ctx->buffer_interface, ctx, size);
    return buffer;
}

static size_t dummy_backend_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    return ctx->alignment;
}

static size_t dummy_backend_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    return ctx->max_buffer_size;
}

static ggml_backend_buffer_t dummy_backend_buffer_type_alloc_buffer_n_custom(
        ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    ctx->custom_alloc_buffer_n_called = true;

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, 64);
    if (buffer == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < n_tensors; i++) {
        tensors[i]->buffer = buffer;
        tensors[i]->data   = (char *) ggml_backend_buffer_get_base(buffer);
    }
    return buffer;
}

static size_t dummy_backend_buffer_type_get_alloc_size_n_custom(
        ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    GGML_UNUSED(tensors);
    GGML_UNUSED(n_tensors);
    ctx->custom_get_alloc_size_n_called = true;
    return 64;
}

static bool dummy_backend_buffer_type_is_host(ggml_backend_buffer_type_t /*buft*/) {
    return true;
}

// ggml_backend_buffer interface

static void dummy_backend_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    dummy_backend_context * ctx = (dummy_backend_context *) buffer->context;

    auto i = std::find(ctx->buffers.begin(), ctx->buffers.end(), buffer);
    GGML_ASSERT(i != ctx->buffers.end());
    ctx->buffers.erase(i);
}

static void * dummy_backend_buffer_get_base(ggml_backend_buffer_t /*buft*/) {
    return alloc_base;
}

static ggml_status dummy_backend_buffer_init_tensor(ggml_backend_buffer_t /*buft*/, ggml_tensor * /*x*/) {
    return GGML_STATUS_SUCCESS;
}

static void dummy_backend_buffer_memset_tensor(ggml_backend_buffer_t /*buft*/, ggml_tensor * /*x*/, uint8_t, size_t, size_t) {}

static void dummy_backend_buffer_set_tensor(ggml_backend_buffer_t /*buft*/, ggml_tensor * /*x*/, const void *, size_t, size_t) {}

static void dummy_backend_buffer_get_tensor(ggml_backend_buffer_t /*buft*/, const ggml_tensor * /*x*/, void *, size_t, size_t) {}

static void dummy_backend_buffer_clear(ggml_backend_buffer_t /*buft*/, uint8_t /*val*/) {}

// ggml_backend_device interface

static enum ggml_backend_dev_type dummy_backend_device_get_type(ggml_backend_dev_t /*dev*/) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;
}

static bool dummy_backend_device_supports_op(ggml_backend_dev_t /*dev*/, const ggml_tensor * /*x*/) {
    return true;
}

static bool dummy_backend_device_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    return device->context == buft->context;
}

// ggml_backend interface

static const char * dummy_backend_get_name(ggml_backend_t /*backend*/) {
    return "dummy_backend";
}

// dummy_backend

struct dummy_backend {
    std::unique_ptr<dummy_backend_context> context;
    ggml_backend_buffer_type               buffer_type;
};

static dummy_backend dummy_backend_init(size_t max_buffer_size, size_t alignment = 8) {
    dummy_backend b{};
    b.context                  = std::make_unique<dummy_backend_context>();
    b.context->alignment       = alignment;
    b.context->max_buffer_size = max_buffer_size;

    b.context->buffer_interface.free_buffer   = dummy_backend_buffer_free_buffer;
    b.context->buffer_interface.get_base      = dummy_backend_buffer_get_base;
    b.context->buffer_interface.init_tensor   = dummy_backend_buffer_init_tensor;
    b.context->buffer_interface.memset_tensor = dummy_backend_buffer_memset_tensor;
    b.context->buffer_interface.set_tensor    = dummy_backend_buffer_set_tensor;
    b.context->buffer_interface.get_tensor    = dummy_backend_buffer_get_tensor;
    b.context->buffer_interface.clear         = dummy_backend_buffer_clear;

    b.context->device.context             = b.context.get();
    b.context->device.iface.get_type      = dummy_backend_device_get_type;
    b.context->device.iface.supports_op   = dummy_backend_device_supports_op;
    b.context->device.iface.supports_buft = dummy_backend_device_supports_buft;

    b.context->backend.context        = b.context.get();
    b.context->backend.device         = &b.context->device;
    b.context->backend.iface.get_name = dummy_backend_get_name;

    b.buffer_type.device              = &b.context->device;
    b.buffer_type.context             = b.context.get();
    b.buffer_type.iface.get_name      = dummy_backend_buffer_type_get_name;
    b.buffer_type.iface.alloc_buffer  = dummy_backend_buffer_type_alloc_buffer;
    b.buffer_type.iface.get_alignment = dummy_backend_buffer_type_get_alignment;
    b.buffer_type.iface.get_max_size  = dummy_backend_buffer_type_get_max_size;
    b.buffer_type.iface.is_host       = dummy_backend_buffer_type_is_host;
    return b;
}

//
// test utilities

struct test_context_with_graph {
    ggml_context *   ctx;
    ggml_cgraph *    graph;
    ggml_context_ptr ctx_ptr;
};

static test_context_with_graph make_context() {
    ggml_init_params params{};
    params.mem_size = 48 * ggml_tensor_overhead() + ggml_graph_overhead();
    params.no_alloc = true;

    ggml_context *   ctx     = ggml_init(params);
    ggml_context_ptr ctx_ptr = ggml_context_ptr(ctx);
    ggml_cgraph *    graph   = ggml_new_graph(ctx);
    return { ctx, graph, std::move(ctx_ptr) };
}

static ggml_tensor * make_input_1d(ggml_context * ctx, int64_t n_elements) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    ggml_set_input(t);
    return t;
}

static ggml_tensor * make_input_with_size(ggml_context * ctx, size_t size_bytes) {
    GGML_ASSERT(size_bytes % 4 == 0);
    return make_input_1d(ctx, size_bytes / 4);
}

static void assign_names(ggml_context * ctx, const char * prefix = "x") {
    int i = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        ggml_format_name(t, "%s%d", prefix, i++);
    }
}

static int get_leaf_id(ggml_cgraph * graph, const char * tensor_name) {
    for (int i = 0; i < graph->n_leafs; ++i) {
        if (strncmp(graph->leafs[i]->name, tensor_name, GGML_MAX_NAME) == 0) {
            return i;
        }
    }
    fprintf(stderr, "leaf not found: %s\n", tensor_name);
    return -1;
}

static int get_node_id(ggml_cgraph * graph, const char * tensor_name) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (strncmp(graph->nodes[i]->name, tensor_name, GGML_MAX_NAME) == 0) {
            return i;
        }
    }
    fprintf(stderr, "node not found: %s", tensor_name);
    return -1;
}

static ggml_gallocr_ptr allocate_graph(ggml_cgraph * graph, ggml_tensor * out, ggml_backend_buffer_type_t buft) {
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_ptr galloc = ggml_gallocr_ptr(ggml_gallocr_new(buft));
    bool             result = ggml_gallocr_alloc_graph(galloc.get(), graph);
    GGML_ASSERT(result);
    return galloc;
}

//
// correctness checks for result allocations

static void check_all_allocated(ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        GGML_ASSERT(t->buffer != nullptr);
        GGML_ASSERT(t->data != nullptr);
    }
}

static void check_max_size(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        auto * buft     = ggml_backend_buffer_get_type(t->buffer);
        size_t max_size = ggml_backend_buft_get_max_size(buft);
        size_t offset   = (char *) t->data - (char *) ggml_backend_buffer_get_base(t->buffer);
        GGML_ASSERT(t->data >= ggml_backend_buffer_get_base(t->buffer));
        GGML_ASSERT((size_t) offset + ggml_nbytes(t) <= max_size);
    }
}

static bool can_reuse_memory(ggml_cgraph * graph, int current_i, ggml_tensor * current, ggml_tensor * other) {
    if (other->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }
    // Check if `other` is still "alive", ie. an input to any node after the `current` op
    for (int i = current_i; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (t == current && ggml_op_can_inplace(t->op)) {
                continue;
            }
            if (t->src[s] == other) {
                return false;
            }
            if (t->src[s] && t->src[s]->view_src == other) {
                return false;
            }
        }
    }
    return true;
}

static bool memory_overlap(ggml_tensor * a, ggml_tensor * b) {
    if (a->buffer != b->buffer) {
        return false;
    }
    int64_t a0 = (int64_t) a->data;
    int64_t a1 = a0 + ggml_nbytes(a);
    int64_t b0 = (int64_t) b->data;
    int64_t b1 = b0 + ggml_nbytes(b);
    return a1 > b0 && b1 > a0;
}

static ggml_tensor * get_view_source(ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

static void check_no_overlap(ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        for (int j = 0; j < i; ++j) {
            ggml_tensor * t = ggml_graph_node(graph, i);
            ggml_tensor * o = ggml_graph_node(graph, j);
            GGML_ASSERT(t != o);

            if (get_view_source(t) == get_view_source(o)) {
                continue;
            }
            if (memory_overlap(t, o)) {
                GGML_ASSERT(can_reuse_memory(graph, i, t, o));
            }
        }
    }
}

//
// test cases

// Scenario where the first backend buffer is completely exhausted and there are further
// tensors which require a second buffer
static void test_max_size_too_many_tensors() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[7];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = ggml_mul(ctx, x[0], x[1]);
    x[4] = ggml_add(ctx, x[1], x[2]);
    x[5] = ggml_add(ctx, x[3], x[0]);
    x[6] = ggml_add(ctx, x[4], x[5]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[6], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 16 + 16);
}

// Scenario where there is some space left in the first buffer, but not enough to accommodate
// a larger tensor, so a second buffer is required
static void test_max_size_tensor_too_large() {
    dummy_backend backend      = dummy_backend_init(32);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 16);    // chunk 0, [0 , 16)
    x[1] = make_input_with_size(ctx, 8);     // chunk 0, [16, 24)
    x[2] = ggml_concat(ctx, x[0], x[1], 0);  // chunk 1, [0 , 24)
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[2], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 32 + 24);
}

// Scenario where a single tensor exceeds the max buffer size - in this case the allocator
// should try to create a bigger buffer anyway, and wait for the backend to throw an error.
// Backends may report an artificially lower max size in some cases for compatibility reasons.
static void test_tensor_larger_than_max_size() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 24);
    x[1] = ggml_scale(ctx, x[0], 2.0f);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[1], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() == 24);
}

// This test assumes a max of 16 buffer chunks, and tries to allocate tensors that would
// require more. Expectation is that the last buffer should grow to fit everything,
// leaving it to the backend to error out if it can't allocate that much.
static void test_not_enough_chunks() {
    const int max_chunks = 16;
    const int max_size   = 8;

    dummy_backend backend      = dummy_backend_init(max_size);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[max_chunks + 1];
    for (int i = 0; i < max_chunks + 1; ++i) {
        x[i] = make_input_with_size(ctx, max_size);
    }
    ggml_tensor * acc = x[0];
    for (int i = 0; i < max_chunks; ++i) {
        acc = ggml_add(ctx, acc, x[i + 1]);
    }
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, acc, &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() > max_chunks * max_size);
}

// Fill up leftover unallocated space of a chunk after allocating a large tensor that
// requires a new chunk.
static void test_fill_leftover_space() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = ggml_pad(ctx, x[0], 2, 0, 0, 0);
    x[3] = ggml_mean(ctx, x[1]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[3], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 12 + 16);
}

// Check that views don't require any extra memory
static void test_view_inplace() {
    dummy_backend backend      = dummy_backend_init(32);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[6];
    x[0] = make_input_1d(ctx, 4);                // chunk 0, [0, 16)
    x[1] = ggml_reshape_2d(ctx, x[0], 2, 2);     // view of x0
    x[2] = ggml_permute(ctx, x[1], 1, 0, 2, 3);  // view of x0
    x[3] = ggml_view_1d(ctx, x[2], 2, 4);        // view of x0
    x[4] = make_input_1d(ctx, 2);                // chunk 0, [16, 24)
    x[5] = ggml_add(ctx, x[3], x[4]);            // reuse (inplace add)
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[5], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 24);
}

static void test_reuse_and_free() {
    dummy_backend backend      = dummy_backend_init(40);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 24);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = ggml_add(ctx, x[1], x[2]);        // reuse, free x2
    x[4] = ggml_pad(ctx, x[0], 2, 0, 0, 0);  // alloc new buffer, free x0
    x[5] = ggml_scale(ctx, x[4], 2.0f);      // alloc from free block
    x[6] = ggml_add(ctx, x[4], x[5]);        // reuse, free x5
    x[7] = ggml_view_1d(ctx, x[6], 2, 8);    // view
    x[8] = ggml_add(ctx, x[3], x[7]);        // reuse
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[8], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 40 + 32 + 32);
}

static void test_merge_free_block(size_t max_buffer_size) {
    dummy_backend backend      = dummy_backend_init(max_buffer_size);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = make_input_with_size(ctx, 16);
    x[2] = make_input_with_size(ctx, 16);
    x[3] = ggml_mean(ctx, x[0]);
    x[4] = ggml_mean(ctx, x[1]);
    x[5] = ggml_pad(ctx, x[2], 2, 0, 0, 0);
    x[6] = ggml_add(ctx, x[3], x[4]);
    x[7] = ggml_pad(ctx, x[6], 5, 0, 0, 0);
    x[8] = ggml_add(ctx, x[5], x[7]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[8], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 32 + 32 + 24);
}

// Check that previously allocated but freed memory is preferred over allocating
// additional memory, even if the remaining space in a chunk would match tensor size better
static void test_prefer_already_allocated_memory() {
    dummy_backend backend      = dummy_backend_init(32, /*align*/ 4);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 24);  // [24b][8b unused]
    x[1] = ggml_mean(ctx, x[0]);           // [24b free][4b][4b unused]
    x[2] = ggml_mean(ctx, x[1]);           // should be allocated in the 24b block
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[2], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() <= 28);
}

// test for allocating on multiple devices with some tensors in the graph
// allocated externally (not by gallocr).
static void test_multiple_buffer_types() {
    dummy_backend backend_a = dummy_backend_init(32);
    dummy_backend backend_b = dummy_backend_init(SIZE_MAX);

    auto [ctx_a, _a, ctx_a_ptr] = make_context();
    auto [ctx_b, _b, ctx_b_ptr] = make_context();
    auto [ctx, graph, ctx_ptr]  = make_context();

    ggml_tensor * a[2];
    a[0] = make_input_with_size(ctx_a, 16);
    a[1] = make_input_with_size(ctx_a, 16);
    assign_names(ctx_a, "a");

    ggml_tensor * b[2];
    b[0] = make_input_with_size(ctx_b, 24);
    b[1] = make_input_with_size(ctx_b, 4);
    assign_names(ctx_b, "b");

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_mul(ctx, x[0], a[0]);
    x[2] = ggml_pad(ctx, x[1], 2, 0, 0, 0);
    x[3] = ggml_mul(ctx, x[2], b[0]);
    x[4] = ggml_mean(ctx, x[3]);
    x[5] = ggml_add(ctx, x[4], b[1]);
    x[6] = ggml_pad(ctx, x[5], 3, 0, 0, 0);
    x[7] = ggml_add(ctx, x[6], a[1]);
    x[8] = ggml_scale(ctx, x[7], 2.0f);
    assign_names(ctx, "x");

    ggml_backend_buffer_ptr    buf_a(ggml_backend_alloc_ctx_tensors_from_buft(ctx_a, &backend_a.buffer_type));
    ggml_backend_buffer_ptr    buf_b(ggml_backend_alloc_ctx_tensors_from_buft(ctx_b, &backend_b.buffer_type));
    ggml_backend_buffer_type_t bufts[2] = { &backend_a.buffer_type, &backend_b.buffer_type };

    // assign buffer types manually to avoid extra complexity from backend scheduler
    ggml_set_output(x[8]);
    ggml_build_forward_expand(graph, x[8]);

    GGML_ASSERT(graph->n_leafs == 5);
    int leaf_buffer_ids[5];
    leaf_buffer_ids[get_leaf_id(graph, "a0")] = 0;
    leaf_buffer_ids[get_leaf_id(graph, "a1")] = 0;
    leaf_buffer_ids[get_leaf_id(graph, "b0")] = 1;
    leaf_buffer_ids[get_leaf_id(graph, "b1")] = 1;
    leaf_buffer_ids[get_leaf_id(graph, "x0")] = 0;

    GGML_ASSERT(graph->n_nodes == 8);
    int node_buffer_ids[8];
    node_buffer_ids[get_node_id(graph, "x1")] = 0;
    node_buffer_ids[get_node_id(graph, "x2")] = 0;
    node_buffer_ids[get_node_id(graph, "x3")] = 1;
    node_buffer_ids[get_node_id(graph, "x4")] = 1;
    node_buffer_ids[get_node_id(graph, "x5")] = 1;
    node_buffer_ids[get_node_id(graph, "x6")] = 1;
    node_buffer_ids[get_node_id(graph, "x7")] = 0;
    node_buffer_ids[get_node_id(graph, "x8")] = 0;

    ggml_gallocr_ptr galloc(ggml_gallocr_new_n(bufts, 2));
    ggml_gallocr_reserve_n(galloc.get(), graph, node_buffer_ids, leaf_buffer_ids);
    ggml_gallocr_alloc_graph(galloc.get(), graph);

    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend_a.context->allocated_total() <= 32 + 32 + 24);
    GGML_ASSERT(backend_b.context->allocated_total() <= 32 + 24);
}

static void test_buffer_size_zero() {
    dummy_backend backend_a    = dummy_backend_init(SIZE_MAX);
    dummy_backend backend_b    = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_scale(ctx, x[0], 2.0f);

    ggml_set_output(x[1]);
    ggml_build_forward_expand(graph, x[1]);

    int leaf_buffer_ids[1] = { 0 };
    int node_buffer_ids[1] = { 0 };

    ggml_backend_buffer_type_t bufts[2] = { &backend_a.buffer_type, &backend_b.buffer_type };
    ggml_gallocr_ptr           galloc   = ggml_gallocr_ptr(ggml_gallocr_new_n(bufts, 2));
    bool                       res1     = ggml_gallocr_reserve_n(galloc.get(), graph, node_buffer_ids, leaf_buffer_ids);
    bool                       res2     = ggml_gallocr_alloc_graph(galloc.get(), graph);
    GGML_ASSERT(res1 && res2);

    check_all_allocated(graph);
    GGML_ASSERT(backend_a.context->allocated_total() == 16);
    GGML_ASSERT(backend_b.context->allocated_total() == 0);
}

// Test re-using gallocr for a different graph. The new graph has the same
// total size, but one of the chunks is larger, so reallocation is required.
static void test_reallocation() {
    dummy_backend    backend = dummy_backend_init(32, /*align*/ 4);
    ggml_gallocr_ptr galloc;
    {
        auto [ctx, graph, ctx_ptr] = make_context();
        ggml_tensor * x[4];
        x[0] = make_input_with_size(ctx, 24);
        x[1] = make_input_with_size(ctx, 16);
        x[2] = ggml_view_1d(ctx, x[0], 4, 0);
        x[3] = ggml_add(ctx, x[2], x[1]);
        assign_names(ctx);

        galloc = allocate_graph(graph, x[3], &backend.buffer_type);
        check_all_allocated(graph);
        GGML_ASSERT(backend.context->allocated_total() == 40);
    }
    {
        auto [ctx, graph, ctx_ptr] = make_context();
        ggml_tensor * x[3];
        x[0] = make_input_with_size(ctx, 20);
        x[1] = make_input_with_size(ctx, 20);
        x[2] = ggml_add(ctx, x[0], x[1]);
        assign_names(ctx);
        ggml_set_output(x[2]);
        ggml_build_forward_expand(graph, x[2]);

        bool result = ggml_gallocr_alloc_graph(galloc.get(), graph);
        GGML_ASSERT(result);
        check_all_allocated(graph);
        GGML_ASSERT(backend.context->allocated_total() == 40);
    }
}

static void test_backend_graph_optimize(ggml_backend_t /*backend*/, ggml_cgraph * graph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(graph->n_nodes == 3);
    params->add_alloc_dep(params->user_data, graph->nodes[0], graph->nodes[2]);
}

static bool graph_reuses_allocation(bool add_alloc_dep) {
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_scale(ctx, x[0], 2.0f);
    x[2] = ggml_scale(ctx, x[1], 2.0f);
    x[3] = ggml_scale(ctx, x[2], 2.0f);

    ggml_set_output(x[3]);
    ggml_build_forward_expand(graph, x[3]);

    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    if (add_alloc_dep) {
        backend.context->backend.iface.graph_optimize = test_backend_graph_optimize;
    }

    ggml_backend_t             backend_ptr = &backend.context->backend;
    ggml_backend_buffer_type_t buft        = &backend.buffer_type;
    ggml_backend_sched_ptr     sched(ggml_backend_sched_new(&backend_ptr, &buft, 1, 8, false, true));
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));

    return x[1]->data == x[2]->data;
}

static void test_graph_optimize_alloc_dep() {
    GGML_ASSERT(graph_reuses_allocation(false));
    GGML_ASSERT(!graph_reuses_allocation(true));
}

// Check that the size reported by ggml_backend_alloc_ctx_tensors_from_buft_size
// matches the actual size of the buffer allocated for the ctx tensors
static ggml_backend_buffer_ptr check_size_matches(ggml_backend_buffer_type_t buft, ggml_context * ctx) {
    std::vector<ggml_tensor *> tensors;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        tensors.push_back(t);
    }

    const size_t expected_size = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft);
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(buft, tensors.data(), (int) tensors.size()) == expected_size);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft));
    GGML_ASSERT((buffer != nullptr) == (expected_size != 0));
    if (buffer) {
        GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == expected_size);
    }
    return buffer;
}

// Check that all tensors are placed in a single buffer when they fit within the
// backend's max size
static void test_buft_alloc_buffer_n_single_buffer() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = ggml_add(ctx, x[0], x[1]);
    assign_names(ctx);

    ggml_tensor * tensors[3] = { x[0], x[1], x[2] };
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 3));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(!ggml_backend_buffer_is_multi_buffer(buffer.get()));
    GGML_ASSERT(backend.context->buffers.size() == 1);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 24);
    for (ggml_tensor * t : tensors) {
        GGML_ASSERT(t->buffer == buffer.get());
        GGML_ASSERT(t->data != nullptr);
    }
    for (int i = 0; i < 3; i++) {
        for (int j = i + 1; j < 3; j++) {
            GGML_ASSERT(!memory_overlap(tensors[i], tensors[j]));
        }
    }
}

// Check that tensors are split across multiple underlying buffers when they don't
// fit within the backend's max size
static void test_buft_alloc_buffer_n_multi_buffer() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[4] = { x[0], x[1], x[2], x[3] };
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 4));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer.get()));
    GGML_ASSERT(backend.context->buffers.size() == 2);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 32);
    for (ggml_tensor * t : tensors) {
        GGML_ASSERT(t->buffer != nullptr);
        GGML_ASSERT(t->data != nullptr);
    }
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            GGML_ASSERT(!memory_overlap(tensors[i], tensors[j]));
        }
    }
}

// Check that allocating tensors with zero total size returns nullptr
static void test_buft_alloc_buffer_n_zero_size() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_1d(ctx, 0);
    x[1] = make_input_1d(ctx, 0);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2) == nullptr);
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft(ctx, &backend.buffer_type) == nullptr);
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, &backend.buffer_type) == 0);
}

// Check that tensors already allocated to a buffer are left in place, and only the
// remaining tensors are allocated to a new buffer
static void test_buft_alloc_buffer_n_already_allocated() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * a[2];
    a[0] = make_input_with_size(ctx, 8);
    a[1] = make_input_with_size(ctx, 8);
    assign_names(ctx, "a");

    ggml_backend_buffer_ptr buf_a(ggml_backend_alloc_ctx_tensors_from_buft(ctx, &backend.buffer_type));
    GGML_ASSERT(buf_a != nullptr);

    ggml_tensor * b = make_input_with_size(ctx, 8);
    assign_names(ctx, "b");

    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, &backend.buffer_type) == 8);
    GGML_ASSERT(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, a, 2) == nullptr);

    ggml_tensor * tensors[3] = { a[0], a[1], b };
    ggml_backend_buffer_ptr buf_b(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 3));
    GGML_ASSERT(buf_b != nullptr);
    GGML_ASSERT(backend.context->buffers.size() == 2);
    GGML_ASSERT(a[0]->buffer == buf_a.get());
    GGML_ASSERT(a[1]->buffer == buf_a.get());
    GGML_ASSERT(b->buffer == buf_b.get());
    GGML_ASSERT(ggml_backend_buffer_get_size(buf_b.get()) == 8);
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft(ctx, &backend.buffer_type) == nullptr);
}

// Check that views don't require any extra memory and share the base tensor's data
static void test_buft_alloc_buffer_n_views() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * base  = make_input_1d(ctx, 4);
    ggml_tensor * view  = ggml_view_1d(ctx, base, 2, 0);
    ggml_tensor * extra = make_input_1d(ctx, 2);
    assign_names(ctx);

    ggml_tensor * tensors[3] = { base, view, extra };
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 3));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(backend.context->buffers.size() == 1);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 24);
    GGML_ASSERT(base->buffer == buffer.get());
    GGML_ASSERT(view->buffer == buffer.get());
    GGML_ASSERT(extra->buffer == buffer.get());
    GGML_ASSERT(view->data == base->data);
}

// Check that the reported size matches the allocated size in various scenarios:
// single buffer, multi buffer, and views
static void test_alloc_ctx_tensors_from_buft_size_matches() {
    {
        dummy_backend backend      = dummy_backend_init(SIZE_MAX, 8);
        auto [ctx, graph, ctx_ptr] = make_context();

        ggml_tensor * x[3];
        x[0] = make_input_with_size(ctx, 4);
        x[1] = make_input_with_size(ctx, 4);
        x[2] = make_input_with_size(ctx, 8);
        assign_names(ctx);

        GGML_UNUSED(x);

        ggml_backend_buffer_ptr buffer = check_size_matches(&backend.buffer_type, ctx);
        GGML_ASSERT(buffer != nullptr);
        GGML_ASSERT(backend.context->allocated_total() == 24);
    }
    {
        dummy_backend backend      = dummy_backend_init(16);
        auto [ctx, graph, ctx_ptr] = make_context();

        ggml_tensor * x[4];
        x[0] = make_input_with_size(ctx, 8);
        x[1] = make_input_with_size(ctx, 8);
        x[2] = make_input_with_size(ctx, 8);
        x[3] = make_input_with_size(ctx, 8);
        assign_names(ctx);

        GGML_UNUSED(x);

        ggml_backend_buffer_ptr buffer = check_size_matches(&backend.buffer_type, ctx);
        GGML_ASSERT(buffer != nullptr);
        GGML_ASSERT(backend.context->allocated_total() == 32);
    }
    {
        dummy_backend backend      = dummy_backend_init(SIZE_MAX);
        auto [ctx, graph, ctx_ptr] = make_context();

        ggml_tensor * base  = make_input_1d(ctx, 4);
        ggml_tensor * view  = ggml_view_1d(ctx, base, 2, 0);
        ggml_tensor * extra = make_input_1d(ctx, 2);
        assign_names(ctx);

        GGML_UNUSED(view);
        GGML_UNUSED(extra);

        ggml_backend_buffer_ptr buffer = check_size_matches(&backend.buffer_type, ctx);
        GGML_ASSERT(buffer != nullptr);
        GGML_ASSERT(backend.context->allocated_total() == 24);
    }
}

// Check that a backend-provided alloc_buffer_n implementation takes precedence over
// the default one
static void test_buft_alloc_buffer_n_custom_override() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    backend.buffer_type.iface.alloc_buffer_n = dummy_backend_buffer_type_alloc_buffer_n_custom;

    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(backend.context->custom_alloc_buffer_n_called);
    GGML_ASSERT(backend.context->buffers.size() == 1);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 64);
    GGML_ASSERT(x[0]->buffer == buffer.get());
    GGML_ASSERT(x[1]->buffer == buffer.get());
}

// Check that get_alloc_size_n predicts a single buffer allocation
static void test_buft_get_alloc_size_n_single_buffer() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 2) == 16);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 16);
}

// Check that get_alloc_size_n accounts for splitting into multiple buffers
static void test_buft_get_alloc_size_n_multi_buffer() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[4] = { x[0], x[1], x[2], x[3] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 4) == 32);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 4));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer.get()));
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 32);
}

// Check that a tensor larger than max_size is still accounted for
static void test_buft_get_alloc_size_n_single_tensor_exceeds_max() {
    dummy_backend backend      = dummy_backend_init(8);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x = make_input_with_size(ctx, 16);
    assign_names(ctx);

    ggml_tensor * tensors[1] = { x };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 1) == 16);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 1));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 16);
}

// Check that zero-size tensors report a total allocation size of 0
static void test_buft_get_alloc_size_n_zero_size() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_1d(ctx, 0);
    x[1] = make_input_1d(ctx, 0);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 2) == 0);
    GGML_ASSERT(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2) == nullptr);
}

// Check that already-allocated tensors are not counted again
static void test_buft_get_alloc_size_n_already_allocated() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * a[2];
    a[0] = make_input_with_size(ctx, 8);
    a[1] = make_input_with_size(ctx, 8);
    assign_names(ctx, "a");

    ggml_backend_buffer_ptr buf_a(ggml_backend_alloc_ctx_tensors_from_buft(ctx, &backend.buffer_type));
    GGML_ASSERT(buf_a != nullptr);

    ggml_tensor * b = make_input_with_size(ctx, 8);
    assign_names(ctx, "b");

    ggml_tensor * tensors[3] = { a[0], a[1], b };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 3) == 8);

    ggml_backend_buffer_ptr buf_b(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 3));
    GGML_ASSERT(buf_b != nullptr);
    GGML_ASSERT(backend.context->buffers.size() == 2);
    GGML_ASSERT(a[0]->buffer == buf_a.get());
    GGML_ASSERT(a[1]->buffer == buf_a.get());
    GGML_ASSERT(b->buffer == buf_b.get());
    GGML_ASSERT(ggml_backend_buffer_get_size(buf_b.get()) == 8);
}

// Check that views do not add to the projected allocation size
static void test_buft_get_alloc_size_n_views() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * base  = make_input_1d(ctx, 4);
    ggml_tensor * view  = ggml_view_1d(ctx, base, 2, 0);
    ggml_tensor * extra = make_input_1d(ctx, 2);
    assign_names(ctx);

    ggml_tensor * tensors[3] = { base, view, extra };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 3) == 24);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 3));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 24);
}

// Check that n_tensors == 0 is handled
static void test_buft_get_alloc_size_n_n_tensors_zero() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, nullptr, 0) == 0);
    GGML_ASSERT(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, nullptr, 0) == nullptr);
}

// Check that get_alloc_size_n respects the backend alignment
static void test_buft_get_alloc_size_n_alignment() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX, 16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 4);
    x[1] = make_input_with_size(ctx, 4);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 2) == 32);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 32);
}

// Check that a backend-provided get_alloc_size_n implementation takes precedence over
// the default one
static void test_buft_get_alloc_size_n_custom_override() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    backend.buffer_type.iface.alloc_buffer_n   = dummy_backend_buffer_type_alloc_buffer_n_custom;
    backend.buffer_type.iface.get_alloc_size_n = dummy_backend_buffer_type_get_alloc_size_n_custom;

    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 2) == 64);
    GGML_ASSERT(backend.context->custom_get_alloc_size_n_called);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(backend.context->custom_alloc_buffer_n_called);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 64);
}

// Check that the custom get_alloc_size_n override is used by the context size helper
static void test_buft_get_alloc_size_n_custom_override_ctx_size() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    backend.buffer_type.iface.get_alloc_size_n = dummy_backend_buffer_type_get_alloc_size_n_custom;

    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    GGML_UNUSED(x);

    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, &backend.buffer_type) == 64);
}

// Check that querying the projected allocation size does not allocate anything
static void test_buft_get_alloc_size_n_does_not_allocate() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 2) == 16);
    GGML_ASSERT(backend.context->buffers.empty());
    GGML_ASSERT(x[0]->data == nullptr);
    GGML_ASSERT(x[1]->data == nullptr);
}

// Rebuilding a different selected branch must not use stack proportional to its ancestor chain.
static ggml_backend_meta_split_state test_meta_split_state_mirrored(const ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

static void test_meta_split_state_rebuild_impl(bool select_other) {
    const int n_chain = 512;

    ggml_backend_load_all();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_dev_t devs[2] = {cpu, cpu};
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2, test_meta_split_state_mirrored, nullptr);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_dev_buffer_type(meta));

    // the same memory for both builds, so that the rebuilt graph reuses the tensor addresses
    std::vector<uint8_t> mem(ggml_tensor_overhead()*(n_chain + 3) + ggml_graph_overhead_custom(n_chain + 3, false));

    for (int build = 0; build < 2; build++) {
        ggml_init_params params = {mem.size(), mem.data(), /*no_alloc =*/ true};
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * cur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_input(cur);
        for (int i = 0; i < n_chain; i++) {
            cur = ggml_scale(ctx, cur, 1.0f);
        }
        ggml_tensor * branches[2] = {ggml_scale(ctx, cur, 1.0f), ggml_scale(ctx, cur, 1.0f)};
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_chain + 3, false);
        ggml_tensor * out = ggml_build_forward_select(gf, branches, 2, select_other ? build : 0);
        GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));

        const float value = 3.0f;
        float result = 0.0f;
        ggml_backend_tensor_set(out, &value, 0, sizeof(value));
        ggml_backend_tensor_get(out, &result, 0, sizeof(result));
        GGML_ASSERT(result == value);
        ggml_free(ctx);
    }
    ggml_gallocr_free(galloc);
}

static void test_meta_split_state_rebuild(bool select_other) {
#ifndef _WIN32
    pthread_attr_t attr;
    GGML_ASSERT(pthread_attr_init(&attr) == 0);
    // 512 uncached nodes exceed 1 MiB recursively; the iterative walk has constant stack use.
    GGML_ASSERT(pthread_attr_setstacksize(&attr, 1024*1024) == 0);
    pthread_t thread;
    GGML_ASSERT(pthread_create(&thread, &attr, [](void * arg) -> void * {
        test_meta_split_state_rebuild_impl(*static_cast<bool *>(arg));
        return nullptr;
    }, &select_other) == 0);
    GGML_ASSERT(pthread_attr_destroy(&attr) == 0);
    GGML_ASSERT(pthread_join(thread, nullptr) == 0);
#else
    test_meta_split_state_rebuild_impl(select_other);
#endif
}

static ggml_backend_meta_split_state test_meta_split_state_rows(const ggml_tensor * tensor, void * userdata) {
    if (strcmp(tensor->name, "cache") == 0 || strcmp(tensor->name, "rows") == 0) {
        return {GGML_BACKEND_SPLIT_AXIS_0, {tensor->ne[0]/2, tensor->ne[0]/2}, {1}, 1};
    }
    return test_meta_split_state_mirrored(tensor, userdata);
}

static void test_meta_host_view(ggml_op op, bool split) {
    ggml_backend_load_all();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_dev_t devs[2] = {cpu, cpu};
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2,
            split ? test_meta_split_state_rows : test_meta_split_state_mirrored, nullptr);
    ggml_backend_ptr backend(ggml_backend_dev_init(meta, nullptr));
    GGML_ASSERT(backend);

    ggml_init_params params = {ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * cache = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 8, 4);
    ggml_set_name(cache, "cache");
    ggml_tensor * rows = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 8, 2);
    ggml_set_name(rows, "rows");
    ggml_tensor * idxs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 2);
    ggml_tensor * view = ggml_view_2d(ctx.get(), cache, 8, 2, cache->nb[1], 0);
    ggml_tensor * out = view;
    if (op == GGML_OP_CPY) {
        out = ggml_cpy(ctx.get(), rows, view);
    } else if (op == GGML_OP_SET_ROWS) {
        out = ggml_set_rows(ctx.get(), cache, rows, idxs);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, out);
    gf->uid = 1;
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    GGML_ASSERT(buffer && ggml_backend_buffer_is_host(buffer.get()));

    // CPU Meta buffers are host buffers, but their views and writes still belong to the Meta plan.
    GGML_ASSERT(out->view_src == cache && cache->op == GGML_OP_NONE);
    const int64_t indices[2] = {1, 3};
    ggml_backend_tensor_set(idxs, indices, 0, sizeof(indices));
    for (int pass = 0; pass < 3; ++pass) {
        float data[16], expected[32] = {}, result[32];
        for (int i = 0; i < 16; ++i) {
            data[i] = float(i + 1 + pass*16);
        }
        ggml_backend_tensor_set(cache, expected, 0, sizeof(expected));
        ggml_backend_tensor_set(rows, data, 0, sizeof(data));
        if (op == GGML_OP_CPY) {
            memcpy(expected, data, sizeof(data));
        } else if (op == GGML_OP_SET_ROWS) {
            memcpy(expected + 8, data, 8*sizeof(float));
            memcpy(expected + 24, data + 8, 8*sizeof(float));
        }
        const ggml_status status = ggml_backend_graph_compute(backend.get(), gf);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(cache, result, 0, sizeof(result));
        GGML_ASSERT(memcmp(expected, result, sizeof(result)) == 0);
    }
}

// Computing an allocated graph again after changing a node's op_params must apply the new values on every device.
static void test_meta_op_params_refresh() {
    ggml_backend_load_all();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_dev_t devs[2] = {cpu, cpu};
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2, test_meta_split_state_mirrored, nullptr);
    ggml_backend_t backend = ggml_backend_dev_init(meta, nullptr);
    GGML_ASSERT(backend != nullptr);

    ggml_init_params params = {ggml_tensor_overhead()*2 + ggml_graph_overhead(), nullptr, /*no_alloc =*/ true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(x);
    ggml_tensor * y = ggml_scale(ctx, x, 2.0f);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));

    // y may be computed in place of x, so x is set before each compute
    const float x_val = 1.0f;
    float y_val = 0.0f;
    ggml_backend_tensor_set(x, &x_val, 0, sizeof(x_val));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(y, &y_val, 0, sizeof(y_val));
    GGML_ASSERT(y_val == 2.0f);

    ggml_set_op_params_f32(y, 0, 3.0f);
    ggml_backend_tensor_set(x, &x_val, 0, sizeof(x_val));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(y, &y_val, 0, sizeof(y_val));
    GGML_ASSERT(y_val == 3.0f);

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static ggml_backend_meta_split_state test_meta_split_state_gather_rows(const ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(userdata);
    if (strcmp(tensor->name, "w") == 0) {
        return {GGML_BACKEND_SPLIT_AXIS_1, {4, 4}, {1}, 1};
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

// Gather slices carry per-rank pad offsets in op_params; the per-compute op_params refresh must keep them.
static void test_meta_gather_op_params_refresh() {
    ggml_backend_load_all();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_dev_t devs[2] = {cpu, cpu};
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2, test_meta_split_state_gather_rows, nullptr);
    ggml_backend_t backend = ggml_backend_dev_init(meta, nullptr);
    GGML_ASSERT(backend != nullptr);

    ggml_context * wctx = ggml_init({ggml_tensor_overhead(), nullptr, /*no_alloc =*/ true});
    ggml_tensor * w = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 8, 8);
    ggml_set_name(w, "w");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    GGML_ASSERT(wbuf != nullptr);
    float wv[64];
    for (int k = 0; k < 64; ++k) {
        wv[k] = float(k % 5 - 2) * 0.125f;
    }
    ggml_backend_tensor_set(w, wv, 0, sizeof(wv));

    // rank 0 computes rows 0-3 and rank 1 rows 4-7 of w*x; the gather pads each into place and the sum completes it
    ggml_context * ctx = ggml_init({8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, /*no_alloc =*/ true});
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    ggml_set_input(x);
    ggml_tensor * gather = ggml_backend_meta_gather(ctx, ggml_mul_mat(ctx, w, x));
    ggml_tensor * y = ggml_scale(ctx, ggml_cont(ctx, gather), 2.0f);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));

    // uid 0: every compute rebuilds the subgraphs and refreshes op_params; the second one also changes the scale
    for (int step = 0; step < 2; ++step) {
        const float scale = step == 0 ? 2.0f : 3.0f;
        ggml_set_op_params_f32(y, 0, scale);
        float xv[16];
        for (int k = 0; k < 16; ++k) {
            xv[k] = float((k + step) % 7 - 3) * 0.25f;
        }
        ggml_backend_tensor_set(x, xv, 0, sizeof(xv));
        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        float yv[16];
        ggml_backend_tensor_get(y, yv, 0, sizeof(yv));
        // host reference; all values are small dyadic fractions, so the sums are exact in any order
        for (int r = 0; r < 2; ++r) {
            for (int n = 0; n < 8; ++n) {
                float expected = 0.0f;
                for (int k = 0; k < 8; ++k) {
                    expected += wv[n*8 + k] * xv[r*8 + k];
                }
                GGML_ASSERT(yv[r*8 + n] == scale * expected);
            }
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(backend);
}

struct test_meta_comm_context {
    ggml_backend_t backends[2];
};

static void * (*test_meta_get_proc)(ggml_backend_reg_t, const char *);
static bool test_meta_merge;
static int test_meta_decline;
static std::vector<size_t> test_meta_runs;

// Mirrors the CUDA collective skip: a repeated tuple of non-zero subgraph UIDs must imply unchanged node properties.
struct test_meta_uid_state {
    std::vector<uint64_t> uids;
    std::vector<uint8_t>  props;
};
static std::map<const ggml_cgraph *, test_meta_uid_state> test_meta_uids;
static size_t test_meta_uid_skips;
static size_t test_meta_uid_scans;

static void test_meta_check_uids(ggml_cgraph ** graphs, size_t n_graphs) {
    std::vector<uint64_t> uids;
    std::vector<uint8_t> props;
    bool uids_set = true;
    auto append = [&](const void * p, size_t n) {
        props.insert(props.end(), (const uint8_t *) p, (const uint8_t *) p + n);
    };
    for (size_t k = 0; k < 2*n_graphs; ++k) {
        uids.push_back(graphs[k]->uid);
        uids_set = uids_set && graphs[k]->uid != 0;
        append(&graphs[k]->n_nodes, sizeof(graphs[k]->n_nodes));
        for (int i = 0; i < graphs[k]->n_nodes; ++i) {
            const ggml_tensor * node = graphs[k]->nodes[i];
            append(node, sizeof(ggml_tensor));
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (node->src[j]) {
                    append(&node->src[j]->data, sizeof(node->src[j]->data));
                    append(node->src[j]->ne, sizeof(node->src[j]->ne));
                    append(node->src[j]->nb, sizeof(node->src[j]->nb));
                }
            }
        }
    }
    // Keyed like the CUDA collective executable: by the run's first graph object.
    test_meta_uid_state & state = test_meta_uids[graphs[0]];
    if (uids_set && uids == state.uids) {
        GGML_ASSERT(props == state.props);
        ++test_meta_uid_skips;
        return;
    }
    ++test_meta_uid_scans;
    state.uids = uids_set ? uids : std::vector<uint64_t>();
    state.props = std::move(props);
}

static void * test_meta_comm_init(ggml_backend_t * backends, size_t n) {
    GGML_ASSERT(n == 2);
    return new test_meta_comm_context{{backends[0], backends[1]}};
}

static void test_meta_comm_free(void * ctx) {
    delete static_cast<test_meta_comm_context *>(ctx);
}

static bool test_meta_comm_allreduce(void *, ggml_tensor **) {
    return false;
}

static bool test_meta_comm_compute(void * ptr, ggml_cgraph ** graphs, size_t n_graphs) {
    auto * ctx = static_cast<test_meta_comm_context *>(ptr);
    GGML_ASSERT(n_graphs > 1);
    const int run = test_meta_runs.size();
    for (size_t g = 0; g + 1 < n_graphs; ++g) {
        for (size_t r = 0; r < 2; ++r) {
            ggml_cgraph * graph = graphs[2*g + r];
            GGML_ASSERT(strcmp(graph->nodes[graph->n_nodes - 1]->name, "gather") != 0);
        }
    }
    test_meta_runs.push_back(n_graphs);
    test_meta_check_uids(graphs, n_graphs);
    if (run == test_meta_decline) {
        return false;
    }
    for (size_t g = 0; g < n_graphs; ++g) {
        ggml_tensor * last[2];
        for (size_t r = 0; r < 2; ++r) {
            ggml_cgraph * graph = graphs[2*g + r];
            GGML_ASSERT(graph->n_nodes > 0);
            GGML_ASSERT(ggml_backend_graph_compute(ctx->backends[r], graph) == GGML_STATUS_SUCCESS);
            last[r] = graph->nodes[graph->n_nodes - 1];
        }
        if (g + 1 == n_graphs) {
            continue;
        }
        const size_t n = ggml_nelements(last[0]);
        std::vector<float> values[2] = {std::vector<float>(n), std::vector<float>(n)};
        for (size_t r = 0; r < 2; ++r) {
            GGML_ASSERT(last[r]->type == GGML_TYPE_F32 && ggml_nelements(last[r]) == (int64_t) n);
            if (last[r]->flags & GGML_TENSOR_FLAG_COMPUTE) {
                ggml_backend_tensor_get(last[r], values[r].data(), 0, n * sizeof(float));
            }
        }
        for (size_t k = 0; k < n; ++k) {
            values[0][k] += values[1][k];
        }
        for (size_t r = 0; r < 2; ++r) {
            ggml_backend_tensor_set(last[r], values[0].data(), 0, n * sizeof(float));
        }
    }
    return true;
}

static void * test_meta_comm_lookup(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *) test_meta_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *) test_meta_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *) test_meta_comm_allreduce;
    }
    if (strcmp(name, "ggml_backend_comm_graph_compute") == 0) {
        return test_meta_merge ? (void *) test_meta_comm_compute : nullptr;
    }
    return test_meta_get_proc(reg, name);
}

static ggml_backend_meta_split_state test_meta_comm_split(const ggml_tensor * t, void *) {
    if (strcmp(t->name, "up") == 0) {
        return {GGML_BACKEND_SPLIT_AXIS_1, {4, 4}, {1}, 1};
    }
    if (strcmp(t->name, "down") == 0) {
        return {GGML_BACKEND_SPLIT_AXIS_0, {4, 4}, {1}, 1};
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

static void test_meta_comm_runs() {
    ggml_backend_load_all();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(cpu);
    test_meta_get_proc = reg->iface.get_proc_address;
    reg->iface.get_proc_address = test_meta_comm_lookup;
    ggml_backend_dev_t devs[2] = {cpu, cpu};
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2, test_meta_comm_split, nullptr);
    // Outer UID per plan, as scheduler splits set it; a property change comes with a new UID, 0 means unset.
    // ggml_graph_next_uid is not GGML_API; these values stay far above its counter.
    const uint64_t plan_uids[]   = {0x7e57000000000001ull, 0x7e57000000000002ull, 0};
    const float    plan_scales[] = {0.5f, 0.25f, 0.25f};
    const size_t   n_plans = sizeof(plan_uids)/sizeof(plan_uids[0]);
    const size_t   n_computes = 2;
    std::vector<std::vector<float>> reference(n_plans*n_computes);
    for (int mode = 0; mode < 4; ++mode) {
        test_meta_merge = mode != 0;
        test_meta_decline = mode >= 2 ? mode - 2 : -1;
        test_meta_runs.clear();
        test_meta_uids.clear();
        test_meta_uid_skips = 0;
        test_meta_uid_scans = 0;
        ggml_backend_t backend = ggml_backend_dev_init(meta, nullptr);
        GGML_ASSERT(backend != nullptr);
        // Only weight buffers take the split callback; activations need a compute buffer so their splits are inferred.
        ggml_context * wctx = ggml_init({2 * ggml_tensor_overhead(), nullptr, true});
        ggml_tensor * up = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 8, 8);
        ggml_tensor * down = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 8, 8);
        ggml_set_name(up, "up");
        ggml_set_name(down, "down");
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(wctx, backend);
        GGML_ASSERT(buffer != nullptr);
        ggml_context * ctx = ggml_init({32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
        ggml_set_input(x);
        ggml_tensor * cur = x;
        for (int l = 0; l < 4; ++l) {
            cur = ggml_add(ctx, cur, ggml_mul_mat(ctx, down, ggml_mul_mat(ctx, up, cur)));
            if (l == 1) {
                ggml_tensor * gather = ggml_backend_meta_gather(ctx, ggml_mul_mat(ctx, up, cur));
                ggml_set_name(gather, "gather");
                cur = ggml_cont(ctx, gather);
            }
        }
        ggml_tensor * out = ggml_scale(ctx, cur, 0.5f);
        ggml_set_output(out);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, graph));
        float weights[64];
        for (size_t k = 0; k < 64; ++k) {
            weights[k] = float(int(k % 5) - 2) * 0.125f;
        }
        ggml_backend_tensor_set(up, weights, 0, sizeof(weights));
        ggml_backend_tensor_set(down, weights, 0, sizeof(weights));
        for (size_t p = 0; p < n_plans; ++p) {
            graph->uid = plan_uids[p];
            ggml_set_op_params_f32(out, 0, plan_scales[p]);
            const size_t runs_before = test_meta_runs.size();
            ggml_backend_graph_plan_t plan = ggml_backend_graph_plan_create(backend, graph);
            GGML_ASSERT(plan != nullptr && test_meta_runs.size() == runs_before);
            for (size_t c = 0; c < n_computes; ++c) {
                const size_t step = p*n_computes + c;
                float input[16];
                for (size_t k = 0; k < 16; ++k) {
                    input[k] = float(int((k + step) % 7) - 3) * 0.25f;
                }
                ggml_backend_tensor_set(x, input, 0, sizeof(input));
                GGML_ASSERT(ggml_backend_graph_plan_compute(backend, plan) == GGML_STATUS_SUCCESS);
                ggml_backend_synchronize(backend);
                std::vector<float> result(16);
                ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
                if (mode == 0) {
                    reference[step] = result;
                } else {
                    GGML_ASSERT(memcmp(reference[step].data(), result.data(), ggml_nbytes(out)) == 0);
                }
            }
            ggml_backend_graph_plan_free(backend, plan);
        }
        if (mode == 0) {
            GGML_ASSERT(test_meta_runs.empty());
        } else {
            GGML_ASSERT(test_meta_runs == std::vector<size_t>(2*n_plans*n_computes, 3));
            // Per run: a repeated non-zero UID skips; a new UID (with the scale change) or UID 0 scans.
            GGML_ASSERT(test_meta_uid_skips == 2*2 && test_meta_uid_scans == 2*4);
        }
        ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_free(wctx);
        ggml_backend_free(backend);
    }
    reg->iface.get_proc_address = test_meta_get_proc;
}

static void run(const char * name, void (*f)()) {
    printf("%s ", name);
    fflush(stdout);
    f();
    printf("PASSED\n");
}

int main(int argc, char ** argv) {
    if (argc == 2 && strcmp(argv[1], "--split-state-same") == 0) {
        test_meta_split_state_rebuild(false);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--split-state-other") == 0) {
        test_meta_split_state_rebuild(true);
        return 0;
    }
    run("test_max_size_too_many_tensors", test_max_size_too_many_tensors);
    run("test_max_size_tensor_too_large", test_max_size_tensor_too_large);
    run("test_tensor_larger_than_max_size", test_tensor_larger_than_max_size);
    run("test_not_enough_chunks", test_not_enough_chunks);
    run("test_fill_leftover_space", test_fill_leftover_space);
    run("test_view_inplace", test_view_inplace);
    run("test_reuse_and_free", test_reuse_and_free);
    run("test_merge_free_block(32)", []() { test_merge_free_block(32); });
    run("test_merge_free_block(SIZE_MAX)", []() { test_merge_free_block(SIZE_MAX); });
    run("test_prefer_already_allocated_memory", test_prefer_already_allocated_memory);
    run("test_multiple_buffer_types", test_multiple_buffer_types);
    run("test_buffer_size_zero", test_buffer_size_zero);
    run("test_reallocation", test_reallocation);
    run("test_graph_optimize_alloc_dep", test_graph_optimize_alloc_dep);
    run("test_buft_alloc_buffer_n_single_buffer", test_buft_alloc_buffer_n_single_buffer);
    run("test_buft_alloc_buffer_n_multi_buffer", test_buft_alloc_buffer_n_multi_buffer);
    run("test_buft_alloc_buffer_n_zero_size", test_buft_alloc_buffer_n_zero_size);
    run("test_buft_alloc_buffer_n_already_allocated", test_buft_alloc_buffer_n_already_allocated);
    run("test_buft_alloc_buffer_n_views", test_buft_alloc_buffer_n_views);
    run("test_alloc_ctx_tensors_from_buft_size_matches", test_alloc_ctx_tensors_from_buft_size_matches);
    run("test_buft_alloc_buffer_n_custom_override", test_buft_alloc_buffer_n_custom_override);
    run("test_buft_get_alloc_size_n_single_buffer", test_buft_get_alloc_size_n_single_buffer);
    run("test_buft_get_alloc_size_n_multi_buffer", test_buft_get_alloc_size_n_multi_buffer);
    run("test_buft_get_alloc_size_n_single_tensor_exceeds_max", test_buft_get_alloc_size_n_single_tensor_exceeds_max);
    run("test_buft_get_alloc_size_n_zero_size", test_buft_get_alloc_size_n_zero_size);
    run("test_buft_get_alloc_size_n_already_allocated", test_buft_get_alloc_size_n_already_allocated);
    run("test_buft_get_alloc_size_n_views", test_buft_get_alloc_size_n_views);
    run("test_buft_get_alloc_size_n_n_tensors_zero", test_buft_get_alloc_size_n_n_tensors_zero);
    run("test_buft_get_alloc_size_n_alignment", test_buft_get_alloc_size_n_alignment);
    run("test_buft_get_alloc_size_n_custom_override", test_buft_get_alloc_size_n_custom_override);
    run("test_buft_get_alloc_size_n_custom_override_ctx_size", test_buft_get_alloc_size_n_custom_override_ctx_size);
    run("test_buft_get_alloc_size_n_does_not_allocate", test_buft_get_alloc_size_n_does_not_allocate);
    run("test_meta_split_state_rebuild(same)", []() { test_meta_split_state_rebuild(false); });
    run("test_meta_split_state_rebuild(other)", []() { test_meta_split_state_rebuild(true); });
    run("test_meta_host_view(mirrored)", []() { test_meta_host_view(GGML_OP_VIEW, false); });
    run("test_meta_host_view(split)", []() { test_meta_host_view(GGML_OP_VIEW, true); });
    run("test_meta_host_cpy(mirrored)", []() { test_meta_host_view(GGML_OP_CPY, false); });
    run("test_meta_host_cpy(split)", []() { test_meta_host_view(GGML_OP_CPY, true); });
    run("test_meta_host_set_rows(mirrored)", []() { test_meta_host_view(GGML_OP_SET_ROWS, false); });
    run("test_meta_host_set_rows(split)", []() { test_meta_host_view(GGML_OP_SET_ROWS, true); });
    run("test_meta_op_params_refresh", test_meta_op_params_refresh);
    run("test_meta_gather_op_params_refresh", test_meta_gather_op_params_refresh);
    run("test_meta_comm_runs", test_meta_comm_runs);
    return 0;
}
