// Host-only contract of ggml-cuda/add-rms-norm.h on synthetic graphs: what the fused kernel may replace.
// Buffers are never dereferenced; data pointers only stand for placement.
#include "add-rms-norm.h"
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

static int failures = 0;

struct chain {
    ggml_context * ctx;
    ggml_cgraph * graph;
    ggml_tensor *x, *y, *w, *add, *norm, *mul;
    ggml_tensor * extra = nullptr;
};

static chain build(int64_t cols, int64_t rows, ggml_type type = GGML_TYPE_F32, bool inplace_add = false) {
    ggml_init_params p = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    chain c{};
    c.ctx = ggml_init(p);
    c.x = ggml_new_tensor_2d(c.ctx, type, cols, rows);
    c.y = ggml_new_tensor_2d(c.ctx, type, cols, rows);
    c.w = ggml_new_tensor_1d(c.ctx, type, cols);
    c.add = inplace_add ? ggml_add_inplace(c.ctx, c.x, c.y) : ggml_add(c.ctx, c.x, c.y);
    c.norm = ggml_rms_norm(c.ctx, c.add, 1e-6f);
    c.mul = ggml_mul(c.ctx, c.norm, c.w);
    c.graph = ggml_new_graph(c.ctx);
    ggml_build_forward_expand(c.graph, c.mul);
    // distinct, non-overlapping placement by default
    const size_t n = size_t(cols)*rows*ggml_type_size(type);
    char * base = (char *) 0x10000000;
    c.x->data = base; c.y->data = base + 1*n + 4096; c.w->data = base + 2*n + 8192;
    c.add->data = inplace_add ? c.x->data : base + 3*n + 12288;
    c.norm->data = base + 4*n + 16384; c.mul->data = base + 5*n + 20480;
    return c;
}

static int node_index(const chain & c, const ggml_tensor * t) {
    for (int i = 0; i < ggml_graph_n_nodes(c.graph); ++i) {
        if (ggml_graph_node(c.graph, i) == t) { return i; }
    }
    return -1;
}

static void expect(const char * name, const chain & c, bool want) {
    const int i = node_index(c, c.add);
    const bool got = i >= 0 && ggml_cuda_can_fuse_add_rms_norm(c.graph, i);
    if (got != want) {
        std::printf("FAIL %s: matcher=%d expected=%d\n", name, got, want);
        ++failures;
    }
}

int main() {
    for (int64_t rows : {1, 2, 6, 8, 9, 12, 16, 17, 33}) {
        for (int64_t cols : {1, 24, 512, 1024, 5120}) {
            auto c = build(cols, rows);
            const std::string name = "rows=" + std::to_string(rows) + " cols=" + std::to_string(cols);
            expect(name.c_str(), c, rows <= 16);
            ggml_free(c.ctx);
        }
    }
    {
        auto c = build(512, 12, GGML_TYPE_F32, true);
        expect("exact in-place residual", c, true);
        ggml_free(c.ctx);
    }
    {
        auto c = build(512, 12);
        c.mul->data = c.x->data;
        expect("result exactly over x", c, true);
        ggml_free(c.ctx);
    }
    {
        auto c = build(512, 12);
        c.mul->data = (char *) c.x->data + 16;
        expect("result partially over x", c, false);
        ggml_free(c.ctx);
    }
    {
        auto c = build(512, 12);
        c.add->data = (char *) c.y->data + 16;
        expect("residual partially over y", c, false);
        ggml_free(c.ctx);
    }
    {
        auto c = build(512, 12);
        c.mul->data = c.add->data;
        expect("result exactly over the residual", c, false);
        ggml_free(c.ctx);
    }
    {
        auto c = build(512, 12);
        c.mul->data = c.w->data;
        expect("result over the weight", c, false);
        ggml_free(c.ctx);
    }
    {
        auto c = build(512, 12);
        c.add->data = c.w->data;
        expect("residual over the weight", c, false);
        ggml_free(c.ctx);
    }
    {
        // The unfused RMS_NORM would write its buffer; fusing would leave that buffer's other owner intact.
        auto c = build(512, 12);
        c.norm->data = c.add->data;
        expect("normalized buffer over the residual", c, false);
        ggml_free(c.ctx);
        c = build(512, 12);
        c.norm->data = c.y->data;
        expect("normalized buffer over y", c, false);
        ggml_free(c.ctx);
        c = build(512, 12);
        c.norm->data = c.mul->data;
        expect("result in the normalized buffer", c, true);
        ggml_free(c.ctx);
    }
    {
        auto c = build(512, 12);
        ggml_set_output(c.norm);
        expect("normalized intermediate is a graph output", c, false);
        ggml_free(c.ctx);
    }
    {
        auto c = build(512, 12);
        ggml_tensor * second = ggml_scale(c.ctx, c.norm, 2.0f);
        ggml_build_forward_expand(c.graph, second);
        expect("normalized intermediate has a second consumer", c, false);
        ggml_free(c.ctx);
    }
    {
        auto c = build(512, 12);
        ggml_tensor * later = ggml_scale(c.ctx, c.add, 2.0f);
        ggml_build_forward_expand(c.graph, later);
        expect("residual has a later consumer", c, true);
        ggml_free(c.ctx);
    }
    {
        auto c = build(512, 12, GGML_TYPE_F16);
        expect("F16 operands", c, false);
        ggml_free(c.ctx);
    }
    {
        auto c = build(512, 12);
        ggml_tensor * w2 = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 512, 2);
        c.mul->src[1] = w2;
        w2->data = (char *) 0x20000000;
        expect("matrix weight", c, false);
        ggml_free(c.ctx);
    }
    {
        // ADD followed by a different second op is not this chain.
        ggml_init_params p = { 32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(p);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 4);
        ggml_tensor * y = ggml_dup_tensor(ctx, x);
        ggml_tensor * add = ggml_add(ctx, x, y);
        ggml_tensor * out = ggml_scale(ctx, add, 2.0f);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);
        if (ggml_cuda_can_fuse_add_rms_norm(g, 0)) {
            std::puts("FAIL ADD, SCALE matched");
            ++failures;
        }
        ggml_free(ctx);
    }
    if (failures) {
        return 1;
    }
    std::puts("M: ADD, RMS_NORM, MUL matcher: row bound 16, types, aliasing, consumers, outputs PASS");
}
