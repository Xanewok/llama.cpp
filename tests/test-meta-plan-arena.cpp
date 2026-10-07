// Meta backend, plan arena: a plan whose arena was rebuilt for a graph with fewer subgraphs than the plan's max_subgraphs
// must still serve a later graph with that many subgraphs. The arena is rebuilt only when a capacity is raised, so that
// later graph reuses it and every cgraph_main it touches must belong to the current arena.
//
// Sequence on one Meta(CPU,CPU) backend (a freed plan becomes the spare that the next plan_create reuses):
//   A: many subgraphs, few nodes   -> max_subgraphs = splits(A)
//   B: 2 subgraphs, many nodes     -> max_nnodes raised, arena rebuilt, only 2 subgraphs created
//   A again                        -> no capacity raised, splits(A) > 2: used cgraph_main entries 2.. of the old arena
// Only plan_create (the build loop) is exercised, nothing is computed. The defect is a use after free, so the test
// perturbs freed memory (glibc M_PERTURB) and runs in a child: unfixed it dies on a signal, a sanitizer build reports it.
// GGML_META_AR_NODE=1 is run too: on CPU it only computes and logs the node plan, the split plan executes, so this
// covers the shared defect, not the CUDA-only node plan that makes it reachable in the server.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#ifdef __GLIBC__
#include <malloc.h>
#endif

static const int64_t D = 64; // model dim
static const int64_t H = 64; // hidden dim, split in two
static const int64_t T = 4;  // tokens
static const int N_LAYERS_MAX = 6;

static ggml_backend_meta_split_state split_state(const ggml_tensor * tensor, void * userdata) {
    (void) userdata;
    if (strncmp(tensor->name, "up", 2) == 0) {
        return {GGML_BACKEND_SPLIT_AXIS_1, {H/2, H/2}, {1}, 1};
    }
    if (strncmp(tensor->name, "down", 4) == 0) {
        return {GGML_BACKEND_SPLIT_AXIS_0, {H/2, H/2}, {1}, 1};
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

struct test_graph {
    std::vector<uint8_t> mem;
    ggml_context *       ctx    = nullptr;
    ggml_cgraph *        gf     = nullptr;
    ggml_gallocr_t       galloc = nullptr;
};

// n_layers x (up, down, residual add): each down is a PARTIAL result, so it ends a subgraph; n_tail MIRRORED scales add
// nodes without adding subgraphs.
static void build_graph(test_graph & g, ggml_backend_buffer_type_t buft, const std::vector<ggml_tensor *> & up,
        const std::vector<ggml_tensor *> & down, int n_layers, int n_tail) {
    const size_t max_nodes = 3*n_layers + n_tail + 8;
    g.mem.resize(ggml_tensor_overhead()*(max_nodes + 8) + ggml_graph_overhead_custom(max_nodes, false));
    g.ctx = ggml_init({g.mem.size(), g.mem.data(), /*no_alloc =*/ true});
    ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, T);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    for (int l = 0; l < n_layers; l++) {
        ggml_tensor * h = ggml_mul_mat(g.ctx, up[l], x);
        ggml_tensor * y = ggml_mul_mat(g.ctx, down[l], h);
        x = ggml_add(g.ctx, y, x);
    }
    for (int i = 0; i < n_tail; i++) {
        x = ggml_scale(g.ctx, x, 1.0f);
    }
    ggml_set_output(x);
    g.gf = ggml_new_graph_custom(g.ctx, max_nodes, false);
    ggml_build_forward_expand(g.gf, x);
    g.galloc = ggml_gallocr_new(buft);
    GGML_ASSERT(ggml_gallocr_alloc_graph(g.galloc, g.gf));
}

static void plan_once(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_graph_plan_t plan = ggml_backend_graph_plan_create(backend, gf);
    GGML_ASSERT(plan != nullptr);
    ggml_backend_graph_plan_free(backend, plan);
}

static void run_sequence(bool ar_node) {
    setenv("GGML_META_AR_NODE", ar_node ? "1" : "0", 1);
    ggml_backend_load_all();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_dev_t devs[2] = {cpu, cpu};
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2, split_state, nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(meta);
    ggml_backend_t backend = ggml_backend_dev_init(meta, nullptr);
    GGML_ASSERT(backend != nullptr);

    ggml_context * ctx_w = ggml_init({ggml_tensor_overhead()*(2*N_LAYERS_MAX + 1), nullptr, /*no_alloc =*/ true});
    std::vector<ggml_tensor *> up, down;
    for (int l = 0; l < N_LAYERS_MAX; l++) {
        up.push_back(ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, D, H));
        ggml_format_name(up.back(), "up%d", l);
        down.push_back(ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, H, D));
        ggml_format_name(down.back(), "down%d", l);
    }
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    GGML_ASSERT(buf_w != nullptr);

    test_graph a, b;
    build_graph(a, buft, up, down, N_LAYERS_MAX, 0);
    build_graph(b, buft, up, down, 1, 200);
    GGML_ASSERT(b.gf->n_nodes > a.gf->n_nodes);

    plan_once(backend, a.gf);
    plan_once(backend, b.gf);
    plan_once(backend, a.gf);

    ggml_gallocr_free(a.galloc);
    ggml_gallocr_free(b.galloc);
    ggml_free(a.ctx);
    ggml_free(b.ctx);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    ggml_backend_free(backend);
}

static void run_in_child(const char * name, bool ar_node) {
    printf("%s ", name);
    fflush(stdout);
    const pid_t pid = fork();
    GGML_ASSERT(pid >= 0);
    if (pid == 0) {
#ifdef __GLIBC__
        mallopt(M_PERTURB, 0xA5);
#endif
        run_sequence(ar_node);
        _exit(0);
    }
    int status = 0;
    GGML_ASSERT(waitpid(pid, &status, 0) == pid);
    if (WIFSIGNALED(status)) {
        printf("FAILED: killed by signal %d\n", WTERMSIG(status));
    }
    GGML_ASSERT(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    printf("PASSED\n");
}

int main() {
    run_in_child("test_meta_plan_arena(split)", false);
    run_in_child("test_meta_plan_arena(ar_node)", true);
    return 0;
}
#else
int main() { return 0; }
#endif // _WIN32
