#define main backend_ops_unused_main
#include "../tests/test-backend-ops.cpp"
#undef main

int main() {
    int cases = 0;
    for (auto & test : make_test_cases_eval()) {
        auto * shared = dynamic_cast<test_mul_mat_shared *>(test.get());
        if (!shared) { continue; }
        auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
        shared->mode = MODE_TEST;
        shared->gf = ggml_new_graph_custom(ctx, 128, false);
        auto * output = shared->build_graph(ctx);
        ggml_build_forward_expand(shared->gf, output);
        ggml_tensor * input = nullptr;
        int consumers = 0;
        int copies = 0;
        for (int i = 0; i < ggml_graph_n_nodes(shared->gf); ++i) {
            auto * node = ggml_graph_node(shared->gf, i);
            if (node->op == GGML_OP_MUL_MAT) {
                if (!input) { input = node->src[1]; }
                GGML_ASSERT(input == node->src[1]);
                GGML_ASSERT(node->src[0]->ne[0] == 768 && node->src[0]->ne[1] == 257);
                ++consumers;
            } else if (node->op == GGML_OP_CPY) {
                GGML_ASSERT(consumers == 2 && node->src[1] == input);
                ++copies;
            }
        }
        GGML_ASSERT(consumers == 3 && copies == int(shared->overwrite));
        GGML_ASSERT(shared->run_whole_graph());
        std::printf("M: MUL_MAT_SHARED(%s) consumers=%d copies=%d\n", shared->vars().c_str(), consumers, copies);
        ggml_free(ctx);
        ++cases;
    }
    if (cases != 60) { std::fprintf(stderr, "FAIL: registered cases=%d expected=60\n", cases); return 1; }
    std::puts("M: registered shared graph contract cases=60 PASS");
}
