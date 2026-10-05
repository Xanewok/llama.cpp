#define main backend_ops_main
#include "../tests/test-backend-ops.cpp"
#undef main

int main(int argc, char ** argv) {
    if (argc != 2) {
        return 1;
    }
    auto cases = make_test_cases_from_file(argv[1]);
    if (cases.empty()) {
        return 1;
    }
    for (auto & test : cases) {
        ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
        ggml_tensor * out = test->build_graph(ctx);
        if (out->op != GGML_OP_FLASH_ATTN_EXT || out->src[1]->ne[2] != 2 || out->src[2]->ne[2] != 2) {
            ggml_free(ctx);
            return 1;
        }
        printf("%s\n", test->vars().c_str());
        ggml_free(ctx);
    }
    return 0;
}
