#!/usr/bin/env python3
"""Instrument the shared host driver; compile against the existing frozen libraries."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def replace_once(source, old, new):
    assert source.count(old) == 1, f"driver source changed: {old}"
    return source.replace(old, new)


def instrument(source):
    seed = '''static uint32_t probe_seed(const ggml_tensor * tensor) {
    uint32_t seed = 2166136261u;
    for (const char * p = tensor->name; *p; ++p) {
        seed = (seed ^ (uint8_t) *p) * 16777619u;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        seed = (seed ^ (uint32_t) tensor->ne[i]) * 16777619u;
    }
    return seed;
}

'''
    source = replace_once(source, "static void init_tensor_uniform(", seed + "static void init_tensor_uniform(")
    source = replace_once(source, "thread_local std::default_random_engine gen(std::random_device{}());",
                          "std::default_random_engine gen(probe_seed(tensor) ^ (uint32_t) start);")
    start = source.index("static void init_tensor_kq_mask(")
    end = source.index("static void init_tensor_kq_mask_sparse(", start)
    mask = replace_once(source[start:end], "std::random_device rd;", "std::mt19937 rd(probe_seed(tensor));")
    source = source[:start] + mask + source[end:]
    source = replace_once(source, "        // Ops with an inplace flag create a view of src[0] as their output.", '''        for (size_t i = 0; i < source_count; ++i) {
            ggml_format_name(source_tensors[i], "probe_src_%zu", i);
        }

        // Ops with an inplace flag create a view of src[0] as their output.''')
    source = replace_once(source, "            if (err > ud->tc->max_err(ud->backend1)) {",
                          '''            if (t1->op == GGML_OP_FLASH_ATTN_EXT && strcmp(t1->name, "out") == 0) {
                printf("[%s] ERR = %.17g > %.17g ", ggml_op_desc(t1), err, ud->tc->max_err(ud->backend1));
            }
            if (err > ud->tc->max_err(ud->backend1)) {''')
    source = replace_once(source, '                printf("[%s] ERR = %.9f > %.9f ", ggml_op_desc(t1), err, ud->tc->max_err(ud->backend1));', "")
    source = replace_once(source, "    ggml_backend_load_all();", '''    const char * probe_lib_dir = std::getenv("Q8_PROBE_LIB_DIR");
    GGML_ASSERT(probe_lib_dir != nullptr);
    ggml_backend_load_all_from_path(probe_lib_dir);''')
    return source


if __name__ == "__main__":
    dest = ROOT / "build-tmp/q8-probe3/driver.cpp"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(instrument((ROOT / "tests/test-backend-ops.cpp").read_text()))
