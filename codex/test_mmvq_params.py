"""Compile the actual launch rules on CPU and compare them with master."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
PATH = "ggml/src/ggml-cuda/mmvq.cu"


def rules(source):
    table = source[source.index("enum mmvq_parameter_table_id {"):source.index("// Per-architecture maximum")]
    params = source[source.index("static constexpr __host__ __device__ int calc_nwarps("):source.index("template <ggml_type type, int ncols_dst, bool has_fusion")]
    return table + params


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    original = subprocess.check_output(["git", "show", "050439614:" + PATH], cwd=ROOT, text=True)
    current = (ROOT / PATH).read_text()
    common = (ROOT / "ggml/src/ggml-cuda/common.cuh").read_text()
    definitions = "\n".join(line for line in common.splitlines() if line.startswith("#define GGML_CUDA_CC_"))
    definitions += "\n" + next(line for line in (ROOT / "ggml/src/ggml-cuda/mmvq.cuh").read_text().splitlines() if line.startswith("#define MMVQ_MAX_BATCH_SIZE"))
    directory = ROOT / "build-tmp/params"
    directory.mkdir(parents=True, exist_ok=True)
    cases = 0
    for arch, extra in ((750, ""), (800, ""), (890, ""), (1200, ""), (1210, ""),
                        (0, "#define GCN\n"), (0, "#define RDNA2\n"), (0, "#define RDNA3_0\n"), (0, "#define RDNA4\n")):
        output = []
        for label, source in (("master", original), ("patch", current)):
            text = f'''#include "ggml.h"
#include <cstdio>
#define __host__
#define __device__
#define __CUDA_ARCH__ {arch}
{extra}
{definitions}
static int ggml_cuda_highest_compiled_arch(int cc) {{ return cc; }}
{rules(source)}
int main() {{
    for (int cc : {{500, 610, 700, 750, 800, 860, 870, 890, 900, 1000, 1010, 1200, 1210, 1300, 0x1000000 + 0x908, 0x1000000 + 0x1100, 0x1000000 + 0x1200}}) {{
        for (int t = 0; t < GGML_TYPE_COUNT; ++t) {{
            for (int n = 1; n <= 8; ++n) {{
                for (bool small : {{false, true}}) {{
                    for (bool halve : {{false, true}}) {{
                        auto type = static_cast<ggml_type>(t);
                        auto host = get_device_table_id(cc);
                        int nw = calc_nwarps(type, n, host, small, halve);
                        int rows = calc_rows_per_block(n, host, small, nw);
                        int device_nw = calc_nwarps(type, n, get_device_table_id(), small, halve);
                        int device_rows = calc_rows_per_block(n, get_device_table_id(), small, device_nw);
                        std::printf("%d %d %d %d %d %d %d %d %d\\n", cc, t, n, small, halve, nw, rows, device_nw, device_rows);
                    }}
                }}
            }}
        }}
    }}
}}
'''
            path = directory / f"{label}.cpp"
            path.write_text("#include <initializer_list>\n" + text)
            binary = directory / label
            subprocess.run(["c++", "-std=c++17", "-I" + str(ROOT / "ggml/include"), str(path), "-o", str(binary)], check=True)
            output.append(subprocess.check_output([str(binary)], text=True).splitlines())
        for before, after in zip(*output):
            b, a = [int(v) for v in before.split()], [int(v) for v in after.split()]
            # Quantization enum IDs are part of the public type contract.
            targeted = a[1] in (12, 13, 14, 23) and a[2] >= 5
            expected = list(b)
            if a[0] == 1200 and targeted:
                expected[5] = 4
            if arch == 1200 and targeted:
                expected[7] = 4
            if a != expected:
                raise AssertionError((arch, extra, b, a, expected))
            cases += 1
    print(f"launch_rule_cases_M={cases}; only sm_120 target types at ncols=5..8 change")


if __name__ == "__main__":
    main()
