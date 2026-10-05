"""Compare every table entry with the pinned general-form parent on the CPU."""
import os
from pathlib import Path
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""
root = Path(__file__).resolve().parents[1]
if os.getpriority(os.PRIO_PROCESS, 0) < 19:
    os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
(root / "build-tmp").mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(root / "build-tmp")
out = root / "build-cpu/mmvq-pertype-tables"
out.mkdir(parents=True, exist_ok=True)
base = subprocess.check_output(["git", "show", "08c1737bc:ggml/src/ggml-cuda/mmvq-params.h"], cwd=root, text=True)
base = base.replace("bool small_k = false, int nwarps = 1)", "bool small_k = false, int nwarps = 1, ggml_type = GGML_TYPE_COUNT)")
(out / "base.h").write_text(base)
(out / "patch.h").write_text((root / "ggml/src/ggml-cuda/mmvq-params.h").read_text())
source = r'''#include "HEADER"
#include <cstdio>
int main() {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        for (int n = 1; n <= 8; ++n) {
            for (int table = 0; table <= MMVQ_PARAMETERS_BLACKWELL; ++table) {
                for (int flags = 0; flags < 4; ++flags) {
                    auto type = static_cast<ggml_type>(t);
                    auto id = static_cast<mmvq_parameter_table_id>(table);
                    int nw = calc_nwarps(type, n, id, flags & 1, flags & 2);
                    int nr = calc_rows_per_block(n, table, flags & 1, nw, type);
                    std::printf("%d %d %d %d %d %d\n", t, n, table, flags, nw, nr);
                }
            }
        }
    }
}
'''
records = {}
for name in ("base", "patch"):
    cpp = out / f"{name}.cpp"
    cpp.write_text(source.replace("HEADER", f"{name}.h"))
    subprocess.run(["c++", "-std=c++17", "-O2", "-I" + str(root / "ggml/include"), str(cpp), "-o", str(out / name)], check=True)
    raw = subprocess.check_output([str(out / name)], text=True)
    (out / f"{name}.txt").write_text(raw)
    records[name] = {tuple(map(int, l.split()[:4])): tuple(map(int, l.split()[4:])) for l in raw.splitlines()}
changes = {k for k, v in records["base"].items() if records["patch"][k] != v}
expected = {(t, n, 7, f) for t, ns in ((12, (5, 6, 7, 8)), (13, (6, 7, 8)), (14, (7, 8)), (23, (5, 6, 7, 8))) for n in ns for f in range(4)}
assert changes == expected, (changes - expected, expected - changes)
print(f"M: {len(records['base'])} entries compared; only {len(changes)} intended Blackwell entries changed")
# Count the baseline with the same counting logic, without enforcing the new table.
test = (root / "tests/test-mmvq-params.cpp").read_text().replace("../ggml/src/ggml-cuda/mmvq-params.h", "base.h").replace("if (table_id == MMVQ_PARAMETERS_BLACKWELL)", "if (false)")
(out / "count.cpp").write_text(test)
subprocess.run(["c++", "-std=c++17", "-O2", "-I" + str(root / "ggml/include"), str(out / "count.cpp"), "-L" + str(root / "build-cpu/bin"), "-lggml-base", "-Wl,-rpath," + str(root / "build-cpu/bin"), "-o", str(out / "count")], check=True)
print("Baseline:", subprocess.check_output([str(out / "count")], text=True).strip())
