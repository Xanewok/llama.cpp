"""The old recursive resolver must fail the same 1 MiB child-stack regression."""
import os
from pathlib import Path
import resource
import subprocess

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
ROOT = Path(__file__).resolve().parent.parent
namespace = {"__file__": str(ROOT / "codex/negative_reuse.py")}
exec((ROOT / "codex/negative_reuse.py").read_text().split("\nresults = []", 1)[0], namespace)
current = (ROOT / "ggml/src/ggml-backend-meta.cpp").read_text()
old = subprocess.check_output(["git", "show", "import/graph-reuse-tandem:ggml/src/ggml-backend-meta.cpp"], text=True)
dest = namespace["mutant"]("meta-recursive", "ggml/src/ggml-backend-meta.cpp", current, old,
    "build-cpu/ggml/src/CMakeFiles/ggml-base.dir", "CMakeFiles/ggml-base.dir/ggml-backend-meta.cpp.o", "libggml-base.so.0")
env = dict(os.environ, LD_LIBRARY_PATH=str(dest) + ":" + str(ROOT / "build-cpu/bin"), OMP_NUM_THREADS="8")
result = subprocess.run([str(ROOT / "build-cpu/bin/test-alloc")], env=env, capture_output=True, text=True)
(dest / "result.log").write_text(result.stdout + result.stderr)
assert result.returncode != 0 and "rebuild killed by signal 11" in result.stderr, (result.returncode, result.stdout, result.stderr)
print("M PASS: recursive negative control rejected under 1 MiB child stack; exit", result.returncode)
