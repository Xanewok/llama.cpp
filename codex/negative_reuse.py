"""Build CPU-only mutations from existing objects; require the contract checks to reject them."""
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
env = dict(os.environ, OMP_NUM_THREADS="8", TMPDIR=str(ROOT / "build-tmp"))


def mutant(name, source, old, new, build_dir, object_name, soname):
    dest = ROOT / ("build-negative-" + name)
    dest.mkdir(exist_ok=True)
    original = (ROOT / source).read_text()
    assert original.count(old) == 1, (name, "mutation site changed")
    changed = dest / Path(source).name
    changed.write_text(original.replace(old, new))
    obj = dest / (changed.name + ".o")
    flags = {}
    for line in (ROOT / build_dir / "flags.make").read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            flags[key] = shlex.split(value)
    subprocess.run(["/usr/bin/c++", *flags["CXX_DEFINES"], *flags["CXX_INCLUDES"], *flags["CXX_FLAGS"],
                    "-c", str(changed), "-o", str(obj)], env=env, check=True)
    link_dir = (ROOT / build_dir).parent.parent
    args = shlex.split((ROOT / build_dir / "link.txt").read_text())
    args[args.index("-o") + 1] = str(dest / soname)
    assert args.count(object_name) == 1
    args[args.index(object_name)] = str(obj)
    subprocess.run(args, cwd=link_dir, env=env, check=True)
    return dest


results = []
for name, source, old, new, directory, obj, soname, mode, message in (
    ("binding", "src/llama-graph.cpp", "    for (const auto & bind : bindings) {\n        bind();\n    }",
     "    for (const auto & bind : bindings) { (void) bind; }", "build-cpu/src/CMakeFiles/llama.dir",
     "CMakeFiles/llama.dir/llama-graph.cpp.o", "libllama.so.0", "binding", "stale bindings or RNG draw count"),
    ("epoch", "ggml/src/ggml-backend.cpp",
     "sched->alloc_graph_uid != graph->uid || sched->alloc_epoch != ggml_gallocr_get_epoch(sched->galloc)",
     "sched->alloc_graph_uid != graph->uid", "build-cpu/ggml/src/CMakeFiles/ggml-base.dir",
     "CMakeFiles/ggml-base.dir/ggml-backend.cpp.o", "libggml-base.so.0", "epoch", "stale epoch escaped public compute guard"),
):
    dest = mutant(name, source, old, new, directory, obj, soname)
    child_env = dict(env, LD_LIBRARY_PATH=str(dest) + ":" + str(ROOT / "build-cpu/bin"))
    result = subprocess.run([str(ROOT / "build-cpu/bin/reuse-contract"), mode], env=child_env, capture_output=True, text=True)
    (dest / "result.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 1 and message in result.stderr, (name, result.returncode, result.stdout, result.stderr)
    results.append({"mutation": name, "exit_M": result.returncode, "failure": result.stderr.strip()})
(ROOT / "codex/negative-results.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
