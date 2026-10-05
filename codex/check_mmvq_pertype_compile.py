import json, os, shlex, subprocess, time
from pathlib import Path
root = Path(__file__).resolve().parents[1]
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
if os.getpriority(os.PRIO_PROCESS, 0) < 19:
    os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
(root / "build-tmp").mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(root / "build-tmp")
(root / "codex/logs-pertype").mkdir(exist_ok=True)
baseline = root / "build-cuda-pertype/compile-ab"
baseline.mkdir(exist_ok=True)
for name in ("mmvq.cu", "mmvq.cuh", "mmvq-params.h"):
    (baseline / name).write_bytes(subprocess.check_output(["git", "show", f"08c1737bc:ggml/src/ggml-cuda/{name}"], cwd=root))
build = root / "build-cuda-pertype/ggml/src/ggml-cuda"
flags = {}
for line in (build / "CMakeFiles/ggml-cuda.dir/flags.make").read_text().splitlines():
    if " = " in line:
        k, v = line.split(" = ", 1)
        flags[k] = shlex.split(v)
common = ["/usr/local/cuda-13.2/bin/nvcc"] + flags["CUDA_DEFINES"] + flags["CUDA_INCLUDES"] + flags["CUDA_FLAGS"]
common += ["-I" + str(root / "ggml/src/ggml-cuda")]
results = {}
for name, source in [("general", root / "build-cuda-pertype/compile-ab/mmvq.cu"), ("pertype", root / "ggml/src/ggml-cuda/mmvq.cu")]:
    while int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:"))) < 10485760:
        time.sleep(10)
    args = common + ["-c", str(source), "-o", str(root / f"build-cuda-pertype/compile-ab/{name}.o")]
    (root / f"codex/logs-pertype/compile-{name}.command.json").write_text(json.dumps(args))
    start = time.monotonic()
    with (root / f"codex/logs-pertype/compile-{name}.log").open("w") as log:
        subprocess.run(args, cwd=build, stdout=log, stderr=log, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    results[name] = time.monotonic() - start
    (root / "codex/logs-pertype/compile-ab.json").write_text(json.dumps(results, indent=2))

counts = {}
for name in results:
    symbols = subprocess.check_output(["nm", "--defined-only", "-C", str(root / f"build-cuda-pertype/compile-ab/{name}.o")], text=True)
    counts[name] = sum(" t void mul_mat_vec_q<" in line and "[clone" not in line for line in symbols.splitlines())
(root / "codex/logs-pertype/compile-counts.json").write_text(json.dumps(counts, indent=2) + "\n")
print(json.dumps({"M_seconds": results, "M_specializations": counts, "D_compile_delta_pct": 100 * (results["pertype"] / results["general"] - 1)}, indent=2))
