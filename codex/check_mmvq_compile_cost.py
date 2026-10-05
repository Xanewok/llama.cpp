import json, os, shlex, subprocess, time
from pathlib import Path
root = Path(__file__).resolve().parents[1]
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
if os.getpriority(os.PRIO_PROCESS, 0) < 19:
    os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
(root / "build-tmp").mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(root / "build-tmp")
(root / "codex/logs-general").mkdir(exist_ok=True)
baseline = root / "build-cuda-general/compile-ab"
baseline.mkdir(exist_ok=True)
for name in ("mmvq.cu", "mmvq.cuh"):
    (baseline / name).write_bytes(subprocess.check_output(["git", "show", f"6ab88b00b:ggml/src/ggml-cuda/{name}"], cwd=root))
build = root / "build-cuda-general/ggml/src/ggml-cuda"
flags = {}
for line in (build / "CMakeFiles/ggml-cuda.dir/flags.make").read_text().splitlines():
    if " = " in line:
        k, v = line.split(" = ", 1)
        flags[k] = shlex.split(v)
common = ["/usr/local/cuda-13.2/bin/nvcc"] + flags["CUDA_DEFINES"] + flags["CUDA_INCLUDES"] + flags["CUDA_FLAGS"]
common += ["-I" + str(root / "ggml/src/ggml-cuda")]
results = {}
for name, source in [("serve", root / "build-cuda-general/compile-ab/mmvq.cu"), ("general", root / "ggml/src/ggml-cuda/mmvq.cu")]:
    while int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:"))) < 10485760:
        time.sleep(10)
    args = common + ["-c", str(source), "-o", str(root / f"build-cuda-general/compile-ab/{name}.o")]
    (root / f"codex/logs-general/compile-{name}.command.json").write_text(json.dumps(args))
    start = time.monotonic()
    with (root / f"codex/logs-general/compile-{name}.log").open("w") as log:
        subprocess.run(args, cwd=build, stdout=log, stderr=log, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    results[name] = time.monotonic() - start
    (root / "codex/logs-general/compile-ab.json").write_text(json.dumps(results, indent=2))
