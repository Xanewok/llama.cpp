"""Compile production CUDA staging code against CPU delayed queues; never link CUDA."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
os.chdir(ROOT)
out = ROOT / "build-tmp/input-contract"
out.mkdir(parents=True, exist_ok=True)
common = (ROOT / "ggml/src/ggml-cuda/common.cuh").read_text()
fields = common.split("    struct input_slot {", 1)[1].split("#endif", 1)[0]
fields = "    struct input_slot {" + fields
source = (ROOT / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text()
function = source[source.index("static bool ggml_backend_cuda_set_tensor_batch_owned("):source.index("static void ggml_backend_cuda_get_tensor_async(")].rsplit("\n#endif", 1)[0]
cleanup = source.split("ggml_backend_cuda_context::~ggml_backend_cuda_context()", 1)[1].split("    ggml_cuda_set_device(device);", 1)[1].split("#endif", 1)[0]
template = (ROOT / "codex/input-staging-fake.cpp").read_text().replace("// PRODUCTION_FIELDS", fields).replace("// PRODUCTION_CLEANUP", cleanup)
mutations = {
    "correct": function,
    "early-reuse": function.replace("CUDA_CHECK(cudaEventSynchronize(slot.uploaded));", "/* mutation: reuse before completion */"),
    "skip-upload-wait": function.replace("CUDA_CHECK(cudaStreamWaitEvent(ctx->stream(), slot.uploaded, 0));", "/* mutation: compute before upload */"),
    "skip-compute-wait": function.replace("CUDA_CHECK(cudaStreamWaitEvent(ctx->input_stream, ctx->input_compute_done, 0));", "/* mutation: overwrite before prior compute */"),
}
results = []
for name, body in mutations.items():
    cpp = out / (name + ".cpp")
    cpp.write_text(template.replace("// PRODUCTION_UPLOAD", body))
    exe = out / name
    subprocess.run(["nice", "-n", "19", "g++", "-std=c++17", "-O0", "-g", str(cpp), "-o", str(exe)], check=True)
    run = subprocess.run(["nice", "-n", "19", str(exe)], capture_output=True, text=True)
    expected = 0 if name == "correct" else 1
    assert run.returncode == expected, (name, run.returncode, run.stdout, run.stderr)
    results.append({"case": name, "exit_M": run.returncode, "stdout": run.stdout.strip(), "stderr": run.stderr.strip()})
(ROOT / "codex/input-contract-results.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
