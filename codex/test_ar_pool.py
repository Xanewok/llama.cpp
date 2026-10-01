#!/usr/bin/env python3
"""DO NOT MERGE. CPU parser and source wiring checks; never load CUDA."""
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "ggml/src/ggml-cuda/allreduce.cu"


def wiring(source):
    assert "GGML_CUDA_AR_POOL_SIZE" not in source
    for pattern, count in (
        (r"p->call_count % p->pool_size", 1),
        (r"p->call_count >= p->pool_size", 1),
        (r"s < p->pool_size", 2),
        (r"\(size_t\)p->pool_size \* n_devices", 1),
        (r"\(size_t\) p->pool_size \* p->buf_bytes", 1),
        (r"ev_pool\[GGML_CUDA_MAX_DEVICES\]\[GGML_CUDA_AR_POOL_MAX\]", 1),
        (r'p->pool_size\s*= pool_size;', 1),
        (r'ggml_cuda_ar_pool_size\(getenv\("GGML_CUDA_AR_POOL"\)\)', 1),
    ):
        assert len(re.findall(pattern, source)) == count, pattern
    assert source.count("p->pool_size") == 8
    assert 'GGML_ABORT("GGML_CUDA_AR_POOL' in source


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    source = SOURCE.read_text()
    wiring(source)
    controls = 0
    for match in list(re.finditer(r"p->pool_size", source)):
        broken = source[:match.start()] + "2" + source[match.end():]
        try:
            wiring(broken)
        except AssertionError:
            controls += 1
        else:
            raise AssertionError("hardcoded ring use escaped")
    # Capacity is the only CUDA wiring change; preserve every API/kernel and its order.
    parent = subprocess.check_output(["git", "show", "2b645a655:ggml/src/ggml-cuda/allreduce.cu"], cwd=ROOT, text=True)
    operations = lambda s: re.findall(r"\b(?:cuda[A-Z]\w*|ggml_cuda_ar_\w+)\s*(?:\(|<)", s)
    assert operations(parent) == [op for op in operations(source) if not op.startswith("ggml_cuda_ar_pool_size")]
    with tempfile.TemporaryDirectory(dir=ROOT / "build-tmp") as directory:
        path = Path(directory)
        cpp = path / "pool.cpp"
        cpp.write_text('''#include "ar-pool.h"
#include <cassert>
#include <string>
int main() {
    assert(ggml_cuda_ar_pool_size(nullptr) == 2);
    for (int n = 2; n <= 16; ++n) {
        assert(ggml_cuda_ar_pool_size(std::to_string(n).c_str()) == n);
    }
    for (const char * s : {"", "0", "1", "17", "-2", "+2", " 2", "2 ", "2x", "2.0", "999999999999999999999999"}) {
        assert(ggml_cuda_ar_pool_size(s) == 0);
    }
}
''')
        subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I" + str(SOURCE.parent), str(cpp), "-o", str(path / "pool")], check=True)
        subprocess.run([str(path / "pool")], check=True)
    print(f"M: parser range/rejections PASS; all ring uses PASS; {controls} hardcoded-use controls rejected; CUDA operation order unchanged")


if __name__ == "__main__":
    main()
