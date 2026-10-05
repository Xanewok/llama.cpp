#!/usr/bin/env python3
"""Compile the actual host dispatch with mocked GPU capabilities; never load CUDA."""
import os
from pathlib import Path
import resource
import subprocess

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "ggml/src/ggml-cuda/fattn.cu"


def function(source, name):
    start = source.rfind("\n", 0, source.index(name)) + 1
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


PREFIX = r'''
#include "ggml.h"
#include "ggml-impl.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <initializer_list>
#define FLASH_ATTN_AVAILABLE
#define FATTN_KQ_STRIDE 256
#define GGML_CUDA_CC_TURING 750
#define GGML_CUDA_CC_ADA_LOVELACE 890
#define GGML_CUDA_CC_IS_NVIDIA(cc) ((cc) < 10000)
static bool sparse = false;
static bool turing_mma_available(int cc) { return cc >= 750 && cc < 10000; }
static bool cp_async_available(int cc) { return cc >= 800 && cc < 10000; }
static bool volta_mma_available(int cc) { return cc == 700; }
static bool amd_mfma_available(int) { return false; }
static bool amd_wmma_available(int) { return false; }
static bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(int, const ggml_tensor *, int, int) { return sparse; }
static bool ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(int, int, int, int) { return false; }
static struct { struct { int cc = 1200; } devices[1]; } info;
static auto & ggml_cuda_info() { return info; }
struct ggml_cuda_flash_attn_ext_f16_extra_data { uintptr_t K, V, end; };
static void * ggml_cuda_get_fattn_vec_case(int64_t, ggml_type k, ggml_type v) {
    return k == v && (k == GGML_TYPE_F16 || k == GGML_TYPE_Q8_0) ? (void *) uintptr_t(1) : nullptr;
}
enum best_fattn_kernel { BEST_FATTN_KERNEL_NONE, BEST_FATTN_KERNEL_TILE, BEST_FATTN_KERNEL_VEC, BEST_FATTN_KERNEL_MMA_F16 };
'''

MAIN = r'''
int main() {
    int cases = 0;
    for (int kv : {8192, 32768, 90112}) {
        for (int d : {128, 256}) {
            for (int gqa : {1, 4, 6, 8, 12}) {
                ggml_tensor q{}, k{}, v{}, m{}, out{};
                q.type = GGML_TYPE_F32; k.type = v.type = GGML_TYPE_Q8_0; m.type = GGML_TYPE_F16;
                q.ne[0] = k.ne[0] = v.ne[0] = d;
                q.ne[1] = 1; q.ne[2] = 2*gqa; q.ne[3] = 1;
                k.ne[1] = v.ne[1] = kv; k.ne[2] = v.ne[2] = 2;
                k.ne[3] = v.ne[3] = 1;
                m.ne[0] = kv; m.ne[1] = m.ne[2] = m.ne[3] = 1;
                for (int i = 1; i < 4; ++i) { q.nb[i] = m.nb[i] = 1024; k.nb[i] = v.nb[i] = d*34/32; }
                k.data = v.data = (void *) uintptr_t(16);
                out.src[0] = &q; out.src[1] = &k; out.src[2] = &v; out.src[3] = &m;
                out.op = GGML_OP_FLASH_ATTN_EXT; out.type = GGML_TYPE_F32;
                out.ne[0] = d; out.ne[1] = 2*gqa; out.ne[2] = out.ne[3] = 1;
                out.nb[0] = 4; out.nb[1] = d*4; out.nb[2] = out.nb[3] = d*4*2*gqa;
                out.data = (void *) uintptr_t(128);
                const bool enabled = d == 256 && gqa > 4;
                assert(ggml_cuda_get_best_fattn_kernel(0, &out) == (enabled ? BEST_FATTN_KERNEL_MMA_F16 : BEST_FATTN_KERNEL_VEC));
                assert(ggml_cuda_flash_attn_ext_get_alloc_size(0, &out) == ggml_nbytes(&out));
                if (enabled) {
                    out.src[3] = nullptr;
                    assert(ggml_cuda_get_best_fattn_kernel(0, &out) == BEST_FATTN_KERNEL_VEC);
                    out.src[3] = &m;
                    q.nb[2] += 4;
                    assert(ggml_cuda_get_best_fattn_kernel(0, &out) == BEST_FATTN_KERNEL_VEC);
                    q.nb[2] -= 4;
                    k.data = (void *) uintptr_t(18);
                    assert(ggml_cuda_get_best_fattn_kernel(0, &out) == BEST_FATTN_KERNEL_VEC);
                    k.data = (void *) uintptr_t(16);
                    ((float *) out.op_params)[1] = 0.1f;
                    assert(ggml_cuda_get_best_fattn_kernel(0, &out) == BEST_FATTN_KERNEL_VEC);
                    ((float *) out.op_params)[1] = 0;
                    ((float *) out.op_params)[2] = 1;
                    assert(ggml_cuda_get_best_fattn_kernel(0, &out) == BEST_FATTN_KERNEL_VEC);
                    ((float *) out.op_params)[2] = 0;
                    sparse = true;
                    assert(ggml_cuda_get_best_fattn_kernel(0, &out) == BEST_FATTN_KERNEL_VEC);
                    sparse = false;
                    v.type = GGML_TYPE_F16;
                    assert(ggml_cuda_get_best_fattn_kernel(0, &out) == BEST_FATTN_KERNEL_VEC);
                    v.type = GGML_TYPE_Q8_0;
                    info.devices[0].cc = 750;
                    assert(ggml_cuda_get_best_fattn_kernel(0, &out) == BEST_FATTN_KERNEL_VEC);
                    info.devices[0].cc = 1200;
                    q.ne[1] = 2;
                    assert(ggml_cuda_get_best_fattn_kernel(0, &out) == BEST_FATTN_KERNEL_VEC);
                }
                q.ne[1] = out.ne[2] = 3;
                const size_t allocation = ggml_cuda_flash_attn_ext_get_alloc_size(0, &out);
                assert(enabled ? allocation == ggml_nbytes(&out) : allocation > ggml_nbytes(&out));
                ++cases;
            }
        }
    }
    printf("M: %d dispatch shapes and fallback gates passed\n", cases);
}
'''


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    source = SOURCE.read_text()
    parts = [function(source, name) for name in (
        "ggml_cuda_flash_attn_ext_mma_q8_0_supported",
        "ggml_cuda_fattn_kv_type_supported", "ggml_cuda_get_best_fattn_kernel")]
    common = (ROOT / "ggml/src/ggml-cuda/fattn-common.cuh").read_text()
    parts.append(function(common, "ggml_cuda_flash_attn_ext_get_f16_extra_data"))
    parts.append(function(source, "ggml_cuda_flash_attn_ext_get_alloc_size"))
    good = PREFIX + "\n".join(parts) + MAIN
    mutant = good.replace("(Q->ne[1] != 1 && Q->ne[1] < 3)", "Q->ne[1] < 3")
    assert mutant != good
    allocation_mutant = good.replace(
        "need_f16_K = need_f16_V = !ggml_cuda_flash_attn_ext_mma_q8_0_supported(ggml_cuda_info().devices[device].cc, dst);",
        "need_f16_K = need_f16_V = true;")
    assert allocation_mutant != good
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GGML_CUDA_FA_MMA_Q8_0="1")
    for name, text in (("dispatch", good), ("dispatch-mutant", mutant), ("allocation-mutant", allocation_mutant)):
        path = ROOT / f"build-tmp/{name}.cpp"
        path.write_text(text)
        binary = path.with_suffix("")
        subprocess.run(["c++", "-O2", "-std=c++17", "-Iggml/include", "-Iggml/src", str(path),
                        "-Lbuild-cpu/bin", "-Wl,-rpath," + str(ROOT / "build-cpu/bin"),
                        "-lggml-base", "-o", str(binary)], cwd=ROOT, env=env, check=True)
        result = subprocess.run([str(binary)], cwd=ROOT, env=env, text=True, capture_output=True)
        if name == "dispatch":
            assert result.returncode == 0, result.stderr
            print(result.stdout, end="")
        else:
            assert result.returncode != 0, "negative dispatch control passed"
            print(f"M: {name} fails the control")


if __name__ == "__main__":
    main()
