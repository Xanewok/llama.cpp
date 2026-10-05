#!/usr/bin/env python3
"""Run the production tile loaders with CPU half arithmetic and CUDA copy shims."""
import os
from pathlib import Path
import resource
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def function(source, name):
    pos = source.index(name)
    start = source.rfind("template<", 0, pos)
    brace = source.index("{", pos)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


PREFIX = r'''
#include "ggml.h"
#include "ggml-quants.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#define __host__
#define __device__
#define __forceinline__ inline
#define CP_ASYNC_AVAILABLE
using half = uint16_t;
struct half2 { half x, y; };
struct { int x, y; } threadIdx;
static char * shared_base;
static constexpr int ggml_cuda_get_physical_warp_size() { return 32; }
static unsigned int ggml_cuda_cvta_generic_to_shared(void * p) { return (char *)p - shared_base; }
template<int> static void cp_async_cg_16(unsigned int dst, const void * src) { memcpy(shared_base + dst, src, 16); }
static half __ushort_as_half(uint16_t v) { return v; }
static half __int2half_rn(int v) { return ggml_fp32_to_fp16(float(v)); }
static half2 __half2half2(half v) { return {v, v}; }
static half2 __halves2half2(half a, half b) { return {a, b}; }
static half2 make_half2(float a, float b) { return {ggml_fp32_to_fp16(a), ggml_fp32_to_fp16(b)}; }
static half2 __hmul2(half2 a, half2 b) { return make_half2(ggml_fp16_to_fp32(a.x)*ggml_fp16_to_fp32(b.x), ggml_fp16_to_fp32(a.y)*ggml_fp16_to_fp32(b.y)); }
static half2 __hsub2(half2 a, half2 b) { return make_half2(ggml_fp16_to_fp32(a.x)-ggml_fp16_to_fp32(b.x), ggml_fp16_to_fp32(a.y)-ggml_fp16_to_fp32(b.y)); }
static uint32_t __byte_perm(uint32_t a, uint32_t b, uint32_t sel) {
    uint64_t both = uint64_t(b)<<32 | a;
    uint32_t out = 0;
    for (int i=0; i<4; ++i) { out |= uint32_t((both >> (8*((sel>>(4*i))&7)))&255) << (8*i); }
    return out;
}
template<bool swz, typename T> static size_t swizzle_bytes(int i, int k, int stride) {
    size_t offset = (i*stride + k)*sizeof(T);
    if constexpr (swz) { offset ^= (i&7)*16; }
    return offset;
}
template<int n> static void ggml_cuda_memcpy_1(void * dst, const void * src) { memcpy(dst, src, n); }
'''

MAIN = r'''
template<int D, bool swz> void check() {
    constexpr int rows = 128, warps = 4;
    constexpr int raw_row = D/32*sizeof(block_q8_0), stride = D/2+8;
    std::vector<char> raw(rows*2*raw_row, char(0x55)), staged(rows*raw_row + 16, char(0x77));
    std::vector<half2> tile(rows*stride + 8, {0x7777, 0x7777});
    for (int i = 0; i < rows; ++i) {
        auto * blocks = (block_q8_0 *)(raw.data() + i*2*raw_row);
        for (int b = 0; b < D/32; ++b) {
            blocks[b].d = ggml_fp32_to_fp16(((i + b)%17 + 1)*0.00001f);
            for (int j = 0; j < 32; ++j) { blocks[b].qs[j] = (i*31 + b*17 + j)%256 - 128; }
        }
    }
    shared_base = staged.data();
    for (int w = 0; w < warps; ++w) for (int l = 0; l < 32; ++l) {
        threadIdx = {l,w};
        flash_attn_ext_f16_load_tile_q8_0<D,warps,rows>(raw.data(), staged.data(), 2*raw_row, 0);
    }
    for (int i = 0; i < rows; ++i) { assert(memcmp(raw.data()+i*2*raw_row, staged.data()+i*raw_row, raw_row)==0); }
    for (int i = rows*raw_row; i < int(staged.size()); ++i) { assert(staged[i] == char(0x77)); }
    for (int w = 0; w < warps; ++w) for (int l = 0; l < 32; ++l) {
        threadIdx = {l,w};
        flash_attn_ext_f16_dequantize_tile_q8_0<D,stride,swz,warps,rows>(staged.data(), tile.data());
    }
    auto verify = [&](int col, int n) {
        for (int i = 0; i < rows; ++i) for (int j = 0; j < n; j += 2) {
            auto * b = (const block_q8_0 *)(raw.data()+i*2*raw_row) + (col+j)/32;
            half2 expected = make_half2(b->qs[(col+j)%32]*ggml_fp16_to_fp32(b->d), b->qs[(col+j+1)%32]*ggml_fp16_to_fp32(b->d));
            auto * actual = (const half2 *)((const char *)tile.data() + swizzle_bytes<swz,half2>(i,j/2,stride));
            assert(memcmp(actual,&expected,sizeof(expected))==0);
        }
    };
    verify(0, D);
    for (int col = 0; col < D; col += 32) {
        for (int w = 0; w < warps; ++w) for (int l = 0; l < 32; ++l) {
            threadIdx = {l,w};
            flash_attn_ext_f16_load_tile_q8_0_partial<stride,swz,warps,rows>(raw.data(), tile.data(), 16, 2*raw_row, col/2, 0);
        }
        verify(col, 32);
    }
}
template<bool swz> void all() {
    check<64,swz>(); check<96,swz>(); check<128,swz>(); check<192,swz>();
    check<256,swz>(); check<320,swz>(); check<512,swz>(); check<576,swz>();
}
int main() { all<false>(); all<true>(); puts("M: 16 head/swizzle tile cases match CPU dequantization bit for bit"); }
'''


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    source = (ROOT / "ggml/src/ggml-cuda/fattn-mma-f16.cuh").read_text()
    good = PREFIX + "\n".join(function(source, name) for name in (
        "flash_attn_ext_f16_q8_0_row_bytes", "flash_attn_ext_f16_load_tile_q8_0(",
        "flash_attn_ext_f16_dequantize_tile_q8_0", "flash_attn_ext_f16_load_tile_q8_0_partial")) + MAIN
    mutant = good.replace("const int col = 2*k0 + 8*k;", "const int col = 8*k;")
    assert mutant != good
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for name, code in (("q8-loader", good), ("q8-loader-mutant", mutant)):
        path = ROOT / f"build-tmp/{name}.cpp"
        path.write_text(code)
        binary = path.with_suffix("")
        subprocess.run(["c++", "-O2", "-std=c++17", "-fno-strict-aliasing", "-Iggml/include", "-Iggml/src", str(path),
                        "-Lbuild-cpu/bin", "-Wl,-rpath," + str(ROOT / "build-cpu/bin"), "-lggml-base", "-o", str(binary)], check=True, env=env)
        result = subprocess.run([str(binary)], env=env, text=True, capture_output=True)
        if name == "q8-loader":
            assert result.returncode == 0, result.stderr
            print(result.stdout, end="")
        else:
            assert result.returncode != 0, "negative loader control passed"
            print("M: dropped dimension offset fails the control")


if __name__ == "__main__":
    main()
