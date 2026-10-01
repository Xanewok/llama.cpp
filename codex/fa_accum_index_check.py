#!/usr/bin/env python3
"""CPU-only index audit. Compile the production tile maps/configs with CUDA qualifiers removed."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "ggml/src/ggml-cuda"


def body(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def expression(source, name):
    return re.search(r"\b" + name + r"\s*=\s*([^;]+);", source)[1]


@lru_cache(None)
def compile_index(expr):
    expr = expr.replace("T_C_VKQ_h::", "h_").replace("T_C_VKQ::", "")
    expr = expr.replace("threadIdx.x", "lane").replace("threadIdx.y", "warp")
    expr = expr.replace("(cols_per_warp == 8 ? get_j(l) : get_i(l))", "(get_j(l) if cols_per_warp == 8 else get_i(l))")
    expr = expr.replace("(cols_per_warp == 8 ? get_i(l) : get_j(l))", "(get_i(l) if cols_per_warp == 8 else get_j(l))")
    # All index division has nonnegative operands except the ignored get_j(-1) argument.
    expr = re.sub(r"(?<!/)/(?!/)", "//", expr)
    return compile(expr, "<production index>", "eval")


def evaluate(expr, **values):
    return eval(compile_index(expr), {"__builtins__": {}}, values)


def source_maps(source, out):
    instances = set()
    for path in (CUDA / "template-instances").glob("fattn-mma-f16-instance-*.cu"):
        for kind, dk, dv, c1, c2 in re.findall(r"DECL_FATTN_MMA_(F16|Q8_0)_CASE\((\d+), (\d+), (\d+), (\d+)\)", path.read_text()):
            instances.add((int(dk), int(dv), int(c1), int(c2), kind))
    assert instances
    mma = (CUDA / "mma.cuh").read_text()
    enum = body(mma, "enum data_layout") + ";"
    tiles = mma[mma.index("template <int I_, int J_, typename T, data_layout ds_"):
                mma.index("template <int I_, int J_>\n    struct tile<I_, J_, nv_bfloat162")]
    configs = source[source.index("struct fattn_mma_config"):
                     source.index("static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_volta")]
    shapes = sorted({(dk, dv, c1*c2) for dk, dv, c1, c2, _ in instances})
    cpp = """
#include <cassert>
#include <cstdint>
#include <iostream>
#include <type_traits>
#define __host__
#define __device__
#define __forceinline__ inline
#define WARP_SIZE 32
#define NO_DEVICE_CODE assert(false)
struct { int x, y; } threadIdx;
struct half2 {
    uint16_t x, y;
    constexpr half2(float a = 0, float b = 0) : x(a), y(b) {}
};
""" + enum + tiles + configs + """
template<int I, int J, typename T> void dump(const char * name) {
    using C = tile<I, J, T>;
    for (threadIdx.x = 0; threadIdx.x < 32; ++threadIdx.x) {
        for (int l = -1; l < C::ne; ++l) {
            std::cout << "MAP " << name << " " << I << " " << J << " " << threadIdx.x << " " << l
                      << " " << C::get_i(l) << " " << C::get_j(l) << "\\n";
        }
    }
}
int main() {
    static_assert(sizeof(half2) == sizeof(float));
    dump<16, 8, float>("f"); dump<16, 16, float>("f");
    dump<16, 4, half2>("h"); dump<16, 8, half2>("h"); dump<8, 8, half2>("h");
"""
    for dk, dv, nc in shapes:
        for arch in (75, 80):
            getter = "turing" if arch == 75 else "ampere"
            cpp += f"""{{ auto c = ggml_cuda_fattn_mma_get_config_{getter}({dk}, {dv}, {nc});
                std::cout << "CFG {arch} {dk} {dv} {nc} " << c.nthreads << " " << c.nbatch_fa << " "
                << c.nbatch_K2 << " " << c.nbatch_V2 << " " << c.nbatch_combine << " "
                << c.nstages_target << " " << c.Q_in_reg << "\\n"; }}\n"""
    cpp += "}\n"
    path = out / "maps.cpp"
    path.write_text(cpp)
    subprocess.run(["c++", "-O2", "-std=c++17", str(path), "-o", str(out / "fa-index-maps")], check=True)
    lines = subprocess.check_output([str(out / "fa-index-maps")], text=True).splitlines()
    maps, configs = {}, {}
    for line in lines:
        parts = line.split()
        if parts[0] == "MAP":
            maps[(parts[1], *map(int, parts[2:6]))] = tuple(map(int, parts[6:]))
        else:
            configs[tuple(map(int, parts[1:5]))] = tuple(map(int, parts[5:]))
    return sorted(instances), maps, configs


def global_inputs(dk, dv, c1, c2, batch, stages, kind):
    checked = 0
    # Multi-stage GQA only runs with padded KV. The scalar tail uses a zero source.
    lengths = (batch, 2*batch) if stages > 1 else (1, batch-1, batch, batch+1)
    for kv in lengths:
        for tokens in sorted({1, 2, 4, 8, max(1, c1-1), c1, c1+1}):
            for gqa in sorted({1, 6, max(1, c2-1), c2, c2+1}):
                heads = 2*gqa
                # Both query layouts, with row padding and a nonzero sequence/head/view offset.
                for head_major in (False, True):
                    width = dk//2+4
                    sq1, sq2 = (width, tokens*width) if head_major else (heads*width, width)
                    size = 2*tokens*heads*width
                    for jt in range((tokens+c1-1)//c1):
                        for zt in range((gqa+c2-1)//c2):
                            for jc in range(c1*c2):
                                j, c = jc//c2, jc % c2
                                if jt*c1+j >= tokens or zt*c2+c >= gqa:
                                    continue
                                for sequence, kh in ((0, 0), (1, 1)):
                                    base = sequence*tokens*heads*width+(kh*gqa+zt*c2)*sq2
                                    first = base+(jt*c1+j)*sq1+c*sq2
                                    assert 0 <= first and first+dk//2 <= size
                                    sink = kh*gqa+zt*c2+c
                                    assert 0 <= sink < heads
                                    dst = ((sequence*tokens+jt*c1+j)*heads+sink)*(dv//2)
                                    assert dst+dv//2 <= 2*tokens*heads*(dv//2)
                                    checked += 3
                    # Mask queries wrap into its padded query dimension, including the last unused tile rows.
                    tile_queries = ((tokens+c1-1)//c1)*c1
                    for j in range(tile_queries):
                        for kb in range((kv+batch-1)//batch):
                            for i in (0, batch-1):
                                if kb*batch+i >= kv:
                                    assert stages <= 1
                                    continue
                                addr = (j % tokens)*kv+kb*batch+i
                                assert addr < tokens*kv
                                checked += 1
                for d in (dk, dv):
                    rowbytes = d//32*34 if kind == "Q8_0" else d*2
                    stride = ((rowbytes+15)//16)*16+16
                    size = 2*2*kv*stride
                    for kb in range((kv+batch-1)//batch):
                        for i in range(batch):
                            if kb*batch+i >= kv:
                                assert stages <= 1  # synchronous loader copies its 16-byte local zero
                                continue
                            first = ((1*2+1)*kv+kb*batch+i)*stride
                            assert first+rowbytes <= size
                            # Sparse gathers either a valid index or a zero/-inf source.
                            for index in (-1, 0, kv-1):
                                if index >= 0:
                                    assert ((1*2+1)*kv+index)*stride+rowbytes <= size
                            checked += 1
    return checked


def audit(source, instances, maps, configs):
    writer = body(source, "if constexpr (use_f32) {")
    j_expr, k_expr, index_expr = (expression(writer, name) for name in ("j", "k", "index"))
    assert "tile_Q_f[index] = VKQ_C[(2*k00 + k1)/tile_width].x[l]" in writer
    assert "k1 < 2*nbatch_combine; k1 += tile_width" in writer
    assert "((const float2 *) tile_Q)[(jc_tile_K + ip*cols_per_warp) * (tile_stride/2) + k]" in source
    assert "jc_tile_K*tile_stride + (use_f32 ? 2*nbatch_combine : nbatch_combine)" in source
    assert "nbatch_combine_h % 16 == 0 ? nbatch_combine_h/2 : nbatch_combine_h" in source
    barrier = re.search(r"if \(([^\n]+)\) \{\s*__syncthreads\(\);\s*\}\s*\}\s*#else\s*GGML_UNUSED_VARS\(Q_f2", source)[1]
    barrier = barrier.replace("||", " or ").replace("&&", " and ")
    counts = {"configs": 0, "f32_words": 0, "metadata_words": 0, "shared_chunks": 0, "output_pairs": 0, "global_ranges": 0}
    maxima = {}
    for arch in (75, 80):
        for dk, dv, c1, c2, kind in instances:
            nc = c1*c2
            if arch == 75 and nc > 32:
                continue  # production kernel guard; nb=8 dispatches width 4 on Turing
            if dk == 192 and c2 not in (8, 16):
                continue
            threads, batch, k2, v2, combine_h, stages_target, qreg = configs[(arch, dk, dv, nc)]
            assert batch > 0
            stages = stages_target if arch >= 80 and c2 >= 2 else 0
            if kind == "Q8_0" and stages <= 1:
                continue
            counts["global_ranges"] += global_inputs(dk, dv, c1, c2, batch, stages, kind)
            thin = nc == 8 or (dk in (256, 512) and nc == 16) or (dk == 512 and nc == 32)
            cpw = 8 if thin else 16
            nw = threads//32
            np = nw*cpw//nc
            assert np >= 1 and nw % np == 0 and nc % cpw == 0
            hj, fj = (4, 8) if thin else (8, 16)
            tile_width = 16
            swz = k2 >= 32 and k2 % 32 == 0 and v2 >= 32 and v2 % 32 == 0
            sk, sv = k2 + (0 if swz else 4), v2 + (0 if swz else 4)
            if dk == 576:
                sv = sk  # V is a K view
            qbytes = nc*(dk//2 + 4)*4
            kvbytes = batch*(sk+sv if stages > 1 else max(sk, sv))*4
            maskbytes = c1*(batch//2 + 4)*4
            rawbytes = batch*(dk//32)*34 if kind == "Q8_0" else 0
            for f32 in (False, True):
                combine = combine_h//2 if f32 and combine_h % 16 == 0 else combine_h
                stride = (2 if f32 else 1)*combine + 4
                combine_bytes = nw*min(nc, 16)*stride*4  # actual host formula, including thin over-allocation
                total = max(combine_bytes, max(qbytes, kvbytes+maskbytes+rawbytes) if qreg else qbytes+kvbytes+maskbytes+rawbytes)
                rows = nw*cpw
                assert rows*stride*4 <= total
                assert (dv//2) % combine == 0 and (not f32 or (2*combine) % tile_width == 0)
                counts["configs"] += 1
                maxima[f"sm_{arch}/{dk}/{c1}/{c2}/{kind}/{'f32' if f32 else 'default'}"] = total
                assert qbytes <= total and batch % (np*16) == 0

                # Each ldmatrix lane supplies an aligned 16-byte row segment.
                for warp in range(nw):
                    for lane in range(32):
                        for k0 in range(0, dk//2, 8):
                            row = warp//np*cpw+lane % cpw
                            col = k0+(lane//cpw*4) % 8
                            addr = (row*(dk//2+4)+col)*4
                            assert row < nc and col+4 <= dk//2 and addr % 16 == 0 and addr+16 <= qbytes
                        for kstart in range(0, dk//2, k2):
                            kstop = min(dk//2, kstart+k2)
                            for k0 in range(kstart, kstop, 8):
                                for i00 in range(0, batch, np*16):
                                    row = i00+(warp % np)*16+lane % 16
                                    col = k0-kstart+lane//16*4
                                    addr = (row*sk+col)*4
                                    if swz:
                                        addr ^= (row & 7) << 4
                                    assert row < batch and col+4 <= kstop-kstart and addr % 16 == 0 and addr+16 <= batch*sk*4
                        for ivstart in range(0, dv, 2*v2):
                            for iv in range(ivstart, ivstart+2*v2, 16):
                                for k00 in range(0, batch//2, np*8):
                                    row = 2*k00+(warp % np)*16+lane % 16
                                    col = (iv-ivstart)//2+lane//16*4
                                    addr = (row*sv+col)*4
                                    if swz:
                                        addr ^= (row & 7) << 4
                                    assert row < batch and col+4 <= v2 and addr % 16 == 0 and addr+16 <= batch*sv*4

                # Address sets cover every scalar in K/V tiles and every 16-byte loader/ldmatrix chunk.
                kvbase = 0 if qreg else qbytes
                for width, ts, offset in ((k2, sk, kvbase), (v2, sv, kvbase + (batch*sk*4 if stages > 1 else 0))):
                    assert offset + batch*ts*4 <= total
                    for row in range(batch):
                        for col in range(0, width, 4):
                            addr = (row*ts + col)*4
                            if swz:
                                addr ^= (row & 7) << 4
                            assert addr % 16 == 0 and row*ts*4 <= addr and addr+16 <= (row+1)*ts*4
                            assert offset+addr+16 <= total
                            counts["shared_chunks"] += 1
                maskbase = kvbase + kvbytes
                assert maskbase+maskbytes+rawbytes <= total
                if kind == "Q8_0":
                    rowbytes = dk//32*34
                    assert stages > 1 and dk == dv and k2 == dk//2 and v2 == dv//2
                    assert rowbytes % 16 == 0 and (maskbase+maskbytes) % 16 == 0
                    for row in range(batch):
                        for pair in range(dk//64):
                            src = row*rowbytes + pair*68
                            # The dequantizer reads src[0..16], including both scale words.
                            assert 0 <= src and src+17*4 <= rawbytes
                            for b in range(2):
                                for cc in range(4):
                                    col = pair*32 + b*16 + cc*4
                                    addr = (row*sk + col)*4
                                    if swz:
                                        addr ^= (row & 7) << 4
                                    assert addr % 16 == 0 and addr+16 <= batch*sk*4

                # Half C packs two adjacent J scalars. Float C must cover the same logical rectangle once.
                half_coords, float_coords = set(), set()
                for lane in range(32):
                    for l in range(16*hj//32):
                        i, j = maps[("h", 16, hj, lane, l)]
                        for h in range(2):
                            coord = (2*j+h, i) if thin else (i, 2*j+h)
                            assert coord not in half_coords
                            half_coords.add(coord)
                    for l in range(16*fj//32):
                        i, j = maps[("f", 16, fj, lane, l)]
                        coord = (j, i) if thin else (i, j)
                        assert coord not in float_coords
                        float_coords.add(coord)
                        col = l % 2 if thin else (l//2) % 2
                        query = 2*(lane % 4)+col if thin else lane//4+8*col
                        assert query == coord[0]  # rescale and sink use this same KQ_max slot
                assert half_coords == float_coords == {(j, k) for j in range(cpw) for k in range(16)}

                # The metadata stays at the packed-half column mapping; float2 addressing uses two words.
                meta = {}
                for warp in range(nw):
                    for lane in range(32):
                        if thin:
                            if lane >= 8:
                                continue
                            col = lane % 8//4
                            row = warp*8 + 2*maps[("h", 16, 4, lane, -1)][1]+col
                        else:
                            if lane % 4 >= 2:
                                continue
                            col = lane % 2
                            row = warp*16 + maps[("h", 16, 8, lane, lane % 4)][0]
                        assert row not in meta
                        word = row*stride + (2*combine if f32 else combine)
                        assert word % 2 == 0 and word+2 <= total//4
                        assert word % stride == stride-4
                        meta[row] = (warp//np*cpw+row % cpw, col)
                        counts["metadata_words"] += 2
                assert set(meta) == set(range(rows))
                nmeta = max(1, np*cpw//32)
                for warp in range(0, nw, np):
                    for lane in range(32):
                        row = warp*cpw + (lane % (np*cpw) if np*cpw < 32 else lane)
                        for im in range(nmeta):
                            r = row + im*32
                            assert r in meta and r*stride+stride-2 <= total//4

                # Every chunk is filled with distinct coordinate tags and read with the production stride.
                for k00 in range(0, dv//2, combine):
                    written = {}
                    for warp in range(nw):
                        for lane in range(32):
                            if f32:
                                for k1 in range(0, 2*combine, tile_width):
                                    for l in range(16*fj//32):
                                        i, j = maps[("f", 16, fj, lane, l)]
                                        env = dict(l=l, lane=lane, warp=warp, k1=k1, cols_per_warp=cpw,
                                                   get_i=lambda _: i, get_j=lambda _: j, tile_stride=stride)
                                        row = evaluate(j_expr, **env)
                                        k = evaluate(k_expr, **env)
                                        idx = evaluate(index_expr, **env, j=row, k=k)
                                        assert 0 <= row < rows and 0 <= k < 2*combine
                                        assert 0 <= idx < total//4, f"OOB sm_{arch} {dk=} {c1=} {c2=} {idx=} capacity={total//4}"
                                        assert idx == row*stride+k, "float writer/reader stride mismatch"
                                        assert idx not in written, "duplicate float C store"
                                        acc = (2*k00+k1)//tile_width
                                        assert 0 <= acc < dv//tile_width
                                        written[idx] = (warp, warp//np*cpw+row % cpw, 2*k00+k)
                                        counts["f32_words"] += 1
                            else:
                                for j in range(cpw):
                                    for k in range(combine):
                                        row = warp*cpw+j
                                        written[row*stride+k] = (warp, warp//np*cpw+j, 2*(k00+k))
                    assert len(written) == rows*combine*(2 if f32 else 1)
                    cross_warp = False
                    output = set()
                    for stride_k in (32, 16, 8, 4):
                        start = 0 if stride_k == 32 else combine-combine % (2*stride_k)
                        stop = combine-combine % stride_k
                        sj = 32//stride_k
                        for warp in range(0, nw, np):
                            for lane in range(32):
                                for jc0 in range(0, nc, (nw//np)*sj):
                                    jc = jc0+(warp//np)*sj+(0 if stride_k == 32 else lane//stride_k)
                                    if jc >= nc:
                                        continue
                                    row = (jc//cpw)*(np*cpw)+jc % cpw
                                    for k0 in range(start, stop, stride_k):
                                        k = k0+(lane if stride_k == 32 else lane % stride_k)
                                        for ip in range(np):
                                            r = row+ip*cpw
                                            idx = r*stride+(2*k if f32 else k)
                                            tag = written[idx]
                                            assert tag[1:] == (jc, 2*(k00+k))
                                            if f32:
                                                assert written[idx+1][1:] == (jc, 2*(k00+k)+1)
                                            cross_warp |= tag[0] != warp
                                            assert r in meta
                                        assert (jc, k00+k) not in output
                                        output.add((jc, k00+k))
                                        # Final-tile guards remove padded query/head columns; fixup has full nc storage.
                                        for tokens, gqa in ((c1, c2), (1, 1), (max(1, c1-1), max(1, c2-1))):
                                            jd, cd = jc//c2, jc % c2
                                            if jd < tokens and cd < gqa:
                                                addr = (jd*gqa+cd)*(dv//2)+k00+k
                                                assert 0 <= addr < tokens*gqa*(dv//2)
                                        # Full fixup allocation = grid*(2*nc + nc*DV/2) float2s.
                                        for grid, block in ((1, 0), (36, 35)):
                                            addr = grid*2*nc + block*nc*(dv//2)+jc*(dv//2)+k00+k
                                            assert addr < grid*(2*nc+nc*(dv//2))
                                            assert block*nc+jc < grid*2*nc
                                            assert (grid+block)*nc+jc < grid*2*nc
                                        counts["output_pairs"] += 1
                    assert output == {(jc, k00+k) for jc in range(nc) for k in range(combine)}
                    if f32 and k00+combine < dv//2 and cross_warp:
                        assert eval(barrier, {"__builtins__": {}}, dict(np=np, use_f32=f32, nbatch_combine=combine, DV=dv)), "cross-warp read/next-chunk write race"
    return counts, maxima


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "build-accum-fix/index")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_VISIBLE_DEVICES must be empty"
    out = args.out.resolve()
    assert out.is_relative_to(ROOT) and out.parts[len(ROOT.parts)].startswith("build")
    out.mkdir(parents=True, exist_ok=True)
    source = (CUDA / "fattn-mma-f16.cuh").read_text()
    start = time.monotonic()
    instances, maps, configs = source_maps(source, out)
    counts, sizes = audit(source, instances, maps, configs)
    print("M CPU index audit:", json.dumps(counts, sort_keys=True))
    for kind in ("F16", "Q8_0"):
        size = sizes[f"sm_80/256/4/8/{kind}/f32"]
        bad_end = ((4*16-1)*(2*132)+127+1)*4
        assert bad_end > size
        print(f"D old width-4 {kind} store end: {bad_end} bytes; allocation: {size} bytes; overrun: {bad_end-size} bytes")
    (out / "shared-bytes.json").write_text(json.dumps(sizes, indent=2, sort_keys=True)+"\n")
    for label, bad in (("double float stride", source.replace("const int index = j*tile_stride + k;", "const int index = j*(2*tile_stride) + k;")),
                       ("missing f32 chunk barrier", source.replace("np > 1 || (use_f32 && nbatch_combine < DV/2)", "np > 1"))):
        assert bad != source
        try:
            audit(bad, instances, maps, configs)
        except AssertionError as error:
            print(f"M negative control ({label}): rejected: {error}")
        else:
            raise AssertionError(f"negative control passed: {label}")
    print(f"M CPU wall time including map compile: {time.monotonic()-start:.3f} s")
    print("D address proof assumes valid tensor strides, padded mask, and in-range sparse indices; no CUDA instruction executed.")


if __name__ == "__main__":
    main()
