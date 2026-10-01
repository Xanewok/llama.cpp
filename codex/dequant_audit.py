#!/usr/bin/env python3
"""CPU arithmetic model; does not execute CUDA or establish pipeline correctness."""
import struct
from pathlib import Path
import numpy as np


def byte_perm(a, b, selector):
    packed = (a & 0xffffffff).to_bytes(4, 'little') + (b & 0xffffffff).to_bytes(4, 'little')
    return sum(packed[(selector >> (4*i)) & 7] << (8*i) for i in range(4))


def unpack_pair(raw):
    src = struct.unpack('<17I', raw)
    scales, quants = [], []
    for b in range(2):
        scales.append(src[0] & 0xffff if b == 0 else src[8] >> 16)
        for cc in range(4):
            for l in range(2):
                m = 2*cc + l
                q = byte_perm(src[m], src[m+1], 0x5432) if b == 0 else src[9+m]
                u = q ^ 0x80808080
                for word in (byte_perm(u, 0x64646464, 0x5140), byte_perm(u, 0x64646464, 0x7362)):
                    quants.extend(struct.unpack('<2e', word.to_bytes(4, 'little')))
    return scales, np.asarray(quants, dtype=np.float16) - np.float16(1152)


def main():
    root = Path(__file__).resolve().parent.parent
    source = (root / 'ggml/src/ggml-cuda/fattn-mma-f16.cuh').read_text()
    for expression in ('make_half2(1152.0f, 1152.0f)', '__byte_perm(src[m], src[m + 1], 0x5432)',
                       '__byte_perm(u, 0x64646464, 0x5140)', '__byte_perm(u, 0x64646464, 0x7362)',
                       '(threadIdx.x % 8)*pairs_per_row + threadIdx.x/8'):
        assert expression in source, expression
    q = np.arange(-128, 128, dtype=np.int16)
    encoded = (0x6400 | ((q + 128).astype(np.uint16))).view(np.float16)
    recovered = encoded - np.float16(1152)
    assert np.array_equal(recovered.view(np.uint16), q.astype(np.float16).view(np.uint16))
    scales_bits = np.arange(65536, dtype=np.uint16)
    scales_bits = scales_bits[(scales_bits & 0x7c00) != 0x7c00]
    count = 0
    with np.errstate(over='ignore', invalid='ignore'):
        for start in range(0, len(scales_bits), 1024):
            scales = scales_bits[start:start+1024].view(np.float16)[:, None]
            actual = (scales.astype(np.float32) * recovered.astype(np.float32)).astype(np.float16)
            reference = (scales.astype(np.float32) * q.astype(np.float32)).astype(np.float16)
            assert np.array_equal(actual.view(np.uint16), reference.view(np.uint16))
            count += actual.size
    # Exercise the half-block boundary and all signed bytes in the packed representation.
    for offset in range(256):
        values = ((np.arange(64) + offset) % 256).astype(np.uint8)
        raw = struct.pack('<H', 0x3555) + values[:32].tobytes() + struct.pack('<H', 0xb155) + values[32:].tobytes()
        scales, actual = unpack_pair(raw)
        assert scales == [0x3555, 0xb155]
        assert np.array_equal(actual, values.view(np.int8).astype(np.float16))
    lanes = [(x % 8)*4 + x//8 for x in range(32)]
    assert sorted(lanes) == list(range(32))
    for first in range(0, 32, 8):
        assert len({p//4 for p in lanes[first:first+8]}) == 8
    broken = encoded - np.float16(1151)
    assert not np.array_equal(broken, q.astype(np.float16)), 'negative control did not fail'
    print(f'M: {count} finite-scale/sign-byte products match f16 bits; 256 packed pairs match; lane permutation is bijective.')
    print('M: bias=1151 negative control differs. Non-finite scales and CUDA scheduling are outside this check.')


if __name__ == '__main__':
    main()
