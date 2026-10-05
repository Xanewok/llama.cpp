"""CPU check of packed F16 tail detection, real-query union and alignment fallback."""
import struct


def bound(data, nk, nq, nbq, warp, offset=0):
    nwords = 256 // (2*warp)
    nbytes = 4*nwords
    if nk % 256 or nbq % nbytes or offset % nbytes:
        return nk
    end = nk
    while end:
        start = end - 256
        all_masked = True
        for lane in range(warp):
            for q in range(nq):
                values = struct.unpack_from('<' + 'I'*nwords, data, offset + q*nbq + (start + lane*2*nwords)*2)
                all_masked &= all(v == 0xfc00fc00 for v in values)
        if not all_masked:
            break
        end = start
    return end


cases = 0
for warp in (32, 64):
    for nq in (1, 4, 6, 8):
        for tail in (0, 1, 255, 256, 257, 4095):
            nk = 8192
            live = nk - tail
            for extra in (0, 16):
                nbq = 2*nk + extra
                # Extra columns and the padding query row are finite; neither may affect the bound.
                data = bytearray(nbq*(nq + 1))
                for q in range(nq):
                    live_q = live - (min(256, live-1) if q % 2 else 0)
                    data[q*nbq:q*nbq + 2*nk] = b'\x00\xfc'*nk
                    data[q*nbq:q*nbq + 2*live_q] = b'\x00\x00'*live_q
                expected = (live + 255)//256*256
                assert bound(data, nk, nq, nbq, warp) == expected
                assert bound(data, nk, nq, nbq, warp) >= live
                assert bound(b'\0\0'+data, nk, nq, nbq, warp, 2) == nk
                assert bound(data, nk-1, nq, nbq, warp) == nk-1
                assert bound(data, nk, nq, nbq+2, warp) == nk
                # A finite value, positive infinity or NaN in the last unit must retain that unit.
                for bits in (0, 0x7c00, 0x7e00, 0xfbff):
                    changed = bytearray(data)
                    struct.pack_into('<H', changed, 2*(nk-1), bits)
                    assert bound(changed, nk, nq, nbq, warp) == nk
                cases += 1
print(f'M: packed-mask cases={cases}; query union, padding exclusion, special values and alignment fallback pass')
