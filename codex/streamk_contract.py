"""CPU arithmetic check of the dynamic stream-k partition and empty-block fixup gates."""

def fast_values(d):
    l = (d - 1).bit_length()
    mp = (((1 << 32) * ((1 << l) - d) // d) + 1) & 0xffffffff
    return mp, l, d


def fastdiv(n, fd):
    mp, l, _ = fd
    return ((n * mp >> 32) + n) >> l


cases = 0
for tiles in (1, 2, 4, 6, 8, 12, 24, 48, 128):
    for live in (1, 2, 3, 7, 16, 32, 128, 353):
        for blocks in range(1, 129):
            total = tiles * live
            spans = [(b * total // blocks, (b + 1) * total // blocks) for b in range(blocks)]
            assert spans[0][0] == 0 and spans[-1][1] == total
            assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
            for tile in range(tiles):
                active = [b for b, (start, stop) in enumerate(spans) if start < stop and start < (tile + 1) * live and stop > tile * live]
                assert active
                if blocks % tiles == 0 and blocks > tiles:
                    bpt = blocks // tiles
                    first, last = tile * bpt, (tile + 1) * bpt - 1
                    expected = [b for b in range(first, last + 1) if (b-first)*live//bpt != (b-first+1)*live//bpt]
                    assert active == expected
                    assert active[-1] == last
                    if live == 1:
                        assert len(active) == 1
                    else:
                        assert spans[last][0] % live > 0
                elif tiles % blocks:
                    last = active[-1]
                    start, stop = spans[last]
                    fixup = start != stop and start % live != 0 and not (start // live == stop // live and stop % live != 0)
                    if start // live == tile:
                        assert fixup == (len(active) > 1)
            for d in (live, live * tiles):
                fd = fast_values(d)
                for n in (0, 1, d-1, d, d+1, total, (1 << 31)-1):
                    assert fastdiv(n, fd) == n // d
            cases += 1
print(f'M: stream-k arithmetic cases={cases}; live coverage, uniform empty blocks, general fixup, dynamic divisors pass')
