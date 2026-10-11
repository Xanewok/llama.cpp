#!/usr/bin/env python3
"""Print type/shape/bytes of GGUF tensors matching a regex, reading only the header (sequential reads, no mmap)."""
import re, struct, sys
# ggml type id -> (name, block elems, block bytes)
T = {0:("F32",1,4),1:("F16",1,2),8:("Q8_0",32,34),10:("Q2_K",256,84),11:("Q3_K",256,110),12:("Q4_K",256,144),
     13:("Q5_K",256,176),14:("Q6_K",256,210),20:("IQ4_NL",32,18),23:("IQ4_XS",256,136),30:("BF16",1,2),
     2:("Q4_0",32,18),3:("Q4_1",32,20),6:("Q5_0",32,22),7:("Q5_1",32,24)}
def main(path, pat):
    f = open(path, "rb")
    r = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    rs = lambda: f.read(r("Q")).decode("utf-8", "replace")
    assert f.read(4) == b"GGUF"
    r("I"); n_t = r("Q"); n_kv = r("Q")
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    def skip(vt):
        if vt == 8: rs()
        elif vt == 9:
            at = r("I"); n = r("Q")
            if at == 8:
                for _ in range(n): rs()
            else: f.seek(sz[at] * n, 1)
        else: f.seek(sz[vt], 1)
    for _ in range(n_kv):
        rs(); skip(r("I"))
    for _ in range(n_t):
        name = rs(); nd = r("I"); ne = [r("Q") for _ in range(nd)]; t = r("I"); r("Q")
        if re.search(pat, name):
            tn, be, bb = T.get(t, (str(t), 0, 0))
            n = 1
            for x in ne: n *= x
            nb = n // be * bb if be else -1
            print(f"{name} {tn} ne={ne} bytes={nb} ({nb/2**30:.3f} GiB)")
main(sys.argv[1], sys.argv[2])
