"""Compare two reuse-server-cpu outputs: token IDs equal? final-logit max |diff|, argmax equal?"""
import struct, sys
import numpy as np
def load(p):
    d = open(p, "rb").read(); n = struct.unpack("<Q", d[:8])[0]
    t = np.frombuffer(d[8:8 + 4 * n], dtype=np.int32); l = np.frombuffer(d[8 + 4 * n:], dtype=np.float32)
    return t, l
a, b = load(sys.argv[1]), load(sys.argv[2])
print(f"tokens_equal={np.array_equal(a[0], b[0])} n={len(a[0])} logits_bitwise={np.array_equal(a[1], b[1])} "
      f"max_abs={np.abs(a[1] - b[1]).max():.3g} argmax_equal={a[1].argmax() == b[1].argmax()} max_logit={np.abs(a[1]).max():.3g}")
