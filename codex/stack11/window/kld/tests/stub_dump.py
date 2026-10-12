"""CPU-only distribution fixture; never loads a model or shared GPU library."""
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import tokens
from kldmath import records_from_logits


def make(path, seqs, tail, name, full_path=None):
    vocab, k = 248320, 64
    logits = np.full(vocab, -18.0, dtype=np.float64)
    logits[:64] = np.linspace(0, -8, 64)
    if name in ("cand", "cand_repeat", "f16"):
        logits[0] += {"cand": 0.001, "cand_repeat": 0.001, "f16": 0.05}[name]
    lp = (logits - np.logaddexp.reduce(logits)).astype("<f4")
    with open(path, "wb") as f:
        f.write(b"KLDO" + struct.pack("<3i", k, vocab, len(seqs)))
        for ts in seqs:
            # All fixture rows share one distribution; avoid a sequence-sized dense matrix.
            rec = np.repeat(records_from_logits(logits[None, :], ts[1:2], k), len(ts)-1)
            rec["lp_next"] = lp[ts[1:]]
            f.write(struct.pack("<i", len(rec)))
            f.write(rec.tobytes())
    if full_path:
        with open(full_path, "wb") as f:
            f.write(b"KLF1" + struct.pack("<2i", vocab, len(seqs)))
            for ts in seqs:
                start = max(0, len(ts)-tail)
                f.write(struct.pack("<2i", start, len(ts)-1-start))
                for _ in range(len(ts)-1-start):
                    f.write(lp.tobytes())


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    if os.environ.get("STUB_SLEEP"):
        time.sleep(float(os.environ["STUB_SLEEP"]))
    if os.environ.get("STUB_FAIL"):
        sys.exit(int(os.environ["STUB_FAIL"]))
    # Verify the real invocation contract rather than accepting arbitrary argv.
    for flag, value in (("-sm", "tensor"), ("-ts", "1,1"), ("-fa", "on"), ("-t", "8"), ("-tb", "8")):
        assert sys.argv[sys.argv.index(flag)+1] == value
    assert os.environ["KLD_DECODE_TAIL_BS"] == "6"
    make(os.environ["KLD_OUT"], tokens(os.environ["KLD_TOKENS"]),
         int(os.environ["KLD_DECODE_TAIL"]), os.environ["STUB_ARM"], os.environ["KLD_TAIL_WRITE"])
