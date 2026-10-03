"""Persist real rejection hashes and measured process wall time after the CPU runner."""
import hashlib
import json
import re
import struct
from pathlib import Path

root = Path(__file__).resolve().parent
results = []
for k in (3, 5):
    for seed in (1729, 2718):
        expected = None
        for arm in ("up", "patch"):
            path = root / f"input-results/server-{k}-{seed}-{arm}.bin"
            data = path.read_bytes()
            if expected is None:
                expected = data
            assert data == expected, (k, seed, arm, "rejection token/logit mismatch")
            count, = struct.unpack_from("<Q", data)
            assert count == 48
            log = path.with_suffix(".log").read_text()
            wall = float(re.search(r"M: CPU wall seconds=([0-9.]+)", log).group(1))
            results.append({"k": k, "seed": seed, "arm": arm, "tokens_M": count,
                            "bytes_M": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                            "process_wall_seconds_M": wall})
(root / "server-input-results.json").write_text(json.dumps(results, indent=2) + "\n")
print("M: real CPU rejection token/final-logit identity; recorded server hashes")
