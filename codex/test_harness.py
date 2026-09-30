#!/usr/bin/env python3
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import gpu_harness as h


def rejects(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("broken frozen snapshot was accepted")


def main():
    for mode in ("trace", "ab"):
        with patch.object(sys, "argv", ["gpu_harness", mode]), patch.object(h.subprocess, "run", side_effect=AssertionError("plan subprocess")), \
                patch.object(h.subprocess, "Popen", side_effect=AssertionError("plan launch")), \
                patch.object(h.socket, "socket", side_effect=AssertionError("plan network")), contextlib.redirect_stdout(io.StringIO()):
            h.main()
    rejects(lambda: execute_without_gate())
    controls = 1
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        root = Path(directory)
        (root / "bin").mkdir()
        (root / "SOURCE").write_text("a" * 40 + "\n")
        binary = root / "bin/round-trace"
        binary.write_bytes(b"not executable\n")
        digest = h.sha(binary)
        (root / "bin/SHA256SUMS").write_text(digest + "  round-trace\n")
        h.frozen(root)
        binary.write_bytes(b"changed\n")
        rejects(lambda: h.frozen(root)); controls += 1
        binary.write_bytes(b"not executable\n")
        extra = root / "bin/unmanifested"
        extra.write_text("extra")
        rejects(lambda: h.frozen(root)); controls += 1
        extra.unlink()
        (root / "SOURCE").write_text("wrong source\n")
        rejects(lambda: h.frozen(root)); controls += 1
        (root / "SOURCE").write_text("a" * 40 + "\n")
        binary.unlink()
        binary.symlink_to(Path(__file__).resolve())
        rejects(lambda: h.frozen(root)); controls += 1
    print(f"M: trace/AB plan has no subprocess/network calls; {controls} execution/frozen controls PASS")


def execute_without_gate():
    with patch.dict(os.environ, {"RUN_GPU": "0", "CUDA_VISIBLE_DEVICES": ""}), \
            patch.object(sys, "argv", ["gpu_harness", "trace", "--execute"]), contextlib.redirect_stdout(io.StringIO()):
        h.main()


if __name__ == "__main__":
    main()
