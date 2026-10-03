#!/usr/bin/env python3
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import copy
from types import SimpleNamespace
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
    with patch.object(sys, "argv", ["gpu_harness", "ab", "--budget", "2401"]):
        rejects(h.main)
    reference = {"response": {"choices": [{"message": {"role": "assistant", "content": "same"}, "finish_reason": "stop"}]},
                 "metrics": {"drafts": 4, "draft_tokens": 20, "accepted_tokens": 16}}
    wrong = copy.deepcopy(reference)
    wrong["response"]["choices"][0]["message"]["content"] = "different"
    assert h.identity(wrong) != h.identity(reference)
    wrong = copy.deepcopy(reference)
    wrong["metrics"]["accepted_tokens"] += 1
    assert h.identity(wrong) != h.identity(reference)
    guard = h.Harness.__new__(h.Harness)
    guard.env = {"XID_SINCE": "2026-10-03 00:00:00"}
    calls = []
    guard.run = lambda cmd, **kwargs: (calls.append(cmd) or SimpleNamespace(stdout="-- No entries --"))
    guard.xid()
    assert calls[0][calls[0].index("--since") + 1] == guard.env["XID_SINCE"]
    guard.run = lambda *args, **kwargs: SimpleNamespace(stdout="NVRM: Xid 31")
    rejects(guard.xid)
    guard.env = {}
    rejects(guard.xid)
    controls = 6
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        root = Path(directory)
        (root / "bin").mkdir()
        (root / "SOURCE").write_text("a" * 40 + "\n")
        binary = root / "bin/round-trace"
        binary.write_bytes(b"not executable\n")
        digest = h.sha(binary)
        (root / "bin/SHA256SUMS").write_text(digest + "  round-trace\n")
        (root / "SHA256SUMS").write_text(h.sha(root / "SOURCE") + "  SOURCE\n" + h.sha(root / "bin/SHA256SUMS") + "  bin/SHA256SUMS\n")
        h.frozen(root)
        out = root / "boots"
        out.mkdir()
        boot = h.Harness(SimpleNamespace(budget=2400, k=5, port=8398), out)
        with patch.object(boot, "idle"), patch.object(h.subprocess, "Popen", side_effect=AssertionError("test launch blocked")):
            try:
                boot.boot(root, "snapshot")
            except AssertionError as error:
                assert str(error) == "test launch blocked"
            else:
                raise AssertionError("mocked boot reached a launch")
        h.frozen(out / "snapshot")
        controls += 1
        binary.write_bytes(b"changed\n")
        rejects(lambda: h.frozen(root)); controls += 1
        binary.write_bytes(b"not executable\n")
        extra = root / "bin/unmanifested"
        extra.write_text("extra")
        rejects(lambda: h.frozen(root)); controls += 1
        extra.unlink()
        (root / "SOURCE").write_text("b" * 40 + "\n")
        rejects(lambda: h.frozen(root)); controls += 1
        (root / "SOURCE").write_text("wrong source\n")
        rejects(lambda: h.frozen(root)); controls += 1
        (root / "SOURCE").write_text("a" * 40 + "\n")
        binary.unlink()
        binary.symlink_to(Path(__file__).resolve())
        rejects(lambda: h.frozen(root)); controls += 1
    print(f"M: trace/AB plan has no subprocess/network calls; {controls} execution/frozen/budget/identity/Xid controls PASS")


def execute_without_gate():
    with patch.dict(os.environ, {"RUN_GPU": "0", "CUDA_VISIBLE_DEVICES": ""}), \
            patch.object(sys, "argv", ["gpu_harness", "trace", "--execute"]), \
            patch.object(h.subprocess, "run", side_effect=AssertionError("gated subprocess")), \
            patch.object(h.subprocess, "Popen", side_effect=AssertionError("gated launch")), contextlib.redirect_stdout(io.StringIO()):
        h.main()


if __name__ == "__main__":
    main()
