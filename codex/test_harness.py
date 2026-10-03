#!/usr/bin/env python3
import contextlib
import io
import json
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
    sampled = {"max_tokens": 300, "metrics": {"drafts": 50, "draft_tokens": 250, "accepted_tokens": 200},
               "response": {"timings": {"draft_n": 250, "draft_n_accepted": 200, "predicted_ms": 1540, "predicted_n": 251}},
               "chunks": [{"timings": {"draft_n": 5 * r, "predicted_ms": 40 + 30 * r, "predicted_n": 1 + 5 * r}} for r in range(1, 51)]}
    measured = h.steady(sampled, 5, 40)
    assert measured["steady_rounds_M"] == 45
    assert measured["steady_ms_per_round_D"] == 30 and measured["generation_excess_ms_D"] == 40
    for field, value in (("drafts", 100), ("draft_tokens", 249), ("accepted_tokens", 199)):
        wrong = copy.deepcopy(sampled)
        wrong["metrics"][field] = value
        rejects(lambda: h.steady(wrong, 5, 40))
    wrong = copy.deepcopy(sampled)
    wrong["chunks"] = wrong["chunks"][:24]
    rejects(lambda: h.steady(wrong, 5, 40))
    wrong = copy.deepcopy(sampled)
    wrong["chunks"][20]["timings"]["predicted_ms"] = 1
    rejects(lambda: h.steady(wrong, 5, 40))
    wrong = copy.deepcopy(sampled)
    wrong["chunks"] = wrong["chunks"][::2]
    rejects(lambda: h.steady(wrong, 5, 40))
    real = json.loads((Path(__file__).parent / "fixtures/swift-long-tool-s1.json").read_text())
    measured = h.steady(real, 5, 1)
    assert measured["steady_rounds_M"] == 24
    assert abs(measured["steady_ms_per_round_D"] - 37.19825) < 1e-9
    try:
        h.steady(real, 5, 40)
    except ValueError as error:
        assert str(error) == "insufficient steady rounds: 24 < 40"
    else:
        raise AssertionError("short real request passed the steady gate")
    wrong = copy.deepcopy(real)
    wrong["chunks"].insert(20, {"timings": {"predicted_n": 1, "predicted_ms": 0}})
    wrong["chunks"].insert(40, {"timings": {"draft_n": None, "predicted_ms": 0}})
    assert h.steady(wrong, 5, 1) == measured
    for field, value in (("draft_n", 0), ("predicted_ms", 0), ("draft_n", 155), ("predicted_ms", 1201)):
        wrong = copy.deepcopy(real)
        wrong["chunks"][20]["timings"][field] = value
        rejects(lambda: h.steady(wrong, 5, 1))
    chunks = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "random-id", "function": {"name": "apply_patch", "arguments": "ab"}}]}}]},
              {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "cd"}}]}, "finish_reason": "length"}],
               "timings": sampled["response"]["timings"]}]
    assert h.completion(chunks)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == "abcd"
    rejects(lambda: h.completion(chunks[:1]))
    rejects(lambda: h.completion([{"error": "broken response"}]))
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        fake = SimpleNamespace(args=SimpleNamespace(k=5, rounds=40, short_requests=8))
        calls = []
        def request(dest, case, number, body=None, stream=False):
            calls.append(copy.deepcopy(body))
            record = copy.deepcopy(sampled)
            record["response"].update(choices=[{"message": {"role": "assistant", "content": "fixture"}, "finish_reason": "length"}])
            record["response"]["timings"].update(cache_n=0, prompt_ms=100)
            record.update(begin_ns=0, end_ns=2000000000)
            return record
        fake.request = request
        rejects(lambda: h.suite(fake, Path(directory)))
        assert len(calls) == 7 and calls[-1]["cache_prompt"]
        def cached_request(*args, **kwargs):
            record = request(*args, **kwargs)
            if args[1].startswith("short-"):
                record["response"]["timings"]["cache_n"] = 100
            return record
        fake.request = cached_request
        result = h.suite(fake, Path(directory))
        assert len(result) == 13
        assert len(calls[-1]["messages"]) > len(calls[-8]["messages"])
        variants = [copy.deepcopy(result) for _ in range(4)]
        for variant in variants[1:3]:
            for record in variant.values():
                record["response"]["timings"]["predicted_ms"] -= 40
                record["end_ns"] -= 40000000
                if "steady" in record:
                    record["steady"]["steady_ms_per_round_D"] -= 1
                    record["steady"]["generation_excess_ms_D"] += 10
        report = h.summarize(variants, 5)
        assert report["paired"][1]["patch_minus_up_D"]["steady_ms_per_round_D"] == -1
        assert report["paired"][0]["patch_minus_up_D"]["generation_excess_ms_D"] == 10
        assert report["paired"][-1]["patch_minus_up_D"]["generation_excess_ms_D"] == 10
        assert report["paired"][-1]["patch_minus_up_D"]["request_fixed_ms_estimate_D"] == 10
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
    controls = 15
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
