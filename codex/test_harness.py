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
    with patch.object(sys, "argv", ["gpu_harness", "ab", "--canary", "--budget", "600"]), \
            patch.object(h.subprocess, "run", side_effect=AssertionError("canary subprocess")), \
            patch.object(h.subprocess, "Popen", side_effect=AssertionError("canary launch")), \
            patch.object(h.socket, "socket", side_effect=AssertionError("canary network")), contextlib.redirect_stdout(io.StringIO()) as output:
        h.main()
    plan = json.loads(output.getvalue())
    assert plan["canary"] and plan["boots"] == ["patch"] and plan["budget_s_G"] == 600
    with patch.object(sys, "argv", ["gpu_harness", "ab", "--canary", "--budget", "601"]):
        rejects(h.main)
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
    for k, run in ((5, "20261003-173851"), (3, "20261003-174710")):
        source = h.ROOT / "codex/runs" / run / "2-patch"
        actual = {case: json.loads((source / name).read_text()) for case, name in (
            ("cold-tool", "cold-tool-1.json"), ("long-tool-s1", "long-tool-s1-1.json"),
            ("long-think-s1", "long-think-s1-1.json"), ("short-tool-1", "short-tool-1-1.json"),
            ("short-tool-2", "short-tool-2-1.json"))}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            assert not h.validate_boot(actual, Path(directory), k, 40)
            assert not h.canary_identity(actual, k)
            broken = copy.deepcopy(actual)
            broken["cold-tool"]["metrics"]["accepted_tokens"] += 1
            assert h.canary_identity(broken, k) == ["cold-tool"]
            for record in actual.values():
                assert h.identity(record) == h.identity(copy.deepcopy(record))
                if record.get("chunks"):
                    parsed = h.completion(record["chunks"])
                    assert parsed["timings"] == record["response"]["timings"]
            wrong = copy.deepcopy(actual)
            wrong["short-tool-2"]["response"]["timings"]["cache_n"] = 0
            assert "short-tool-2" in h.validate_boot(wrong, Path(directory), k, 40)
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
    for case, thinking in (("tool", False), ("think", True)):
        body = json.loads((h.INPUTS / f"long-{case}.json").read_text())
        assert body["chat_template_kwargs"]["enable_thinking"] == thinking
        assert body["max_tokens"] >= 2048 and body["tool_choice"] == "auto" and not body["cache_prompt"]
        assert len(body["messages"]) == 2 and len(body["tools"]) == 1
        tool = body["tools"][0]["function"]
        assert tool["name"] == "write_file" and tool["parameters"]["required"] == ["content"]
        prompt = body["messages"][-1]["content"]
        literal = prompt.split("BEGIN LITERAL FILE\n", 1)[1].split("\nEND LITERAL FILE", 1)[0]
        lines = literal.splitlines()
        assert len(lines) == 256 and len(literal.encode()) == 34047
        for number, line in enumerate(lines, 1):
            assert line == f"{number:04d}: This is numbered record {number:04d} in the benchmark fixture. Preserve this complete sentence and its distinct record number exactly."
        assert "visible content argument" in prompt and "not reasoning" in prompt
        if thinking:
            assert body["chat_template_kwargs"]["reasoning_effort"] == "low"
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
            if case.startswith("long-"):
                record["max_tokens"] = record["response"]["timings"]["predicted_n"] = 2048
                record["response"]["choices"][0]["message"]["tool_calls"] = [
                    {"type": "function", "function": {"name": "write_file", "arguments": '{"content":"0001: fixture'}}]
            record.update(begin_ns=0, end_ns=2000000000)
            return record
        fake.request = request
        result = h.suite(fake, Path(directory))
        failures = h.validate_boot(result, Path(directory), 5, 40)
        assert len(calls) == 13 and calls[-1]["cache_prompt"]
        assert set(failures) == {f"short-tool-{number}" for number in range(2, 9)}
        def cached_request(*args, **kwargs):
            record = request(*args, **kwargs)
            if args[1].startswith("short-"):
                record["response"]["timings"]["cache_n"] = 100
            return record
        fake.request = cached_request
        result = h.suite(fake, Path(directory))
        assert not h.validate_boot(result, Path(directory), 5, 40)
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
        invalid = copy.deepcopy(result)
        invalid["long-think-s1"]["response"]["choices"][0]["message"].pop("tool_calls")
        invalid["long-think-s2"]["response"]["timings"]["predicted_n"] = 1024
        failures = h.validate_boot(invalid, Path(directory), 5, 40)
        assert set(failures) == {"long-think-s1", "long-think-s2"}
        def short_request(*args, **kwargs):
            record = cached_request(*args, **kwargs)
            if args[1] == "long-tool-s1":
                record = copy.deepcopy(real)
                record.update(begin_ns=0, end_ns=2000000000)
            return record
        fake.request = short_request
        out = Path(directory) / "run"
        boots, stopped, xids = [], [], []
        def boot(root, label, trace=False):
            boots.append(label)
            dest = out / label
            dest.mkdir()
            return dest
        fake.boot, fake.stop, fake.xid = boot, lambda: stopped.append(True), lambda: xids.append(True)
        fake.capturing = False
        with patch.dict(os.environ, {"RUN_GPU": "1", "CUDA_VISIBLE_DEVICES": "fixture-only"}), \
                patch.object(sys, "argv", ["gpu_harness", "ab", "--execute", "--out", str(out)]), \
                patch.object(h, "Harness", return_value=fake), patch.object(h, "frozen"), \
                patch.object(h, "sha", return_value=h.MODEL_SHA), patch.object(h.signal, "signal"), \
                patch.object(h.subprocess, "run", side_effect=AssertionError("test subprocess")), \
                patch.object(h.subprocess, "Popen", side_effect=AssertionError("test launch")), \
                patch.object(h.socket, "socket", side_effect=AssertionError("test network")), contextlib.redirect_stdout(io.StringIO()):
            try:
                h.main()
            except ValueError as error:
                assert "boot validation failed after writing results" in str(error)
            else:
                raise AssertionError("short boot passed")
        saved = json.loads((out / "ab-results.json").read_text())
        validation = json.loads((out / "1-up/boot-validation.json").read_text())
        assert len(saved) == 1 and len(saved[0]) == 13 and saved[0]["long-tool-s1"]["response"] == real["response"]
        assert not validation["passed"] and set(validation["failures"]) == {"long-tool-s1"}
        assert all((out / f"1-up/long-{case}-s{seed}-steady.json").is_file() for case in ("tool", "think") for seed in (1, 2))
        assert boots == ["1-up"] and len(stopped) == 2 and len(xids) == 1
        # Replay the decisive GPU captures through the complete canary path.
        out = Path(directory) / "canary"
        boots.clear(); stopped.clear(); xids.clear()
        fake.args.canary = True
        fake.deadline = h.time.monotonic() + 600
        def actual_request(dest, case, number, body=None, stream=False):
            names = {"cold-tool": "cold-tool-1.json", "long-tool-s1": "long-tool-s1-1.json",
                     "long-think-s1": "long-think-s1-1.json", "short-tool-1": "short-tool-1-1.json",
                     "short-tool-2": "short-tool-2-1.json"}
            return json.loads((h.ROOT / "codex/runs/20261003-173851/2-patch" / names[case]).read_text())
        fake.request = actual_request
        with patch.dict(os.environ, {"RUN_GPU": "1", "CUDA_VISIBLE_DEVICES": "fixture-only"}), \
                patch.object(sys, "argv", ["gpu_harness", "ab", "--execute", "--canary", "--budget", "600", "--out", str(out)]), \
                patch.object(h, "Harness", return_value=fake), patch.object(h, "frozen"), \
                patch.object(h, "sha", return_value=h.MODEL_SHA), patch.object(h.signal, "signal"), \
                patch.object(h.subprocess, "run", side_effect=AssertionError("canary test subprocess")), \
                patch.object(h.subprocess, "Popen", side_effect=AssertionError("canary test launch")), \
                patch.object(h.socket, "socket", side_effect=AssertionError("canary test network")), contextlib.redirect_stdout(io.StringIO()):
            h.main()
        validation = json.loads((out / "canary-validation.json").read_text())
        assert validation["passed"] and len(validation["cases"]) == 5
        assert boots == ["1-patch"] and len(xids) == 1
        assert not (out / "ab-summary.json").exists()
    guard = h.Harness.__new__(h.Harness)
    guard.env = {"XID_SINCE": "2026-10-03 00:00:00"}
    calls = []
    guard.run = lambda cmd, **kwargs: (calls.append(cmd) or SimpleNamespace(stdout="-- No entries --"))
    guard.xid()
    assert calls[0][calls[0].index("--since") + 1] == guard.env["XID_SINCE"]
    guard.run = lambda *args, **kwargs: SimpleNamespace(stdout="NVRM: Xid 31")
    rejects(guard.xid)
    real_xids = Path("/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/xid-lines.txt").read_text()
    guard.run = lambda *args, **kwargs: SimpleNamespace(stdout=real_xids)
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
