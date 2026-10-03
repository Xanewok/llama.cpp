"""CPU-only replay of real GPU artifacts and tandem contract failure controls."""
import contextlib
import copy
import io
from functools import lru_cache
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import urllib.error
from unittest.mock import patch

import gpu_harness as h
from test_harness import rejects

FIXTURES = Path("/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures")


@lru_cache(maxsize=4)
def recorded(k, canary=False):
    run = "20261003-173851" if k == 5 else "20261003-174710"
    root = h.ROOT / "codex/runs" / run / "1-up"
    cases = ["cold-tool", *[f"long-{case}-s{seed}" for seed in ((1,) if canary else (1, 2)) for case in ("tool", "think")],
             *[f"short-tool-{n}" for n in range(1, 3 if canary else 9)]]
    return {case: json.loads((root / f"{case}-1.json").read_text()) for case in cases}


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    observed = json.loads((FIXTURES / "llama-sse/long-tool-s1-1.json").read_text())
    sample = h.steady(observed, 5, 1)
    assert sample["steady_rounds_M"] == 24
    rejects(lambda: h.steady(observed, 5, 40))
    changed = copy.deepcopy(observed)
    changed["chunks"].extend([{}, {"timings": None}, {"timings": {"draft_n": None}}, {"timings": {"draft_n": 150}}])
    # Unknown rows and usage-only finals do not erase the last complete timing.
    changed["chunks"][1:1] = [{"timings": {"draft_n": None}}, {"timings": {"draft_n": 5}}]
    assert h.steady(changed, 5, 1) == sample
    assert h.completion(observed["chunks"])["timings"] == observed["response"]["timings"]
    final = copy.deepcopy(observed["chunks"])
    final.extend([{"timings": {"draft_n": None}}, {"choices": [], "usage": {"completion_tokens": 2048}}])
    assert h.completion(final)["timings"] == observed["response"]["timings"]
    rejects(lambda: h.completion(observed["chunks"][:1]))
    with tempfile.TemporaryDirectory(dir=h.ROOT / "codex") as directory:
        dest = Path(directory)
        assert "long-tool-s1" in h.validate_boot({"long-tool-s1": observed}, dest, 5, 40)
        cold = json.loads((FIXTURES / "llama-sse/cold-tool-1.json").read_text())
        assert h.identity(cold) == h.identity(copy.deepcopy(cold))
        for k in (5, 3):
            result = recorded(k)
            assert not h.validate_boot(result, dest, k, 40)
            boots = [copy.deepcopy(result) for _ in h.TANDEM_ORDER]
            for arm, boot in zip(h.TANDEM_ORDER, boots):
                delta = {"up": 0, "graph-prefill": 1, "graph-tandem": 2}[arm]
                for record in boot.values():
                    record["end_ns"] -= delta * 1_000_000
                    record["response"]["timings"]["prompt_ms"] -= delta
                    if "steady" in record:
                        record["steady"]["steady_ms_per_round_D"] -= delta
            report = h.summarize(boots, k, h.TANDEM_ORDER)
            assert len(report["paired"]) == 26 and len(report["request_classes"]) == 12
            rows = [row for row in report["paired"] if row["case"] == "long-tool-s1"]
            assert rows[0]["patch_minus_up_D"]["steady_ms_per_round_D"] == -1
            assert rows[1]["patch_minus_up_D"]["steady_ms_per_round_D"] == -2
            assert all(len(row["pair_differences_D"]) == 2 for row in report["paired"])
            wrong = copy.deepcopy(result)
            wrong["short-tool-2"]["response"]["timings"]["cache_n"] = 0
            assert "short-tool-2" in h.validate_boot(wrong, dest, k, 40)
            wrong["cold-tool"]["metrics"]["accepted_tokens"] += 1
            assert h.identity(wrong["cold-tool"]) != h.identity(result["cold-tool"])
        request = h.Harness(SimpleNamespace(budget=600, k=3, port=8398), dest)
        body = {"max_tokens": 768, "messages": []}
        for stream in (False, True):
            error = urllib.error.HTTPError("fixture", 500, "tool parse failure", {}, io.BytesIO(b"Failed to parse tool call arguments"))
            with patch.object(request, "metrics", return_value={}), patch.object(request, "http", side_effect=error), \
                    patch("urllib.request.urlopen", side_effect=error):
                response = request.request(dest, f"error-{stream}", 1, body=body, stream=stream)
            assert response["error"]["status"] == 500 and response["error"]["body"] == "Failed to parse tool call arguments"
            assert (dest / f"error-{stream}-1-error.json").exists()
        metrics = "\n".join(f"llamacpp:spec_decode_num_{key}_total {value}" for key, value in
                            (("drafts", cold["metrics"]["drafts"]), ("draft_tokens", cold["metrics"]["draft_tokens"]),
                             ("accepted_tokens", cold["metrics"]["accepted_tokens"])))
        with patch.object(request, "http", return_value=metrics):
            assert request.metrics() == cold["metrics"]
        with patch.object(request, "http", return_value=metrics.replace("accepted_tokens", "unknown")):
            rejects(request.metrics)
        # A failing request still records the rest of the independent requests before validation.
        fake = SimpleNamespace(args=SimpleNamespace(canary=True, short_requests=8))
        calls = []
        def failed_request(dest, case, number, body=None, stream=False):
            calls.append(case)
            return {"error": {"message": "saved HTTP failure"}} if case == "long-tool-s1" else recorded(3, True)[case]
        fake.request = failed_request
        result = h.suite(fake, dest)
        assert len(calls) == 5 and "long-tool-s1" in h.validate_boot(result, dest, 3, 40)
    # Full canary and full-run control flow, using decisive real requests at both K values.
    for canary in (True, False):
        with tempfile.TemporaryDirectory(dir=h.ROOT / "codex") as directory:
            out = Path(directory) / "run"
            args = SimpleNamespace(k=3, canary=canary, short_requests=8)
            fake = SimpleNamespace(args=args, deadline=h.time.monotonic() + 3600)
            boots, xids, stops = [], [], []
            def boot(root, label):
                args.k = int(label.split("-")[0][1:])
                boots.append(label)
                dest = out / label
                dest.mkdir()
                return dest
            fake.boot = boot
            fake.stop = lambda: stops.append(True)
            fake.xid = lambda: xids.append(True)
            fake.timeout = lambda **kwargs: 1
            fake.request = lambda dest, case, number, **kwargs: copy.deepcopy(recorded(args.k, canary)[case])
            argv = ["gpu_harness", "tandem", "--execute", "--out", str(out)] + (["--canary"] if canary else [])
            with patch.dict(os.environ, {"RUN_GPU": "1", "CUDA_VISIBLE_DEVICES": "fixture-only"}), \
                    patch.object(sys, "argv", argv), patch.object(h, "Harness", return_value=fake), \
                    patch.object(h, "frozen", return_value={}), patch.object(h, "sha", return_value=h.MODEL_SHA), \
                    patch.object(h.signal, "signal"), patch.object(h.subprocess, "run", side_effect=AssertionError("subprocess")), \
                    patch.object(h.subprocess, "Popen", side_effect=AssertionError("launch")), \
                    patch.object(h.socket, "socket", side_effect=AssertionError("network")), contextlib.redirect_stdout(io.StringIO()):
                h.main()
            assert len(boots) == (3 if canary else 12) and len(xids) == len(boots)
            assert json.loads((out / "verdict.json").read_text())["verdict"] == "PASS"
            report = json.loads((out / ("canary-summary.json" if canary else "ab-summary.json")).read_text())
            assert len(report) == (1 if canary else 2)
    for broken in ("identity", "steady", "cache", "HTTP"):
        with tempfile.TemporaryDirectory(dir=h.ROOT / "codex") as directory:
            out = Path(directory) / "negative"
            args = SimpleNamespace(k=3, canary=True, short_requests=8)
            fake = SimpleNamespace(args=args)
            labels = []
            def boot(root, label):
                labels.append(label)
                dest = out / label
                dest.mkdir()
                return dest
            def request(dest, case, number, **kwargs):
                record = copy.deepcopy(recorded(3, True)[case])
                if "graph-tandem" in labels[-1]:
                    if broken == "identity" and case == "cold-tool":
                        record["metrics"]["accepted_tokens"] += 1
                    elif broken == "steady" and case == "long-tool-s1":
                        record["chunks"] = record["chunks"][:20]
                    elif broken == "cache" and case == "short-tool-2":
                        record["response"]["timings"]["cache_n"] = 0
                    elif broken == "HTTP" and case == "long-tool-s1":
                        record = {"error": {"message": "saved HTTP 500 body"}}
                return record
            fake.boot, fake.request, fake.stop, fake.xid = boot, request, lambda: None, lambda: None
            argv = ["gpu_harness", "tandem", "--execute", "--canary", "--out", str(out)]
            with patch.dict(os.environ, {"RUN_GPU": "1", "CUDA_VISIBLE_DEVICES": "fixture-only"}), \
                    patch.object(sys, "argv", argv), patch.object(h, "Harness", return_value=fake), \
                    patch.object(h, "frozen", return_value={}), patch.object(h, "sha", return_value=h.MODEL_SHA), \
                    patch.object(h.signal, "signal"), patch.object(h.subprocess, "run", side_effect=AssertionError("subprocess")), \
                    patch.object(h.subprocess, "Popen", side_effect=AssertionError("launch")), \
                    patch.object(h.socket, "socket", side_effect=AssertionError("network")), contextlib.redirect_stdout(io.StringIO()):
                rejects(h.main)
            assert json.loads((out / "verdict.json").read_text())["verdict"] == "INVALID"
            assert not (out / "canary-summary.json").exists()
            assert len(json.loads((out / "k3-ab-results.json").read_text())) == 3
    # PID reuse must not let cleanup signal a different process.
    for start, signals in (("123", [h.signal.SIGTERM]), ("999", [])):
        process = h.Harness.__new__(h.Harness)
        process.pid, process.binary, process.child, process.session = (789, "123"), None, None, None
        stats = iter(["789 (round-trace) " + " ".join(["S", *["0"] * 18, start]),
                      "789 (round-trace) " + " ".join(["Z", *["0"] * 18, start]),
                      "789 (round-trace) " + " ".join(["Z", *["0"] * 18, start])])
        with patch.object(h.Path, "read_text", side_effect=lambda *args, **kwargs: next(stats)), \
                patch.object(h.os, "kill") as kill, patch.object(h.time, "sleep"):
            process.stop()
        assert [call.args[1] for call in kill.call_args_list] == signals
    # A model/port guard and deadline failure are exercised without launching any process.
    for argv in (["gpu_harness", "tandem", "--canary", "--budget", "601"],
                 ["gpu_harness", "tandem", "--canary", "--k", "both"],
                 ["gpu_harness", "tandem", "--budget", "1000"]):
        with patch.object(sys, "argv", argv):
            rejects(h.main)
    # Idle memory from the real recorded device fixture, then a simulated delayed release.
    guard = h.Harness.__new__(h.Harness)
    guard.env = {"CUDA_VISIBLE_DEVICES": "fixture-only"}
    guard.args = SimpleNamespace(port=8398)
    guard.xid = lambda: None
    guard.timeout = lambda: 1
    memory = [int(line.split(",")[2].strip().split()[0]) for line in (FIXTURES / "nvidia-smi-query.csv").read_text().splitlines()[1:]]
    polls = iter(["16000, 0\n16000, 0\n", "".join(f"{n}, 0\n" for n in memory)])
    guard.run = lambda *args, **kwargs: SimpleNamespace(stdout=next(polls))
    with patch.object(h.socket, "socket") as mocked, patch.object(h.time, "sleep"):
        guard.idle()
        mocked.return_value.__enter__.return_value.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    guard.env = {"XID_SINCE": "fixture-only"}
    guard.run = lambda *args, **kwargs: SimpleNamespace(stdout=(FIXTURES / "xid-lines.txt").read_text())
    rejects(lambda: h.Harness.xid(guard))
    guard.deadline = h.time.monotonic() - 1
    try:
        h.Harness.timeout(guard)
    except TimeoutError as error:
        assert "budget" in str(error)
    else:
        raise AssertionError("expired budget passed")
    assert h.stage_estimate(True, 8)["boot_total_s_G"] * 3 <= .7 * 600
    assert h.stage_estimate(False, 8)["boot_total_s_G"] * 12 <= .7 * 3600
    print("M: real llama-sse/ab3 replays, three-arm summary, all-arm canary, 12-boot K=5/K=3 flow, HTTP error persistence, partial SSE, identity/cache/size/budget/Xid/idle controls PASS")


if __name__ == "__main__":
    main()
