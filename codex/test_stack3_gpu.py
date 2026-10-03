"""Parser/gate replay on recorded Swift GPU artifacts; no launches or sockets."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import stack3_gpu as s

FIXTURES = Path("/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures")


def reject(fn, message):
    try:
        fn()
    except (ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
        return
    raise AssertionError("negative control escaped: " + message)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    short = json.loads((FIXTURES / "llama-sse/long-tool-s1-1.json").read_text())
    assert s.h.steady(short, 5, 1)["steady_rounds_M"] == 24
    reject(lambda: s.h.steady(short, 5, 40), "insufficient steady")
    real = json.loads(Path("/home/xanewok/repos/graph-reuse/codex/runs/graph-reuse-tandem/full-a2/k5-1-up/long-tool-s1-1.json").read_text())
    parsed = s.h.completion(real["chunks"])
    assert parsed["timings"] == real["response"]["timings"]
    assert s.output_bytes(dict(real, response=parsed)) == s.output_bytes(real)
    timing = s.h.steady(real, 5, 40)
    assert timing["steady_rounds_M"] >= 40
    changed = copy.deepcopy(real)
    changed["response"]["choices"][0]["message"]["content"] += "broken"
    reject(lambda: s.same_outputs({"case": real}, {"case": changed}), "identity failed")
    changed = copy.deepcopy(real)
    changed["chunks"] = changed["chunks"][:5]
    reject(lambda: s.h.steady(changed, 5, 40), "insufficient steady")
    changed = copy.deepcopy(real)
    changed["metrics"]["accepted_tokens"] += 1
    reject(lambda: s.h.steady(changed, 5, 40), "draft counters")
    changed = copy.deepcopy(real)
    changed["chunks"].insert(2, {"timings": {"draft_n": None}})
    assert s.h.steady(changed, 5, 40) == timing
    changed = copy.deepcopy(real)
    changed["chunks"].append({"choices": [{"delta": {}, "finish_reason": None}]})
    assert s.h.completion(changed["chunks"])["timings"] == parsed["timings"]
    reject(lambda: s.h.completion(real["chunks"][:5]), "missing final")
    memory = (FIXTURES / "nvidia-smi-query.csv").read_text()
    assert [row[2] for row in s.h.gpu_memory(memory)] == [108, 15]
    reject(lambda: s.h.gpu_memory(memory.replace("108 MiB", "20000 MiB")), "memory bounds")
    reject(lambda: s.h.gpu_memory(memory.replace("108 MiB", "unknown")), "memory values")
    reject(lambda: s.h.gpu_memory(memory.splitlines()[1]), "not a pair")
    with tempfile.TemporaryDirectory(dir=s.ROOT / "build-tmp") as directory:
        harness = s.h.Harness(SimpleNamespace(budget=960, port=8403), Path(directory))
        harness.env["XID_SINCE"] = "fixture-cutoff"
        harness.env["CUDA_VISIBLE_DEVICES"] = "mock-pair"
        busy_memory = memory.replace("108 MiB", "16000 MiB")
        with patch.object(harness, "xid"), patch.object(harness, "run", side_effect=[SimpleNamespace(stdout=busy_memory), SimpleNamespace(stdout=memory)]), \
             patch.object(s.h.socket, "socket") as sock, patch.object(s.h.time, "sleep") as pause:
            harness.idle()
            assert pause.call_count == 1
            assert sock.return_value.__enter__.return_value.bind.called
        def run_result(text): return SimpleNamespace(stdout=text)
        with patch.object(harness, "run", return_value=run_result((FIXTURES / "xid-lines.txt").read_text())):
            reject(harness.xid, "Xid since")
        with patch.object(harness, "run", return_value=run_result("ordinary kernel message stop_xid_hash_s_G")):
            harness.xid()
        with patch.object(harness, "run", return_value=run_result("")):
            reject(harness.xid, "empty/unreadable")
        metrics = "\n".join("llamacpp:spec_decode_num_" + key + "_total " + str(value) for key, value in real["metrics"].items())
        with patch.object(harness, "http", return_value=metrics):
            assert harness.metrics() == real["metrics"]
        with patch.object(harness, "http", return_value=""):
            reject(harness.metrics, "missing authoritative metric")
        harness.deadline = 0
        try:
            harness.timeout()
        except TimeoutError as error:
            assert "queue wall budget" in str(error)
        else:
            raise AssertionError("budget gate escaped")
    # Replay the real SSE through the new size/overlap gates. The extra fields are test inputs, not GPU measurements.
    record = copy.deepcopy(real)
    record["response"]["timings"].update(prompt_n=32768, cache_n=0)
    record["chunk_ns"] = [i * 10000000 for i in range(len(record["chunks"]))]
    records = {key: copy.deepcopy(record) for key in ("single-t0", "single-t1", "np2-0", "np2-1")}
    for key, value in records.items():
        if key != "single-t1": value["response"]["timings"]["prompt_n"] = 1024
    pair = {key: 2 * value for key, value in real["metrics"].items()}
    assert not s.validate(records, pair, 5)["failures"]
    for kind in ("prompt", "cache", "length", "overlap", "counter", "rate"):
        broken = copy.deepcopy(records)
        counters = dict(pair)
        if kind == "prompt": broken["single-t0"]["response"]["timings"]["prompt_n"] = 100
        if kind == "cache": broken["single-t0"]["response"]["timings"]["cache_n"] = 100
        if kind == "length": broken["single-t0"]["response"]["timings"]["predicted_n"] = 32
        if kind == "overlap": broken["np2-1"]["chunk_ns"] = [t + 10**15 for t in record["chunk_ns"]]
        if kind == "counter": counters["accepted_tokens"] += 1
        if kind == "rate": broken["np2-0"]["response"]["timings"]["predicted_ms"] = 1
        assert s.validate(broken, counters, 5)["failures"], kind
    with patch.object(s, "pins", return_value=[]), patch.object(s.h.subprocess, "Popen", side_effect=AssertionError("launch")), \
         patch.object(s.h.socket, "socket", side_effect=AssertionError("socket")), \
         patch.object(s.sys, "argv", ["stack3_gpu"]), contextlib.redirect_stdout(io.StringIO()) as output:
        s.main()
        assert json.loads(output.getvalue())["boots"] == list(s.ORDER)
    with patch.object(s, "pins", return_value=[]), patch.object(s.sys, "argv", ["stack3_gpu", "--execute"]):
        reject(s.main, "parent gpuq environment")
    with patch.object(s.sys, "argv", ["stack3_gpu", "--canary", "--budget", "100"]):
        reject(s.main, "queue wall budget")
    assert "round-trace" in s.h.frozen(s.ROOT / "codex/frozen/stack2")
    with tempfile.TemporaryDirectory(dir=s.ROOT / "build-tmp") as directory:
        root = Path(directory)
        (root / "bin").mkdir()
        (root / "SOURCE").write_text("0" * 40 + "\n")
        (root / "bin/round-trace").write_bytes(b"never execute")
        (root / "bin/SHA256SUMS").write_text(s.h.sha(root / "bin/round-trace") + "  round-trace\n")
        (root / "SHA256SUMS").write_text("".join(s.h.sha(root / name) + "  " + name + "\n" for name in ("SOURCE", "bin/SHA256SUMS")))
        s.h.frozen(root)
        (root / "bin/round-trace").write_bytes(b"changed")
        reject(lambda: s.h.frozen(root), "frozen bytes changed")
        (root / "bin/round-trace").write_bytes(b"never execute")
        (root / "SOURCE").write_text("1" * 40 + "\n")
        reject(lambda: s.h.frozen(root), "frozen metadata changed")
    for canary in (False, True):
        with tempfile.TemporaryDirectory(dir=s.ROOT / "build-tmp") as directory:
            out = Path(directory) / "attempt"
            boot_count = 1 if canary else 6
            zero = {key: 0 for key in pair}
            def boot(harness, root, label, *args):
                dest = harness.out / label
                dest.mkdir()
                return dest
            def request(harness, dest, case, number, body=None, stream=False):
                result = copy.deepcopy(record)
                result["case"] = case
                if case != "single-t1":
                    result["response"]["timings"]["prompt_n"] = 1024
                (dest / f"{case}-{number}.json").write_text(json.dumps(result))
                return result
            argv = ["stack3_gpu", "--execute", "--out", str(out)] + (["--canary", "--budget", "240"] if canary else [])
            with patch.dict(os.environ, RUN_GPU="1", CUDA_VISIBLE_DEVICES="0,1", XID_SINCE="mock-cutoff"), \
                 patch.object(s, "pins", return_value=[s.h.MODEL]), patch.object(s.h, "sha", return_value=s.h.MODEL_SHA), \
                 patch.object(s.h, "frozen", return_value={}), patch.object(s.h.Harness, "boot", boot), \
                 patch.object(s.h.Harness, "request", request), patch.object(s.h.Harness, "metrics", side_effect=[zero, pair] * boot_count), \
                 patch.object(s.h.Harness, "stop"), patch.object(s.h.Harness, "xid"), \
                 patch.object(s.h.subprocess, "Popen", side_effect=AssertionError("mock flow launch")), \
                 patch.object(s.h.socket, "socket", side_effect=AssertionError("mock flow socket")), patch.object(s.sys, "argv", argv):
                s.main()
            assert json.loads((out / "verdict.json").read_text())["verdict"] == "PASS"
            assert len(json.loads((out / "boots.json").read_text())) == boot_count
    with tempfile.TemporaryDirectory(dir=s.ROOT / "build-tmp") as directory:
        out = Path(directory) / "interrupted"
        original_handler = s.signal.getsignal(s.signal.SIGTERM)
        def stop_signal(*args):
            s.signal.getsignal(s.signal.SIGTERM)(s.signal.SIGTERM, None)
        with patch.dict(os.environ, RUN_GPU="1", CUDA_VISIBLE_DEVICES="0,1", XID_SINCE="mock-cutoff"), \
             patch.object(s, "pins", return_value=[s.h.MODEL]), patch.object(s.h, "sha", return_value=s.h.MODEL_SHA), \
             patch.object(s.h.Harness, "boot", boot), patch.object(s, "suite", stop_signal), \
             patch.object(s.h.Harness, "stop") as cleanup, patch.object(s.sys, "argv", ["stack3_gpu", "--execute", "--out", str(out)]):
            try:
                s.main()
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("termination did not interrupt")
        assert cleanup.call_count >= 1
        assert json.loads((out / "verdict.json").read_text())["verdict"] == "interrupted"
        assert s.signal.getsignal(s.signal.SIGTERM) == original_handler
    print("M PASS: recorded Swift completion/steady/counters, journal, identity, size, overlap, budget, and execution guard controls")


if __name__ == "__main__": main()
