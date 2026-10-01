#!/usr/bin/env python3
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import gpu_harness as base
import pool_harness as h
from attrib import analyze
from test_attrib import fixture


def rejects(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("negative control escaped")


def synthetic():
    boots = []
    for trace in (False, True):
        for index, pool in enumerate(h.ORDER):
            record = {"response": {"choices": [{"message": {"content": "same"}, "finish_reason": "length"}],
                                   "timings": {"predicted_ms": 300 if pool == 2 else 200}},
                      "metrics": {"drafts": 10, "draft_tokens": 29, "accepted_tokens": 20}}
            boots.append({"boot": str(index), "pool": pool, "trace": trace,
                          "records": {case: [copy.deepcopy(record)] for case in ("tool", "think")},
                          "sampled_peak_MiB_M": [12000, 12000]})
    return boots


def main():
    plan = io.StringIO()
    with patch.object(base.subprocess, "run", side_effect=AssertionError("plan subprocess")), \
            patch.object(base.subprocess, "Popen", side_effect=AssertionError("plan launch")), \
            patch.object(base.socket, "socket", side_effect=AssertionError("plan network")), \
            patch.object(h, "sha", side_effect=AssertionError("plan model or input hash")), \
            patch.object(sys, "argv", ["pool_harness"]), contextlib.redirect_stdout(plan):
        # Frozen validation has its own hash function; plan never opens model/inputs or probes GPUs.
        h.main()
    assert json.loads(plan.getvalue())["common_env"] == {"LLAMA_ARG_LOG_VERBOSITY": "4"}
    for env in ({"RUN_GPU": "0", "CUDA_VISIBLE_DEVICES": "0,1"}, {"RUN_GPU": "1", "CUDA_VISIBLE_DEVICES": ""}):
        with patch.dict(os.environ, env), patch.object(sys, "argv", ["pool_harness", "--execute"]), \
                patch.object(base.subprocess, "run", side_effect=AssertionError("ungated subprocess")), \
                patch.object(base.subprocess, "Popen", side_effect=AssertionError("ungated launch")), \
                patch.object(base.socket, "socket", side_effect=AssertionError("ungated network")), \
                patch.object(h, "sha", side_effect=AssertionError("ungated model hash")), \
                contextlib.redirect_stdout(io.StringIO()):
            rejects(h.main)
    boots = synthetic()
    result = h.summary(boots)
    assert len(result["per_boot"]) == 16
    assert all(c["pool8_minus_pool2_ms_per_round_D"] == -10 for c in result["comparisons"])
    assert result["falsification_rule"] == h.FALSIFICATION
    attrib = analyze(fixture(), skip=0)
    mechanisms = h.trace_summary(attrib)
    assert mechanisms["ar_waits_per_round_D"] == 5
    assert abs(mechanisms["ar_wait_host_us_per_round_D"] - 0.5) < 1e-9
    assert mechanisms["paired_gpu_boundary_idle_us_per_round_D"] is None
    for field in ("completion", "drafts", "draft_tokens", "accepted_tokens"):
        broken = copy.deepcopy(boots)
        record = broken[1]["records"]["tool"][0]
        if field == "completion":
            record["response"]["choices"][0]["message"]["content"] = "different"
        else:
            record["metrics"][field] += 1
        rejects(lambda: h.summary(broken))
    rejects(lambda: h.summary(boots[:-1]))
    cmd = h.command(h.ROOT / "codex/frozen/ar-pool/bin/round-trace", 5, 8398)
    assert cmd[cmd.index("--spec-draft-n-max") + 1] == "5"
    for flag in ("--spec-draft-vocab", "--spec-draft-backend-sampling", "--backend-sampling", "--spec-draft-sampling"):
        assert flag in cmd
    assert cmd[cmd.index("-c") + 1] == "102400" and cmd[cmd.index("-sm") + 1] == "tensor"
    with tempfile.TemporaryDirectory(dir=h.ROOT / "build-tmp") as directory:
        dest = Path(directory)
        with patch.dict(os.environ, {"LLAMA_ARG_LOG_VERBOSITY": "0", "GGML_CUDA_AR_POOL": "16"}):
            harness = h.PoolHarness(SimpleNamespace(budget=2400, patch=h.ROOT / "codex/frozen/ar-pool"), dest)
        assert harness.env["LLAMA_ARG_LOG_VERBOSITY"] == "4"
        assert "GGML_CUDA_AR_POOL" not in harness.env
        (dest / "command.json").write_text('{}')
        with patch.object(harness, "boot", return_value=dest):
            for log in ("generic fallback", "initialized AllReduce pipeline: 2 GPUs, pool=2", "initialized AllReduce pipeline: 2 GPUs, pool=8", "initialized AllReduce pipeline: 2 GPUs, pool=8\ninitialized AllReduce pipeline: 2 GPUs, pool=2"):
                (dest / "server.log").write_text(log)
                rejects(lambda: harness.pool_boot("test", 8, False))
            (dest / "server.log").write_text("initialized AllReduce pipeline: 2 GPUs, pool=8\ninitialized AllReduce pipeline: 2 GPUs, pool=8")
            harness.pool_boot("test", 8, False)
            assert harness.env["GGML_CUDA_AR_POOL"] == "8"
            config = json.loads((dest / "command.json").read_text())
            assert config["GGML_CUDA_AR_POOL"] == "8" and config["LLAMA_ARG_LOG_VERBOSITY"] == "4"
    print("M: plan/gates make no subprocess/network calls; completion/counter/missing-boot controls rejected; authoritative period arithmetic and S3/K flags PASS")


if __name__ == "__main__":
    main()
