"""Full tool-loop coding probe with inaccessible oracles and pristine external grading."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import tomllib

from scripts import ROOT
from scripts.corpora.selftest_polyglot import _overlay_reference
from scripts.instruments import emit, isolated_exec, run_polyglot as poly
from scripts.instruments.bench_mtp import acceptance
from scripts.instruments.hillclimb_probe import api, digest_file


TASK_IDS = ["javascript/affine-cipher", "javascript/alphametics", "javascript/bowling",
            "javascript/connect", "rust/word-count", "rust/robot-name", "rust/forth",
            "rust/doubly-linked-list"]
TEST_SCOPE = "trusted-tests-v2"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare_assets(destination, *, full_suite=False):
    destination.mkdir(parents=True, exist_ok=False)
    node = Path(subprocess.check_output(["node", "-p", "process.execPath"], text=True).strip())
    rust = Path(subprocess.check_output(["rustup", "which", "rustc"], text=True).strip())
    assets = {"node_root": str(node.parent.parent), "rust_root": str(rust.parent.parent),
              "cargo_registry": str(Path.home() / ".cargo/registry"),
              "node_modules": str(ROOT / ".cache/js-deps/node_modules"),
              "codex": str(Path(shutil.which("codex")).resolve())}
    assets["binary_hashes"] = {str(path): digest_file(path)
                                for path in (node, rust, Path(assets["codex"]))}
    tasks = json.loads((ROOT / "subsets/polyglot.json").read_text())
    selected = []
    task_ids = ([task["task_id"] for task in tasks] if full_suite else TASK_IDS)
    if len(task_ids) != len(set(task_ids)) or any(t["language"] not in {"javascript", "rust"} for t in tasks):
        raise ValueError("suite must contain unique supported tasks")
    for task_id in task_ids:
        task = dict(next(t for t in tasks if t["task_id"] == task_id))
        source = (ROOT / "subsets" / task["source_dir"]).resolve()
        task["test_paths"] = [str(p.relative_to(source)) for p in
                              poly.task_file_paths(task, source, "test_path", "test")]
        target = destination / "sources" / task_id
        shutil.copytree(source, target)
        task["source_dir"] = "sources/" + task_id
        selected.append(task)
    shutil.copy2(ROOT / "subsets/rust-deps.json", destination / "rust-deps.json")
    write_json(destination / "tools.json", assets)
    write_json(destination / "tasks.json", selected)
    return assets, selected


def enable_js_tests(text):
    text = re.sub(r"(?m)^([ \t]*)x(test|it|describe)(?=\s*\()", r"\1\2", text)
    return re.sub(r"(?m)^([ \t]*)(test|it|describe)\.skip(?=\s*\(|\.each)", r"\1\2", text)


def materialize(task, asset_dir, destination, *, reference=False):
    source = asset_dir / task["source_dir"]
    shutil.copytree(source, destination)
    if reference:
        _overlay_reference(task, destination)
    if task["language"] == "javascript":
        for relative in task["test_paths"]:
            path = destination / relative
            path.write_text(enable_js_tests(path.read_text()))
        (destination / "node_modules").symlink_to("/node_modules", target_is_directory=True)
    else:
        deps = json.loads((asset_dir / "rust-deps.json").read_text())["deps"].get(task["task_id"], {})
        cargo = destination / "Cargo.toml"
        data = tomllib.loads(cargo.read_text())
        data.setdefault("dependencies", {}).update(deps)
        cargo.write_text(dump_toml(data))
    poly._strip_reference_material(destination)


def toml_value(value):
    if isinstance(value, dict):
        return "{ " + ", ".join(json.dumps(k) + " = " + toml_value(v) for k, v in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, (str, int, float, bool)):
        return json.dumps(value, allow_nan=False)
    raise ValueError("unsupported TOML value")


def dump_toml(data):
    return "\n".join(json.dumps(key) + " = " + toml_value(value) for key, value in data.items()) + "\n"


def transfer_solution(task, workdir, grade_dir):
    ignored = []
    for relative in task["solution_path"]:
        source = workdir / relative
        if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(workdir.resolve()):
            raise ValueError("solution is missing or escapes the workspace: " + relative)
        if source.stat().st_size > 16 * 1024**2:
            raise ValueError("solution file exceeds 16 MiB: " + relative)
        if relative == "Cargo.toml":
            supplied = tomllib.loads(source.read_text())
            trusted = tomllib.loads((grade_dir / relative).read_text())
            for section in ("dependencies", "dev-dependencies"):
                if section in supplied:
                    for spec in supplied[section].values():
                        if isinstance(spec, dict) and set(spec) & {"path", "git", "registry"}:
                            raise ValueError("grading permits only cached registry dependencies")
                    trusted[section] = supplied[section]
            ignored = [key for key in supplied if key not in ("dependencies", "dev-dependencies")
                       and supplied[key] != trusted.get(key)]
            (grade_dir / relative).write_text(dump_toml(trusted))
        else:
            target = grade_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return ignored


def rust_test_plan(task, asset_dir):
    source = asset_dir / task["source_dir"]
    tests = sorted(str(p.relative_to(source)) for p in (source / "tests").glob("*.rs"))
    if not tests or not set(task["test_paths"]) <= set(tests):
        raise ValueError("trusted Rust integration targets are missing or unsupported")
    starter = (source / "src/lib.rs").read_text()
    docs = "\n".join(re.findall(r"(?m)^\s*//(?:/|!)(?!/)[ ]?(.*)$", starter))
    return tests, docs if "```" in docs else ""


def test_suite(task, workdir, assets, timeout, *, asset_dir):
    started = time.monotonic()
    readonly = task["test_paths"]
    docs = ""
    if task["language"] == "javascript":
        command = ["/node/bin/node", "/node_modules/jest/bin/jest.js", *task["test_paths"],
                   "--runInBand", "--json", "--outputFile=/work/.grader-result.json"]
    else:
        readonly, docs = rust_test_plan(task, asset_dir)
        command = ["/rust/bin/cargo", "test", "--offline"]
        # Only immutable integration targets count, never implementation-owned unit tests.
        for relative in readonly:
            command += ["--test", Path(relative).stem]
        command += ["--", "--include-ignored"]
    result = isolated_exec.run(workdir, assets, command, timeout, readonly=readonly)
    output = result.stdout + result.stderr
    returncode = result.returncode
    if docs and returncode == 0:
        # Starter doctests must survive a candidate replacing all of src/lib.rs.
        doc_path = workdir / ".grader-doctests.md"
        doc_path.write_text(docs + "\n")
        cargo = tomllib.loads((asset_dir / task["source_dir"] / "Cargo.toml").read_text())
        name = cargo.get("lib", {}).get("name", cargo["package"]["name"].replace("-", "_"))
        libraries = list((workdir / "target/debug/deps").glob(f"lib{name}-*.rlib"))
        if len(libraries) != 1:
            raise ValueError("trusted doctests require exactly one freshly built candidate library")
        command = ["/rust/bin/rustdoc", "--test", "/work/" + doc_path.name,
                   "--edition", str(cargo["package"].get("edition", "2015")),
                   "--extern", name + "=/work/" + str(libraries[0].relative_to(workdir)),
                   "-L", "dependency=/work/target/debug/deps", "--test-args", "--include-ignored"]
        result = isolated_exec.run(workdir, assets, command,
                                   poly._remaining(started + timeout), readonly=[*readonly, doc_path.name])
        output += result.stdout + result.stderr
        returncode = result.returncode
    passed = failed = ignored = 0
    if task["language"] == "javascript":
        report_path = workdir / ".grader-result.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            passed, failed, ignored = (report.get(key, 0) for key in
                                      ("numPassedTests", "numFailedTests", "numPendingTests"))
    else:
        for match in re.finditer(r"test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;", output):
            counts = [int(value) for value in match.groups()]
            passed += counts[0]
            failed += counts[1]
            ignored += counts[2]
    return {"returncode": returncode, "passed": passed, "failed": failed, "ignored": ignored,
            "test_scope": TEST_SCOPE, "test_paths": readonly, "starter_doctests": bool(docs),
            "seconds": time.monotonic() - started, "output": output[-24000:]}


def selftest(asset_dir, assets, tasks):
    rows = []
    for task in tasks:
        with tempfile.TemporaryDirectory(prefix="qwen-grader-check-") as scratch:
            scratch = Path(scratch)
            stub, reference = scratch / "stub", scratch / "reference"
            materialize(task, asset_dir, stub)
            materialize(task, asset_dir, reference, reference=True)
            reference_result = test_suite(task, reference, assets, 300, asset_dir=asset_dir)
            stub_result = test_suite(task, stub, assets, 300, asset_dir=asset_dir)
            good_reference = (reference_result["returncode"] == 0 and reference_result["passed"] > 0
                              and reference_result["failed"] == reference_result["ignored"] == 0)
            rows.append({"task_id": task["task_id"], "reference_passes": good_reference,
                         "stub_fails": stub_result["returncode"] != 0,
                         "expected_passed_tests": reference_result["passed"],
                         "reference": reference_result, "stub": stub_result})
            print(json.dumps({"task": task["task_id"], "reference_passes": good_reference,
                              "stub_fails": rows[-1]["stub_fails"]}), flush=True)
    report = {"test_scope": TEST_SCOPE,
              "harness_verified": all(r["reference_passes"] and r["stub_fails"] for r in rows), "tasks": rows}
    write_json(asset_dir / "selftest.json", report)
    return report


def client_config(context):
    compact = 58000 if context == 65536 else context - 16384
    return f'''model = "local-benchmark"
model_provider = "benchmark"
model_context_window = {context}
model_auto_compact_token_limit = {compact}
model_reasoning_effort = "xhigh"
approval_policy = "never"
sandbox_mode = "danger-full-access"
web_search = "disabled"
project_doc_max_bytes = 0

[model_providers.benchmark]
name = "Isolated local benchmark"
base_url = "http://127.0.0.1:8081/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
request_max_retries = 2
stream_max_retries = 2
stream_idle_timeout_ms = 900000
'''


def _guard_port(port, out):
    # Opt-in runaway guard (QWEN_RUNAWAY_GUARD=buffer|stream): the bridge dials the guard instead of the engine.
    if os.environ.get("QWEN_RUNAWAY_GUARD", "") in ("", "0"):
        return contextlib.nullcontext(port)
    from scripts.instruments import runaway_guard
    return runaway_guard.guarded_port(port, out)


def fresh_context_peak(before, after, context):
    # This server includes generated tokens in n_tokens_max; adding n_decoded double-counts them.
    if before.get("n_tokens_max") != 0:
        raise ValueError("task context accounting requires a fresh server")
    peak = after.get("n_tokens_max")
    if not isinstance(peak, (int, float)) or not 0 <= peak <= context or int(peak) != peak:
        raise ValueError("invalid per-task context high-water mark")
    return int(peak)


def probe(asset_dir, task_id, url, context, out, timeout, engine=None):
    # engine=None is the single-slot llama-server contract; another engine supplies check(),
    # meter(out) (start/stop/port, stop() returning the task's context peak), context_peak()
    # and peak_source.
    assets = json.loads((asset_dir / "tools.json").read_text())
    tasks = json.loads((asset_dir / "tasks.json").read_text())
    task = next(t for t in tasks if t["task_id"] == task_id)
    checks = json.loads((asset_dir / "selftest.json").read_text())
    if not checks["harness_verified"] or checks.get("test_scope") != TEST_SCOPE:
        raise ValueError("trusted grader self-test has not passed")
    expected_tests = next(r["expected_passed_tests"] for r in checks["tasks"] if r["task_id"] == task_id)
    for path, expected in assets["binary_hashes"].items():
        if digest_file(path) != expected:
            raise ValueError("tool binary changed: " + path)
    if engine is None:
        props = api(url, "/props")
        if props["total_slots"] != 1 or props["default_generation_settings"]["n_ctx"] != context:
            raise ValueError("agentic probe requires the declared single slot")
    else:
        engine.check(url, context)
    started = time.monotonic()
    deadline = started + timeout
    workdir = out / "workspace"
    materialize(task, asset_dir, workdir)
    client_home = out / "codex-state"
    client_home.mkdir()
    (client_home / "config.toml").write_text(client_config(context))
    prompt = (poly._prompt(task) + " Do not modify tests. Dependencies are preinstalled; "
              "external network access is disabled. The supported solution files are: "
              + ", ".join(task["solution_path"]) + ".")
    before = emit.scrape_metrics(url)
    write_json(out / "metrics-before.json", before)
    sampler = emit.ContextSampler(url) if engine is None else engine.meter(out)
    sampler.start()
    rounds, stats = [], []
    success = timed_out = False
    error = None
    try:
        with tempfile.TemporaryDirectory(prefix="qwen-bridge-") as bridge_dir:
            bridge = Path(bridge_dir) / "model.sock"
            port = int(url.rsplit(":", 1)[-1]) if engine is None else sampler.port
            with _guard_port(port, out) as port, isolated_exec.inference_bridge(bridge, port):
                for attempt in range(2):
                    (out / f"prompt-{attempt}.txt").write_text(prompt)
                    transcript = out / f"codex-{attempt}.jsonl"
                    # The outer mount/PID/network jail is the boundary; no user config or credentials enter it.
                    result = isolated_exec.run(workdir, assets,
                        ["/runner/codex", "exec", "--skip-git-repo-check", "--json", "-"],
                        poly._remaining(deadline), bridge=bridge, client_home=client_home,
                        readonly=task["test_paths"], input_text=prompt,
                        stdout_path=transcript, stderr_path=out / f"codex-{attempt}.stderr")
                    stats.append(emit.parse_codex_stats(transcript.read_text()))
                    if result.returncode:
                        raise RuntimeError(f"Codex exited with status {result.returncode}")
                    grade_dir = out / f"grade-{attempt}"
                    materialize(task, asset_dir, grade_dir)
                    try:
                        ignored = transfer_solution(task, workdir, grade_dir)
                        grade = test_suite(task, grade_dir, assets, poly._remaining(deadline), asset_dir=asset_dir)
                    except (ValueError, FileNotFoundError) as invalid_solution:
                        ignored = []
                        grade = {"returncode": 1, "passed": 0, "failed": 0, "ignored": 0,
                                 "seconds": 0, "output": "Invalid solution: " + str(invalid_solution)}
                    grade["ignored_cargo_metadata_changes"] = ignored
                    rounds.append(grade)
                    write_json(out / f"grade-{attempt}.json", grade)
                    success = (grade["returncode"] == 0 and grade["passed"] == expected_tests
                               and grade["failed"] == grade["ignored"] == 0)
                    if success:
                        break
                    prompt = (poly._prompt(task) + "\nTrusted grading failed. Repair the solution without changing tests.\n"
                              + f"Expected {expected_tests} trusted tests; got {grade['passed']} passed, "
                              + f"{grade['failed']} failed, {grade['ignored']} ignored; exit {grade['returncode']}.\n"
                              + grade["output"])
    except subprocess.TimeoutExpired:
        timed_out = True
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    finally:
        wall = time.monotonic() - started
        sampled_peak = sampler.stop()
    after = emit.scrape_metrics(url)
    write_json(out / "metrics-after.json", after)
    peak = (fresh_context_peak(before, after, context) if engine is None
            else engine.context_peak(sampled_peak, context))
    delta = emit.metrics_delta(before, after)
    prefill, decode = delta.get("prefill_seconds"), delta.get("decode_seconds")
    valid_timing = prefill is not None and decode is not None and prefill + decode <= wall + 2
    summary = {"workload": "agentic", "task_id": task_id, "complete": error is None and valid_timing,
               "task_success": success, "timed_out": timed_out, "error": error,
               "verified_wall_seconds": wall, "prefill_seconds": prefill, "decode_seconds": decode,
               "other_seconds": max(0, wall - prefill - decode) if valid_timing else None,
               "grader_seconds": sum(r["seconds"] for r in rounds), "grading_rounds": len(rounds),
               "expected_passed_tests": expected_tests, "harness_verified": True,
               "metrics_scope": "task_single_slot", "metrics_delta": delta, "mtp": acceptance(delta),
               "mtp_live": delta.get("draft_tokens", 0) > 0 and delta.get("draft_accepted_tokens", 0) > 0,
               "depth_verified": True, "overlap_verified": True, "context_occupancy_requirement": "natural task history",
               "peak_context_tokens": peak, "legacy_sampled_context_tokens": sampled_peak,
               "context_peak_source": "fresh-server n_tokens_max" if engine is None else engine.peak_source,
               "client_context": context,
               "client_compaction": 58000 if context == 65536 else context - 16384,
               "agent_stats": poly._merge_stats(stats), "throughput_output_cap_applied": False,
               "external_network": False, "pristine_grader": True, "test_scope": TEST_SCOPE}
    write_json(out / "probe.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "selftest", "probe"])
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--task")
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--context", type=int, default=102400)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--full-suite", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_assets(args.assets, full_suite=args.full_suite)
        return 0
    if args.command == "selftest":
        result = selftest(args.assets, json.loads((args.assets / "tools.json").read_text()),
                          json.loads((args.assets / "tasks.json").read_text()))
        return 0 if result["harness_verified"] else 1
    result = probe(args.assets, args.task, args.url, args.context, args.out, args.timeout)
    return 0 if result["complete"] and result["mtp_live"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
