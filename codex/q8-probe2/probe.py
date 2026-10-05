#!/usr/bin/env python3
"""Frozen one-row FA A/B. GPU modes are for the queue; preflight is CPU-only."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import struct
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FROZEN_ROOT = Path("/home/xanewok/repos/q8-1row/codex/frozen")
FROZEN = {"up": FROZEN_ROOT / "q8-1row-base", "patch": FROZEN_ROOT / "q8-1row"}
COMMITS = {"up": "98c695111df5b5517245b184ef4e818333cf5776", "patch": "797465a5c7f0a840ba7a8b6f8931d3e9ca8318e7"}
DRIVER = HERE / "bin/q8-probe-driver"
ERR = re.compile(r"ERR = ([0-9.eE+-]+) > ([0-9.eE+-]+)")
STAGE_S = {"test": 60, "controls": 20, "perf": 90}
DEPTHS = (8192, 32768, 90112)
CONTROLS = ((8, 1, 256), (4, 2, 256), (2, 4, 256), (1, 8, 256), (2, 6, 256), (3, 6, 256), (4, 6, 256), (8, 6, 256))
GQAS = (1, 4, 6, 8)
DIMS = (128, 256)
TYPES = {"f16": (1, 2, 1), "q8_0": (8, 34, 32)}
FIELDS = ("stamp", "commit", "backend", "op", "params", "mode", "supported", "passed", "error", "us", "flops", "bw", "mem", "runs", "device", "reg")
HAZARD = re.compile(r"\bXid\b|fallen off the bus")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


class Gate(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def fa_op():
    enum = re.search(r"enum ggml_op\s*\{(.*?)\};", (ROOT / "ggml/include/ggml.h").read_text(), re.S).group(1)
    return re.findall(r"\bGGML_OP_\w+\b", enum).index("GGML_OP_FLASH_ATTN_EXT")


def tensor(typ, shape, strides=None):
    tid, size, block = (0, 4, 1) if typ == "f32" else TYPES[typ]
    d, n, h, b = shape
    nb = strides or (size, d * size // block, d * n * size // block, d * n * h * size // block)
    return " ".join(map(str, (tid, *shape, *nb)))


def case_name(typ, kv, nb=1, gqa=6, d=256):
    return f"q8-fa:{typ}:{kv}:q{nb}:g{gqa}:d{d}"


def case_line(typ, kv, nb=1, gqa=6, d=256):
    _, size, block = TYPES[typ]
    row = d * size // block
    scale = struct.unpack("i", struct.pack("f", 1 / math.sqrt(d)))[0]
    sources = [tensor("f32", (d, nb, 2 * gqa, 1)),
               tensor(typ, (d, kv, 2, 1), (size, row, 2 * row * kv, 4 * row * kv)),
               tensor(typ, (d, kv, 2, 1), (size, row, 2 * row * kv, 4 * row * kv)),
               tensor("f16", (kv, nb, 1, 1))]
    return f"{fa_op()} 0 {d} {2 * gqa} {nb} 1 4 {scale} 0 0 10 4 " + " ".join(sources) + " " + case_name(typ, kv, nb, gqa, d)


def generate(out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.txt").write_text("\n".join(case_line(t, k, 1, g, d) for k in DEPTHS for t in TYPES for g in GQAS for d in DIMS) + "\n")
    (out / "controls.txt").write_text("\n".join(case_line(t, 8192, nb, gqa, d) for t in TYPES for nb, gqa, d in CONTROLS) + "\n")


def sql_rows(raw):
    rows = []
    for line in raw.splitlines():
        if "INSERT INTO test_backend_ops" not in line:
            continue
        body = line.split(" VALUES (", 1)[1].rsplit(");", 1)[0]
        values = re.findall(r"'((?:[^']|'')*)'", body)
        if len(values) != len(FIELDS):
            raise Gate(6, "SQL record field count changed")
        row = dict(zip(FIELDS, (v.replace("''", "'") for v in values)))
        metrics = ERR.findall(line.split("INSERT INTO", 1)[0])
        if len(metrics) > 1:
            raise Gate(6, "duplicate ERR record")
        if metrics:
            row.update(nmse=float(metrics[0][0]), tolerance=float(metrics[0][1]))
        rows.append(row)
    return rows


def stock_shape(params):
    match = re.match(r"hsk=(\d+),hsv=(\d+),nh=(\d+),nr23=\[(\d+),(\d+)\],kv=(\d+),nb=(\d+),.*type_K=(\w+),type_V=(\w+),", params)
    if not match:
        raise Gate(6, "unrecognized recorded FA shape")
    return match.groups()


def recorded_fa(raw, mode):
    rows = []
    for line in ANSI.sub("", raw).splitlines():
        match = re.search(r"FLASH_ATTN_EXT\((.*)\):\s*(.*)", line)
        if not match:
            continue
        shape = stock_shape(match[1])
        if mode == "perf":
            metric = re.search(r"(\d+) runs -\s*([0-9.]+) us/run", match[2])
            if metric:
                rows.append((shape, float(metric[2]), int(metric[1])))
        elif match[2].strip() == "OK":
            rows.append((shape, True))
        elif match[2].strip().startswith("not supported"):
            continue
        else:
            raise Gate(8, "recorded FA correctness failed")
    if not rows:
        raise Gate(5, "recorded FA rows missing")
    return rows


def check_device(raw):
    found = re.findall(r"Device (\d+): (.*?), compute capability ([0-9.]+),", raw)
    if len(found) != 1 or found[0][0] != "0" or "RTX 5060 Ti" not in found[0][1] or found[0][2] != "12.0":
        raise Gate(6, "expected one visible RTX 5060 Ti at cc 12.0")


def gpu_memory(raw):
    try:
        rows = list(csv.DictReader(raw.splitlines()))
        result = {int(r["index"]): int(r[" memory.used [MiB]"].strip().split()[0]) for r in rows}
    except (KeyError, ValueError, IndexError) as exc:
        raise Gate(4, "GPU inventory format changed") from exc
    if not result:
        raise Gate(4, "GPU inventory missing")
    return result


def case_params(typ, kv, nb=1, gqa=6, d=256):
    _, size, block = TYPES[typ]
    row = d * size // block
    source = f"{typ}[{d},{kv},2,1]nb[{size},{row},{2 * row * kv},{4 * row * kv}]"
    scale = struct.unpack("i", struct.pack("f", 1 / math.sqrt(d)))[0]
    return (f"name={case_name(typ, kv, nb, gqa, d)},type=f32,ne=[{d},{2 * gqa},{nb},1],op_params=[0:{scale},3:10],"
            f"sources=f32[{d},{nb},{2 * gqa},1],{source},{source},f16[{kv},{nb},1,1]")


def shape_key(params):
    match = re.match(r"name=q8-fa:(f16|q8_0):(\d+):q(\d+):g(\d+):d(\d+),", params)
    if not match:
        raise Gate(6, "FA case does not match the requested geometry")
    typ, kv, nb, gqa, d = match.groups()
    kv, nb, gqa, d = int(kv), int(nb), int(gqa), int(d)
    if kv not in DEPTHS or d not in DIMS or params != case_params(typ, kv, nb, gqa, d):
        raise Gate(6, "FA source shape mismatch")
    return typ, kv, nb, gqa, d


def validate(raw, mode, commit, controls=False):
    rows = []
    for row in sql_rows(raw):
        if row["op"] != "FLASH_ATTN_EXT":
            continue
        typ, kv, nb, gqa, d = shape_key(row["params"])
        if row["mode"] != mode or row["backend"] != "CUDA0" or row["commit"] != commit[:9]:
            raise Gate(6, "backend, mode or frozen commit mismatch")
        if row["supported"] != "1":
            raise Gate(8, "FA case unsupported")
        if mode == "test":
            if "nmse" not in row or not math.isfinite(row["nmse"]) or row["nmse"] < 0:
                raise Gate(6, "missing or invalid ERR record")
            tolerance = 5e-4 * max(1, kv / 20000)
            if abs(row["tolerance"] - tolerance) > 1e-12:
                raise Gate(6, "FA tolerance mismatch")
            passed = row["nmse"] <= tolerance
            if row["passed"] != str(int(passed)) or row["error"] != ("" if passed else "test failed"):
                raise Gate(6, "ERR and SQL verdict disagree")
            if typ == "q8_0" and not passed:
                raise Gate(8, "q8_0 CUDA-vs-CPU correctness failed")
        elif row["passed"] != "1" or row["error"]:
            raise Gate(8, "FA perf failed")
        row.update(type=typ, kv=kv, nb=nb, gqa=gqa, d=d)
        if mode == "perf":
            us = float(row["us"])
            if not math.isfinite(us) or us <= 0 or int(row["runs"]) < 2:
                raise Gate(5, "invalid perf sample")
            _, size, block = TYPES[typ]
            nbytes = kv * 2 * d * 2 * size / block
            row.update(us=us, kv_bytes=nbytes, effective_gbs=nbytes / us / 1000,
                       cache_note="fits 32 MiB L2" if nbytes <= 32 * 1024**2 else "exceeds 32 MiB L2")
        rows.append(row)
    want = {(t, 8192, nb, g, d) for t in TYPES for nb, g, d in CONTROLS} if controls else {(t, k, 1, g, d) for t in TYPES for k in DEPTHS for g in GQAS for d in DIMS}
    keys = [(r["type"], r["kv"], r["nb"], r["gqa"], r["d"]) for r in rows]
    if len(keys) != len(want) or set(keys) != want:
        raise Gate(5, "missing or duplicate FA cases")
    return rows


def f16_integrity(rows, require_pair=True):
    groups = {}
    for row in rows:
        if row["type"] == "f16":
            key = (row["gpu"], row["kind"], row["kv"], row["nb"], row["gqa"], row["d"])
            groups.setdefault(key, {}).setdefault(row["arm"], []).append(row["nmse"])
    for arms in groups.values():
        if set(arms) != set(FROZEN):
            if require_pair:
                raise Gate(6, "f16 control arm missing")
            continue
        values = arms["up"] + arms["patch"]
        if max(values) - min(values) > 1e-6:
            raise Gate(6, "f16 control ERR differs between arms")
    if require_pair and not groups:
        raise Gate(6, "f16 controls missing")


def checksums(root, binary=None):
    for base, manifest in ((root, root / "SHA256SUMS"), (root / "bin", root / "bin/SHA256SUMS")):
        for line in manifest.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            path = base / name.lstrip("*")
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise Gate(6, "frozen bytes changed")
    result = subprocess.run(["ldd", str(binary or root / "bin/test-backend-ops")],
                            env=dict(os.environ, LD_LIBRARY_PATH=str(root / "bin")), text=True, capture_output=True, check=True)
    libs = re.findall(r"(libggml\S+) => (\S+)", result.stdout)
    if not libs or any(Path(p).resolve().parent != (root / "bin").resolve() for _, p in libs):
        raise Gate(6, "ggml libs resolve outside frozen dir")
    return (root / "SOURCE").read_text().splitlines()[0]


def selftest():
    fixtures = HERE / "fixtures"
    sql = sql_rows((fixtures / "real-mmvq.sql").read_text())
    assert len(sql) == 40 and all(r["backend"] == "CUDA0" and r["passed"] == "1" for r in sql)
    perf_raw = (fixtures / "real-fa-perf.log").read_text()
    perf = recorded_fa(perf_raw, "perf")
    assert any(s[:7] == ("256", "256", "2", "6", "1", "8192", "1") and us == 14.59 for s, us, _ in perf)
    assert len(recorded_fa((fixtures / "real-fa-eval.log").read_text(), "test")) > 100
    check_device(perf_raw)
    assert gpu_memory((fixtures / "real-memory.csv").read_text()) == {0: 108, 1: 15}
    assert HAZARD.search((fixtures / "real-kernel-errors.txt").read_text())
    assert not HAZARD.search("stop_xid_hash_s_G")
    canary = sql_rows((fixtures / "canary-test.sql").read_text())
    assert len(canary) == 48
    failed = [r for r in canary if r["passed"] == "0"]
    assert len(failed) == 1 and shape_key(failed[0]["params"]) == ("f16", 90112, 1, 6, 128)
    assert failed[0]["nmse"] == 0.002350528 and failed[0]["tolerance"] == 0.002252800
    controls = [dict(failed[0], type="f16", kv=90112, nb=1, gqa=6, d=128, gpu=0, kind="test", arm=a) for a in FROZEN]
    f16_integrity(controls)
    controls[1]["nmse"] += 2e-6
    try:
        f16_integrity(controls)
    except Gate as exc:
        assert exc.code == 6
    else:
        raise AssertionError("f16 integrity negative control passed")
    raw = []
    for typ, kv, nb, gqa, d in sorted({(t, k, 1, g, d) for t in TYPES for k in DEPTHS for g in GQAS for d in DIMS}):
        params = case_params(typ, kv, nb, gqa, d)
        assert shape_key(params) == (typ, kv, nb, gqa, d)
        assert len(case_line(typ, kv, nb, gqa, d).split()) == 49
        values = ["time", "123456789", "CUDA0", "FLASH_ATTN_EXT", params, "perf", "1", "1", "", "500", "0", "0", "0", "100", "", ""]
        raw.append("INSERT INTO test_backend_ops VALUES (" + ", ".join("'" + v + "'" for v in values) + ");")
    good = "\n".join(raw)
    assert len(validate(good, "perf", "123456789")) == 48
    eval_lines = []
    for line in raw:
        kv = shape_key(sql_rows(line)[0]["params"])[1]
        eval_lines.append(f"[FLASH_ATTN_EXT] ERR = 0.0001 > {5e-4 * max(1, kv / 20000)} " + line.replace("'perf'", "'test'"))
    assert len(validate("\n".join(eval_lines), "test", "123456789")) == 48
    mutant = "\n".join(line.replace("ERR = 0.0001", "ERR = 0.1").replace("'1', '1', ''", "'1', '0', 'test failed'")
                       if "name=q8-fa:q8_0:" in line else line for line in eval_lines)
    try:
        validate(mutant, "test", "123456789")
    except Gate as exc:
        assert exc.code == 8
    else:
        raise AssertionError("q8_0 correctness negative control passed")
    mutants = [good.replace("'1', '1'", "'1', '0'"), good.replace("'500'", "'nan'"),
               "\n".join(raw[:-1]), good + "\n" + raw[0], good.replace("'CUDA0'", "'CPU'"),
               good.replace("ne=[128,2,1,1]", "ne=[128,3,1,1]"), good.replace("'123456789'", "'000000000'")]
    for mutant in mutants:
        try:
            validate(mutant, "perf", "123456789")
        except Gate:
            continue
        raise AssertionError("negative parser control passed")
    print("M: recorded-artifact parsers and negative controls passed")


def command(args, out, tag, deadline, env=None, allow_test_failure=False):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise Gate(7, "budget expired")
    (out / f"{tag}.command.json").write_text(json.dumps(args) + "\n")
    with (out / f"{tag}.sql").open("w") as stdout_file, (out / f"{tag}.err").open("w") as stderr_file:
        child = subprocess.Popen(args, env=env, stdout=stdout_file, stderr=stderr_file, text=True)
        (out / f"{tag}.pid").write_text(str(child.pid) + "\n")
        try:
            child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            raise Gate(7, "budget expired")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    (out / f"{tag}.returncode").write_text(str(child.returncode) + "\n")
    if child.returncode and not (allow_test_failure and child.returncode == 1):
        raise Gate(8, f"{tag} exited {child.returncode}; raw output saved")
    return (out / f"{tag}.sql").read_text(), (out / f"{tag}.err").read_text()


def kernel_guard(deadline, cutoff):
    left = deadline - time.monotonic()
    if left <= 0:
        raise Gate(7, "budget expired")
    result = subprocess.run(["journalctl", "-k", "-b", "--since", cutoff, "--no-pager"],
                            capture_output=True, text=True, timeout=min(10, left))
    if result.returncode:
        raise Gate(4, "kernel log unreadable")
    if HAZARD.search(result.stdout):
        raise Gate(3, "Xid since boot cutoff; stop GPU work")


def run(mode, out, budget):
    if mode != "preflight" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise Gate(4, "queue must provide CUDA visibility")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "SUMMARY.json").exists():
        raise Gate(6, "fresh output directory required")
    selftest()
    generate(out)
    digest, name = (HERE / "DRIVER-SHA256SUMS").read_text().split()
    if name != "codex/q8-probe2/bin/q8-probe-driver" or hashlib.sha256(DRIVER.read_bytes()).hexdigest() != digest:
        raise Gate(6, "probe driver bytes changed")
    commits = {arm: checksums(root, DRIVER) for arm, root in FROZEN.items()}
    if commits != COMMITS:
        raise Gate(6, "A/B source identity mismatch")
    if mode == "preflight":
        print("M: freeze identity, library resolution and case generation passed; no CUDA executable run")
        return
    deadline = time.monotonic() + budget
    cutoff = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    correctness = []
    outcome = {"status": "running", "commits": commits, "rows": rows, "correctness": correctness}
    # A kernel canary needs one paired A/B cycle to exercise the control-integrity gate.
    stages = [(0, "up"), (0, "patch")] if mode == "canary" else [(g, a) for g in (0, 1) for a in ("up", "patch", "patch", "up")]
    try:
        kernel_guard(deadline, cutoff)
        for index, (gpu, arm) in enumerate(stages):
            kernel_guard(deadline, cutoff)
            for poll in range(45):
                inventory, _ = command(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv"], out, f"inventory-{index}-{poll}", deadline)
                memory = gpu_memory(inventory)
                if gpu not in memory:
                    raise Gate(4, "selected GPU missing")
                if memory[gpu] <= 512:
                    break
                kernel_guard(deadline, cutoff)
                time.sleep(min(1, max(0, deadline - time.monotonic())))
            else:
                raise Gate(4, "GPU memory did not release; queue should wait")
            checksums(FROZEN[arm])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="8", GGML_CUDA_FA_MMA_Q8_0="1",
                       LD_LIBRARY_PATH=str(FROZEN[arm] / "bin"), Q8_PROBE_LIB_DIR=str(FROZEN[arm] / "bin"))
            env.pop("GGML_BACKEND_PATH", None)
            binary = DRIVER
            # Kernel suites use fresh processes; each arm's eval gate precedes its perf numbers.
            for kind in ("test", "controls", "perf"):
                kernel_guard(deadline, cutoff)
                tag = f"{index:02d}-gpu{gpu}-{arm}-{kind}"
                test_file = "controls.txt" if kind == "controls" else "cases.txt"
                test_mode = "test" if kind == "controls" else kind
                cmd = ["nice", "-n", "19", str(binary), test_mode, "-o", "FLASH_ATTN_EXT", "-b", "CUDA0", "--output", "sql", "--test-file", str(out / test_file)]
                stdout, stderr = command(cmd, out, tag, min(deadline, time.monotonic() + STAGE_S[kind]), env, allow_test_failure=test_mode == "test")
                check_device(stderr + stdout)
                parsed = validate(stdout, test_mode, commits[arm], controls=kind == "controls")
                expected_rc = int(test_mode == "test" and any(r["passed"] == "0" for r in parsed))
                if int((out / f"{tag}.returncode").read_text()) != expected_rc:
                    raise Gate(6, "driver exit code and case verdicts disagree")
                if kind == "perf":
                    rows.extend(dict(r, arm=arm, gpu=gpu, boot=index) for r in parsed)
                else:
                    correctness.extend(dict(r, arm=arm, gpu=gpu, boot=index, kind=kind) for r in parsed)
                    f16_integrity(correctness, require_pair=False)
                kernel_guard(deadline, cutoff)
            checksums(FROZEN[arm])
        f16_integrity(correctness)
        outcome["status"] = "ok"
    except Gate as exc:
        try:
            kernel_guard(deadline, cutoff)
        except Gate as guard:
            if guard.code == 3:
                exc = guard
        outcome.update(status="invalid", exit_code=exc.code, error=str(exc))
        raise exc
    finally:
        (out / "SUMMARY.json").write_text(json.dumps(outcome, indent=2) + "\n")
        if correctness:
            with (out / "correctness.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(correctness[0]))
                writer.writeheader()
                writer.writerows(correctness)
        if rows:
            with (out / "results.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        if outcome["status"] == "ok":
            lines = ["M: elapsed us/run; D: effective GB/s = unique KV bytes / elapsed time. Small cases can hit L2.", "", "| GPU | type | D | GQA | KV | up us | patch us | up GB/s | patch GB/s |", "|---|---|---|---|---|---|---|---|---|"]
            for gpu in sorted({r["gpu"] for r in rows}):
                for typ, kv, nb, gqa, d in sorted({(r["type"], r["kv"], r["nb"], r["gqa"], r["d"]) for r in rows}):
                    groups = {a: [r for r in rows if r["gpu"] == gpu and r["arm"] == a and (r["type"], r["kv"], r["nb"], r["gqa"], r["d"]) == (typ, kv, nb, gqa, d)] for a in FROZEN}
                    values = [f"{statistics.median(r[field] for r in groups[a]):.2f}" if groups[a] else "-" for field in ("us", "effective_gbs") for a in FROZEN]
                    lines.append(f"| {gpu} | {typ} | {d} | {gqa} | {kv} | " + " | ".join(values) + " |")
            (out / "results.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "canary", "run", "selftest", "gen"))
    parser.add_argument("--out", type=Path, default=ROOT / "codex/runs/q8-probe2/preflight")
    parser.add_argument("--budget-s", type=int, default=1200)
    args = parser.parse_args()
    if args.mode == "selftest":
        selftest()
    elif args.mode == "gen":
        generate(args.out)
    else:
        run(args.mode, args.out, args.budget_s)


if __name__ == "__main__":
    def stop(signum, frame):
        raise Gate(7, "queue stopped the item")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        main()
    except Gate as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(exc.code)
