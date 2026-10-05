#!/usr/bin/env python3
"""Frozen MMVQ A/B. GPU modes are for the queue; preflight is CPU-only."""
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
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FROZEN = {"up": Path("/home/xanewok/repos/q8-1row/codex/frozen/q8-1row-base"), "patch": ROOT / "codex/frozen/mmvq-w2r4-general"}
TYPES = {"q4_K": (12, 144, 256), "q5_K": (13, 176, 256), "q6_K": (14, 210, 256), "iq4_xs": (23, 136, 256), "q8_0": (8, 34, 32)}
FIELDS = ("stamp", "commit", "backend", "op", "params", "mode", "supported", "passed", "error", "us", "flops", "bw", "mem", "runs", "device", "reg")
HAZARD = re.compile(r"\bXid\b|fallen off the bus")


class Gate(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def sql_rows(raw):
    rows = []
    for line in raw.splitlines():
        if not line.startswith("INSERT INTO test_backend_ops"):
            continue
        body = line.split(" VALUES (", 1)[1].rsplit(");", 1)[0]
        values = re.findall(r"'((?:[^']|'')*)'", body)
        if len(values) != len(FIELDS):
            raise Gate(6, "SQL record field count changed")
        rows.append(dict(zip(FIELDS, (v.replace("''", "'") for v in values))))
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


def checksums(root):
    for base, manifest in ((root, root / "SHA256SUMS"), (root / "bin", root / "bin/SHA256SUMS")):
        for line in manifest.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            path = base / name.lstrip("*")
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise Gate(6, "frozen bytes changed")
    result = subprocess.run(["ldd", str(root / "bin/test-backend-ops")], text=True, capture_output=True, check=True)
    libs = re.findall(r"(libggml\S+) => (\S+)", result.stdout)
    if not libs or any(Path(p).resolve().parent != (root / "bin").resolve() for _, p in libs):
        raise Gate(6, "ggml libs resolve outside frozen dir")
    return (root / "SOURCE").read_text().splitlines()[0]


def command(args, out, tag, deadline, env=None):
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
    if child.returncode:
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


def mul_mat_op():
    enum = re.search(r"enum ggml_op\s*\{(.*?)\};", (ROOT / "ggml/include/ggml.h").read_text(), re.S).group(1)
    return re.findall(r"\bGGML_OP_\w+\b", enum).index("GGML_OP_MUL_MAT")


def geometry(typ, n, m, k):
    return f"type=f32,ne=[{m},{n},1,1],op_params=[],sources={typ}[{k},{m},1,1],f32[{k},{n},1,1]"


def case_params(typ, n, m, k):
    return f"name=mmvq-general:{typ}:n{n}:m{m}:k{k}," + geometry(typ, n, m, k)


def case_line(typ, n, m, k):
    tid, size, block = TYPES[typ]
    row = k * size // block
    sources = (f"{tid} {k} {m} 1 1 {size} {row} {row*m} {row*m}",
               f"0 {k} {n} 1 1 4 {k*4} {k*n*4} {k*n*4}")
    return f"{mul_mat_op()} 0 {m} {n} 1 1 0 2 " + " ".join(sources) + f" mmvq-general:{typ}:n{n}:m{m}:k{k}"


def cases(kind):
    if kind == "partial":
        return {(t, n, m, 512) for t in TYPES for n in (5, 6, 7, 8) for m in (5, 6, 7)}
    if kind == "aligned":
        return {(t, n, 8, 512) for t in TYPES for n in range(1, 9)}
    return {(t, n, 17408, 5120) for t in TYPES for n in range(1, 9)}


def generate(out):
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("partial", "aligned", "perf"):
        (out / f"{kind}.txt").write_text("\n".join(case_line(*c) for c in sorted(cases(kind))) + "\n")


def validate(raw, mode, commit, expected):
    want = {case_params(*c) for c in expected}
    rows = []
    for row in sql_rows(raw):
        if row["op"] != "MUL_MAT" or not row["params"].startswith("name=mmvq-general:"):
            continue
        if row["params"] not in want or row["mode"] != mode or row["backend"] != "CUDA0" or row["commit"] != commit[:9]:
            raise Gate(6, "backend, geometry, mode or frozen commit mismatch")
        if row["supported"] != "1" or row["passed"] != "1" or row["error"]:
            raise Gate(8, "CUDA-vs-CPU correctness or perf failed")
        if mode == "perf":
            try:
                us, runs = float(row["us"]), int(row["runs"])
            except ValueError as exc:
                raise Gate(5, "invalid perf sample") from exc
            if not math.isfinite(us) or us <= 0 or runs < 2:
                raise Gate(5, "invalid perf sample")
            row.update(us=us, runs=runs)
        rows.append(row)
    keys = [r["params"] for r in rows]
    if len(keys) != len(want) or set(keys) != want:
        raise Gate(5, "missing or duplicate MMVQ cases")
    return rows


def selftest():
    fixtures = HERE / "fixtures"
    raw = (fixtures / "real-mmvq.sql").read_text()
    recorded = sql_rows(raw)
    assert len(recorded) == 40
    # Exercise the same validator on recorded SQL, changing only the case-name prefix.
    expected = cases("perf")
    for typ in TYPES:
        for n in range(1, 9):
            raw = raw.replace(f"name=mmvq:gen:gateup:{typ}:n{n},", f"name=mmvq-general:{typ}:n{n}:m17408:k5120,")
    good = validate(raw, "perf", "94cceb8b8", expected)
    assert len(good) == 40 and good[0]["us"] == 148.242720
    check_device((fixtures / "real-mmvq.err").read_text())
    assert gpu_memory((fixtures / "real-memory.csv").read_text()) == {0: 108, 1: 15}
    assert HAZARD.search((fixtures / "real-kernel-errors.txt").read_text())
    assert not HAZARD.search("stop_xid_hash_s_G")
    mutants = [raw.replace("'1', '1'", "'1', '0'"), raw.replace("148.242720", "nan"),
               raw.replace("CUDA0", "CPU"), raw.replace("94cceb8b8", "000000000"),
               raw.replace("ne=[17408,1,1,1]", "ne=[17409,1,1,1]"),
               "\n".join(l for l in raw.splitlines() if l != next(r for r in raw.splitlines() if r.startswith("INSERT"))), raw + "\n" + next(l for l in raw.splitlines() if l.startswith("INSERT"))]
    for index, mutant in enumerate(mutants):
        try:
            validate(mutant, "perf", "94cceb8b8", expected)
        except Gate:
            continue
        raise AssertionError(f"negative parser control {index} passed")
    test_raw = raw.replace("perf", "test")
    assert len(validate(test_raw, "test", "94cceb8b8", expected)) == 40
    assert all(len(case_line(*c).split()) == 27 for c in cases("partial") | expected)
    print("M: recorded MMVQ SQL, device, memory, kernel-log parsers and negative controls passed")


def run(mode, out, budget, only):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "SUMMARY.json").exists():
        raise Gate(6, "fresh output directory required")
    selftest()
    generate(out)
    commits = {a: checksums(r) for a, r in FROZEN.items()}
    if not commits["up"].startswith("98c695111") or commits["patch"] == commits["up"]:
        raise Gate(6, "A/B source identity mismatch")
    if mode == "preflight":
        print("M: freeze identity and library resolution passed; no CUDA executable run")
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise Gate(4, "queue must provide CUDA visibility")
    deadline = time.monotonic() + budget
    cutoff = time.strftime("%Y-%m-%d %H:%M:%S")
    stages = [(0, "patch")] if mode == "canary" else [(g, a) for g in (0, 1) for a in ("up", "patch", "patch", "up")]
    rows, failures = [], []
    outcome = {"status": "running", "commits": commits, "rows": rows, "failures": failures}
    try:
        for index, (gpu, arm) in enumerate(stages):
            kernel_guard(deadline, cutoff)
            for poll in range(60):
                inventory, _ = command(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv"], out, f"memory-{index}-{poll}", deadline)
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
            kinds = ["aligned"] + (["partial"] if arm == "patch" else []) + ["perf"]
            if mode != "canary" and only != "all":
                kinds = [k for k in kinds if (k == "perf") == (only == "perf")]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="8")
            for kind in kinds:
                kernel_guard(deadline, cutoff)
                tag = f"{index:02d}-gpu{gpu}-{arm}-{kind}"
                mode_arg = "perf" if kind == "perf" else "test"
                args = ["nice", "-n", "19", "taskset", "-c", CPUSET, str(FROZEN[arm] / "bin/test-backend-ops"), mode_arg,
                        "-o", "MUL_MAT", "-b", "CUDA0", "--output", "sql", "--test-file", str(out / f"{kind}.txt")]
                try:
                    stdout, stderr = command(args, out, tag, min(deadline, time.monotonic() + 120), env)
                    check_device(stderr + stdout)
                    parsed = validate(stdout, mode_arg, commits[arm], cases(kind))
                    rows.extend(dict(r, arm=arm, gpu=gpu, boot=index, kind=kind) for r in parsed)
                except Gate as exc:
                    failures.append({"stage": tag, "code": exc.code, "error": str(exc)})
                    kernel_guard(deadline, cutoff)
                    if exc.code not in (5, 8):
                        raise
                kernel_guard(deadline, cutoff)
            checksums(FROZEN[arm])
            (out / "SUMMARY.json").write_text(json.dumps(outcome, indent=2) + "\n")
            if failures:
                raise Gate(failures[0]["code"], "boot gates failed; raw outputs saved")
        outcome["status"] = "ok"
        if mode == "run" and only != "correctness":
            comparison = []
            for gpu in (0, 1):
                for shape in sorted(cases("perf")):
                    values = {a: [r["us"] for r in rows if r["arm"] == a and r["gpu"] == gpu and r["kind"] == "perf" and r["params"] == case_params(*shape)] for a in FROZEN}
                    if any(len(v) != 2 for v in values.values()):
                        raise Gate(5, "paired perf stages missing")
                    up, patch = (statistics.median(values[a]) for a in ("up", "patch"))
                    comparison.append({"gpu": gpu, "shape": shape, "M_up_us": up, "M_patch_us": patch, "D_delta_pct": 100 * (patch / up - 1)})
            (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
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


CPUSET = ",".join(map(str, sorted(os.sched_getaffinity(0))[:8]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "canary", "run", "selftest", "gen"))
    parser.add_argument("--out", type=Path, default=ROOT / "codex/runs/w2r4-general-preflight")
    parser.add_argument("--budget-s", type=int, default=1800)
    parser.add_argument("--only", choices=("all", "correctness", "perf"), default="all")
    args = parser.parse_args()
    if args.mode == "selftest":
        selftest()
    elif args.mode == "gen":
        generate(args.out)
    else:
        run(args.mode, args.out, args.budget_s, args.only)


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
