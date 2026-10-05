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
FROZEN = {"up": ROOT / "codex/frozen/draft-attn-base", "patch": ROOT / "codex/frozen/draft-attn"}
DEPTHS = (8192, 32768, 90112)
CONTROLS = ((8, 1), (4, 2), (2, 4), (1, 8))
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


def case_name(typ, kv, nb=1, gqa=6):
    return f"draft-fa:{typ}:{kv}" + (f":q{nb}:g{gqa}" if (nb, gqa) != (1, 6) else "")


def case_line(typ, kv, nb=1, gqa=6):
    _, size, block = TYPES[typ]
    row = 256 * size // block
    scale = struct.unpack("i", struct.pack("f", 1 / 16))[0]
    sources = [tensor("f32", (256, nb, 2 * gqa, 1)),
               tensor(typ, (256, kv, 2, 1), (size, row, 2 * row * kv, 4 * row * kv)),
               tensor(typ, (256, kv, 2, 1), (size, row, 2 * row * kv, 4 * row * kv)),
               tensor("f16", (kv, nb, 1, 1))]
    return f"{fa_op()} 0 256 {2 * gqa} {nb} 1 4 {scale} 0 0 10 4 " + " ".join(sources) + " " + case_name(typ, kv, nb, gqa)


def generate(out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.txt").write_text("\n".join(case_line(t, k) for k in DEPTHS for t in TYPES) + "\n")
    (out / "controls.txt").write_text("\n".join(case_line("f16", 8192, nb, gqa) for nb, gqa in CONTROLS) + "\n")


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


def case_params(typ, kv, nb=1, gqa=6):
    _, size, block = TYPES[typ]
    row = 256 * size // block
    source = f"{typ}[256,{kv},2,1]nb[{size},{row},{2 * row * kv},{4 * row * kv}]"
    return (f"name={case_name(typ, kv, nb, gqa)},type=f32,ne=[256,{2 * gqa},{nb},1],op_params=[0:1031798784,3:10],"
            f"sources=f32[256,{nb},{2 * gqa},1],{source},{source},f16[{kv},{nb},1,1]")


def shape_key(params):
    match = re.match(r"name=draft-fa:(f16|q8_0):(\d+)(?::q(\d+):g(\d+))?,", params)
    if not match:
        raise Gate(6, "FA case does not match the requested geometry")
    typ, kv, nb, gqa = match.groups()
    kv, nb, gqa = int(kv), int(nb or 1), int(gqa or 6)
    if kv not in DEPTHS or params != case_params(typ, kv, nb, gqa):
        raise Gate(6, "FA source shape mismatch")
    return typ, kv, nb, gqa


def validate(raw, mode, commit, controls=False):
    rows = []
    for row in sql_rows(raw):
        if row["op"] != "FLASH_ATTN_EXT":
            continue
        typ, kv, nb, gqa = shape_key(row["params"])
        if row["mode"] != mode or row["backend"] != "CUDA0" or row["commit"] != commit[:9]:
            raise Gate(6, "backend, mode or frozen commit mismatch")
        if row["supported"] != "1" or row["passed"] != "1" or row["error"]:
            raise Gate(8, "CUDA-vs-CPU correctness or perf failed")
        row.update(type=typ, kv=kv, nb=nb, gqa=gqa)
        if mode == "perf":
            us = float(row["us"])
            if not math.isfinite(us) or us <= 0 or int(row["runs"]) < 2:
                raise Gate(5, "invalid perf sample")
            _, size, block = TYPES[typ]
            nbytes = kv * 2 * 256 * 2 * size / block
            row.update(us=us, kv_bytes=nbytes, effective_gbs=nbytes / us / 1000,
                       cache_note="fits 32 MiB L2" if nbytes <= 32 * 1024**2 else "exceeds 32 MiB L2")
        rows.append(row)
    want = {("f16", 8192, nb, gqa) for nb, gqa in CONTROLS} if controls else {(t, k, 1, 6) for t in TYPES for k in DEPTHS}
    keys = [(r["type"], r["kv"], r["nb"], r["gqa"]) for r in rows]
    if len(keys) != len(want) or set(keys) != want:
        raise Gate(5, "missing or duplicate FA cases")
    return rows


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
    raw = []
    for kv in DEPTHS:
        for typ in TYPES:
            params = case_params(typ, kv)
            assert shape_key(params) == (typ, kv, 1, 6)
            assert len(case_line(typ, kv).split()) == 49
            values = ["time", "123456789", "CUDA0", "FLASH_ATTN_EXT", params, "perf", "1", "1", "", "500", "0", "0", "0", "100", "", ""]
            raw.append("INSERT INTO test_backend_ops VALUES (" + ", ".join("'" + v + "'" for v in values) + ");")
    good = "\n".join(raw)
    assert len(validate(good, "perf", "123456789")) == 6
    mutants = [good.replace("'1', '1'", "'1', '0'"), good.replace("'500'", "'nan'"),
               "\n".join(raw[:-1]), good + "\n" + raw[0], good.replace("'CUDA0'", "'CPU'"),
               good.replace("ne=[256,12,1,1]", "ne=[256,13,1,1]"), good.replace("'123456789'", "'000000000'")]
    for mutant in mutants:
        try:
            validate(mutant, "perf", "123456789")
        except Gate:
            continue
        raise AssertionError("negative parser control passed")
    print("M: recorded-artifact parsers and negative controls passed")


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


def run(mode, out, budget):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "SUMMARY.json").exists():
        raise Gate(6, "fresh output directory required")
    selftest()
    generate(out)
    commits = {arm: checksums(root) for arm, root in FROZEN.items()}
    if not commits["up"].startswith("98c695111") or commits["up"] == commits["patch"]:
        raise Gate(6, "A/B source identity mismatch")
    if mode == "preflight":
        print("M: freeze identity, library resolution and case generation passed; no CUDA executable run")
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise Gate(4, "queue must provide CUDA visibility")
    deadline = time.monotonic() + budget
    cutoff = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    outcome = {"status": "running", "commits": commits, "rows": rows}
    stages = [(0, "patch")] if mode == "canary" else [(g, a) for g in (0, 1) for a in ("up", "patch", "patch", "up")]
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
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="8")
            binary = FROZEN[arm] / "bin/test-backend-ops"
            # Kernel suites use fresh processes; each arm's eval gate precedes its perf numbers.
            for kind in ("test", "controls", "perf"):
                kernel_guard(deadline, cutoff)
                tag = f"{index:02d}-gpu{gpu}-{arm}-{kind}"
                test_file = "controls.txt" if kind == "controls" else "cases.txt"
                test_mode = "test" if kind == "controls" else kind
                cmd = ["nice", "-n", "19", str(binary), test_mode, "-o", "FLASH_ATTN_EXT", "-b", "CUDA0", "--output", "sql", "--test-file", str(out / test_file)]
                stdout, stderr = command(cmd, out, tag, min(deadline, time.monotonic() + 75), env)
                check_device(stderr + stdout)
                parsed = validate(stdout, test_mode, commits[arm], controls=kind == "controls")
                if kind == "perf":
                    rows.extend(dict(r, arm=arm, gpu=gpu, boot=index) for r in parsed)
                kernel_guard(deadline, cutoff)
            checksums(FROZEN[arm])
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
        if rows:
            with (out / "results.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        if outcome["status"] == "ok":
            lines = ["M: elapsed us/run; D: effective GB/s = KV bytes / elapsed time. Small cases can hit L2.", "", "| GPU | type | KV | up us | patch us | up GB/s | patch GB/s |", "|---|---|---|---|---|---|---|"]
            for gpu in sorted({r["gpu"] for r in rows}):
                for typ in TYPES:
                    for kv in DEPTHS:
                        groups = {a: [r for r in rows if r["gpu"] == gpu and r["arm"] == a and r["type"] == typ and r["kv"] == kv] for a in FROZEN}
                        values = [f"{statistics.median(r[field] for r in groups[a]):.2f}" if groups[a] else "-" for field in ("us", "effective_gbs") for a in FROZEN]
                        lines.append(f"| {gpu} | {typ} | {kv} | " + " | ".join(values) + " |")
            (out / "results.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "canary", "run", "selftest", "gen"))
    parser.add_argument("--out", type=Path, default=ROOT / "codex/runs/draft-attn-preflight")
    parser.add_argument("--budget-s", type=int, default=600)
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
