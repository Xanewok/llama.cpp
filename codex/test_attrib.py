#!/usr/bin/env python3
import copy
import json
import sqlite3
import tempfile
from pathlib import Path

from attrib import analyze, union_ns, validate, wait_histograms


def fixture():
    rows = []
    for rnd in range(1, 7):
        base = rnd * 5000
        phases = [{"phase": name, "n_tokens": 4 if i < 2 else 1, "result": 0,
                   "begin_ns": base + i * 1000, "end_ns": base + i * 1000 + 600}
                  for i, name in enumerate(("V", "C", "D1", "D2", "D3"))]
        events = []
        for i, p in enumerate(phases):
            start = p["begin_ns"]
            for kind, a, b, device, count, size in (
                ("input", 100, 200, 0, 2, 64), ("ar_wait", 300, 400, 0, 1, 0),
                ("sampling", 610, 700, -1, 1, 0), ("sampling", 630, 650, -1, 1, 0),
                ("graph_hit", 0, 0, -1, 1, 0)):
                events.append(dict(kind=kind, phase=i, device=device, count=count, bytes=size,
                                   begin_ns=start + a, end_ns=start + b))
            counters = []
            for kind in {e["kind"] for e in events if e["phase"] == i}:
                es = [e for e in events if e["phase"] == i and e["kind"] == kind]
                counters.append(dict(kind=kind, device=es[0]["device"], count=sum(e["count"] for e in es),
                                     bytes=sum(e["bytes"] for e in es),
                                     host_us=union_ns([(e["begin_ns"], e["end_ns"]) for e in es]) / 1000))
            p["counters"] = counters
        rows.append(dict(schema=1, pid=123, tid=125, round=rnd, next_begin_ns=base + 5000,
                         phases=phases, events=events))
    return rows


def reject(fn, label):
    try:
        fn()
    except (ValueError, KeyError):
        return
    raise AssertionError("negative control escaped: " + label)


def make_sqlite(path, rows):
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER,globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER,correlationId INTEGER,globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,correlationId INTEGER,globalPid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER,correlationId INTEGER,globalPid INTEGER);
    """)
    db.execute("INSERT INTO StringIds VALUES (1,'cudaGraphLaunch')")
    offset = 10_000_000
    tag = 1 << 48
    pid = tag | (123 << 24)
    tid = pid | 125
    other_pid = tag | (456 << 24)
    for row in rows:
        for i, p in enumerate(row["phases"]):
            a, b = p["begin_ns"] + offset, p["end_ns"] + offset
            label = f"round_trace:125:{row['round']}:{p['phase']}:{i}"
            db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)", (a, b, label, None, tid))
            for device in (0, 1):
                cid = row["round"] * 100 + i * 2 + device
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)", (a + 250, a + 251, 1, cid, tid))
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)", (a + 260, b - 10 + device * 5, device, cid, pid))
                # This process deliberately reuses the CUDA correlation ID.
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)", (a + 250, a + 251, 1, cid, other_pid | 457))
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)", (a + 260, b + 50_000, device, cid, other_pid))
    db.commit()
    db.close()


def main():
    rows = fixture()
    validate(rows)
    result = analyze(rows, skip=0, min_rounds=5)
    assert result["rounds_M"] == 5
    hist = result["ar_wait_histogram"]["by_phase_device"]
    assert len(hist) == 5 and all(h["bins_M"] == [5, 0, 0, 0, 0] for h in hist)
    edges = [0, 999, 1000, 9999, 10000, 99999, 100000, 999999, 1000000]
    synthetic = [{"phases": [{"phase": "V"}], "events": [
        dict(kind="ar_wait", phase=0, device=1, count=1, begin_ns=0, end_ns=t) for t in edges]}]
    edge_hist = wait_histograms(synthetic)["by_phase_device"][0]
    assert edge_hist["bins_M"] == [2, 2, 2, 2, 1] and edge_hist["p95_wait_us_D"] == 1000
    synthetic[0]["events"][0]["count"] = 2
    reject(lambda: wait_histograms(synthetic), "aggregated wait hid histogram samples")
    call = next(r for r in result["tables"] if r["window"] == "decode_call" and r["phase"] == "V")
    gap = next(r for r in result["tables"] if r["window"] == "host_boundary" and r["phase"] == "V->C")
    assert abs(call["inputs"] - 0.1) < 1e-9 and abs(call["ar_wait"] - 0.1) < 1e-9
    assert abs(gap["sampling"] - 0.09) < 1e-9, "nested sampling was summed"
    controls = 0
    for label, mutate in (
        ("reversed decode", lambda r: r[0]["phases"][0].update(end_ns=0)),
        ("duplicate round", lambda r: r[1].update(round=1)),
        ("wrong counter", lambda r: r[0]["phases"][0]["counters"][0].update(count=99)),
        ("wrong bytes", lambda r: r[0]["events"][0].update(bytes=-1)),
        ("event crosses phase", lambda r: r[0]["events"][0].update(end_ns=99999)),
        ("double-count sampling", lambda r: next(c for c in r[0]["phases"][0]["counters"] if c["kind"] == "sampling").update(host_us=0.11)),
        ("wrong phase owner", lambda r: r[0]["events"][0].update(phase=99)),
    ):
        broken = copy.deepcopy(rows)
        mutate(broken)
        reject(lambda: validate(broken), label)
        controls += 1
    reject(lambda: analyze(rows, skip=0, min_rounds=6), "missing steady rounds")
    controls += 1
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        path = Path(directory) / "synthetic.sqlite"
        make_sqlite(path, rows)
        result = analyze(rows, skip=0, min_rounds=5, sqlite=path)
        assert result["nsys"]["gap_samples_M"] == 25, "process-scoped correlation failed"
        gpu = next(r for r in result["tables"] if r["window"] == "gpu_kernel_boundary" and r["phase"] == "D3->V")
        assert abs(gpu["inputs"] - 0.1) < 1e-9, "next-round upload omitted"
        assert abs(gpu["total_us"] - 0.665) < 1e-9
        db = sqlite3.connect(path)
        db.execute("UPDATE NVTX_EVENTS SET text=replace(text,':V:',':D9:')")
        db.commit()
        db.close()
        reject(lambda: analyze(rows, skip=0, sqlite=path), "NVTX phase mismatch")
        controls += 1
    # The controls fail if interval union or process ownership is deliberately broken.
    print(f"M: JSONL/accounting/SQLite tests PASS; {controls} rejection controls; nested-scope and correlation-collision controls PASS")


if __name__ == "__main__":
    main()
