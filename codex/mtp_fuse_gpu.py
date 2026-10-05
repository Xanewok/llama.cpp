"""Depth adapter for mtp-fuse-depth-j5q.

Gates: within-arm token identity; per boot, every complete round has 5 draft-side forwards in the patch arm (V,D1..D5,
138 AllReduce calls) and 6 in the base arm (V,C,D1..D5, 140).
Report-only: cross-arm acceptance; the acceptance-matched ms/round delta (primary result: the arms' tokens diverge, and
round cost rises with the accepted count); CUDA graph capture/key churn per round; minimum free VRAM per GPU per boot."""
import copy
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, "/home/xanewok/repos/qwen38-eval")
from harness import depth, lib
from harness.__main__ import main

EXPERIMENT = Path(__file__).with_name("mtp-fuse-depth-j5q.experiment.toml")
FIXTURE = Path("/home/xanewok/repos/qwen38-eval/results/2026-10-05-stack4-depth-j5q/canary-a1/0-up")
# Real fused boot (GPU canary T56): each request opens with a prompt-tail round V(4),C,D1(1 row)..D5, then fused rounds.
FUSED_FIXTURE = Path("/home/xanewok/repos/qwen38-eval/results/2026-10-05-mtp-fuse-depth-j5q/canary-a1/1-patch")
EVALUATE, FINISH = depth.evaluate_boot, depth.finish
FUSED_ARMS = ("patch",)
AR_PER_ROUND = {True: 138, False: 140}
CUDA_KINDS = ("cuda_capture", "cuda_instantiate", "cuda_changed", "cuda_warmup_reset", "graph_miss", "graph_stale_alloc")
MIN_BUCKET = 10
SMI_MS = 1000
# Per depth: rounds[arm] = [(ms, n_acc)], acceptance[arm], cuda[arm] = [per-boot stats], d1_vs_record[arm]; "vram": {arm: [...]}.
EXTRA = {}


def round_shape(fused):
    return ["V"] + ([] if fused else ["C"]) + [f"D{i}" for i in range(1, 6)]


def in_window(row, window):
    phases = row.get("phases", [])
    return bool(phases) and window[0] <= phases[0].get("begin_ns", -1) <= window[1]


def forward_proof(rows, fused, minimum, window):
    """Every complete k=5 round in the window has the arm's phase shape and AllReduce count; once drafting has begun, a
    fused arm never runs C before D1."""
    shape, complete, drafting = round_shape(fused), 0, False
    for row in rows:
        if not in_window(row, window):
            continue
        phases = row["phases"]
        names = [p.get("phase") for p in phases]
        # Exempt: prompt ubatches (no draft yet; the last one's C precedes the request's first D1, which has nothing
        # deferred) and the release flush (a C with no D1 after the last verify).
        if fused and drafting and "C" in names and "D1" in names:
            raise lib.Fail(f"forward-count proof failed: catch-up phase C in a fused round {names}", "integrity")
        drafting = drafting or "D1" in names
        if names != shape or phases[0].get("n_tokens") != 6 or any(p.get("result", -1) for p in phases):
            continue
        count = sum(c["count"] for p in phases for c in p.get("counters", []) if c["kind"] == "ar_call")
        if count != AR_PER_ROUND[fused]:
            raise lib.Fail(f"forward-count proof failed: {count} AllReduce calls, expected {AR_PER_ROUND[fused]}", "integrity")
        complete += 1
    if complete < minimum:
        raise lib.Fail(f"forward-count proof inconclusive: {complete} complete rounds < {minimum}", "insufficient_data")
    return {"complete_rounds_M": complete, "draft_forwards_per_round_M": 5 if fused else 6, "ar_calls_per_round_M": AR_PER_ROUND[fused]}


def read_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def round_costs(record, k):
    """(ms, n_acc) of the steady full-K interior rounds, on exactly lib.steady_intervals' boundaries: n_acc is the
    draft_n_accepted step between two boundaries; each round must emit n_acc + 1 tokens."""
    chunks = [c["timings"] for c in record["chunks"] if isinstance(c.get("timings"), dict) and
              all(isinstance(c["timings"].get(key), (int, float)) for key in ("predicted_ms", "predicted_n", "draft_n", "draft_n_accepted"))]
    final, points = chunks[-1], {}
    for t in chunks:
        n = t["draft_n"]
        if n > 0 and n % k == 0 and t["predicted_n"] <= final["predicted_n"] - (k + 1):
            points[n // k] = (t["predicted_ms"], t["draft_n_accepted"], t["predicted_n"])
    keys, out = sorted(points), []
    for a, b in zip(keys, keys[1:]):
        if a >= 3 and b == a + 1 and points[b][0] > points[a][0]:
            acc, emitted = points[b][1] - points[a][1], points[b][2] - points[a][2]
            # The last boundary may stop short of its round's final token (the predicted_n cut above).
            if not 0 <= acc <= k or (emitted != acc + 1 if b != keys[-1] else not 1 <= emitted <= acc + 1):
                raise lib.Fail(f"per-round acceptance unreadable: {acc} accepted, {emitted} emitted", "deterministic")
            out.append((points[b][0] - points[a][0], acc))
    return out


def ols(rounds):
    xs, ys = [a for _, a in rounds], [m for m, _ in rounds]
    if len(rounds) < 2 or len(set(xs)) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return {"a_ms_D": my - slope * mx, "b_ms_per_acc_D": slope, "n_M": len(rounds)}


def matched(up, patch, k=5):
    """Bucket-weighted patch - up ms/round over the n_acc buckets both arms reached, weights pooled over both arms."""
    buckets, common = {}, []
    for b in range(k + 1):
        u, p = [m for m, a in up if a == b], [m for m, a in patch if a == b]
        buckets[b] = {"up_n_M": len(u), "patch_n_M": len(p), "up_ms_D": statistics.mean(u) if u else None,
                      "patch_ms_D": statistics.mean(p) if p else None, "under_min": min(len(u), len(p)) < MIN_BUCKET}
        if u and p:
            common.append(b)

    def weighted(sel):
        total = sum(buckets[b]["up_n_M"] + buckets[b]["patch_n_M"] for b in sel)
        if not total:
            return None
        return sum((buckets[b]["up_n_M"] + buckets[b]["patch_n_M"]) / total * (buckets[b]["patch_ms_D"] - buckets[b]["up_ms_D"]) for b in sel)

    pooled = len(up) + len(patch)
    return {"primary_matched_patch_minus_up_ms_D": weighted(common),
            "matched_min10_patch_minus_up_ms_D": weighted([b for b in common if not buckets[b]["under_min"]]),
            "matched_coverage_D": sum(buckets[b]["up_n_M"] + buckets[b]["patch_n_M"] for b in common) / pooled if pooled else None,
            "plain_patch_minus_up_ms_D": statistics.mean(m for m, _ in patch) - statistics.mean(m for m, _ in up) if up and patch else None,
            "buckets": buckets, "ols": {"up": ols(up), "patch": ols(patch)}}


def cuda_stats(rows, window):
    """Per round inside the window: counter sums per device, distinct CUDA graph keys per device (a cuda_key event's
    bytes field is the key); distinct keys over the whole window."""
    n, per, keys_round, keys_all = 0, {}, {}, {}
    for row in rows:
        if not in_window(row, window):
            continue
        n += 1
        for p in row["phases"]:
            for c in p.get("counters", []):
                if c["kind"] in CUDA_KINDS:
                    k = f"{c['kind']}@{c.get('device', -1)}"
                    per[k] = per.get(k, 0) + c["count"]
        seen = {}
        for e in row.get("events", []):
            if e.get("kind") == "cuda_key":
                seen.setdefault(e.get("device", -1), set()).add(e.get("bytes"))
        for dev, keys in seen.items():
            keys_round.setdefault(dev, []).append(len(keys))
            keys_all.setdefault(dev, set()).update(keys)
    if not n:
        return {"rounds_M": 0}
    return {"rounds_M": n, "per_round_D": {k: v / n for k, v in sorted(per.items())},
            "distinct_keys_per_round_D": {str(d): statistics.mean(v) for d, v in sorted(keys_round.items())},
            "distinct_keys_window_M": {str(d): len(v) for d, v in sorted(keys_all.items())}}


def d1_hist(rows, window):
    """Fused arm: n_acc = D1 rows - 2 per complete round, set against the record's histogram (report-only)."""
    out = {}
    for row in rows:
        if in_window(row, window) and [p.get("phase") for p in row["phases"]] == round_shape(True):
            a = row["phases"][1]["n_tokens"] - 2
            out[a] = out.get(a, 0) + 1
    return dict(sorted(out.items()))


def hist(rounds):
    out = {}
    for _, a in rounds:
        out[a] = out.get(a, 0) + 1
    return dict(sorted(out.items()))


def analysis(x):
    out = {}
    for n, e in sorted((n, e) for n, e in EXTRA.items() if isinstance(n, int)):
        r = e.get("rounds", {})
        out[n] = {"acceptance_report_M": e.get("acceptance", {}), "cuda_report": e.get("cuda", {}), "d1_vs_record": e.get("d1_vs_record", {})}
        if r.get("up") and r.get("patch"):
            out[n]["matched"] = matched(r["up"], r["patch"], x["depth"]["k"])
    out["vram_min_free_mib_M"] = EXTRA.get("vram", {})
    return out


def evaluate(x, dest, arm, results, canary=False):
    local = copy.deepcopy(x)
    local["depth"]["identity"] = "within-arm"
    failures = EVALUATE(local, dest, arm, results, canary)
    d, fused, proofs = x["depth"], arm in FUSED_ARMS, {}
    try:
        rows = read_rows(dest / "round-trace.jsonl")
    except OSError as error:
        return failures + [lib.Fail(f"unreadable round trace: {error}", "insufficient_data")]
    for n in depth.depths_of(x, canary):
        try:
            record = lib.read_json(dest / f"depth-{n}.json")
            window = record["mono_ns"]
            proofs[n] = forward_proof(rows, fused, d["min_steady_intervals"], window)
            e = EXTRA.setdefault(n, {})
            rounds = round_costs(record, d["k"])
            e.setdefault("rounds", {}).setdefault(arm, []).extend(rounds)
            t = lib.token_identity(record)[1]
            e.setdefault("acceptance", {})[arm] = t["draft_n_accepted"] / t["draft_n"] if t["draft_n"] else None
            stats = cuda_stats(rows, window)
            e.setdefault("cuda", {}).setdefault(arm, []).append(stats)
            if fused:
                e.setdefault("d1_vs_record", {})[arm] = {"trace_d1": d1_hist(rows, window), "record": hist(rounds)}
            proofs[n] |= {"rounds_by_n_acc_M": hist(rounds), "cuda": stats}
        except lib.Fail as error:
            failures.append(lib.Fail(f"depth {n}: {error}", error.cls))
        except (OSError, ValueError, KeyError) as error:
            failures.append(lib.Fail(f"depth {n}: unreadable proof input: {error}", "deterministic"))
    try:
        vram = lib.min_free_vram(dest / "smi.csv", x["devices"])
    except lib.Fail as error:
        vram = {"missing": str(error)}
    EXTRA.setdefault("vram", {}).setdefault(arm, []).append(vram)
    proofs["vram_min_free_mib_M"] = vram
    (dest / "forward-counts.json").write_text(json.dumps(proofs, indent=2))
    (dest.parent / "matched-so-far.json").write_text(json.dumps(analysis(x), indent=2))
    return failures


def finish(x, results):
    out = FINISH(x, results)
    report = analysis(x)
    for n, entry in out.items():
        entry.update(report.get(n, {}))
    out["vram_min_free_mib_M"] = report["vram_min_free_mib_M"]
    return out


class SmiServer(lib.Server):
    """Free-VRAM sampler beside each boot (the harness samples only for streams > 1); same cadence in both arms."""

    def start(self):
        pid = super().start()
        self.smi = lib.start_smi(self.log.parent / "smi.csv", interval_ms=SMI_MS)
        return pid

    def stop(self, grace=60):
        super().stop(grace)
        lib.stop_child(getattr(self, "smi", None), 5)


def expect_fail(what, fn, *args):
    try:
        fn(*args)
    except lib.Fail:
        return
    raise AssertionError(f"negative control escaped: {what}")


def close(a, b, tol=1e-9):
    return a is not None and abs(a - b) <= tol


def fixtures():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    rows = read_rows(FIXTURE / "round-trace.jsonl")
    out = {}
    for n in (8192, 32768):
        record = lib.read_json(FIXTURE / f"depth-{n}.json")
        window = record["mono_ns"]
        proof = forward_proof(rows, False, 40, window)
        fused = copy.deepcopy(rows)
        for row in fused:
            row["phases"] = [p for p in row.get("phases", []) if p.get("phase") != "C"]
        assert forward_proof(fused, True, 40, window)["complete_rounds_M"] == proof["complete_rounds_M"]
        expect_fail("eager trace as fused", forward_proof, rows, True, 40, window)
        expect_fail("fused trace as eager", forward_proof, fused, False, 40, window)
        extra = copy.deepcopy(fused)
        for row in extra:
            for p in row["phases"]:
                if p.get("phase") == "D1":
                    p["counters"] = p.get("counters", []) + [{"kind": "ar_call", "count": 2}]
        expect_fail("fused shape with a sixth forward's AllReduces", forward_proof, extra, True, 40, window)
        inside = [i for i, r in enumerate(fused) if in_window(r, window) and [p.get("phase") for p in r["phases"]] == round_shape(True)]
        flushed = copy.deepcopy(fused)
        flushed[inside[-1]]["phases"] = flushed[inside[-1]]["phases"][:1] + [dict(rows[inside[-1]]["phases"][1], counters=[])]
        forward_proof(flushed, True, 40, window)
        catchup = copy.deepcopy(fused)
        mid = inside[len(inside) // 2]
        catchup[mid]["phases"].insert(1, copy.deepcopy(rows[mid]["phases"][1]))
        expect_fail("one catch-up in a fused trace", forward_proof, catchup, True, 40, window)
        expect_fail("window without rounds", forward_proof, rows, False, 40, (0, 1))

        rounds = round_costs(record, 5)
        steady = lib.steady_intervals(record, 40, 5)
        assert len(rounds) == steady["intervals_M"] and close(statistics.mean(m for m, _ in rounds), steady["steady_ms_per_round_D"], 1e-6)
        broken = copy.deepcopy(record)
        for c in broken["chunks"]:
            t = c.get("timings")
            if isinstance(t, dict) and isinstance(t.get("predicted_n"), int) and isinstance(t.get("draft_n"), int) and t["draft_n"] >= 50:
                t["predicted_n"] += 1
        expect_fail("round emitting n_acc + 2 tokens", round_costs, broken, 5)

        stats = cuda_stats(rows, window)
        injected = copy.deepcopy(rows)
        for row in injected:
            if in_window(row, window):
                row["phases"][0]["counters"] += [{"kind": "cuda_capture", "device": 0, "count": 1}, {"kind": "cuda_capture", "device": 1, "count": 1}]
        more = cuda_stats(injected, window)
        for dev in (0, 1):
            k = f"cuda_capture@{dev}"
            assert close(more["per_round_D"][k], stats["per_round_D"].get(k, 0) + 1)
        frows = read_rows(FUSED_FIXTURE / "round-trace.jsonl")
        fwindow = lib.read_json(FUSED_FIXTURE / f"depth-{n}.json")["mono_ns"]
        fproof = forward_proof(frows, True, 40, fwindow)
        expect_fail("real fused boot as eager", forward_proof, frows, False, 40, fwindow)
        finside = [i for i, r in enumerate(frows) if in_window(r, fwindow) and [p.get("phase") for p in r["phases"]] == round_shape(True)]
        fbad = copy.deepcopy(frows)
        fbad[finside[len(finside) // 2]]["phases"].insert(1, {"phase": "C", "n_tokens": 6, "result": 0, "counters": []})
        expect_fail("catch-up inside a real fused request", forward_proof, fbad, True, 40, fwindow)
        frounds = round_costs(lib.read_json(FUSED_FIXTURE / f"depth-{n}.json"), 5)
        out[n] = {"eager_proof": proof, "fused_proof": fproof, "fused_rounds_by_n_acc_M": hist(frounds), "rounds_by_n_acc_M": hist(rounds), "cuda": stats}

    # Same cost model in both arms with the patch accepting less: the plain mean shows a gain, the matched delta must be 0;
    # a real -0.9 ms shift must come back as -0.9.
    cost = lambda a: 40 + 0.3 * a
    up = [(cost(a), a) for a in [5] * 60 + [0] * 20 + [3] * 20]
    lower = [(cost(a), a) for a in [5] * 20 + [0] * 60 + [3] * 20]
    m0 = matched(up, lower)
    assert m0["plain_patch_minus_up_ms_D"] < -0.4 and close(m0["primary_matched_patch_minus_up_ms_D"], 0)
    m1 = matched(up, [(ms - 0.9, a) for ms, a in lower])
    assert close(m1["primary_matched_patch_minus_up_ms_D"], -0.9) and close(m1["ols"]["up"]["b_ms_per_acc_D"], 0.3)
    m2 = matched(up, [(cost(a), a) for a in [5] * 5 + [0] * 95])
    assert m2["buckets"][5]["under_min"] and not m2["buckets"][0]["under_min"] and close(m2["matched_min10_patch_minus_up_ms_D"], 0)
    print(json.dumps({"recorded_artifact": str(FIXTURE), "depths": out,
                      "matched_controls": {"lower_acceptance_plain_D": m0["plain_patch_minus_up_ms_D"],
                                           "lower_acceptance_matched_D": m0["primary_matched_patch_minus_up_ms_D"],
                                           "shift_minus_0.9_matched_D": m1["primary_matched_patch_minus_up_ms_D"]},
                      "negative_controls": "PASS"}, indent=2))
    return 0


def run(argv):
    if argv == ["--fixtures"]:
        return fixtures()
    if "--preflight" in argv:
        fixtures()
    if any(a in argv for a in ("--execute", "--canary")) and os.environ.get("GPUQ_ACTIVE") != "1":
        raise lib.Fail("GPU execution requires gpuq", "deterministic")
    depth.evaluate_boot, depth.finish, lib.Server = evaluate, finish, SmiServer
    return main(["depth", str(EXPERIMENT), *argv])


if __name__ == "__main__":
    sys.exit(lib.run_main(lambda: run(sys.argv[1:])))
