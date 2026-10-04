"""Read-only replay of every depth-harness parser on recorded GPU artifacts."""
import copy
import json
import os
from pathlib import Path
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, "/home/xanewok/repos/qwen38-eval")
from harness import depth, lib

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path("/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures")
RUN = Path("/home/xanewok/repos/stack3-rebase/codex/runs/stack3-rebase-cccl-ab/canary-a1/up/0-up")
OUT = ROOT / "codex/host-results/parsers"
OUT.mkdir(parents=True, exist_ok=True)
depth.fixture_test(FIXTURES)
results = []
for n in (8192, 32768, 90112):
    path = RUN / f"depth-{n}.json"
    record = json.loads(path.read_text())
    measured = lib.steady_intervals(record, 40, 5)
    identity = lib.token_identity(record)
    assert len(identity[0]) == measured["timings"]["predicted_n"]
    noisy = copy.deepcopy(record)
    noisy["chunks"].extend([{}, {"timings": None}, {"timings": {"draft_n": None}}, record["chunks"][-1]])
    assert lib.steady_intervals(noisy, 40, 5) == measured
    results.append({"artifact": str(path), "tokens_M": len(identity[0]), "steady_intervals_M": measured})
    missing = copy.deepcopy(record)
    for row in missing["chunks"]:
        row.pop("tokens", None)
    try:
        lib.token_identity(missing)
    except lib.Fail:
        pass
    else:
        raise AssertionError("missing token identity escaped")
counts = lib.trace_counts(RUN / "round-trace.jsonl")
staged = lib.check_staging(counts)
try:
    lib.check_staging({"input_fence_staged": 0, "input_fence_sync": 1})
except lib.Fail:
    pass
else:
    raise AssertionError("disabled staging escaped")
assert lib.record_journal(0, OUT / "recorded-journal.txt", FIXTURES / "xid-lines.txt") > 0
sys.path.insert(0, str(ROOT / "codex"))
from host_bundle_canary import reference_check, REFERENCE
x = lib.read_experiment(ROOT / "codex/host-bundle.experiment.toml")
identities = reference_check(x)
native = OUT / "native-identity"
native.mkdir(exist_ok=True)
records = {}
for n in x["depth"]["depths"]:
    records[n] = lib.read_json(REFERENCE / f"depth-{n}.json")
    (native / f"depth-{n}.json").write_text(json.dumps(records[n]))
def seeded():
    return {n: {"identity": value, "runs": []} for n, value in identities.items()}
assert not depth.evaluate_boot(x, native, "patch", seeded())
n = x["depth"]["depths"][0]
changed = copy.deepcopy(records[n])
next(row for row in changed["chunks"] if row.get("tokens"))["tokens"][0] ^= 1
(native / f"depth-{n}.json").write_text(json.dumps(changed))
failures = depth.evaluate_boot(x, native, "patch", seeded())
assert any("cross-arm identity failed" in str(f) for f in failures)
changed = copy.deepcopy(records[n])
final = next(row for row in reversed(changed["chunks"]) if row.get("stop") is True)
final["timings"]["draft_n_accepted"] += 1
(native / f"depth-{n}.json").write_text(json.dumps(changed))
failures = depth.evaluate_boot(x, native, "patch", seeded())
assert any("cross-arm identity failed" in str(f) for f in failures)
(native / f"depth-{n}.json").write_text(json.dumps(records[n]))
summary = {"real_native_records": results, "staging_M": staged,
           "negative_controls": "PASS", "source": str(RUN)}
(OUT / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
