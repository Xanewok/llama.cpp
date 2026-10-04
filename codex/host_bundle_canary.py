"""One-boot canary: compare patch to recorded requests from the exact frozen upstream bytes."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, "/home/xanewok/repos/qwen38-eval")
from harness import depth, lib
from harness.__main__ import main

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = ROOT / "codex/host-bundle.experiment.toml"
REFERENCE = Path("/home/xanewok/repos/stack3-rebase/codex/runs/stack3-rebase-cccl-ab/canary-a1/patch/0-patch")


def reference_check(x):
    pins = lib.read_json(REFERENCE.parent / "pins.json")["pins"]
    command = lib.read_json(REFERENCE / "command.json")
    old_root = Path(command[0]).parent
    up = x["arms"]["up"]
    lib.check_frozen(up)
    for path in up["bindir"].iterdir():
        if path.name == "round-trace" or ".so" in path.name:
            if pins.get(str(old_root / path.name)) != lib.file_sha(path):
                raise lib.Fail(f"frozen bytes changed: recorded upstream library {path.name} differs", "integrity")
    for name in ("model", "draft_vocab"):
        spec = x[name]
        if pins.get(str(spec["path"])) != spec["sha"]:
            raise lib.Fail(f"inputs changed: recorded {name} digest differs", "integrity")
    port = int(command[command.index("--port") + 1])
    expected_command = [str(up["exe"]), *lib.expand_args(x["server"]["args"], up, x, port)]
    if [str(up["exe"]), *command[1:]] != expected_command:
        raise lib.Fail("effective params mismatch: recorded upstream command differs", "integrity")
    identities = {}
    for n in x["depth"]["depths"]:
        record = lib.read_json(REFERENCE / f"depth-{n}.json")
        if record["request"] != depth.request_body(x, n):
            raise lib.Fail(f"canary prompt differs: recorded request at depth {n}", "integrity")
        lib.steady_intervals(record, x["depth"]["min_steady_intervals"], x["depth"]["k"])
        identities[n] = lib.token_identity(record)
    return identities


def run(argv):
    x = lib.read_experiment(EXPERIMENT)
    identities = reference_check(x)
    if argv == ["--reference-check"]:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        print("PASS M: recorded upstream frozen code, flags, model/vocab digests and seeded request identities")
        return 0
    else:
        if "--canary" not in argv or os.environ.get("GPUQ_ACTIVE") != "1":
            raise lib.Fail("one-boot canary requires gpuq and --canary", "deterministic")
        evaluate = depth.evaluate_boot
        def evaluate_reference(x, dest, arm, results, canary=False):
            if canary:
                for n, identity in identities.items():
                    results.setdefault(n, {"identity": identity, "runs": []})
            return evaluate(x, dest, arm, results, canary)
        depth.evaluate_boot = evaluate_reference
        return main(["depth", str(EXPERIMENT), *argv])


if __name__ == "__main__":
    sys.exit(lib.run_main(lambda: run(sys.argv[1:])))
