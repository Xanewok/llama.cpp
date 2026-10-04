"""gpuq manifest + identity pins for gpu_swe_fresh.sh. Draft generator: prints TOML, submits nothing.

Conventions follow two-agent/codex/harness_support.py (digest, pin inventory, manifest keys); that
file is read by path, not edited, and is itself pinned.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HS_PATH = Path("/home/xanewok/repos/two-agent/codex/harness_support.py")
CODE = ["run_swe_fresh.py", "relay.py", "egress_scan.py", "gpu_swe_fresh.sh", "make_manifest.py",
        "preflight.sh", "test_swe_fresh.py", "stub_responses_server.py"]
# gpuq runs items with PATH=/usr/local/bin:/usr/bin:/bin, where `uvx` (Harbor) does not exist.
RUN_PATH = str(Path.home() / ".local/bin") + ":/usr/local/bin:/usr/bin:/bin"
# gpu_swe_fresh.sh exits: 3 kernel hazard, 4 busy GPU/port, 5 Docker/disk precondition, 6 identity, 7 budget.
EXIT_CODES = {"3": "hazard", "4": "transient", "5": "deterministic", "6": "integrity", "7": "timeout"}
MIN_FREE_GB = {"canary": 5, "measure": 30, "all": 30}  # the canary pulls one 1.2 GB image; a measure pass retains about 65 GB
# Wall estimate per task-run (seconds): image + container ~50, in-container Codex install ~130 (both measured
# on the stub proof, CPU only), verifier ~60; the agent part is the cap in the worst case.
SETUP_S = 240


def support():
    spec = importlib.util.spec_from_file_location("harness_support", HS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def engine_files(engine, hs):
    q, hq, sp = hs.Q, hs.HQ, hs.SP
    files = [q / "corpora/templates/swift-leading-instructions-v1.jinja", HS_PATH]
    if engine == "llama":
        frozen = hs.FROZEN
        files += [frozen / "SOURCE", *sorted(p for p in (frozen / "build/bin").iterdir() if p.is_file())]
        files += [Path.home() / "models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf",
                  Path.home() / "models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json"]
    else:
        files += [hq / "single-user/start_qwen.sh", sp / "stop_hq.sh", sp / "vllm-ar/build/libhsar_sm120.so"]
        files += sorted((sp / "vllm-both/overlay").rglob("*.py"))
        files += sorted((hq / "venv/lib/python3.12/site-packages/vllm").rglob("*.py"))
        files += sorted(p for p in (Path.home() / "models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-g32-attn-int8-codexh").iterdir()
                        if p.is_file())
    return files


def pin(args):
    hs = support()
    files = {*(HERE / name for name in CODE), args.tasks_file, *engine_files(args.engine, hs)}
    snapshot = Path(args.dataset_path)
    ids = [t["instance_id"] for t in json.loads(args.tasks_file.read_text())["tasks"]] + [args.canary_task]
    for task in ids:  # a task.toml/test.sh edit in the local snapshot must not slip into a pinned run
        found = [p for p in (snapshot / task).rglob("*") if p.is_file()]
        if not found:
            raise SystemExit(f"task missing from snapshot: {task}")
        files |= set(found)
    document = {"engine": args.engine, "tasks_file": str(args.tasks_file.resolve()),
                "dataset_path": str(snapshot.resolve()), "canary_task": args.canary_task,
                "files": {str(p.resolve()): hs.digest(p) for p in sorted(files)}}
    with args.pins.open("x") as stream:
        json.dump(document, stream, indent=1)
        stream.write("\n")


def verify(args):
    hs = support()
    document = json.loads(args.pins.read_text())
    for path, expected in document["files"].items():
        if hs.digest(Path(path)) != expected:
            raise SystemExit("pinned inputs changed: " + path)
    # New files must not appear silently in the code set or the engine inventory.
    current = {str(p.resolve()) for p in [*(HERE / n for n in CODE), *engine_files(document["engine"], hs)]}
    if not current <= set(document["files"]):
        raise SystemExit("pinned inputs changed (file inventory): " + ", ".join(sorted(current - set(document["files"]))[:3]))


def manifest(args):
    document = json.loads(args.pins.read_text())
    engine = document["engine"]
    tasks = json.loads(Path(document["tasks_file"]).read_text())["tasks"]
    n = 1 if args.stage == "canary" else len(tasks) * args.runs
    cap = args.cap_s
    boot = 900 if engine == "hq" else 180
    if args.stage == "canary":
        est = boot + SETUP_S + args.canary_cap_s + 60
        budget = boot + 1500 + 300
    else:
        # est assumes the average agent phase is half the cap (guess); budget assumes every run hits the cap.
        est = boot + n * (SETUP_S + cap // 2)
        budget = boot + n * (SETUP_S + cap) + 600
    env = {"ENGINE": engine, "PINS": str(args.pins.resolve()), "TASKS_FILE": document["tasks_file"],
           "CANARY_TASK": args.canary_task, "OUT": str(args.out.resolve()), "ONLY": args.stage,
           "BUDGET_S": str(budget - 300), "CAP_S": str(cap), "CANARY_CAP_S": str(args.canary_cap_s),
           "AGENTS": "1", "REPEAT": str(args.runs), "PROVIDER": "local", "DATASET_PATH": document["dataset_path"],
           "XID_SINCE": args.xid_since,
           "MIN_FREE_GB": str(args.min_free_gb if args.min_free_gb is not None else MIN_FREE_GB[args.stage]), "PATH": RUN_PATH}
    quoted = json.dumps
    lines = [f'id = "{args.item_id}"',
             f'title = "swe-fresh {args.stage}: Harbor + Codex on SWE-rebench 2026_03 ({engine}, 1 agent)"',
             f"budget_s = {budget}", f"est_s = {est}", "ab = false", 'retry = "fresh"',
             "exit_codes = { " + ", ".join(f"{quoted(k)} = {quoted(v)}" for k, v in EXIT_CODES.items()) + " }"]
    if engine == "hq":
        lines += ["hq = true", "startup_grace_s = 900"]
    lines += ['run = { cmd = ' + quoted(["bash", str(HERE / "gpu_swe_fresh.sh")]) + ', cwd = ' + quoted(str(HERE)) +
              ', env = { ' + ", ".join(f"{quoted(k)} = {quoted(v)}" for k, v in env.items()) + ' }, ports = [8081] }',
              'preflight = { cmd = ["bash", ' + quoted(str(HERE / "preflight.sh")) + '], env = { CUDA_VISIBLE_DEVICES = "" } }',
              'outputs = { paths = ' + quoted([str(args.out.resolve())]) + ' }',
              'pins = { files = ' + quoted(sorted({str(args.pins.resolve()), *document["files"]})) + ' }']
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pin")
    p.add_argument("--engine", choices=["llama", "hq"], required=True)
    p.add_argument("--tasks-file", type=Path, required=True)
    p.add_argument("--dataset-path", type=Path, required=True)
    p.add_argument("--canary-task", required=True)
    p.add_argument("--pins", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--pins", type=Path, required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--pins", type=Path, required=True)
    m.add_argument("--stage", choices=["canary", "measure", "all"], required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--xid-since", required=True)
    m.add_argument("--canary-task", required=True)
    m.add_argument("--id", dest="item_id", required=True)
    m.add_argument("--runs", type=int, default=1, help="repetitions per task inside one item")
    m.add_argument("--cap-s", type=int, default=1200)
    m.add_argument("--canary-cap-s", type=int, default=300)
    m.add_argument("--min-free-gb", type=int, default=None, help="default by stage: canary 5, otherwise 30")
    args = parser.parse_args()
    {"pin": pin, "verify": verify, "manifest": manifest}[args.command](args)


if __name__ == "__main__":
    main()
