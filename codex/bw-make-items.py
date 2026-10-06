"""np2 IQ4 depth items for host-trims-v2 (build window); harness.mkitem owns the queue manifests."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parent.parent
FROZEN = Path.home() / "repos/qwen38-eval-worktrees/frozen/host-trims-v2"
EVAL = Path.home() / "repos/qwen38-eval"
BASE = "ac7465885dc9f13de396ccd91b4be5eaffde3730"


def value(item):
    if isinstance(item, bool):
        return str(item).lower()
    if isinstance(item, dict):
        return "{ " + ", ".join(f"{key} = {value(val)}" for key, val in item.items()) + " }"
    if isinstance(item, list):
        return "[" + ", ".join(map(value, item)) + "]"
    return json.dumps(item)


def dump(data):
    lines = []

    def table(items, prefix=""):
        if prefix:
            lines.extend(["", f"[{prefix}]"])
        for key, item in items.items():
            if not isinstance(item, dict):
                lines.append(f"{key} = {value(item)}")
        for key, item in items.items():
            if isinstance(item, dict):
                table(item, prefix + "." + key if prefix else key)

    table(data)
    return "\n".join(lines) + "\n"


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    fixed = (FROZEN / "SOURCE").read_text().splitlines()[0]
    stack6 = Path.home() / "repos/qwen38-eval-worktrees/frozen/stack6"
    items = Path(__file__).resolve().parent / "items"
    items.mkdir(exist_ok=True)
    gate = {"GGML_SCHED_INPUT_STAGING": "1", "LLAMA_ARG_LOG_VERBOSITY": "4", "LD_LIBRARY_PATH": "{root}/build/bin"}
    proofs = [r"host_trims: kv_seq_rm_indexed engaged\b", r"host_trims: mtp_verify_bulk rows=[1-9]\d*\b"]
    base_log = [r"copy-engine H2D streams per GPU: 2\b", r"n_kv_pad += 4096\b"]
    # "stack6": the arms of the original np2 item (stack6 vs the fixed build). "same": one binary, switches off vs default,
    # so a collapse that follows the schedule and not the code shows up in both arms.
    variants = {
        "stack6": dict(title="stack6 vs host-trims-v2 (fixed build): np2 iq4, ABBA", same=False),
        "same": dict(title="host-trims-v2 same binary: both switches off (stack6 path) vs default, np2 iq4, ABBA", same=True),
    }
    for tag, v in variants.items():
        x = tomllib.loads((EVAL / "harness/experiments/duo-np2-depth.toml").read_text())
        name = f"host-trims-v2-np2-iq4-{tag}"
        x.update(id=name, title=v["title"] + "; inherited ignore_eos target fallback",
                 out=str(ROOT / "codex/gpu-results" / name),
                 cpu_evidence={"path": str(ROOT / "codex/bw-results/summary.json"),
                               "command": "bw-host-trims-v2.sh: CPU and Meta(CPU,CPU) np2, per-stream counters, switch bisect, mutants"},
                 extra_pins=[str(ROOT / "codex/bw-host-trims-v2.sh"), str(ROOT / "codex/bw-np2-driver.cpp")])
        x["arms"] = {}
        for arm in ("up", "patch"):
            if v["same"]:
                root, commit = FROZEN, fixed
                env = dict(gate, LLAMA_HOST_TRIMS_SEQRM="0", LLAMA_HOST_TRIMS_BULKH="0") if arm == "up" else dict(gate, LLAMA_HOST_TRIMS_PROOF="1")
                label = "switches-off" if arm == "up" else "switches-default"
            else:
                root, commit = (stack6, BASE) if arm == "up" else (FROZEN, fixed)
                env = dict(gate) if arm == "up" else dict(gate, LLAMA_HOST_TRIMS_PROOF="1")
                label = "stack6" if arm == "up" else "host-trims-v2"
            x["arms"][arm] = dict(label=label, commit=commit, root=str(root), repo=str(ROOT), bin="build/bin",
                                  exe="build/bin/round-trace", env=env, flags=["--kv-pad", "4096"],
                                  canary_log=base_log + (proofs if arm == "patch" else []))
        args = x["server"]["args"]
        args[args.index("-ub") + 1] = "512"
        args.insert(args.index("-sm"), "{arm_flags}")
        x["server"]["knobs"] += ["LLAMA_ARG_LOG_VERBOSITY", "LLAMA_HOST_TRIMS_PROOF", "LLAMA_HOST_TRIMS_SEQRM", "LLAMA_HOST_TRIMS_BULKH"]
        x["depth"]["streams"] = 2
        if v["same"]:
            x["depth"].pop("distinct_builds", None)
        x["depth"]["identity_off_reason"] = "Arrival-dependent np2 batches; exactness is checked on the CPU np2 matrix of bw-host-trims-v2.sh."
        x["depth"]["wall_s"] = 1100
        x["canary"].update(plan=["patch"], depths=[8192], proofs=["staging"])
        x["budget"].update(canary_s=600, margin=1.5, est_s=5000)
        experiment = items / f"{name}.toml"
        experiment.write_text(dump(x))
        subprocess.run(["python3", "-m", "harness.mkitem", str(experiment), "--out", str(experiment.with_suffix(".gpuq.toml"))],
                       cwd=EVAL, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), check=True)
        print(f"item {experiment.with_suffix('.gpuq.toml')}")


if __name__ == "__main__":
    main()
