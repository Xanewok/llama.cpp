"""Prepare local experiments; harness.mkitem owns the queue manifests."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parent.parent
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
    patch = (ROOT / "codex/frozen/host-trims/SOURCE").read_text().splitlines()[0]
    for np, quant in ((2, "iq4"), (2, "j5q"), (1, "j5q")):
        x = tomllib.loads((EVAL / "harness/experiments/duo-np2-depth.toml").read_text())
        reference = tomllib.loads((EVAL / "harness/experiments/stack6-depth-j5q.toml").read_text())
        name = f"host-trims-np{np}-{quant}"
        x.update(id=name, title=f"stack6 vs host-trims: np{np} {quant}, indexed seq_rm and bulk MTP verify hidden rows, ABBA; inherited ignore_eos target fallback",
                 out=str(ROOT / "codex/gpu-results" / name),
                 cpu_evidence={"path": str(ROOT / "codex/REPORT-host-trims.md"),
                               "command": "host-gate.sh: CPU and Meta(CPU,CPU), deterministic np1/np2 MTP identity and production mutants"},
                 extra_pins=[str(ROOT / "codex/gpu_ab.sh"), str(ROOT / "codex/host-parser-check.py")])
        x["model"] = copy.deepcopy(reference["model"] if quant == "j5q" else x["model"])
        x["arms"] = {}
        for arm, commit, root in (("up", BASE, Path.home() / "repos/qwen38-eval-worktrees/frozen/stack6"),
                                  ("patch", patch, ROOT / "codex/frozen/host-trims")):
            x["arms"][arm] = dict(label="stack6" if arm == "up" else "host-trims",
                                  commit=commit, root=str(root), repo=str(ROOT), bin="build/bin", exe="build/bin/round-trace",
                                  env={"GGML_SCHED_INPUT_STAGING": "1", "LLAMA_ARG_LOG_VERBOSITY": "4",
                                       "LD_LIBRARY_PATH": "{root}/build/bin"},
                                  flags=["--kv-pad", "4096"],
                                  canary_log=[r"copy-engine H2D streams per GPU: 2\b", r"n_kv_pad += 4096\b"])
            if arm == "patch":
                x["arms"][arm]["env"]["LLAMA_HOST_TRIMS_PROOF"] = "1"
                x["arms"][arm]["canary_log"] += [r"host_trims: kv_seq_rm_indexed engaged\b", r"host_trims: mtp_verify_bulk rows=[1-9]\d*\b"]
        args = x["server"]["args"]
        args[args.index("-ub") + 1] = "512"
        args.insert(args.index("-sm"), "{arm_flags}")
        x["server"]["knobs"] += ["LLAMA_ARG_LOG_VERBOSITY", "LLAMA_HOST_TRIMS_PROOF"]
        x["depth"]["streams"] = np
        if np == 1:
            x["depth"]["identity"] = "cross-arm"
            x["depth"].pop("identity_off_reason", None)
        else:
            x["depth"]["identity_off_reason"] = "Arrival-dependent np2 batches; exactness checked on deterministic CPU np2 batches, np1 GPU cross-arm identity is a separate item."
        x["depth"]["wall_s"] = 1100 if np == 2 else 500
        x["canary"].update(plan=["patch"], depths=[8192], proofs=["staging"])
        x["budget"].update(canary_s=600, margin=1.5, est_s=5000 if np == 2 else 2200)
        experiment = ROOT / f"codex/{name}.toml"
        experiment.write_text(dump(x))
        subprocess.run(["python3", "-m", "harness.mkitem", str(experiment), "--out", str(experiment.with_suffix(".gpuq.toml"))],
                       cwd=EVAL, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), check=True)


if __name__ == "__main__":
    main()
