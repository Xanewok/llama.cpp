#!/usr/bin/env python3
"""Write the two gpuq items (one per model) into qwen38-eval/orchestration/gpuq/items/. Re-run after any change to a pinned input; pins are hashed by gpuq at submit.
Usage: gen_item.py [--check]   (--check exits 1 when a written item is stale)"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("AB_ITEM", "iq4")
from ab_common import ARMS, ASSETS, EVAL, FIT_TASK, HERE, IDS, MEMOC_GUARD, MODELS, RUNNER, TASKS, XID_RE, XID_SINCE  # noqa: E402

ITEMS = EVAL / "orchestration/gpuq/items"
OUT = HERE / "runs"
PLAN = "ABBA"
# Slowest measured campaign wall per model: IQ4 = knob-real a2 arm A (1557, 1438); J5q = weekend S5 (1605, 1462).
WALL_S = {"iq4": 1557, "j5q": 1605}
MARGIN = 1.25
CLEANUP_S = 150
CANARY_TIMEOUT_S = 900  # gpuq maximum; the canary self-caps at 850 s
CANARY_MEASURED_S = 600  # knob-real's one-boot canary took 294 s including a 300 s task; here two boots plus a 33.8k-token request each


def budget(key):
    run = math.ceil((len(PLAN) * WALL_S[key] * MARGIN + CLEANUP_S) / 50) * 50
    return {"run_s": run, "budget_s": run + CANARY_TIMEOUT_S + CLEANUP_S, "est_s": len(PLAN) * WALL_S[key] + CANARY_MEASURED_S}


def imported_modules():
    code = ("import sys; sys.path.insert(0, %r); import runpy; runpy.run_path(%r, run_name='pins'); "
            "import importlib; importlib.import_module('scripts.instruments.agentic_probe'); "  # run as a -m subprocess by the campaign
            "print('\\n'.join(sorted(getattr(m, '__file__', '') or '' for m in sys.modules.values())))" % (str(EVAL), str(RUNNER)))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}).stdout
    return [p for p in out.splitlines() if p.startswith(str(EVAL / "scripts")) and p.endswith(".py")]


def tree(root):
    return sorted(p for p in Path(root).rglob("*") if (p.is_file() or p.is_symlink()) and "__pycache__" not in p.parts)


def pins(key):
    model = MODELS[key]["path"]
    a, b = ARMS["A"], ARMS["B"]
    files = [model, IDS, RUNNER, MEMOC_GUARD, MEMOC_GUARD.with_name("memoc_nvml.py"),
             EVAL / "subsets/polyglot.json", EVAL / "corpora/js-ts.txt", EVAL / "corpora/rust.txt"]
    files += [a["root"] / "SOURCE"] + sorted(p for p in a["bin"].iterdir() if p.is_file())
    files += tree(b["root"]) + tree(b["runner_root"])
    files += [Path(p) for p in imported_modules()]
    files += [ASSETS / n for n in ("tasks.json", "tools.json", "selftest.json", "selftest-initial-skips.json", "rust-deps.json")]
    for task in TASKS + [FIT_TASK]:
        files += sorted(p for p in (ASSETS / "sources" / task).rglob("*") if p.is_file())
    files += [Path(p) for p in json.loads((ASSETS / "tools.json").read_text())["binary_hashes"]]
    files += sorted(HERE.glob("*.py")) + [HERE / "run.sh", HERE / "make_runroot.sh"]
    seen, result = set(), []
    for path in files:
        if path.name == "gen_item.py":
            continue
        if str(path) not in seen:
            seen.add(str(path))
            result.append(str(path))
    return result


def j(value):
    return json.dumps(value)


def render(key):
    spec, b = MODELS[key], budget(key)
    item_id = f"stack3-mask-ab-{key}"
    run = HERE / "run.sh"
    out = OUT / item_id
    env = {"AB_ITEM": key, "DRY_RUN": "0", "MODE": "run", "OUT": str(out / "a{attempt}"), "XID_SINCE": XID_SINCE, "AB_PORT": "8081", "BUDGET_S": str(b["run_s"])}
    sigs = [("hazard", XID_RE),
            ("integrity", r"frozen bytes changed|inputs changed|model SHA changed|effective params mismatch|frozen file set changed|cross-arm identity failed|"
                          r"path-engaged proof failed|staging-engaged proof failed|canary prompt differs"),
            ("timeout", r"canary hit its wall cap|did not finish inside its"),
            ("insufficient_data", r"too few verify steps|proof inconclusive|missing final token identity|canary prompt has \d+ tokens|reached only \d+ cached|R2 reused only")]
    lines = [
        f"id = {j(item_id)}",
        f'title = "Real Codex traffic A/B on {spec["label"]}: stack2-metafix vs stack3-mask+staging, k5 p0, campaigns {PLAN}"',
        'kind = "measure"', "ab = true", 'retry = "fresh"', f"budget_s = {b['budget_s']}", f"est_s = {b['est_s']}", "startup_grace_s = 900",
        'exit_codes = { "3" = "hazard", "4" = "transient", "5" = "insufficient_data", "6" = "integrity", "7" = "timeout", "8" = "deterministic" }',
        "signatures = [", *[f"  {{ class = {j(c)}, regex = {j(r)} }}," for c, r in sigs], "]", "",
        "[run]", f"cmd = [\"bash\", {j(str(run))}]", f"cwd = {j(str(HERE))}",
        "env = { " + ", ".join(f"{k} = {j(v)}" for k, v in env.items()) + " }", "ports = [8081]", "",
        "[preflight]", f"cmd = [\"bash\", {j(str(run))}]",
        f"env = {{ AB_ITEM = {j(key)}, DRY_RUN = \"1\", OUT = {j(str(out / 'preflight'))} }}", "timeout_s = 900", "",
        "[canary]", f"cmd = [\"bash\", {j(str(run))}]",
        f"env = {{ AB_ITEM = {j(key)}, DRY_RUN = \"0\", MODE = \"canary\", OUT = {j(str(out / 'canary-a{attempt}'))}, XID_SINCE = {j(XID_SINCE)}, AB_PORT = \"8081\" }}",
        f"timeout_s = {CANARY_TIMEOUT_S}", "",
        "[outputs]", f"paths = [{j(str(out / 'a{attempt}'))}]", "",
        "[fix]", "editable = []", 'protected = ["REFUSE", "frozen", "budget"]', "",
        "[pins]", "files = [", *[f"  {j(p)}," for p in pins(key)], "]", ""]
    return item_id, "\n".join(lines)


def main():
    stale = False
    for key in MODELS:
        item_id, text = render(key)
        path = ITEMS / f"{item_id}.toml"
        if "--check" in sys.argv:
            stale |= not path.exists() or path.read_text() != text
        else:
            path.write_text(text)
            print(path)
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
