"""experiment.toml -> gpuq item TOML. The item's contract (exit codes, signatures, canary, budget, pins) is fixed here,
never hand-edited: python -m harness.mkitem experiment.toml [--out ITEM.toml | --check ITEM.toml]
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path

from . import lib, snapshot

H = Path(__file__).resolve().parent
EVAL = H.parent
SCRATCH = ("/.cache/claude-code-tmp/", "/scratchpad/", "/tmp/", "/.cache/agent-tmp/")


def boot_wall(x):
    """wall_s is one boot with one text per depth (boot, prefills, decodes); texts > 1 add (N - 1) x depths x text_s, the decode
    of one more text on the cached prompt."""
    s = x[x["kind"]]
    n = s.get("texts", 1)
    if n > 1 and "text_s" not in s:
        raise lib.Fail("inputs changed: depth.texts > 1 needs depth.text_s (seconds of one extra text's decode)", "integrity")
    return s["wall_s"] + (n - 1) * len(s.get("depths", [])) * s.get("text_s", 0)


def budget(x):
    """run_s covers the planned boots or campaigns with margin plus cleanup; budget_s adds the canary stage and a cleanup."""
    b, wall = x["budget"], boot_wall(x)
    run_s = math.ceil((len(x["plan"]) * wall * b["margin"] + b["cleanup_s"]) / 50) * 50
    return {"run_s": run_s, "budget_s": b["canary_s"] + run_s + b["cleanup_s"], "est_s": b.get("est_s", run_s)}


def tree(root):
    return sorted(p for p in Path(root).rglob("*") if (p.is_file() or p.is_symlink()) and not {"__pycache__", ".git"} & set(p.parts))


def imported_modules(runner):
    """Every qwen38-eval module the runner imports, plus the probe it spawns with -m. No runner (agents > 1): the probe's own imports."""
    code = ("import sys; sys.path.insert(0, %r); import runpy; %s"
            "import importlib; importlib.import_module('scripts.instruments.agentic_probe'); "
            "print('\\n'.join(sorted(getattr(m, '__file__', '') or '' for m in sys.modules.values())))" % (str(EVAL), f"runpy.run_path({str(runner)!r}, run_name='pins'); " if runner else ""))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
                              "GIT_NO_LAZY_FETCH": "1", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}).stdout
    return [Path(p) for p in out.splitlines() if p.startswith(str(EVAL / "scripts")) and p.endswith(".py")]


def flag_files(arm):
    """Every existing absolute file an arm's flags name (`--flag PATH` or `--flag=PATH`): an input the arm's bytes depend on."""
    out = []
    for flag in arm.get("flags", []):
        path = Path(flag.split("=", 1)[-1]) if flag.startswith("-") else Path(flag)
        if path.is_absolute() and path.is_file():
            out.append(path)
    return out


def harness_snapshot(x):
    files = {f"harness/{p.name}": p.read_bytes() for p in [*sorted(H.glob("*.py")), *sorted(H.glob("*.sh"))]}
    experiment = x["file"].read_bytes()
    if hashlib.sha256(experiment).hexdigest() != x["_file_sha"]:
        raise lib.Fail("inputs changed: experiment changed during rendering", "integrity")
    files["experiment.toml"] = experiment
    files["source-root.txt"] = (str(EVAL) + "\n").encode()
    return snapshot.create(files, EVAL / "orchestration/gpuq/snapshots", "harness")


def pins(x, code=None):
    files = [x["model"]["path"]] + ([x["draft_vocab"]["path"]] if "draft_vocab" in x else [])
    c = x.get("campaign")
    if c and c.get("runner"):
        files.append(Path(c["runner"]))
    files += [Path(p) for p in x.get("extra_pins", [])]
    if "cpu_evidence" in x:
        files.append(Path(x["cpu_evidence"]["path"]))
    for arm in x["arms"].values():
        files += tree(arm["root"]) + (tree(arm["runroot"]) if arm["runroot"] else [])
        files += flag_files(arm)
    if c:
        assets = Path(c["assets"])
        files += imported_modules(c.get("runner"))
        files += [assets / n for n in ("tasks.json", "tools.json", "selftest.json", "selftest-initial-skips.json", "rust-deps.json")]
        for task in c["tasks"] + ([c["fit_task"]] if c.get("fit_task") else []):
            files += sorted(p for p in (assets / "sources" / task).rglob("*") if p.is_file())
        files += [Path(p) for p in json.loads((assets / "tools.json").read_text())["binary_hashes"]]
    if x["kind"] == "depth" and x["depth"].get("fixtures"):
        files += sorted(p for p in Path(x["depth"]["fixtures"]).iterdir() if p.is_file())
    code = code or harness_snapshot(x)
    files += [p for p in tree(code) if not p.name.startswith(".gpuq-") and ".cache" not in p.relative_to(code).parts]
    seen, result = set(), []
    for path in map(str, files):
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def scratch_paths(text):
    return sorted({line.strip() for line in text.splitlines() if any(s in line for s in SCRATCH)})


def wrap_stages(text, wrapper):
    """A custom entry receives saved run.sh, kind, experiment and the unchanged stage arguments."""
    item = tomllib.loads(text)
    commands = {}
    for stage in ('run', 'preflight', 'canary'):
        cmd = item[stage]['cmd']
        saved = Path(cmd[1])
        if cmd[0] != 'bash' or saved.name != 'run.sh' or Path(cmd[3]).parent != saved.parent.parent:
            raise lib.Fail(f'inputs changed: {stage} is not a saved harness entry', 'integrity')
        if str(Path(wrapper).resolve()) not in item['pins']['files']:
            raise lib.Fail('inputs changed: custom entry must be pinned', 'integrity')
        commands[stage] = ['bash', str(Path(wrapper).resolve()), *cmd[1:]]
    section = None
    lines = []
    for line in text.splitlines():
        if line.startswith('['):
            section = line.strip('[]')
        if section in commands and line.startswith('cmd = '):
            line = 'cmd = ' + json.dumps(commands.pop(section))
        lines.append(line)
    if commands:
        raise lib.Fail('inputs changed: missing stage command', 'integrity')
    return '\n'.join(lines) + '\n'


def render(x):
    if type(x.get("priority")) is not int:
        raise lib.Fail("inputs changed: priority must be an explicit integer for new items", "integrity")
    j = json.dumps
    b, out = budget(x), Path(x["out"])
    code = harness_snapshot(x)
    entry_h = code / "harness"
    o = {"run": "a{attempt}", "canary": "canary-a{attempt}", "preflight": "preflight"} | x.get("outputs", {})
    port = x["server"].get("port")
    entry = ["bash", str(entry_h / "run.sh"), x["kind"], str(code / "experiment.toml")]
    port_args = ["--port", str(port) if port else "{port}"]
    run = entry + ["--execute", "--out", str(out / o["run"])] + port_args + (["--budget-s", str(b["run_s"])] if x["kind"] == "campaign" else [])
    canary = entry + ["--canary", "--out", str(out / o["canary"])] + port_args
    preflight = entry + ["--preflight", "--out", str(out / o["preflight"])]
    m = x["budget"]
    s = x[x["kind"]]
    wall = f"({s['wall_s']} + {s['texts'] - 1} x {len(s['depths'])} depths x {s['text_s']} s text decode = {boot_wall(x)})" if s.get("texts", 1) > 1 else s["wall_s"]
    arithmetic = (f"# {''.join(x['plan']) if x['kind'] == 'campaign' else ' '.join(x['plan'])}: {len(x['plan'])} x {wall} s x {m['margin']} "
                  f"+ {m['cleanup_s']} = run {b['run_s']}; + canary {m['canary_s']} + cleanup {m['cleanup_s']} = {b['budget_s']}.")
    exit_codes = "{ " + ", ".join(f'"{rc}" = {j(cls)}' for rc, cls in lib.EXIT.items()) + " }"
    lines = [
        f"# experiment = {j(str(x['file']))}", f"id = {j(x['id'])}", f"priority = {x['priority']}", f"title = {j(x['title'])}", arithmetic,
        *([f"# cpu_evidence: {x['cpu_evidence']['command']}"] if "cpu_evidence" in x else []),
        'kind = "measure"', f"ab = {'true' if len(x['arms']) > 1 else 'false'}", 'retry = "fresh"',
        f"budget_s = {b['budget_s']}", f"est_s = {b['est_s']}", f"startup_grace_s = {m.get('startup_grace_s', 900)}",
        f"exit_codes = {exit_codes}",
        "signatures = [", *[f"  {{ class = {j(c)}, regex = {j(r)} }}," for c, r in lib.SIGNATURES], "]", "",
        "[run]", f"cmd = {j(run)}", f"cwd = {j(str(entry_h))}", 'env = { GPUQ_ACTIVE = "1" }',
        f"ports = {j([port]) if port else j('dynamic')}", "",
        "[preflight]", f"cmd = {j(preflight)}", f"timeout_s = {m.get('preflight_s', 900)}", "",
        "[canary]", f"cmd = {j(canary)}", 'env = { GPUQ_ACTIVE = "1" }', f"timeout_s = {m['canary_s']}", "",
        "[outputs]", f"paths = {j([str(out / o['run'])])}", "",
        "[fix]", "editable = []", f"protected = {j(x.get('protected', ['frozen', 'budget']))}", "",
        "[pins]", "files = [", *[f"  {j(p)}," for p in pins(x, code)], "]", ""]
    text = "\n".join(lines)
    leaked = scratch_paths(text)
    if leaked:
        raise lib.Fail(f"inputs changed: scratchpad paths in the item: {leaked[:3]}", "integrity")
    if b["budget_s"] < m["canary_s"] + b["run_s"] or b["est_s"] > b["budget_s"] or m["canary_s"] > 900:
        raise lib.Fail(f"budget arithmetic: {b} with canary {m['canary_s']}", "integrity")
    return text


def stale(directory):
    changed = False
    for item in sorted(directory.glob("*.toml")):
        try:
            text = item.read_text()
            header = next((line for line in text.splitlines() if line.startswith("# experiment = ")), None)
            source = json.loads(header.split(" = ", 1)[1]) if header else tomllib.loads(text)["run"]["cmd"][3]
            differs = text != render(lib.read_experiment(source))
        except (OSError, ValueError, KeyError, IndexError) as error:
            print(f"{item}: stale ({error})")
            changed = True
            continue
        if differs:
            print(f"{item}: stale")
            changed = True
    return int(changed)


def build_parser():
    p = argparse.ArgumentParser(prog="mkitem", description="Render a fresh gpuq item from an experiment; --submit checks its contract before submitting.",
                                epilog="--stale DIR lists old drafts without deleting them; rc 1 means stale drafts were found.")
    p.add_argument("experiment", nargs="?", type=Path, help="the experiment.toml to render (required except with --stale)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--out", type=Path, help="write the item here (default: stdout)")
    group.add_argument("--check", type=Path, help="exit 1 when this item differs from the rendered one")
    group.add_argument("--submit", action="store_true", help="render in orchestration/gpuq/items, contract-check, then submit; propagate gpuq rc")
    group.add_argument("--stale", type=Path, metavar="DIR", help="list drafts differing from their experiment's fresh rendering; never delete")
    return p


def main(argv=None):
    parser = build_parser()
    a = parser.parse_args(argv)
    if a.stale:
        if a.experiment:
            parser.error("--stale does not take an experiment")
        return stale(a.stale)
    if not a.experiment:
        parser.error("experiment is required")
    try:
        x = lib.read_experiment(a.experiment)
        text = render(x)
        if a.check:
            return 0 if a.check.exists() and a.check.read_text() == text else 1
        if a.submit:
            item = EVAL / "orchestration/gpuq/items" / f"{x['id']}.toml"
            item.parent.mkdir(parents=True, exist_ok=True)
            item.write_text(text)
            rc = subprocess.run(["gpuq", "submit", "--check", str(item)]).returncode
            if rc:
                return rc
            rc = subprocess.run(["gpuq", "submit", str(item)]).returncode
            if not rc:
                print(f"submitted {x['id']} priority {x['priority']}")
            return rc
        if a.out:
            a.out.write_text(text)
        else:
            sys.stdout.write(text)
        return 0
    except lib.Fail as error:
        print(error, file=sys.stderr)
        return error.rc


if __name__ == "__main__":
    sys.exit(main())
