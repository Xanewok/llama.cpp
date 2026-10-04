"""Render the canonical depth item, with the recorded-reference one-boot canary entry."""
import json
from pathlib import Path
import sys

sys.path.insert(0, "/home/xanewok/repos/qwen38-eval")
from harness import lib, mkitem

ROOT = Path(__file__).resolve().parent.parent
x = lib.read_experiment(ROOT / "codex/host-bundle.experiment.toml")
text = mkitem.render(x)
outputs = {"canary": "canary-a{attempt}"} | x.get("outputs", {})
port = x["server"].get("port") or "{port}"
old = ["bash", str(mkitem.H / "run.sh"), "depth", str(x["file"]), "--canary", "--out", str(Path(x["out"]) / outputs["canary"]), "--port", str(port)]
new = ["python3", x["canary"]["entry"], *old[4:]]
assert text.count("cmd = " + json.dumps(old)) == 1
text = text.replace("cmd = " + json.dumps(old), "cmd = " + json.dumps(new))
print(text, end="")
