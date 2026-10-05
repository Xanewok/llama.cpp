"""Render mtp-fuse-depth-j5q's gpuq item with harness.mkitem, entries pointed at the adapter (mtp_fuse_run.sh)."""
import json
from pathlib import Path
import sys

sys.path.insert(0, "/home/xanewok/repos/qwen38-eval")
from harness import lib, mkitem

HERE = Path(__file__).resolve().parent
x = lib.read_experiment(HERE / "mtp-fuse-depth-j5q.experiment.toml")
text = mkitem.render(x)
old = json.dumps(["bash", str(mkitem.H / "run.sh"), "depth", str(x["file"])])[:-1] + ", "
new = json.dumps(["bash", str(HERE / "mtp_fuse_run.sh")])[:-1] + ", "
assert text.count(old) == 3, text.count(old)
text = text.replace(old, new)
(HERE / "mtp-fuse-depth-j5q.toml").write_text(text)
print(HERE / "mtp-fuse-depth-j5q.toml")
