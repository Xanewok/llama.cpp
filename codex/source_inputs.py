#!/usr/bin/env python3
"""Package pinned local source inputs for a rented machine without Git access."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
root = Path(__file__).resolve().parent.parent
env = dict(os.environ, GIT_DIR=str(root / "codex/git"), GIT_WORK_TREE=str(root), GIT_NO_LAZY_FETCH="1")
inputs = {}
for arm, revision in (("up", "869034b4b"), ("scope", "4ab70fe9d7159352a8a1bbcdf26eeec247654c6e")):
    sha = subprocess.check_output(["git", "-c", "protocol.allow=never", "rev-parse", revision], env=env, text=True).strip()
    archive = root / "codex" / f"source-{arm}.tar.gz"
    proc = subprocess.Popen(["git", "-c", "protocol.allow=never", "archive", "--format=tar", sha], env=env, stdout=subprocess.PIPE)
    with gzip.open(archive, "wb", compresslevel=1) as stream:
        shutil.copyfileobj(proc.stdout, stream)
    assert proc.wait() == 0
    inputs[arm] = {"revision": sha, "archive": archive.name, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
(root / "codex/source-inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
print("M: pinned source archives packaged and hashed; no network used.")
