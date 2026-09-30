#!/usr/bin/env python3
"""Record read-only runner inputs once, before queueing; no GPU calls."""
import json
from weekend_common import ASSETS, EVAL, IDS, ROOT, RUNNER, SP, sha

paths = [IDS, RUNNER, SP / "memoc/memoc_guard.sh", SP / "memoc/memoc_nvml.py",
         EVAL / "corpora/js-ts.txt", EVAL / "corpora/rust.txt"]
paths += sorted(p for p in (EVAL / "scripts").rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
paths += sorted(p for p in (EVAL / "subsets").rglob("*") if p.is_file())
paths += sorted(p for p in ASSETS.rglob("*") if p.is_file())
pins = {str(p): sha(p) for p in paths}
(ROOT / "codex/input-pins.json").write_text(json.dumps(pins, indent=2) + "\n")
print(f"pinned {len(pins)} runner/asset inputs; no model, GPU, or binary execution")
