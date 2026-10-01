#!/usr/bin/env python3
"""Compare compile-only evidence before/after the local packing alignment fix."""
import json
import os
from pathlib import Path
import re
import subprocess

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
root = Path(__file__).resolve().parent.parent
out = root / "build-arch-check"
differences = []
for before in sorted((out / "pre-align").glob("fattn-mma-f16-instance-q8_0-*.o")):
    after = out / before.name
    dumps = []
    for tag, obj in (("before", before), ("after", after)):
        log = root / "codex" / f"alignment-{tag}-{obj.stem}.sass"
        with log.open("w") as stream:
            subprocess.run(["/usr/local/cuda-13.2/bin/cuobjdump", "--dump-sass", str(obj)], stdout=stream, check=True)
        functions = {}
        for block in log.read_text().split("Function : ")[1:]:
            symbol = block.splitlines()[0].strip()
            functions[symbol] = [re.sub(r"^\s*/\*[0-9a-f]+\*/", "", line).strip() for line in block.splitlines()
                                 if re.search(r"/\*[0-9a-f]+\*/", line) or re.search(r"/\* 0x[0-9a-f]+ \*/", line)]
        dumps.append(functions)
    if dumps[0] != dumps[1]:
        differences.append(before.name)
summary = {"paired_objects_M": len(list((out / "pre-align").glob("fattn-mma-f16-instance-q8_0-*.o"))), "differences": differences}
(root / "codex/alignment-identity.json").write_text(json.dumps(summary, indent=2) + "\n")
assert summary["paired_objects_M"] == 54 and not differences, differences
print("M: all 54 raw q8_0 object SASS bodies, including encodings, match before/after explicit alignment.")
