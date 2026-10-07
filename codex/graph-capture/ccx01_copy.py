"""CCX0+1-pinned copy of a rendered depth item (shape of work/stack7-items/*-ccx01.gpuq.toml). usage: ccx01_copy.py IN OUT"""
import sys
import tomllib
from pathlib import Path

CCX = "/home/xanewok/repos/qwen38-eval-worktrees/work/aa-load/ccx01.sh"
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
lines = src.read_text().splitlines(keepends=True)
out, table, done = [], None, set()
for line in lines:
    if line.startswith("["):
        table = line.strip()
    if line.startswith("host_sensitive"):
        continue
    if table in ("[run]", "[canary]") and line.startswith("cmd = [") and table not in done:
        line = line.replace("cmd = [", f'cmd = ["bash", "{CCX}", ', 1)
        done.add(table)
    if table == "[pins]" and line.startswith("files = ["):
        out.append(line)
        out.append(f'  "{CCX}",\n')
        continue
    out.append(line)
    if table is None and line.startswith("priority"):
        out.append("host_sensitive = false\n")
text = "".join(out)
item = tomllib.loads(text)
assert done == {"[run]", "[canary]"}, done
assert item["run"]["cmd"][:2] == ["bash", CCX] and item["canary"]["cmd"][:2] == ["bash", CCX]
assert item["preflight"]["cmd"][1] != CCX
assert item["host_sensitive"] is False and CCX in item["pins"]["files"]
dst.write_text(text)
print(f"wrote {dst}")
