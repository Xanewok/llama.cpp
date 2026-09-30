#!/usr/bin/env python3
import hashlib
import pathlib
import shutil
import sys

source, destination, commit = sys.argv[1:]
source = pathlib.Path(source).resolve()
destination = pathlib.Path(destination)
assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
assert not destination.exists(), f"refuse existing frozen directory: {destination}"
bin_dir = destination / "bin"
bin_dir.mkdir(parents=True)
shutil.copy2(source / "llama-server", bin_dir / "vocab-ab")
for path in sorted(source.iterdir()):
    if path.name.startswith("lib") and ".so" in path.name:
        if path.is_symlink():
            assert path.resolve().parent == source, f"external library link: {path}"
            (bin_dir / path.name).symlink_to(path.resolve().name)
        else:
            shutil.copy2(path, bin_dir / path.name)
manifest = []
for path in sorted(bin_dir.iterdir()):
    digest = hashlib.sha256()
    with path.open("rb") as data:
        for block in iter(lambda: data.read(1024 * 1024), b""):
            digest.update(block)
    manifest.append(f"{digest.hexdigest()}  {path.name}\n")
(bin_dir / "SHA256SUMS").write_text("".join(manifest))
(destination / "SOURCE").write_text(commit + "\nCopied from " + str(source) + "\nUse LD_LIBRARY_PATH=bin to select only frozen project libraries.\n")
