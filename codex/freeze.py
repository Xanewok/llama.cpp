#!/usr/bin/env python3
"""Freeze a committed CUDA build without executing its binaries."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('freeze must run with CUDA hidden')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT):
        raise RuntimeError('commit source before freezing')
    frozen = ROOT / 'codex/frozen/capture'
    if frozen.exists():
        raise RuntimeError('refuse existing frozen bytes')
    subprocess.run(['git', 'clone', '--shared', '--no-checkout', str(ROOT), str(frozen)], check=True)
    subprocess.run(['git', '-C', str(frozen), 'update-ref', '--no-deref', 'HEAD', commit], check=True)
    archive = ROOT / 'build-tmp/gguf-py.tar'
    with archive.open('wb') as stream:
        subprocess.run(['git', 'archive', commit, 'gguf-py'], cwd=ROOT, stdout=stream, check=True)
    with tarfile.open(archive) as stream:
        stream.extractall(frozen, filter='data')
    dest = frozen / 'build/bin'
    dest.mkdir(parents=True)
    build = ROOT / 'build-cuda/bin'
    for path in [build / 'llama-server', *sorted(build.glob('*.so*'))]:
        shutil.copy2(path, dest / path.name, follow_symlinks=True)
        subprocess.run(['patchelf', '--set-rpath', '$ORIGIN', str(dest / path.name)], check=True)
    shutil.copy2(dest / 'llama-server', dest / 'capture-srv')
    files = sorted(dest.iterdir())
    (dest / 'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in files))
    (frozen / 'SOURCE').write_text(commit + '\nDO NOT MERGE: request capture; stack2 base 0dba27b90\n'
                                'CUDA 13.2; NCCL OFF; arch 120; nice 19; j4; compiler forced (no GPU compiler probe)\n')
    print(str(frozen))


if __name__ == '__main__':
    main()
