"""Content-addressed code copies; an existing snapshot is verified, never repaired."""
import hashlib
import tempfile
from pathlib import Path

from . import lib


def create(files, directory, prefix):
    """files maps relative names to bytes. Readers only see complete snapshots."""
    digest = hashlib.sha256()
    for name, body in sorted(files.items()):
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise lib.Fail(f'inputs changed: snapshot path escapes: {name}', 'integrity')
        digest.update(name.encode() + b'\0' + hashlib.sha256(body).digest())
    root = Path(directory) / f'{prefix}-{digest.hexdigest()}'
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root.parent) as temporary:
            staged = Path(temporary) / 'snapshot'
            staged.mkdir()
            for name, body in files.items():
                target = staged / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
                target.chmod(0o555 if target.suffix in ('.py', '.sh') else 0o444)
            try:
                staged.rename(root)
            except OSError:
                if not root.is_dir():
                    raise
    for name, body in files.items():
        path = root / name
        if not path.is_file() or path.read_bytes() != body:
            raise lib.Fail(f'inputs changed: snapshot differs: {path}', 'integrity')
    return root
