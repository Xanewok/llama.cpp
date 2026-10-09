"""Keep mkitem's content-addressed snapshot writes inside the authorized worktree."""
import os
from pathlib import Path

if 'S10_IDENTITY_SNAPSHOTS' in os.environ:
    from harness import snapshot

    original_create = snapshot.create

    def create_local(files, directory, prefix):
        files = {name: body for name, body in files.items() if not Path(name).name.startswith('.gpuq-')}
        return original_create(files, Path(os.environ['S10_IDENTITY_SNAPSHOTS']), prefix)

    snapshot.create = create_local
