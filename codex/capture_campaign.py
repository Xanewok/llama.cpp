#!/usr/bin/env python3
"""Parent-only adapter of the read-only weekend runner. All writes use this worktree."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
WEEKEND = Path('/home/xanewok/repos/qwen38-weekend/codex')


def main():
    if os.environ.get('RUN_PARENT_GPU') != '1' or os.environ.get('PLAN', '1') != '0':
        raise SystemExit('parent execution gates are required')
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--deadline', required=True, type=float)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT) or args.deadline <= time.time():
        raise ValueError('campaign output/deadline is invalid')
    sys.path.insert(0, str(WEEKEND))
    import weekend_common as common
    frozen = ROOT / 'codex/frozen/capture'
    common.FROZEN = frozen
    common.COMMIT = (frozen / 'SOURCE').read_text().splitlines()[0]
    h, s, _ = common.configure('S4', common.MODEL_A, common.SHA_A)
    s.LONG_TAIL_TASKS[:] = ['rust/alphametics', 'rust/nucleotide-codons']
    h.SERVER_EXTRA += ['--no-context-shift', '--fit', 'off']
    original_stop = h.Runner.stop_child
    original_spawn, original_acquire, original_model = h.Runner.spawn, h.Runner.acquire, h.Runner.ensure_model
    servers = []

    def spawn(self, command, log, env=None):
        if Path(command[0]).name in ('curl', 'wget'):
            raise h.StopCampaign('capture_download_forbidden')
        if command[0] == str(frozen / 'build/bin/llama-server'):
            from parent_gpu import verify_frozen, idle
            verify_frozen(frozen / 'build/bin', 'capture-srv')
            idle(8081)
            capture = Path(log).parent / 'capture'
            capture.mkdir(exist_ok=True)
            command = [str(frozen / 'build/bin/capture-srv'), *command[1:]]
            env = dict(env or self.env, LLAMA_CAPTURE_DIR=str(capture), LLAMA_CAPTURE_PROBS='1',
                       LLAMA_CAPTURE_LIMIT='40000', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1',
                       TMPDIR=str(ROOT / 'build-tmp'), GGML_CUDA_FA_MMA_Q8_0='1')
            launch = Path(log).with_name('launch.json')
            data = h.read_sealed(launch)
            data.update(executed_command=command, capture_env={key: env[key] for key in
                        ('LLAMA_CAPTURE_DIR', 'LLAMA_CAPTURE_PROBS', 'LLAMA_CAPTURE_LIMIT')})
            h.sealed_json(launch, data)
        child = original_spawn(self, command, log, env)
        if Path(command[0]).name == 'capture-srv':
            servers.append((self, child))
        return child

    def ensure_model(self, model_id):
        model = next(m for m in self.manifest['models'] if m['id'] == model_id)
        if not Path(model['path']).is_file():
            raise h.StopCampaign('capture_local_model_missing')
        return original_model(self, model_id)

    def acquire(self):
        # Existing shared locks are opened read-only; local runner locks live inside ROOT.
        for name in ('qwen-hillclimb.lock', 'qwen-local-server.lock'):
            handle = (Path(f'/run/user/{os.getuid()}') / name).open('r')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lockfiles.append(handle)
        return original_acquire(self)

    def stop_child(self, child):
        original_stop(self, child)
        if any(owned is child for _, owned in servers):
            from parent_gpu import xid
            xid()

    h.Runner.spawn, h.Runner.acquire, h.Runner.ensure_model, h.Runner.stop_child = spawn, acquire, ensure_model, stop_child
    runtime = ROOT / 'build-parent-runtime'
    runtime.mkdir(exist_ok=True)
    os.environ['XDG_RUNTIME_DIR'] = str(runtime)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CARGO_NET_OFFLINE'] = 'true'
    def interrupted(*_):
        raise h.StopCampaign('capture_signal')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    sys.argv = [str(s.__file__), '--campaign', str(args.out), '--assets', str(common.ASSETS),
                '--deadline-epoch', str(args.deadline)]
    try:
        s.main()
    finally:
        # Runner Popen handles identify the exact owned children, even after a task exception.
        for runner, child in servers:
            if child.poll() is None:
                runner.stop_child(child)


if __name__ == '__main__':
    main()
