import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'build-upstream-pack.git'
OUT = ROOT / 'codex/upstream-out'
BASE = '7049ff0cbeb1f5ead231de4522af6b75d8d773c0'
os.environ.update(CUDA_VISIBLE_DEVICES='', GIT_NO_LAZY_FETCH='1')


def git(*args, data=None, check=True, index=None):
    env = os.environ.copy()
    if index:
        env['GIT_INDEX_FILE'] = str(index)
    return subprocess.run(['git', '--git-dir', str(DB), *args], input=data,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          env=env, check=check)


def start(name):
    index = DB / (name + '.index')
    git('read-tree', BASE, index=index)
    return index, BASE


def record(index, parent, message, source):
    tree = git('write-tree', index=index).stdout.strip().decode()
    # Preserve the original owner's identity and message for the rebased mbox.
    identity = git('show', '-s', '--format=%an%n%ae%n%aI', source).stdout.decode().splitlines()
    env = os.environ.copy()
    env.update(GIT_AUTHOR_NAME=identity[0], GIT_AUTHOR_EMAIL=identity[1],
               GIT_AUTHOR_DATE=identity[2], GIT_COMMITTER_NAME=identity[0],
               GIT_COMMITTER_EMAIL=identity[1])
    result = subprocess.run(['git', '--git-dir', str(DB), 'commit-tree', tree, '-p', parent],
                            input=message, env=env, capture_output=True, check=True)
    return result.stdout.strip().decode()


def apply_commit(index, parent, source, paths=()):
    patch = git('diff', source + '^', source, '--', *paths).stdout
    result = git('apply', '--cached', '--3way', '-', data=patch, check=False, index=index)
    (OUT / 'logs' / (source[:9] + '-apply.log')).write_bytes(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('apply conflict: ' + source + '\n' + result.stderr.decode())
    message = git('show', '-s', '--format=%B', source).stdout
    return record(index, parent, message, source)


def apply_mbox(index, parent, path):
    patch = path.read_bytes()
    result = git('apply', '--cached', '--3way', '-', data=patch, check=False, index=index)
    if result.returncode:
        raise RuntimeError(str(path) + '\n' + result.stderr.decode())
    source = patch.splitlines()[0].split()[1].decode()
    return record(index, parent, git('show', '-s', '--format=%B', source).stdout, source)


def emit(name, base, tip, subdir=''):
    directory = OUT / 'patches/upstream/llama.cpp' / name / subdir
    directory.mkdir(parents=True, exist_ok=True)
    result = git('format-patch', '--no-signature', '--no-cover-letter',
                 '-o', str(directory), base + '..' + tip)
    print(result.stdout.decode(), end='')
    git('update-ref', 'refs/heads/' + name + ('-' + subdir if subdir else ''), tip)
    return directory


if __name__ == '__main__':
    import sys
    name, *commits = sys.argv[1:]
    index, tip = start(name)
    for commit in commits:
        tip = apply_commit(index, tip, commit)
    emit(name, BASE, tip)
    print('TIP', tip)
