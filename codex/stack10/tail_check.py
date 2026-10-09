"""Integrity precheck of the stack10 tail window: may the gates of an earlier attempt (a1) be reused for this tree and build?

Contract: reuse is refused (exit 6, one line per problem, each tagged with the sub-check that refused it) unless
  stages  every stage the full window runs before the tail's own (40-65) has rc 0 in a1's stages.tsv, exactly once, under the
          expected name, and a1's summary names only the tail's own gates as failed;
  tree    `src ggml tests common` at HEAD are the same git trees as at a1's commit, and a1's commit is an ancestor of HEAD;
  stamp   the files stage 41 and 42 recorded still say what they said (cuda-stamps.log; the stage-42 SHA256SUMS agrees with
          the files it summarises);
  hash    every library and test-backend-ops in build-cuda/bin hashes (patchelf-normalised, as stage 42 wrote them) to the
          SHA256SUMS of stage 42, and the set of such files is the same;
  mtime   no file of build-cuda/bin is newer than a1's stage 41 log (covers the binaries no hash was recorded for).
Stage 41 recorded no hashes (only DeviceTopK_count_M); stage 42's assemble_bin SHA256SUMS of the k8-bin copy are the only
recorded hashes, and llama-server, round-trace and the small test binaries have none: for those only mtime and the stamp
strings stand. `controls` proves each sub-check refuses: the baseline passes and every control changes one input.

usage: tail_check.py check --root ROOT --a1 DIR --expected 'CODE:name,...' [--out FILE]
       tail_check.py controls --root ROOT --a1 DIR --expected 'CODE:name,...'
"""
import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

TREES = ['src', 'ggml', 'tests', 'common']
TAIL_OWN_STAGES = {66}  # a1's full-suite failed by a gate defect; the tail reruns it


@dataclass(frozen=True)
class Cfg:
    root: Path
    a1: Path
    expected: tuple  # ((code, name), ...)
    stages_tsv: Path = None
    summary: Path = None
    stamps_log: Path = None
    sums: Path = None
    k8_bin: Path = None
    build_bin: Path = None
    a1_sha: str = None  # the build's commit
    tree_sha: str = None  # the commit whose trees HEAD must equal

    def filled(self):
        a1 = self.a1
        summary = self.summary or a1 / 'summary.json'
        a1_sha = self.a1_sha or json.loads(summary.read_text())['commit']
        stock = Path(re.search(r'^STOCK_DIR=(.*)$', (a1 / 'stock-dir.txt').read_text(), re.M).group(1))
        k8_bin = self.k8_bin or stock / 'k8-bin'
        return replace(self, stages_tsv=self.stages_tsv or a1 / 'stages.tsv', summary=summary,
                       stamps_log=self.stamps_log or a1 / 'cuda-stamps.log', k8_bin=k8_bin, sums=self.sums or k8_bin / 'SHA256SUMS',
                       build_bin=self.build_bin or self.root / 'build-cuda/bin', a1_sha=a1_sha, tree_sha=self.tree_sha or a1_sha)


def parse_expected(text):
    return tuple((int(c), n) for c, n in (p.split(':') for p in text.split(',') if p))


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], text=True, capture_output=True, env={**os.environ, 'GIT_NO_LAZY_FETCH': '1'})


def check_stages(cfg):
    out = []
    rows = [line.split('\t') for line in cfg.stages_tsv.read_text().splitlines()]
    if any(len(r) != 4 for r in rows):
        return ['stages: malformed stages.tsv row']
    got = {}
    for code, name, rc, _ in rows:
        if int(code) in got:
            out.append(f'stages: stage {code} appears twice')
        got[int(code)] = (name, rc)
    want = dict(cfg.expected)
    for code, name in sorted(want.items()):
        if code not in got:
            out.append(f'stages: stage {code} {name} did not run')
        elif got[code] != (name, '0'):
            out.append(f'stages: stage {code} is {got[code][0]} rc={got[code][1]}, want {name} rc=0')
    for code in sorted(set(got) - set(want) - TAIL_OWN_STAGES):
        out.append(f'stages: unexpected stage {code} {got[code][0]}')
    summary = json.loads(cfg.summary.read_text())
    if summary.get('commit') != cfg.a1_sha:
        out.append(f"stages: a1 summary commit {summary.get('commit')} != {cfg.a1_sha}")
    if summary.get('failed_gates') != ['full-suite']:
        out.append(f"stages: a1 failed gates are {summary.get('failed_gates')}, want only ['full-suite']")
    payload = cfg.a1 / 'payload.log'
    if f'commit={cfg.a1_sha}' not in payload.read_text().splitlines()[0]:
        out.append('stages: a1 payload.log first line does not name the a1 commit')
    return out


def check_tree(cfg):
    out = []
    root = cfg.root
    if git(root, 'cat-file', '-e', cfg.tree_sha + '^{commit}').returncode:
        return [f'tree: commit {cfg.tree_sha} is not in the repository']
    if git(root, 'merge-base', '--is-ancestor', cfg.tree_sha, 'HEAD').returncode:
        out.append(f'tree: {cfg.tree_sha[:9]} is not an ancestor of HEAD')
    if git(root, 'status', '--porcelain', '--untracked-files=no').stdout.strip():
        out.append('tree: tracked files are dirty')
    for d in TREES:
        a, b = (git(root, 'rev-parse', f'{rev}:{d}') for rev in ('HEAD', cfg.tree_sha))
        if a.returncode or b.returncode or a.stdout != b.stdout:
            out.append(f'tree: {d}/ at HEAD {a.stdout.strip()[:12]} != at {cfg.tree_sha[:9]} {b.stdout.strip()[:12]}')
    return out


def strings(path):
    return subprocess.run(['strings', str(path)], text=True, capture_output=True, check=True).stdout.splitlines()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_sums(path):
    return {name: digest for digest, name in (line.split(None, 1) for line in path.read_text().splitlines())}


def check_stamp(cfg):
    """The stage 41 stamps re-run on the current files, plus agreement of the stage 42 record with its own files."""
    out = []
    b = cfg.build_bin
    recorded = dict(re.findall(r'^(\w+)=(\d+)$', cfg.stamps_log.read_text(), re.M))
    n = sum('DeviceTopK' in s for s in strings(b / 'libggml-cuda.so'))
    if recorded.get('DeviceTopK_count_M') != str(n) or n != 25:
        out.append(f"stamp: DeviceTopK count {n}, stage 41 recorded {recorded.get('DeviceTopK_count_M')}, want 25")
    short = cfg.a1_sha[:9]
    if short not in strings(b / 'libggml-base.so'):
        out.append(f'stamp: libggml-base.so does not embed {short}')
    if b'mmq_pdl:' not in (b / 'libggml-cuda.so').read_bytes():
        out.append('stamp: libggml-cuda.so has no mmq_pdl: string')
    if not any('backend busy at a skipped sync' in s for s in strings(b / 'libllama.so')):
        out.append('stamp: libllama.so has no K11a skipped-sync string')
    sums = parse_sums(cfg.sums)
    for name, digest in sorted(sums.items()):
        f = cfg.k8_bin / name
        if not f.exists() or sha256(f) != digest:
            out.append(f'stamp: stage 42 SHA256SUMS disagrees with {cfg.k8_bin / name}')
    return out


def normalised_hashes(build_bin, names):
    """sha256 of build_bin's files after the RUNPATH rewrite assemble_bin applies (the only difference to the build output)."""
    result = {}
    with tempfile.TemporaryDirectory(prefix='tailhash.', dir=os.environ.get('TAIL_TMP')) as tmp:
        for name in sorted(names):
            real = Path(os.path.realpath(build_bin / name))
            if real.name not in result:
                copy = Path(tmp) / real.name
                shutil.copy2(real, copy)
                subprocess.run(['patchelf', '--set-rpath', '$ORIGIN', str(copy)], check=True)
                result[real.name] = sha256(copy)
                copy.unlink()
            result[name] = result[real.name]
    return {n: result[n] for n in names}


def hashed_names(build_bin):
    return {p.name for p in build_bin.iterdir() if fnmatch.fnmatch(p.name, '*.so*') or p.name == 'test-backend-ops'}


def check_hash(cfg, record=None):
    out = []
    sums = parse_sums(cfg.sums)
    names = hashed_names(cfg.build_bin)
    if names != set(sums):
        out.append(f'hash: file set differs from stage 42: only in build {sorted(names - set(sums))}, only in record {sorted(set(sums) - names)}')
    got = normalised_hashes(cfg.build_bin, names & set(sums))
    for name in sorted(got):
        if got[name] != sums[name]:
            out.append(f'hash: {name} {got[name][:12]} != recorded {sums[name][:12]}')
    if record is not None:
        record.update(got)
    return out


def check_mtime(cfg):
    limit = cfg.stamps_log.stat().st_mtime
    return [f'mtime: {p.name} is newer than a1 stage 41 ({p.stat().st_mtime - limit:.0f} s)'
            for p in sorted(cfg.build_bin.iterdir()) if p.stat().st_mtime > limit]


CHECKS = {'stages': check_stages, 'tree': check_tree, 'stamp': check_stamp, 'hash': check_hash, 'mtime': check_mtime}


def run(cfg, only=None, record=None):
    problems = []
    for name, fn in CHECKS.items():
        if only and name not in only:
            continue
        try:
            problems += fn(cfg, record) if name == 'hash' else fn(cfg)
        except Exception as e:  # an unreadable input is a refusal, never a pass
            problems.append(f'{name}: cannot check: {type(e).__name__}: {e}')
    return problems


def controls(cfg):
    """Each control differs from the passing baseline in one input and must be refused by the sub-check that owns it."""
    results = []
    base = run(cfg)
    results.append(('baseline passes', not base, base))
    with tempfile.TemporaryDirectory(prefix='tailctl.') as t:
        t = Path(t)
        lines = cfg.stages_tsv.read_text().splitlines()
        code, name, rc, wall = lines[len(lines) // 2].split('\t')
        bad_rc = t / 'rc.tsv'
        bad_rc.write_text('\n'.join(l if i != len(lines) // 2 else f'{code}\t{name}\t8\t{wall}' for i, l in enumerate(lines)) + '\n')
        missing = t / 'missing.tsv'
        missing.write_text('\n'.join(l for i, l in enumerate(lines) if i != len(lines) // 2) + '\n')
        extra_gate = t / 'summary.json'
        extra_gate.write_text(json.dumps({**json.loads(cfg.summary.read_text()), 'failed_gates': ['full-suite', name]}))
        # the parent of the newest commit that changes one of the four trees: its trees differ from HEAD's by construction
        newest = git(cfg.root, 'log', '-1', '--format=%H', 'HEAD', '--', *TREES).stdout.strip()
        older = git(cfg.root, 'rev-parse', f'{newest}~1').stdout.strip()
        stamps = t / 'cuda-stamps.log'
        stamps.write_text(cfg.stamps_log.read_text().replace('=25', '=24'))
        sums = t / 'SHA256SUMS'
        text = cfg.sums.read_text()
        digest = text.split()[0]
        sums.write_text(text.replace(digest, ('0' if digest[0] != '0' else '1') + digest[1:]))
        binary = t / 'bin'
        binary.mkdir()
        for p in cfg.build_bin.iterdir():
            if p.name != 'test-backend-ops':
                os.symlink(os.path.realpath(p), binary / p.name)
        data = bytearray((cfg.build_bin / 'test-backend-ops').read_bytes())
        data[len(data) // 2] ^= 1
        (binary / 'test-backend-ops').write_bytes(data)
        os.utime(binary / 'test-backend-ops', (cfg.stamps_log.stat().st_mtime - 5,) * 2)
        early = t / 'early-stamps.log'
        early.write_text(cfg.stamps_log.read_text())
        os.utime(early, (cfg.stamps_log.stat().st_mtime - 3600,) * 2)
        table = [
            ('one stage rc changed to 8', replace(cfg, stages_tsv=bad_rc), 'stages', f'stage {code} is {name} rc=8'),
            ('one stage row removed', replace(cfg, stages_tsv=missing), 'stages', f'stage {code} {name} did not run'),
            ('a1 summary lists a second failed gate', replace(cfg, summary=extra_gate), 'stages', 'failed gates'),
            ('source trees differ from the a1 commit', replace(cfg, tree_sha=older), 'tree', 'at HEAD'),
            ('stage 41 stamp changed (DeviceTopK 25 -> 24)', replace(cfg, stamps_log=stamps), 'stamp', 'DeviceTopK count'),
            ('stage 42 hash record changed in one digest', replace(cfg, sums=sums), 'hash', 'recorded'),
            ('same changed record, seen by the self-consistency stamp', replace(cfg, sums=sums), 'stamp', 'SHA256SUMS disagrees'),
            ('one byte of test-backend-ops flipped', replace(cfg, build_bin=binary), 'hash', 'test-backend-ops'),
            ('build files newer than the a1 stamps', replace(cfg, stamps_log=early), 'mtime', 'newer than a1 stage 41'),
        ]
        for label, c, tag, needle in table:
            problems = [p for p in run(c, only={tag}) if p.startswith(tag + ':')]
            results.append((f'{label} -> {tag} refuses', any(needle in p for p in problems), problems))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['check', 'controls'])
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--a1', type=Path, required=True)
    ap.add_argument('--expected', required=True)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    try:
        cfg = Cfg(root=a.root, a1=a.a1, expected=parse_expected(a.expected)).filled()
    except Exception as e:
        print(f'tail precheck REFUSED: cannot read the reused attempt: {type(e).__name__}: {e}')
        sys.exit(6)
    if a.mode == 'controls':
        results = controls(cfg)
        for label, ok, problems in results:
            print(f"{'PASS' if ok else 'FAIL'} control: {label}")
            for p in (problems[:1] if ok and not label.startswith('baseline') else problems):
                print(f'    {p}')
        sys.exit(0 if all(ok for _, ok, _ in results) else 6)
    record = {}
    problems = run(cfg, record=record)
    for p in problems:
        print(f'tail precheck REFUSED: {p}')
    if problems:
        sys.exit(6)
    if a.out:
        a.out.write_text(json.dumps(dict(
            a1=str(cfg.a1), a1_commit=cfg.a1_sha, trees={d: git(cfg.root, 'rev-parse', f'HEAD:{d}').stdout.strip() for d in TREES},
            stages=[dict(code=c, name=n) for c, n in cfg.expected], hash_source=str(cfg.sums),
            hash_note='stage 41 recorded no hashes; these are the stage 42 assemble_bin SHA256SUMS (RUNPATH-normalised) re-derived from build-cuda/bin now',
            unhashed_by_mtime_only=sorted(p.name for p in cfg.build_bin.iterdir() if p.name not in record and not p.is_symlink()),
            sha256=record), indent=2) + '\n')
    print(f'tail precheck PASS: {len(cfg.expected)} stages rc 0, trees equal at {cfg.a1_sha[:9]}, {len(record)} hashed files, stamps and mtimes hold')


if __name__ == '__main__':
    main()
