"""Record report evidence without promoting a ledger draft or committing files."""
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from . import calibration, lib

REPO = Path(__file__).resolve().parent.parent
SECTIONS = ('llama-cpp', 'vllm-hq', 'both-hardware', 'roadmap')


def data_root():
    saved = REPO / 'source-root.txt'
    fallback = saved.read_text().strip() if saved.exists() else REPO
    root = Path(os.environ.get('HARNESS_DATA_ROOT', fallback)).resolve()
    if any((parent / 'source-root.txt').is_file() and (parent / 'harness/run.sh').is_file()
           for parent in (root, *root.parents)):
        raise lib.Fail('record destination is inside the harness snapshot', 'integrity')
    return root


def append_once(path, addition, marker):
    previous = path.read_text() if path.exists() else ''
    if marker in previous:
        return
    if path.name == 'INDEX.md':
        heading = next((line for line in addition.splitlines() if line.startswith('## [')), None)
        if heading and heading in previous:
            addition = addition.replace(f'\n{heading}\n', '', 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.record-tmp')
    temporary.write_text(previous + addition)
    temporary.replace(path)


def record(out, rendered, *, lever, axis, cause, relation, verdict, section='llama-cpp'):
    if out['exit_code']:
        raise lib.Fail('report gates failed; nothing recorded', lib.EXIT[out['exit_code']])
    fields = dict(lever=lever, axis=axis, cause=cause, relation=relation, verdict=verdict)
    if section not in SECTIONS or any(not v or '\n' in v or '\r' in v or '|' in v for v in fields.values()):
        raise lib.Fail('record requires non-empty, single-line metadata without table separators', 'integrity')
    lines = rendered.splitlines()
    evidence = next(line.removeprefix('notes - ') for line in lines if line.startswith('notes - '))
    row = next(line.removeprefix('ledger ') for line in lines if line.startswith('ledger | '))
    for key, value in fields.items():
        row = row.replace(f'<{key}>', value)
    identity = dict(run=str(Path(out['run_dir']).resolve()), section=section, **fields)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    marker = f'<!-- harness-record:{key} -->'
    root = data_root()
    records = root / 'docs/ledger/drafts/.records'
    records.mkdir(parents=True, exist_ok=True)
    with (records / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        intent = records / f'{key}.json'
        if intent.exists():
            additions = json.loads(intent.read_text())
        else:
            now = datetime.now(ZoneInfo('Europe/Warsaw'))
            day, clock = now.strftime('%Y-%m-%d'), now.strftime('%H:%M')
            notes = root / 'research/2026-09-28-dual-5060ti/notes'
            entry, index = notes / f'{day}.md', notes / 'INDEX.md'
            draft = root / 'docs/ledger/drafts' / f'{section}-{day}.md'
            heading = f'## [{day}.md]({day}.md)'
            prefix = '' if index.exists() and heading in index.read_text() else f'\n{heading}\n'
            additions = {
                str(entry): f'\n{marker}\n## {clock} {lever}\n\n- {evidence}\n',
                str(index): prefix + f'- {day} {clock} {lever} ([{day}.md]({day}.md)) {marker}\n',
                str(draft): row + f' {marker}\n',
            }
            cal = calibration.assess(out)
            if 'row' in cal:
                additions[str(root / 'docs/calibration.md')] = calibration.row(day, out.get('experiment_id') or Path(out['run_dir']).parent.name,
                                                                               lever, cal, verdict).rstrip('\n') + f' {marker}\n'
            # Save destinations before the first append so a midnight retry repairs the original record.
            lib.write_json(intent, additions)
        for path, addition in additions.items():
            if path.endswith('docs/calibration.md') and not Path(path).exists():
                addition = calibration.HEADER + addition
            append_once(Path(path), addition, marker)
    return list(additions)
