#!/usr/bin/env python3
import collections
import hashlib
import json
import pathlib
import re

root = pathlib.Path('/home/xanewok/repos/qwen38-weekend/results/weekend-20260930-main')
result = []
for spec_path in sorted(root.glob('*phase1*/attempts/*/spec.json')):
    spec = json.loads(spec_path.read_text())
    if spec.get('task') != 'rust/alphametics':
        continue
    attempt = spec_path.parent
    lines = (attempt / 'server.log').read_text().splitlines()
    launch = json.loads((attempt / 'launch.json').read_text())
    probe = json.loads((attempt / 'probe.json').read_text())
    warnings = collections.Counter(re.sub(r'^\S+ W ', '', line) for line in lines if ' W ' in line)
    requests = {}
    for number, line in enumerate(lines, 1):
        match = re.search(r'task (\d+) \|', line)
        if not match:
            continue
        task = match[1]
        if any(s in line for s in ('prompt eval time =', '       eval time =', 'draft acceptance =', 'stop processing:')):
            requests.setdefault(task, []).append({'line': number, 'text': line})
    record = {
        'campaign': attempt.parent.parent.name,
        'attempt': attempt.name,
        'path': str(attempt),
        'source_command': launch['executed_command'],
        'spec_seed': spec['seed'],
        'probe': probe,
        'requests': requests,
        'warnings': dict(warnings),
        'errors': [{'line': n, 'text': l} for n, l in enumerate(lines, 1) if ' E ' in l],
        'restore_replay': [{'line': n, 'text': l} for n, l in enumerate(lines, 1) if re.search(r'restor|replay|checkpoint', l, re.I)],
        'server_log_sha256': hashlib.sha256((attempt / 'server.log').read_bytes()).hexdigest(),
        'template_sha256': hashlib.sha256((attempt / 'chat-template.jinja').read_bytes()).hexdigest(),
        'prompt_sha256': hashlib.sha256((attempt / 'prompt-0.txt').read_bytes()).hexdigest(),
    }
    record['reasoning'] = []
    for rollout in sorted((attempt / 'codex-state/sessions').rglob('*.jsonl')):
        for number, line in enumerate(rollout.open(), 1):
            event = json.loads(line)
            payload = event.get('payload', {})
            if event.get('type') != 'response_item' or payload.get('type') != 'reasoning':
                continue
            text = '\n'.join(part.get('text', '') for key in ('content', 'summary') for part in payload.get(key, []) if isinstance(part, dict))
            spans = list(re.finditer(r'(?:[A-Z][,.] ?){16,}', text))
            longest = max(spans, key=lambda span: len(span[0]), default=None)
            record['reasoning'].append({
                'rollout': str(rollout), 'line': number, 'chars': len(text), 'tail': text[-160:],
                'letter_span_chars': len(longest[0]) if longest else 0,
                'letter_span_start': longest.start() if longest else None,
                'letter_span_sample': longest[0][:120] if longest else None,
            })
    result.append(record)
print(json.dumps(result, indent=2))
