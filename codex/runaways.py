#!/usr/bin/env python3
"""Read-only campaign census. Character offsets refer to decoded JSON fields."""
import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOTS = [
    Path('/home/xanewok/repos/qwen38-eval/results/session2/pair-5060ti-x2'),
    Path('/home/xanewok/repos/qwen38-eval/results/session2/pair-5060ti-2060s'),
    Path('/home/xanewok/repos/qwen38-weekend/results'),
]
FACTORS = ['engine', 'checkpoint', 'quant', 'k', 'subset', 'draft', 'backend',
           'draft_backend', 'min_p', 'kv', 'build', 'workload']
LETTER = re.compile(r'(?:[A-Z][,.]\s*){32,}')
TOKEN = re.compile(r'\w+|[^\w\s]')
PARSE = re.compile(r'(?i)(failed to parse|error parsing|invalid json|invalid tool|tool.*parse error|failed to deserialize)')


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def spans(text):
    """Flag sustained low-diversity output, not total response length."""
    found = []
    for m in LETTER.finditer(text):
        if m.end() - m.start() >= 4096:
            found.append({'kind': 'letter_span', 'start': m.start(), 'end': m.end()})
    if len(text) < 4096:
        return found
    tokens = list(TOKEN.finditer(text))
    begin = None
    for i in range(0, len(tokens) + 256, 256):
        block = tokens[i:i + 256]
        words = [t.group() for t in block]
        grams = set(zip(words, words[1:], words[2:], words[3:]))
        low = len(block) == 256 and len(grams) <= 63 and len(set(words)) <= 64
        if low and begin is None:
            begin = i
        if not low and begin is not None:
            a, b = tokens[begin].start(), tokens[i - 1].end()
            if i - begin >= 1024 and b - a >= 4096:
                found.append({'kind': 'ngram_span', 'start': a, 'end': b})
            begin = None
    lines = text.splitlines(keepends=True)
    start = 0
    begin = None
    for i in range(0, len(lines) + 64, 64):
        block = lines[i:i + 64]
        low = len(block) == 64 and len(set(s.strip() for s in block)) <= 8
        if low and begin is None:
            begin = start
        if not low and begin is not None:
            if start - begin >= 4096:
                found.append({'kind': 'line_span', 'start': begin, 'end': start})
            begin = None
        start += sum(map(len, block))
    for f in found:
        f.update(chars=f['end'] - f['start'], sample=text[f['start']:f['start'] + 240])
    return found


def generated(payload):
    typ = payload.get('type')
    if typ == 'function_call':
        yield 'arguments', payload.get('arguments', '')
    elif typ == 'custom_tool_call':
        yield 'input', payload.get('input', '')
    elif typ == 'reasoning' or (typ == 'message' and payload.get('role') == 'assistant'):
        for field in ['content', 'summary']:
            for i, c in enumerate(payload.get(field) or []):
                if isinstance(c, dict) and c.get('text'):
                    yield f'{field}.{i}.text', c['text']


def discover(roots):
    found = set()
    for root in roots:
        for base, dirs, _ in os.walk(root):
            if 'attempts' in dirs:
                campaign = Path(base)
                for a in (campaign / 'attempts').iterdir():
                    if a.is_dir():
                        found.add(a.resolve())
                dirs.remove('attempts')
            dirs[:] = [d for d in dirs if d not in ['snapshot', 'workspace', 'grade-0', 'codex-state',
                                                  '.git', 'node_modules', 'sp-archive-20260929']]
    return sorted(found)


def option(cmd, *names, default='unknown'):
    for name in names:
        if name in cmd:
            i = cmd.index(name)
            return cmd[i + 1] if i + 1 < len(cmd) else default
    return default


def flag(cmd, positive, negative, default='unknown'):
    if positive in cmd:
        return 'on'
    if negative in cmd:
        return 'off'
    return default


def configuration(a, probe):
    launch = load(a / 'launch.json')
    manifest = {}
    for p in a.parents:
        if (p / 'manifest.json').exists():
            manifest = load(p / 'manifest.json')
            break
    launch = launch or manifest.get('launch', {})
    cmd = launch.get('executed_command', launch.get('command', []))
    engine = 'vllm' if 'vllm' in str(cmd).lower() or manifest.get('engine') == 'vllm' else 'llama.cpp'
    model = option(cmd, '-m', '--model', default='')
    if engine == 'vllm':
        model = cmd[cmd.index('serve') + 1] if 'serve' in cmd else str(manifest.get('model', 'unknown'))
    spec = load(a / 'spec.json')
    model = model or spec.get('model', 'unknown')
    quant = 'unknown'
    for q in ['IQ3_XXS', 'IQ4_XS', 'Q4_K_XL', 'Q4_K_M', 'Q5_K_S', 'Q5_K_M', 'Q6_K', 'J5q', 'J4', 'J1', 'W4A16']:
        if q.lower().replace('_', '-') in model.lower().replace('_', '-'):
            quant = q
            break
    if 'j1' in spec.get('model', '').lower() or 'ssmout6' in model:
        quant = 'J1'
    checkpoint = 'Swift1.5' if re.search(r'(swift[-_]?1[._-]?5|swift15)', model, re.I) else 'Swift1.0' if 'swift' in model.lower() else 'BaseQwen3.8' if 'qwen3.8' in model.lower() else 'unknown'
    sampling = probe.get('sampling', {})
    for sample in probe.get('slot_samples', [])[:5]:
        for slot in sample.get('slots', []):
            if slot.get('params'):
                sampling = slot['params']
                break
    build = manifest.get('engine_commit', 'unknown')
    recorded_commit = build
    # A recorded immutable executable hash distinguishes same-commit local builds.
    pins = manifest.get('binary_pins', {})
    binary = launch.get('command', ['unknown'])[0] if launch.get('command') else 'unknown'
    bin_hash = pins.get(binary, '')
    if bin_hash:
        build += '/' + bin_hash[:12]
    k = option(cmd, '--spec-draft-n-max', default=str(spec.get('config', {}).get('maximum', 'unknown')))
    draft = option(cmd, '--spec-draft-sampling')
    backend = flag(cmd, '--backend-sampling', '--no-backend-sampling')
    draft_backend = flag(cmd, '--spec-draft-backend-sampling', '--no-spec-draft-backend-sampling')
    subset = 'on' if '--spec-draft-vocab' in cmd else 'off' if cmd else 'unknown'
    min_p = str(sampling.get('min_p', option(cmd, '--min-p')))
    kv = launch.get('target_kv', option(cmd, '-ctk', '--cache-type-k') + '/' + option(cmd, '-ctv', '--cache-type-v'))
    defaults = recorded_commit[:9] in ['03fa73cb2', '0dba27b90', '3ec79cbfa', '68d99c67d',
                                     '92dc59bd4', 'd2cbc6290', 'dba0e6bf3', 'df5f4fa3e', 'ed7ac35e1']
    inferred = []
    if engine == 'llama.cpp' and cmd and defaults:
        current = dict(draft=draft, backend=backend, draft_backend=draft_backend, min_p=min_p)
        for name, value in [('draft', 'greedy'), ('backend', 'off'), ('draft_backend', 'on'), ('min_p', '0.05')]:
            if current[name] == 'unknown':
                inferred.append(name)
                if name == 'draft': draft = value
                elif name == 'backend': backend = value
                elif name == 'draft_backend': draft_backend = value
                else: min_p = value
    if min_p != 'unknown':
        min_p = format(float(min_p), '.6g')
    if engine == 'vllm':
        conf = json.loads(option(cmd, '--speculative-config', default='{}'))
        gen = json.loads(option(cmd, '--override-generation-config', default='{}'))
        k = str(conf.get('num_speculative_tokens', 0))
        draft = conf.get('draft_sample_method', 'unknown')
        subset = 'engine-default/unknown'
        backend = draft_backend = 'vllm'
        min_p = str(gen.get('min_p', sampling.get('min_p', 'unknown')))
        kv = option(cmd, '--kv-cache-dtype')
        build = 'vllm-' + launch.get('vllm_version', 'unknown')
    if option(cmd, '--spec-type') == 'none':
        k, draft = '0', 'none'
    return dict(engine=engine, checkpoint=checkpoint, quant=quant, k=k, subset=subset, draft=draft,
                backend=backend, draft_backend=draft_backend, min_p=min_p, kv=kv, build=build,
                binary=binary, model=model, model_sha=launch.get('model_sha256', 'unknown'),
                inferred_source_defaults=inferred,
                workload=probe.get('workload', spec.get('config', {}).get('workload', 'profile' if probe.get('requests') else 'unknown')),
                context=option(cmd, '-c', '--ctx-size', '--max-model-len', default=str(probe.get('client_context', 'unknown'))))


def inspect(a):
    probe = load(a / 'probe.json')
    result = load(a / 'result.json')
    spec = load(a / 'spec.json')
    row = dict(attempt=str(a), campaign=str(a.parent.parent), task=probe.get('task_id', spec.get('task', 'unknown')))
    row.update(configuration(a, probe))
    row['completion_tokens'] = probe.get('agent_stats', {}).get('completion_tokens', probe.get('metrics_delta', {}).get('decode_tokens', 0)) or 0
    row['complete'] = probe.get('complete')
    row['terminal_status'] = result.get('status', 'unrecorded')
    row['censored'] = probe.get('complete') is not True and not probe.get('requests')
    row['max_response_tokens'] = max((x.get('usage', {}).get('completion_tokens', 0) for x in probe.get('requests', [])), default=0)
    row['peak_context'] = probe.get('peak_context_tokens', 0)
    evidence, errors = [], []
    texts = chars = bad_lines = 0
    rollouts = sorted(a.glob('codex-state/sessions/**/rollout-*.jsonl'))
    identities = []
    for p in rollouts:
        for line_num, line in enumerate(p.open(errors='replace'), 1):
            try:
                o = json.loads(line)
            except ValueError:
                bad_lines += 1
                continue
            if o.get('type') != 'response_item':
                if o.get('type') == 'session_meta':
                    identities.append(o['payload'].get('id', o['payload'].get('session_id')))
                continue
            for field, text in generated(o.get('payload', {})):
                texts += 1
                chars += len(text)
                for f in spans(text):
                    evidence.append(dict(f, path=str(p), line=line_num, field=field))
    for p in sorted(a.glob('codex-*.jsonl')):
        for line_num, line in enumerate(p.open(errors='replace'), 1):
            try:
                o = json.loads(line)
            except ValueError:
                bad_lines += 1
                continue
            item = o.get('item', {})
            if item.get('type') == 'error' or o.get('type') in ['error', 'turn.failed']:
                if PARSE.search(line):
                    errors.append(dict(path=str(p), line=line_num, text=line[:500]))
            if not rollouts:
                for key in ['text', 'command']:
                    if item.get(key) and o.get('type') == 'item.completed':
                        text = item[key]
                        texts += 1
                        chars += len(text)
                        evidence.extend(dict(f, path=str(p), line=line_num, field=key) for f in spans(text))
    for p in a.glob('codex-*.stderr'):
        for i, line in enumerate(p.open(errors='replace'), 1):
            if PARSE.search(line):
                errors.append(dict(path=str(p), line=i, text=line[:500]))
    # Profile SSE saves deltas; never inspect tool results or input prompts as generated output.
    if not texts:
        for p in a.glob('request-*.sse'):
            parts = []
            for line in p.open(errors='replace'):
                if not line.startswith('data: '):
                    continue
                try:
                    o = json.loads(line[6:])
                except ValueError:
                    continue
                for c in o.get('choices', []):
                    d = c.get('delta', {})
                    parts.extend(d[k] for k in ['content', 'reasoning_content'] if isinstance(d.get(k), str))
            text = ''.join(parts)
            if text:
                texts += 1
                chars += len(text)
                evidence.extend(dict(f, path=str(p), line=0, field='joined_deltas') for f in spans(text))
    log = (a / 'server.log').read_text(errors='replace') if (a / 'server.log').exists() else ''
    row['truncated'] = bool(re.search(r'\btruncated\s*=\s*1\b', log))
    row['max_response_tokens'] = max([row['max_response_tokens']] + [int(n) for line in log.splitlines()
        if 'prompt eval time' not in line for n in re.findall(r'eval time\s*=.*?/\s*(\d+) tokens', line)])
    for p in a.glob('model-calls.jsonl'):
        for line in p.open():
            try:
                o = json.loads(line)
                row['max_response_tokens'] = max(row['max_response_tokens'], o.get('output_tokens', 0))
            except ValueError:
                bad_lines += 1
    if row['workload'] == 'unknown' and rollouts:
        row['workload'] = 'agentic'
    identity = 'sessions:' + ','.join(sorted(set(identities))) if identities else ''
    if not identity and probe.get('requests'):
        starts = [x.get('started_monotonic') for x in probe['requests']]
        if all(x is not None for x in starts):
            identity = 'requests:' + json.dumps(starts)
    row.update(exposure_id=identity or str(a), texts=texts, generated_chars=chars, rollout_files=len(rollouts), malformed_lines=bad_lines,
               repetition=bool(evidence), evidence=evidence, parse_errors=errors,
               long_completion=row['completion_tokens'] >= 50000,
               long_response=row['max_response_tokens'] >= 32768,
               observed_output=bool(chars or row['completion_tokens']))
    return row


def deduplicate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['exposure_id']].append(r)
    unique = []
    for group in groups.values():
        # Prefer an archived copy with metrics; aliases retain the whole provenance.
        canonical = max(group, key=lambda r: (r['complete'] is True, r['completion_tokens'], r['generated_chars']))
        canonical['aliases'] = [r['attempt'] for r in group if r is not canonical]
        canonical['alias_config_conflict'] = len({tuple(r[k] for k in FACTORS if k != 'workload') for r in group}) > 1
        unique.append(canonical)
    return unique


def ci(x, n):
    from scipy.stats import beta
    if not n:
        return None, None
    return (0.0 if x == 0 else float(beta.ppf(.025, x, n - x + 1)),
            1.0 if x == n else float(beta.ppf(.975, x + 1, n - x)))


def tables(rows, out):
    fields = FACTORS + ['attempts', 'output_attempts', 'text_attempts', 'rollout_attempts', 'censored', 'repetition', 'ci_low', 'ci_high',
                        'truncated', 'long_completion', 'long_response', 'parse_error_attempts']
    grouped = defaultdict(list)
    for r in rows:
        grouped[tuple(r[k] for k in FACTORS)].append(r)
    with (out / 'rates.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for key, group in sorted(grouped.items()):
            eligible = [r for r in group if r['observed_output']]
            x = sum(r['repetition'] for r in eligible)
            low, high = ci(x, len(eligible))
            writer.writerow(dict(zip(FACTORS, key), attempts=len(group), output_attempts=len(eligible),
                                 text_attempts=sum(bool(r['texts']) for r in eligible), rollout_attempts=sum(bool(r['rollout_files']) for r in eligible),
                                 censored=sum(r['censored'] for r in eligible),
                                 repetition=x, ci_low=low, ci_high=high,
                                 **{k: sum(bool(r[k]) for r in eligible) for k in ['truncated', 'long_completion', 'long_response']},
                                 parse_error_attempts=sum(bool(r['parse_errors']) for r in eligible)))
    for factors, name in [(['engine', 'checkpoint', 'quant', 'workload'], 'quant'), (['campaign'], 'campaign')]:
        groups = defaultdict(list)
        for r in rows:
            groups[tuple(r[k] for k in factors)].append(r)
        with (out / f'{name}-rates.csv').open('w') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(factors + ['attempts_M', 'output_M', 'text_M', 'censored_M', 'repeat_M', 'low_D', 'high_D', 'truncated_M', 'long_M', 'parse_M'])
            for key, group in sorted(groups.items()):
                eligible = [r for r in group if r['observed_output']]
                x = sum(r['repetition'] for r in eligible)
                w.writerow(list(key) + [len(group), len(eligible), sum(bool(r['texts']) for r in eligible), sum(r['censored'] for r in eligible), x, *ci(x, len(eligible)),
                                       sum(r['truncated'] for r in eligible), sum(r['long_completion'] for r in eligible), sum(bool(r['parse_errors']) for r in eligible)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', action='append', type=Path)
    ap.add_argument('--out', type=Path, default=Path('codex/census'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    attempts = discover(args.root or ROOTS)
    with (args.out / 'attempts.jsonl').open('w') as f:
        for i, a in enumerate(attempts):
            row = inspect(a)
            rows.append(row)
            f.write(json.dumps(row) + '\n')
            if i % 100 == 0:
                print(f'M: inspected {i}/{len(attempts)} attempts', flush=True)
    unique = deduplicate(rows)
    with (args.out / 'unique.jsonl').open('w') as f:
        for r in unique:
            f.write(json.dumps(r) + '\n')
    tables(unique, args.out)
    summary = dict(roots=list(map(str, args.root or ROOTS)), attempts_M=len(rows),
                   started_utc_M=started, finished_utc_M=datetime.now(timezone.utc).isoformat(),
                   output_M=sum(r['observed_output'] for r in rows), text_M=sum(bool(r['texts']) for r in rows),
                   repetition_M=sum(r['repetition'] for r in rows), seconds_M=time.monotonic() - start,
                   malformed_M=sum(r['malformed_lines'] for r in rows),
                   unique_M=len(unique), duplicates_M=len(rows) - len(unique),
                   unique_output_M=sum(r['observed_output'] for r in unique),
                   alias_conflicts_M=sum(r['alias_config_conflict'] for r in unique),
                   detector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
