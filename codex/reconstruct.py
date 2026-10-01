#!/usr/bin/env python3
"""Extract response text and provenance, without claiming original token IDs."""
import argparse
import hashlib
import json
import pathlib
import re

CAMPAIGN = pathlib.Path('/home/xanewok/repos/qwen38-weekend/results/weekend-20260930-main')
LETTERS = re.compile(r'(?:[A-Z][,.] ?){16,}')
ZEROS = re.compile(r'(?:\[0, 0, 0, 0\],\s*){16,}')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(item):
    return ''.join(p.get('text', '') for p in item.get('content', []))


def cmd_prefix(arguments):
    match = re.search(r'"cmd"\s*:\s*"', arguments)
    if not match:
        raise ValueError('runaway tool call has no cmd string')
    value = arguments[match.end():]
    for trim in range(7):
        try:
            return json.loads('"' + (value[:-trim] if trim else value) + '"')
        except json.JSONDecodeError:
            pass
    raise ValueError('cannot decode truncated cmd string')


def response_stream(items):
    result = ''
    for item in items:
        kind = item['type']
        if kind == 'reasoning':
            result += text(item) + '\n</think>\n\n'
        elif kind == 'message':
            result += text(item)
        elif kind == 'function_call':
            args = json.loads(item['arguments'])
            result += '<tool_call>\n<function=' + item['name'] + '>\n'
            for key, value in args.items():
                value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                result += '<parameter=' + key + '>\n' + value + '\n</parameter>\n'
            result += '</function>\n</tool_call>'
        else:
            raise ValueError('unexpected generated item: ' + kind)
    return result


def request(meta, items):
    clean = [{k: v for k, v in item.items() if k != 'internal_chat_message_metadata_passthrough'}
             for item in items]
    return {'instructions': meta['base_instructions']['text'], 'input': clean}


def candidates(attempt, runaway):
    for rollout in sorted((attempt / 'codex-state/sessions').rglob('*.jsonl')):
        events = [(n, json.loads(line)) for n, line in enumerate(rollout.open(), 1)]
        meta = next(e['payload'] for _, e in events if e['type'] == 'session_meta')
        items = []
        start = 0
        for line, event in events:
            if event['type'] != 'response_item':
                continue
            item = event['payload']
            kind = item['type']
            if (kind == 'message' and item.get('role') != 'assistant') or kind == 'function_call_output':
                items.append(item)
                start = len(items)
                continue
            if kind not in ('reasoning', 'message', 'function_call'):
                raise ValueError('unsupported rollout item: ' + kind)
            raw = text(item) if kind != 'function_call' else item['arguments']
            if kind == 'reasoning':
                spans = list(LETTERS.finditer(raw))
                generation = response_stream(items[start:]) + raw
                shift = len(generation) - len(raw)
            elif kind == 'function_call' and len(raw) > 100000:
                value = cmd_prefix(raw)
                spans = list(ZEROS.finditer(value))
                generation = response_stream(items[start:]) + '<tool_call>\n<function=' + item['name'] + '>\n<parameter=cmd>\n' + value
                shift = len(generation) - len(value)
            else:
                spans = []
                generation = response_stream(items[start:]) + (raw if kind == 'message' else '')
                shift = 0
            longest = max(spans, key=lambda s: len(s[0]), default=None)
            if (runaway and longest) or (not runaway and kind in ('reasoning', 'message')):
                onset = shift + longest.start() if longest else max(0, len(generation) - 160)
                yield {
                    'request': request(meta, items[:start]), 'generation': generation,
                    'onset_char': onset, 'runaway': runaway,
                    'provenance': {'attempt': str(attempt), 'rollout': str(rollout), 'line': line,
                                   'rollout_sha256': digest(rollout), 'event_type': kind,
                                   'span_chars_M': len(longest[0]) if longest else 0,
                                   'template_sha256': digest(attempt / 'chat-template.jinja'),
                                   'launch_sha256': digest(attempt / 'launch.json'),
                                   'spec_sha256': digest(attempt / 'spec.json')},
                }
            items.append(item)


def extract(root, out):
    out.mkdir(parents=True, exist_ok=True)
    cases = [('S4-pair' + str(i), f'*phase1-S4-A-IQ4_XS-*pair{i}', 'rust/alphametics', True) for i in (1, 2, 3)]
    cases += [('S5-codons', '*phase2-S5-A-IQ4_XS-*pair2', 'rust/nucleotide-codons', True)]
    cases += [('S0-alphametics', '*phase1-S0-A-IQ4_XS-*pair2', 'rust/alphametics', False),
              ('S0-list', '*phase1-S0-A-IQ4_XS-*pair3', 'rust/doubly-linked-list', False)]
    manifest = []
    for name, pattern, task, runaway in cases:
        attempts = [p.parent for p in root.glob(pattern + '/attempts/*/spec.json')
                    if json.loads(p.read_text())['task'] == task]
        if len(attempts) != 1:
            raise ValueError(f'expected one attempt for {name}: {attempts}')
        attempt = attempts[0]
        found = list(candidates(attempt, runaway))
        if not found:
            raise ValueError('missing response for ' + name)
        case = max(found, key=lambda c: c['provenance']['span_chars_M'] if runaway else
                   len(json.dumps(c['request'], ensure_ascii=False)) + len(c['generation']))
        case['name'] = name
        case['onset_utf8_bytes'] = len(case['generation'][:case['onset_char']].encode('utf-8'))
        launch = json.loads((attempt / 'launch.json').read_text())['executed_command']
        case['configured_rs'] = int(launch[launch.index('--spec-draft-n-max') + 1])
        case['context_size'] = int(launch[launch.index('-c') + 1])
        case['template'] = (attempt / 'chat-template.jinja').read_text()
        case['fidelity'] = {
            'original_token_ids': False, 'raw_requests': False, 'tool_schemas': 'not archived; omitted',
            'instructions': 'session_meta.base_instructions plus ordered response_item input messages',
            'current_generation': 'reasoning/content text; tool XML inferred from saved template',
            'reasoning_effort': 'xhigh inferred from turn_context; template default agrees',
            'onset': 'start of longest qualifying span; reproducible heuristic, not causal onset',
            'tool_arguments': 'JSON string unescaped for generated cmd; XML whitespace is inferred',
            'sampling': 'ungrammared configured filters; grammar state and original RNG unavailable',
        }
        path = out / (name + '.history.json')
        path.write_text(json.dumps(case, ensure_ascii=False, indent=2) + '\n')
        manifest.append({'name': name, 'history': path.name, 'sha256': digest(path),
                         'provenance': case['provenance'], 'fidelity': case['fidelity']})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps([{k: m[k] for k in ('name', 'history', 'sha256')} for m in manifest], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=pathlib.Path, default=CAMPAIGN)
    parser.add_argument('--out', type=pathlib.Path, default=pathlib.Path('build-histories'))
    args = parser.parse_args()
    extract(args.campaign, args.out)
