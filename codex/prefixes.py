#!/usr/bin/env python3
"""Extract observed histories. This cannot recover missing request/token bytes."""
import json
import shutil
from pathlib import Path


def text(item):
    c = item.get('content', [])
    return c if isinstance(c, str) else ''.join(x.get('text', '') for x in c)


def messages(items, instructions):
    out = [{'role': 'system', 'content': instructions}]
    for item in items:
        typ = item['type']
        if typ == 'message' and item.get('role') in ['system', 'developer', 'user']:
            role = item['role']
            if role in ['system', 'developer']:
                out[0]['content'] += '\n\n' + text(item)
            else:
                out.append({'role': role, 'content': text(item)})
        elif typ in ['reasoning', 'function_call', 'message']:
            if out[-1]['role'] != 'assistant':
                out.append({'role': 'assistant', 'content': '', 'reasoning_content': ''})
            msg = out[-1]
            if typ == 'reasoning':
                msg['reasoning_content'] = text(item)
            elif typ == 'message':
                msg['content'] += text(item)
            else:
                msg.setdefault('tool_calls', []).append(dict(type='function', id=item['call_id'],
                    function=dict(name=item['name'], arguments=json.loads(item['arguments']))))
        elif typ == 'function_call_output':
            out.append(dict(role='tool', content=item['output'], tool_call_id=item['call_id']))
        else:
            raise ValueError('unsupported history item: ' + typ)
    return out


def main():
    out = Path('codex/prefixes')
    out.mkdir(exist_ok=True)
    rows = [json.loads(s) for s in Path('codex/census/unique.jsonl').open()]
    manifest = []
    for r in rows:
        if not r['repetition']:
            continue
        ev = min(r['evidence'], key=lambda f: (f['line'], f['start']))
        p = Path(ev['path'])
        objects = [json.loads(l) for l in p.open()]
        meta = next(o['payload'] for o in objects if o['type'] == 'session_meta')
        items = [o['payload'] for o in objects[:ev['line'] - 1] if o['type'] == 'response_item']
        target = objects[ev['line'] - 1]['payload']
        content = target.get('arguments') if ev['field'] == 'arguments' else text(target)
        cut = max(0, ev['start'] - 512)
        # The current assistant turn begins after the last tool result.
        boundary = max(i for i, item in enumerate(items) if item['type'] == 'function_call_output') + 1
        history, active = items[:boundary], items[boundary:]
        continuation = ''
        if ev['field'] == 'arguments':
            argument_prefix = content[:cut]
            if not argument_prefix.startswith('{"cmd":"'):
                raise ValueError('cannot reconstruct tool parameter prefix')
            # JSON arguments are parser output; the model emits XML parameter text.
            cmd_fragment = json.loads('"' + argument_prefix[len('{"cmd":"'):] + '"')
            reasoning = ''.join(text(i) for i in active if i['type'] == 'reasoning')
            prose = ''.join(text(i) for i in active if i['type'] == 'message')
            continuation = reasoning + '\n</think>\n\n' + prose + '\n\n<tool_call>\n<function=' + target['name'] + '>\n<parameter=cmd>\n' + cmd_fragment
        else:
            continuation = content[:cut]
        name = ('swift15-' if r['checkpoint'] == 'Swift1.5' else 'swift10-') + r['engine'].replace('.', '') + '-' + r['k'] + '-' + p.stem[-8:]
        instructions = meta.get('base_instructions', {}).get('text', '')
        fixture = dict(id=name, task=r['task'], source=str(p), source_line=ev['line'],
            source_field=ev['field'], character_cut_M=cut, onset_character_M=ev['start'],
            exact_prefix=False, fidelity='reconstructed text; missing original token IDs, request tool schemas, input filtering, grammar and sampler state',
            responses_input=history, instructions=instructions,
            messages=messages(history, instructions), continuation_text=continuation,
            template='chat-template.jinja', tools=[], request_overrides={},
            primary=('phase1-S4' in r['campaign'] and 'pair1' in r['campaign']) or ev['field'] == 'arguments')
        (out / (name + '.json')).write_text(json.dumps(fixture, indent=2) + '\n')
        manifest.append(dict(id=name, file=name + '.json', primary=fixture['primary'], source=str(p), exact_prefix=False))
        if fixture['primary'] and not (out / 'chat-template.jinja').exists():
            shutil.copyfile(Path(r['attempt']) / 'chat-template.jinja', out / 'chat-template.jinja')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('M: reconstructed', len(manifest), 'prefixes;', sum(p['primary'] for p in manifest), 'primary')


if __name__ == '__main__':
    main()
