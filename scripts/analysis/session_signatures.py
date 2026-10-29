"""Post-hoc signatures from sealed coding results, rollouts and server counters."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import time

from scripts.analysis.overnight_summary import campaign_terminal, coding_rows
from scripts.instruments.hillclimb import atomic_json, digest_file, read_sealed


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp() if value else None


def text_size(content):
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(item.get('text', '')) for item in content if isinstance(item, dict))
    return 0


def command_text(arguments):
    command = arguments.get('cmd', arguments.get('command', ''))
    if isinstance(command, list):
        return command[-1] if len(command) >= 3 and command[1] in ('-c', '-lc') else shlex.join(command)
    return command if isinstance(command, str) else ''


def action_tags(name, arguments):
    if name == 'write_stdin':
        return ['poll' if not arguments.get('chars') else 'interactive_input']
    if 'apply_patch' in name:
        return ['edit']
    command = command_text(arguments)
    # Categories describe shell text, not proven intent; mixed calls can have several labels.
    patterns = {
        'edit': r'\b(?:apply_patch|tee)\b|\bcat\s*>|\bsed\s+-i|\.write_(?:text|bytes)\s*\(|open\([^\n]*,[ ]*[\x22\x27][wax]|\bwriteFile',
        'test': r'\bcargo\s+test\b|\b(?:npx\s+(?:--no-install\s+)?)?jest\b|\bnpm\s+(?:run\s+)?test\b|\bpytest\b|\bpython\S*\s+-m\s+unittest\b',
        'lint_format': r'\b(?:eslint|prettier|rustfmt)\b|\bcargo\s+(?:fmt|clippy)\b',
        'build': r'\bcargo\s+(?:build|check)\b|\bnpm\s+run\s+build\b',
        'dependencies': r'\bnpm\s+(?:install|ci)\b|\bpip\s+install\b|\bcargo\s+add\b',
        'inspect': r'\b(?:ls|find|rg|grep|cat|head|tail|wc)\b|\bsed\s+-n\b|\.read_(?:text|bytes)\s*\(',
        'git': r'\bgit\s+(?:diff|status|log|show)\b',
        'cleanup': r'\brm\s',
    }
    # Do not classify embedded source code as shell commands after a heredoc opener.
    shell_head = command.split('\n', 1)[0] if '<<' in command else command
    tags = [tag for tag, pattern in patterns.items() if re.search(pattern, command if tag == 'edit' else shell_head)]
    return tags or ['other']


def interval_union_seconds(intervals):
    total = 0.0
    end = None
    for start, stop in sorted(intervals):
        if stop < start:
            continue
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end) if end is not None else stop
    return total


def server_requests(log):
    records = {}
    for number, line in enumerate(log.splitlines(), 1):
        task = re.search(r'\| task\s+(\d+)\s*\|', line)
        if not task:
            continue
        key = task[1]
        match = re.search(r'(prompt eval time|(?<!prompt )\beval time)\s*=\s*([\d.]+) ms /\s*(\d+) tokens', line)
        if match:
            row = records.setdefault(key, {'server_task_id': key})
            prefix = 'prefill' if match[1] == 'prompt eval time' else 'decode'
            row[prefix + '_seconds'] = float(match[2]) / 1000
            row[prefix + '_tokens'] = int(match[3])
            row[prefix + '_log_line'] = number
        release = re.search(r'stop processing: n_tokens = (\d+), truncated = (\d+)', line)
        if release and key in records:
            records[key].update(end_context_tokens=int(release[1]), truncated=bool(int(release[2])))
    return [r for r in records.values() if 'prefill_seconds' in r and 'decode_seconds' in r]


def parse_rollouts(paths):
    requests, calls, outputs, command_items, compactions, compact_items = {}, {}, {}, {}, [], {}
    task_starts, task_completions, client_windows, cli_versions = set(), set(), set(), set()
    messages = Counter()
    content_ids = set()
    for path in paths:
        pending = {'reasoning_chars': 0, 'assistant_chars': 0, 'tool_argument_chars': 0, 'call_ids': []}
        thread = str(path)
        for number, line in enumerate(path.read_text().splitlines(), 1):
            row = json.loads(line)
            payload = row.get('payload', {})
            kind = row.get('type')
            now = timestamp(row.get('timestamp'))
            evidence = {'source': str(path), 'line': number}
            if kind == 'session_meta':
                thread = payload.get('id', thread)
                cli_versions.add(payload.get('cli_version', 'unknown'))
            if kind == 'token_usage_record':
                response_id = payload.get('response_id')
                if not response_id:
                    raise ValueError('token usage record has no response ID: ' + str(path))
                usage = payload['usage']
                if response_id in requests:
                    if requests[response_id]['usage'] != usage:
                        raise ValueError('conflicting repeated response usage: ' + response_id)
                    continue
                requests[response_id] = {**evidence, 'response_id': response_id, 'completed_at': now,
                                         'usage': usage, **pending}
                for call_id in pending['call_ids']:
                    calls[call_id]['response_id'] = response_id
                pending = {'reasoning_chars': 0, 'assistant_chars': 0, 'tool_argument_chars': 0, 'call_ids': []}
            elif kind == 'response_item':
                subtype = payload.get('type')
                identity = (thread, subtype, payload.get('id') or number)
                if identity in content_ids:
                    continue
                content_ids.add(identity)
                if subtype in ('function_call', 'custom_tool_call'):
                    call_id = payload['call_id']
                    raw = payload.get('arguments', payload.get('input', ''))
                    try:
                        arguments = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        arguments = {}
                    if not isinstance(arguments, dict):
                        arguments = {}
                    command = command_text(arguments)
                    calls[call_id] = {**evidence, 'call_id': call_id, 'name': payload['name'], 'called_at': now,
                                      'actions_inferred': action_tags(payload['name'], arguments),
                                      'process_session_id': str(arguments['session_id']) if 'session_id' in arguments else None,
                                      'command_preview': command[:300], 'command_chars': len(command),
                                      'command_sha256': hashlib.sha256(command.encode()).hexdigest() if command else None}
                    pending['call_ids'].append(call_id)
                    pending['tool_argument_chars'] += len(raw) if isinstance(raw, str) else len(json.dumps(raw))
                elif subtype in ('function_call_output', 'custom_tool_call_output'):
                    output = payload.get('output', '')
                    text = output if isinstance(output, str) else json.dumps(output)
                    process = re.search(r'Process running with session ID (\d+)', text)
                    outputs[payload['call_id']] = {'returned_at': now, 'output_chars': len(text),
                        'running_process_session_id': process[1] if process else None,
                        'network_error_in_output': bool(re.search(r'\b(?:ENOTFOUND|EAI_AGAIN|ENETUNREACH|ECONNREFUSED)\b', text)),
                        'rejected_or_tool_error': bool(re.search(r'^(?:\w+ failed:|Error parsing|Error:)', text)),
                        'error_preview': text[:220] if re.search(r'^(?:\w+ failed:|Error parsing|Error:)', text) else None}
                elif subtype == 'reasoning':
                    pending['reasoning_chars'] += text_size(payload.get('content'))
                elif subtype == 'message' and payload.get('role') == 'assistant':
                    pending['assistant_chars'] += text_size(payload.get('content'))
            elif kind == 'compacted':
                compactions.append({**evidence, 'completed_at': now, 'response_id': payload.get('compaction_response_id'),
                                    'summary_chars': len(payload.get('message', ''))})
            elif kind == 'event_msg':
                subtype = payload.get('type')
                if subtype == 'task_started':
                    task_starts.add((thread, payload.get('turn_id')))
                    if payload.get('model_context_window'):
                        client_windows.add(payload['model_context_window'])
                elif subtype == 'task_complete':
                    task_completions.add((thread, payload.get('turn_id')))
                elif subtype == 'item_completed':
                    item = payload.get('item', {})
                    if item.get('type') == 'CommandExecution':
                        command_items[item['id']] = {'exit_code': item.get('exit_code'), 'status': item.get('status')}
                    elif item.get('type') == 'ContextCompaction':
                        compact_items[item['id']] = (payload.get('started_at_ms'), payload.get('completed_at_ms'))
                    elif item.get('type') == 'AgentMessage':
                        messages[item.get('phase') or 'unspecified'] += 1
    compact_ids = {r['response_id'] for r in compactions}
    for request in requests.values():
        request['is_compaction'] = request['response_id'] in compact_ids
    for call in calls.values():
        call.update(outputs.get(call['call_id'], {}))
        call['command_completed_event'] = call['call_id'] in command_items
        call.update(command_items.get(call['call_id'], {}))
    process_origins = {r['running_process_session_id']: r for r in calls.values()
                       if r['name'] == 'exec_command' and r.get('running_process_session_id')}
    for call in calls.values():
        origin = process_origins.get(call.get('process_session_id'))
        if origin:
            call['origin_call_id'] = origin['call_id']
            call['origin_command_preview'] = origin['command_preview']
    return {'requests': list(requests.values()), 'tools': list(calls.values()), 'compactions': compactions,
            'compaction_intervals': [(a / 1000, b / 1000) for a, b in compact_items.values() if a is not None and b is not None],
            'outer_turns_started': len(task_starts), 'outer_turns_completed': len(task_completions),
            'agent_messages': dict(messages), 'client_reported_windows': sorted(client_windows),
            'cli_versions': sorted(cli_versions), 'command_completed_events': len(command_items)}


def session_signature(campaign, row):
    out = campaign / row['attempt']
    probe = row['probe']
    paths = sorted((out / 'codex-state/sessions').rglob('rollout-*.jsonl'))
    if not paths:
        raise ValueError('missing rollout for measured task: ' + str(out))
    parsed = parse_rollouts(paths)
    requests, tools = parsed['requests'], parsed['tools']
    warnings = []
    before = json.loads((out / 'metrics-before.json').read_text())
    after = json.loads((out / 'metrics-after.json').read_text())
    peak = after.get('n_tokens_max')
    if before.get('n_tokens_max') != 0 or peak is None or peak > probe['client_context']:
        warnings.append('fresh-server context high-water could not be validated')
        peak = None
    usage = {key: sum(r['usage'].get(key, 0) for r in requests)
             for key in ('input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_output_tokens')}
    delta = probe['metrics_delta']
    if usage['output_tokens'] != delta['decode_tokens'] or usage['input_tokens'] - usage['cached_input_tokens'] != delta['prefill_tokens']:
        warnings.append('rollout/server token totals differ')
    backend = server_requests((out / 'server.log').read_text())
    matched = len(backend) == len(requests) and all(a['usage']['output_tokens'] == b['decode_tokens'] for a, b in zip(requests, backend))
    if matched:
        for request, timing in zip(requests, backend):
            request.update(timing)
    else:
        warnings.append('per-response server timing alignment failed')
    tool_intervals = [(r['called_at'], r['returned_at']) for r in tools if r.get('called_at') is not None and r.get('returned_at') is not None]
    requests_by_id = {r['response_id']: r for r in requests}
    first_edit = next((r for r in tools if 'edit' in r['actions_inferred']), None)
    edit_index = next((i for i, r in enumerate(requests) if first_edit and first_edit.get('response_id') == r['response_id']), None)
    commands = Counter(r['command_sha256'] for r in tools if r['command_sha256'])
    compaction_requests = [r for r in requests if r['is_compaction']]
    compact_before_after = []
    for i, request in enumerate(requests):
        if request['is_compaction']:
            compact_before_after.append({'response_id': request['response_id'],
                'input_tokens': request['usage']['input_tokens'], 'output_tokens': request['usage']['output_tokens'],
                'next_request_input_tokens': requests[i + 1]['usage']['input_tokens'] if i + 1 < len(requests) else None})
    return {'model': row['model'], 'task': row['task'], 'attempt': str(out), 'solved': probe['task_success'],
            'rollout_sessions': len(paths),
            'wall_seconds': probe['verified_wall_seconds'], 'prefill_seconds': probe['prefill_seconds'],
            'decode_seconds': probe['decode_seconds'], 'other_seconds': probe['other_seconds'],
            'outer_turns': parsed['outer_turns_completed'], 'model_responses': len(requests),
            'normal_model_responses': len(requests) - len(compaction_requests),
            'tool_requests': len(tools), 'tool_names': dict(Counter(r['name'] for r in tools)),
            'command_completed_events': parsed['command_completed_events'],
            'actions_inferred': dict(Counter(tag for r in tools for tag in r['actions_inferred'])),
            'action_sequence_inferred': [r['actions_inferred'][0] for r in tools],
            'nonzero_command_exits': sum(r.get('exit_code') not in (None, 0) for r in tools),
            'rejected_or_tool_errors': sum(r.get('rejected_or_tool_error', False) for r in tools),
            'network_error_outputs': sum(r.get('network_error_in_output', False) for r in tools),
            'exact_repeated_commands': sum(n - 1 for n in commands.values()),
            'tool_wait_union_seconds': interval_union_seconds(tool_intervals),
            'generated_tokens': usage['output_tokens'], 'input_tokens_cumulative': usage['input_tokens'],
            'cached_input_tokens_cumulative': usage['cached_input_tokens'],
            'max_response_output_tokens': max((r['usage']['output_tokens'] for r in requests), default=0),
            'max_request_input_tokens': max((r['usage']['input_tokens'] for r in requests), default=0),
            'visible_reasoning_chars': sum(r['reasoning_chars'] for r in requests if not r['is_compaction']),
            'output_tokens_through_first_edit_request': sum(r['usage']['output_tokens'] for r in requests[:edit_index + 1]) if edit_index is not None else None,
            'first_edit_request_response_output': requests_by_id.get(first_edit.get('response_id'), {}).get('usage', {}).get('output_tokens') if first_edit else None,
            'peak_context_tokens': peak, 'configured_context': probe['client_context'],
            'legacy_double_counted_peak': probe.get('legacy_sampled_context_tokens', probe.get('peak_context_tokens')),
            'client_compaction_threshold': probe['client_compaction'],
            'client_reported_windows': parsed['client_reported_windows'],
            'compactions': len(parsed['compactions']), 'compaction_details': compact_before_after,
            'compaction_elapsed_seconds': interval_union_seconds(parsed['compaction_intervals']),
            'compaction_inference_seconds': sum(r['prefill_seconds'] + r['decode_seconds'] for r in compaction_requests) if matched else None,
            'backend_timings_aligned': matched, 'warnings': warnings, 'agent_messages': parsed['agent_messages'],
            'cli_versions': parsed['cli_versions'],
            'sources': [{'path': str(p), 'sha256': digest_file(p)} for p in paths],
            'requests': requests, 'tools': tools, 'compaction_events': parsed['compactions']}


def paired_explanations(signatures):
    baseline = {r['task']: r for r in signatures if r['model'] == 'iq3-control' and r['solved']}
    results = []
    for model in sorted({r['model'] for r in signatures} - {'iq3-control'}):
        pairs = [(baseline[r['task']], r) for r in signatures
                 if r['model'] == model and r['solved'] and r['task'] in baseline]
        totals = dict.fromkeys(['wall_delta', 'prefill_delta', 'decode_token_volume_delta', 'decode_rate_delta', 'other_delta'], 0.0)
        for base, candidate in pairs:
            nb, nc = base['generated_tokens'], candidate['generated_tokens']
            if nb <= 0 or nc <= 0:
                raise ValueError('paired decode decomposition needs positive token counts')
            sb, sc = base['decode_seconds'] / nb, candidate['decode_seconds'] / nc
            # Average both replacement orders so token volume and rate terms sum exactly to decode delta.
            totals['decode_token_volume_delta'] += (nc - nb) * (sb + sc) / 2
            totals['decode_rate_delta'] += (nc + nb) * (sc - sb) / 2
            for target, field in [('wall_delta', 'wall_seconds'), ('prefill_delta', 'prefill_seconds'), ('other_delta', 'other_seconds')]:
                totals[target] += candidate[field] - base[field]
        results.append({'model': model, 'paired_tasks': len(pairs), **totals})
    return results


def analyze(root, corrected):
    original = root / 'campaign-agentic-20260913'
    signatures, excluded = [], []
    for name, row in coding_rows(original, corrected):
        if row['status'] != 'passed':
            excluded.append({'job': name, 'model': row.get('model'), 'reason': row.get('reason')})
            continue
        campaign = corrected if row['model'] == 'bartowski-iq4-xs' else original
        signatures.append(session_signature(campaign, row))
    state = json.loads((corrected / 'state.json').read_text())
    return {'schema_version': 1, 'analyzer_sha256': digest_file(Path(__file__)),
            'updated': datetime.now().astimezone().isoformat(), 'bartowski_state': state['status'],
            'sessions': signatures, 'excluded': excluded, 'paired_explanations': paired_explanations(signatures),
            'scope': 'post-hoc eight-task pilot; natural histories; no causal or full-suite claim'}


def analyze_campaign(campaign):
    manifest = read_sealed(campaign / 'manifest.json')
    signatures, excluded = [], []
    for job in manifest['queue']:
        path = campaign / ('job-' + job['id'] + '.json')
        if not path.exists():
            excluded.append({'job': job['id'], 'model': job['model'], 'reason': 'pending'})
            continue
        row = read_sealed(path)
        if row['status'] != 'passed':
            excluded.append({'job': job['id'], 'model': row.get('model'), 'reason': row.get('reason')})
            continue
        signatures.append(session_signature(campaign, row))
    return {'schema_version': 1, 'analyzer_sha256': digest_file(Path(__file__)),
            'updated': datetime.now().astimezone().isoformat(),
            'campaign_state': json.loads((campaign / 'state.json').read_text())['status'],
            'sessions': signatures, 'excluded': excluded, 'paired_explanations': paired_explanations(signatures),
            'scope': 'Post-hoc declared full-suite sessions; natural histories; descriptive, not causal.'}


def render(result):
    rows = result['sessions']
    lines = ['# Coding-session signatures', '', 'Updated: ' + result['updated'], '',
             f"Measured sessions: {len(rows)}; stage: {result.get('campaign_state', result.get('bartowski_state'))}.",
             result['scope'], '',
             '## Aggregate signatures', '',
             '| Model | Sessions | Model responses | Exec / poll requests | Output tokens | Longest response | Largest context | Compactions |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model in sorted({r['model'] for r in rows}):
        group = [r for r in rows if r['model'] == model]
        lines.append('| ' + ' | '.join(map(str, [model, len(group), sum(r['model_responses'] for r in group),
            f"{sum(r['tool_names'].get('exec_command', 0) for r in group)} / {sum(r['tool_names'].get('write_stdin', 0) for r in group)}",
            sum(r['generated_tokens'] for r in group), max(r['max_response_output_tokens'] for r in group),
            max((r['peak_context_tokens'] for r in group if r['peak_context_tokens'] is not None), default='unknown'),
            sum(r['compactions'] for r in group)])) + ' |')
    lines += ['', '## Output-length distribution by model response', '',
              'Each observation is one server response, including compaction; responses within a task are not independent trials.', '',
              '| Model | Median output | <512 tokens | 512–2,047 | 2,048–8,191 | >=8,192 |',
              '|---|---:|---:|---:|---:|---:|']
    for model in sorted({r['model'] for r in rows}):
        values = [q['usage']['output_tokens'] for r in rows if r['model'] == model for q in r['requests']]
        bins = [sum(n < 512 for n in values), sum(512 <= n < 2048 for n in values),
                sum(2048 <= n < 8192 for n in values), sum(n >= 8192 for n in values)]
        lines.append('| ' + ' | '.join(map(str, [model, statistics.median(values), *bins])) + ' |')
    lines += ['', '## Paired time-difference accounting', '',
              'Candidate minus single IQ3, in seconds, on matching solved tasks; negative is less time.',
              'Decode is split into output-volume and measured-rate terms by averaging both replacement orders. These add to the observed decode difference, not a causal experiment; depth, MTP and content also affect the measured rate.', '',
              '| Model | Pairs | Wall delta | Prefill delta | Decode: token volume | Decode: rate | Other delta |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in result['paired_explanations']:
        cells = [f"{r[k]:+.1f}" for k in ['wall_delta', 'prefill_delta', 'decode_token_volume_delta', 'decode_rate_delta', 'other_delta']]
        lines.append('| ' + ' | '.join([r['model'], str(r['paired_tasks']), *cells]) + ' |')
    lines += ['', '## Per-session signature', '',
              '| Model / task | Wall s | Outer / model turns | Tool requests | Output | Max context | Compactions | Tokens through first requested edit |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} / {r['task']} | {r['wall_seconds']:.1f} | {r['outer_turns']} / {r['model_responses']} | {r['tool_requests']} | {r['generated_tokens']} | {r['peak_context_tokens']} | {r['compactions']} | {r['output_tokens_through_first_edit_request']} |")
    lines += ['', '## Inferred tool-action distribution', '',
              'Labels are heuristic and multi-label: a combined edit/test command can count in both columns. Polling is separate from issuing a command.', '',
              '| Model | Inspect | Edit | Test | Lint/format | Poll | Rejected/tool errors | Nonzero exits | Exact repeated commands |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model in sorted({r['model'] for r in rows}):
        group = [r for r in rows if r['model'] == model]
        counts = [sum(r['actions_inferred'].get(k, 0) for r in group) for k in ['inspect', 'edit', 'test', 'lint_format', 'poll']]
        counts += [sum(r[k] for r in group) for k in ['rejected_or_tool_errors', 'nonzero_command_exits', 'exact_repeated_commands']]
        lines.append('| ' + ' | '.join(map(str, [model, *counts])) + ' |')
    lines += ['', '## Compaction observations', '']
    for r in rows:
        if r['compactions']:
            inference = r['compaction_inference_seconds']
            lines.append(f"- {r['model']} / {r['task']}: {r['compactions']} compaction(s), {r['compaction_elapsed_seconds']:.1f} s elapsed; " +
                         (f"{inference:.1f} s backend inference." if inference is not None else 'backend attribution unavailable.'))
            for c in r['compaction_details']:
                lines.append(f"  Compaction input {c['input_tokens']:,}, output {c['output_tokens']:,}; next request input {c['next_request_input_tokens']}.")
    if not any(r['compactions'] for r in rows):
        lines.append('No persisted compaction events in the measured sessions.')
    lines += ['', '## Tool-wait and environment observations', '',
              'Call-to-result intervals are unioned to avoid double-counting overlap. These include transport and waiting, not just CPU execution.', '',
              '| Model | Observed tool-wait union s | Non-inference wall s | Outputs containing network errors |',
              '|---|---:|---:|---:|']
    for model in sorted({r['model'] for r in rows}):
        group = [r for r in rows if r['model'] == model]
        lines.append(f"| {model} | {sum(r['tool_wait_union_seconds'] for r in group):.1f} | {sum(r['other_seconds'] for r in group):.1f} | {sum(r['network_error_outputs'] for r in group)} |")
    lines += ['', 'Some optional `npx eslint` invocations tried to fetch an uncached package and failed in the deliberately offline jail. Poll records retain their originating command when its session ID is available. This overhead depends on both model tool choices and the benchmark environment.', '']
    lines += ['', '## Largest single model responses', '',
              '| Model / task | Output tokens | Context entering request | Backend decode s | Compaction |',
              '|---|---:|---:|---:|---|']
    largest = sorted([(r, q) for r in rows for q in r['requests']], key=lambda x: x[1]['usage']['output_tokens'], reverse=True)[:10]
    for r, q in largest:
        value = q.get('decode_seconds')
        seconds = 'unknown' if value is None else f'{value:.1f}'
        lines.append(f"| {r['model']} / {r['task']} | {q['usage']['output_tokens']} | {q['usage']['input_tokens']} | {seconds} | {q['is_compaction']} |")
    lines += ['', '## Measurement contract and caveats', '',
              '- One outer turn can contain many model responses and tool calls. Responses are deduplicated by response ID; compaction requests are included and separately identified.',
              '- Context is the fresh, single-task server high-water counter, checked against its zero starting value and configured capacity. The old sampler added decoded tokens twice; its values are preserved only as excluded legacy data. This does not change task wall time or trusted grades.',
              '- Cumulative input tokens include repeatedly cached prefixes and are not session context size. Client-reported effective windows are stored separately from configured server context and compaction thresholds.',
              '- Compaction evidence comes from persisted compacted records, not keyword searches or guessed context drops. Recorded compaction duration overlaps inference/tool time and must not be added to wall time.',
              '- Tool wait is the union of call-to-result timestamp intervals, including polling/transport. It is not pure CPU time; command exit codes can mask failed tests behind shell pipelines. Trusted grades remain authoritative.',
              '- Visible reasoning is counted in characters, not tokens. The server reports reasoning-token counts as zero despite visible reasoning; output tokens combine reasoning, answers and tool arguments.',
              '- Command categories and the first-edit marker are heuristic. Exact repeated commands need not be wasted work. Examine the linked raw evidence before assigning intent or causality.',
              '- The JSON contains per-request tokens/timings, tool categories/outcomes, timestamps, source lines and rollout hashes. Original results and running snapshots are never edited.',
              '- [Official Codex JSONL event documentation](https://learn.chatgpt.com/docs/non-interactive-mode) and [compaction event documentation](https://learn.chatgpt.com/docs/app-server) establish the public event distinction; this parser uses the actual persisted CLI rollout schema.', '']
    warnings = [(r['model'], r['task'], r['warnings']) for r in rows if r['warnings']]
    lines += ['Validation warnings: ' + ('none.' if not warnings else json.dumps(warnings)), '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--corrected-bartowski', type=Path, required=True)
    parser.add_argument('--after-completion', action='store_true')
    parser.add_argument('--deadline-epoch', type=float)
    args = parser.parse_args()
    if args.after_completion and args.deadline_epoch is None:
        parser.error('--after-completion requires a deadline')
    if args.after_completion:
        while not campaign_terminal(args.corrected_bartowski) and time.time() < args.deadline_epoch:
            time.sleep(15)
    result = analyze(args.root, args.corrected_bartowski)
    atomic_json(args.root / 'SESSION-SIGNATURES.json', result)
    temporary = args.root / 'SESSION-SIGNATURES.md.tmp'
    temporary.write_text(render(result))
    temporary.replace(args.root / 'SESSION-SIGNATURES.md')


if __name__ == '__main__':
    main()
