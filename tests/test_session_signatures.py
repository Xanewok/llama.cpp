import json
from pathlib import Path
import tempfile
import unittest

from scripts.analysis import session_signatures as s
from scripts.instruments.hillclimb import sealed_json


class SessionSignatureTests(unittest.TestCase):
    def test_shell_categories_are_multi_label_and_polling_is_separate(self):
        self.assertEqual(s.action_tags('write_stdin', {'chars': ''}), ['poll'])
        self.assertEqual(s.action_tags('write_stdin', {'chars': 'yes\n'}), ['interactive_input'])
        self.assertIn('inspect', s.action_tags('exec_command', {'cmd': 'cat src/lib.rs'}))
        tags = s.action_tags('exec_command', {'cmd': 'cat > src/lib.rs << EOF\n// run cargo test\nEOF'})
        self.assertIn('edit', tags)
        self.assertNotIn('test', tags)
        self.assertIn('test', s.action_tags('exec_command', {'cmd': 'cd /work && cargo test --offline'}))
        self.assertEqual(s.command_text({'command': ['/bin/bash', '-lc', 'pwd']}), 'pwd')

    def test_overlapping_tool_wait_is_not_added_twice(self):
        self.assertEqual(s.interval_union_seconds([(0, 4), (2, 6), (6, 8)]), 8)
        self.assertEqual(s.interval_union_seconds([]), 0)

    def test_paired_accounting_matches_tasks_and_sums_to_wall_difference(self):
        baseline = {'model': 'iq3-control', 'task': 'one', 'solved': True, 'generated_tokens': 100,
                    'decode_seconds': 10, 'prefill_seconds': 2, 'other_seconds': 3, 'wall_seconds': 15}
        candidate = {**baseline, 'model': 'candidate', 'generated_tokens': 50, 'decode_seconds': 4,
                     'prefill_seconds': 3, 'other_seconds': 4, 'wall_seconds': 11}
        row = s.paired_explanations([baseline, candidate, {**candidate, 'task': 'unpaired'}])[0]
        self.assertEqual(row['paired_tasks'], 1)
        self.assertAlmostEqual(row['decode_token_volume_delta'], -4.5)
        self.assertAlmostEqual(row['decode_rate_delta'], -1.5)
        self.assertAlmostEqual(row['wall_delta'], sum(row[k] for k in [
            'prefill_delta', 'decode_token_volume_delta', 'decode_rate_delta', 'other_delta']))

    def fixture(self, root):
        attempt = root / 'attempts/one'
        directory = attempt / 'codex-state/sessions'
        directory.mkdir(parents=True)
        usage1 = {'response_id': 'r1', 'usage': {'input_tokens': 10, 'cached_input_tokens': 0, 'output_tokens': 20}}
        usage2 = {'response_id': 'r2', 'usage': {'input_tokens': 20, 'cached_input_tokens': 0, 'output_tokens': 30}}
        raw = [
            ('session_meta', {'id': 's1', 'cli_version': 'test'}),
            ('event_msg', {'type': 'task_started', 'turn_id': 't1', 'model_context_window': 120}),
            ('response_item', {'type': 'function_call', 'id': 'f1', 'call_id': 'c1', 'name': 'exec_command',
                               'arguments': json.dumps({'cmd': 'cat > solution.js << EOF\ncode\nEOF'})}),
            ('token_usage_record', usage1),
            ('token_usage_record', usage1),
            ('response_item', {'type': 'function_call_output', 'call_id': 'c1', 'output': 'exec_command failed: Rejected'}),
            ('event_msg', {'type': 'token_count', 'info': {'last_token_usage': {'output_tokens': 99999}}}),
            ('token_usage_record', usage2),
            ('compacted', {'compaction_response_id': 'r2', 'message': 'summary',
                           'replacement_history': [{'type': 'function_call', 'call_id': 'do-not-count'}]}),
            ('event_msg', {'type': 'item_completed', 'item': {'type': 'ContextCompaction', 'id': 'comp'},
                           'started_at_ms': 1000, 'completed_at_ms': 4000}),
            ('event_msg', {'type': 'task_complete', 'turn_id': 't1'}),
        ]
        path = directory / 'rollout-test.jsonl'
        path.write_text('\n'.join(json.dumps({'timestamp': f'2026-09-13T00:00:{i:02d}Z', 'type': kind, 'payload': payload})
                                  for i, (kind, payload) in enumerate(raw)) + '\n')
        (attempt / 'metrics-before.json').write_text(json.dumps({'n_tokens_max': 0}))
        (attempt / 'metrics-after.json').write_text(json.dumps({'n_tokens_max': 110}))
        lines = []
        for key, inputs, outputs in [(1, 10, 20), (2, 20, 30)]:
            lines += [f'I slot print_timing: id  0 | task {key} | prompt eval time = 100.00 ms / {inputs} tokens',
                      f'I slot print_timing: id  0 | task {key} |        eval time = 200.00 ms / {outputs} tokens',
                      f'I slot release: id  0 | task {key} | stop processing: n_tokens = 80, truncated = 0']
        (attempt / 'server.log').write_text('\n'.join(lines))
        row = {'model': 'test', 'task': 'javascript/test', 'attempt': 'attempts/one', 'probe': {
            'task_success': True, 'verified_wall_seconds': 100, 'prefill_seconds': 30,
            'decode_seconds': 50, 'other_seconds': 20, 'client_context': 128,
            'client_compaction': 100, 'peak_context_tokens': 200,
            'metrics_delta': {'decode_tokens': 50, 'prefill_tokens': 30}}}
        return path, row

    def test_responses_deduplicate_but_token_count_notifications_do_not_add_turns(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.fixture(Path(temp))
            result = s.parse_rollouts([path])
            self.assertEqual(len(result['requests']), 2)
            self.assertEqual(sum(r['usage']['output_tokens'] for r in result['requests']), 50)
            self.assertEqual(result['outer_turns_completed'], 1)
            self.assertEqual(len(result['tools']), 1)
            self.assertTrue(result['tools'][0]['rejected_or_tool_error'])
            self.assertFalse(result['tools'][0]['command_completed_event'])
            self.assertTrue(result['requests'][1]['is_compaction'])
            self.assertEqual(s.interval_union_seconds(result['compaction_intervals']), 3)

    def test_poll_links_to_its_command_and_preserves_network_failure_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.fixture(Path(temp))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            output = next(r for r in rows if r['payload'].get('type') == 'function_call_output')
            output['payload']['output'] = 'Process running with session ID 42'
            rows += [
                {'timestamp': '2026-09-13T00:00:11Z', 'type': 'response_item', 'payload': {
                    'type': 'function_call', 'id': 'f2', 'call_id': 'c2', 'name': 'write_stdin',
                    'arguments': '{"session_id":42,"chars":""}'}},
                {'timestamp': '2026-09-13T00:00:12Z', 'type': 'response_item', 'payload': {
                    'type': 'function_call_output', 'call_id': 'c2', 'output': 'npm error code ENOTFOUND'}},
            ]
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            result = s.parse_rollouts([path])
            poll = result['tools'][1]
            self.assertEqual(poll['origin_call_id'], 'c1')
            self.assertTrue(poll['network_error_in_output'])
            self.assertEqual(poll['actions_inferred'], ['poll'])

    def test_peak_context_uses_fresh_server_counter_and_timings_align(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, row = self.fixture(root)
            result = s.session_signature(root, row)
            self.assertEqual(result['peak_context_tokens'], 110)
            self.assertEqual(result['legacy_double_counted_peak'], 200)
            self.assertEqual(result['compactions'], 1)
            self.assertAlmostEqual(result['compaction_inference_seconds'], .3)
            self.assertEqual(result['output_tokens_through_first_edit_request'], 20)
            self.assertTrue(result['backend_timings_aligned'])
            self.assertEqual(result['warnings'], [])

    def test_nonfresh_counter_or_missing_rollout_stays_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, row = self.fixture(root)
            (root / 'attempts/one/metrics-before.json').write_text('{"n_tokens_max": 10}')
            result = s.session_signature(root, row)
            self.assertIsNone(result['peak_context_tokens'])
            self.assertTrue(result['warnings'])
            row['attempt'] = 'absent'
            with self.assertRaisesRegex(ValueError, 'missing rollout'):
                s.session_signature(root, row)

    def test_full_campaign_analysis_and_corrected_probe_peak(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, row = self.fixture(root)
            row.update(status='passed')
            row['probe'].update(peak_context_tokens=110, legacy_sampled_context_tokens=200)
            sealed_json(root / 'manifest.json', {'queue': [{'id': 'one', 'model': 'test'},
                                                        {'id': 'pending', 'model': 'test'}]})
            sealed_json(root / 'job-one.json', row)
            (root / 'state.json').write_text('{"status": "running"}')
            result = s.analyze_campaign(root)
            self.assertEqual(result['campaign_state'], 'running')
            self.assertEqual(result['sessions'][0]['legacy_double_counted_peak'], 200)
            self.assertEqual(result['excluded'][0]['reason'], 'pending')
            self.assertIn('stage: running', s.render(result))


if __name__ == '__main__':
    unittest.main()
