#!/usr/bin/env python3
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import statistics
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gpu_harness as h
import sweep as s

FIXTURES = Path('/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures')
RECORDED = Path('/home/xanewok/repos/graph-reuse/codex/runs/20261003-173851/1-up')
MATRIX = Path(__file__).parent / 'matrix.toml'


def read(name, root=RECORDED):
    return json.loads((root / name).read_text())


def sampled(k=12, count=70, zero=False):
    chunks = []
    log = []
    n = a = 0
    out = 1
    for index in range(1, count + 1):
        d = 0 if zero else 1 + index % 4
        accepted = max(0, d - 1)
        n += d
        a += accepted
        out += accepted + 1
        chunks.append({'choices': [{'delta': {'content': 'x'}}], 'timings': {
            'draft_n': n, 'draft_n_accepted': a, 'predicted_ms': 30 * index, 'predicted_n': out}})
        if d:
            log.append(f'0.01.000.000 I slot accepted: id 0 | task 1 | accepted {accepted:2}/{d:2} draft tokens')
    r = {'max_tokens': out + 100, 'metrics': {'drafts': 0 if zero else count, 'draft_tokens': n, 'accepted_tokens': a},
         'response': {'choices': [{'message': {'role': 'assistant', 'content': 'x'}, 'finish_reason': 'stop'}],
                      'timings': {'draft_n': n, 'draft_n_accepted': a, 'predicted_n': out, 'predicted_ms': 30 * count,
                                  'prompt_n': 8192, 'prompt_ms': 100, 'cache_n': 0}},
         'begin_ns': 0, 'end_ns': 4000000000, 'chunks': chunks, 'round_log': '\n'.join(log)}
    if zero:
        r['response']['timings'].pop('draft_n')
        r['response']['timings'].pop('draft_n_accepted')
        for chunk in r['chunks']:
            chunk['timings'].pop('draft_n')
            chunk['timings'].pop('draft_n_accepted')
    return r


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.cfg = s.load_matrix(MATRIX)

    def test_real_short_capture_and_too_few_rounds(self):
        r = read('long-tool-s1-1.json', FIXTURES / 'llama-sse')
        result = s.steady(r, 5, 1)
        self.assertEqual(result['steady_rounds_M'], 23)
        self.assertAlmostEqual(result['steady_ms_per_round_D'], 37.35504347826087)
        with self.assertRaisesRegex(ValueError, 'insufficient steady rounds'):
            s.steady(r, 5, 40)

    def test_real_long_captures(self):
        for kind, rounds in [('tool', 337), ('think', 341)]:
            r = read(f'long-{kind}-s1-1.json')
            self.assertEqual(h.completion(r['chunks'])['timings'], r['response']['timings'])
            self.assertEqual(s.steady(r, 5, 40)['steady_rounds_M'], rounds)
            s.tool_gate(r, True)

    def test_unknown_rows_are_skipped(self):
        r = read('long-tool-s1-1.json')
        expected = s.steady(r, 5, 40)
        r['chunks'].insert(20, {'timings': {'predicted_n': 1, 'predicted_ms': 0}})
        r['chunks'].insert(40, {'timings': {'draft_n': None, 'predicted_ms': 0}})
        self.assertEqual(s.steady(r, 5, 40), expected)

    def test_real_reversed_rows_rejected(self):
        r = read('long-tool-s1-1.json')
        r['chunks'][100]['timings']['predicted_ms'] = 0
        with self.assertRaisesRegex(ValueError, 'reversed'):
            s.steady(r, 5, 40)

    def test_counter_disagreement_rejected(self):
        for key in ('draft_tokens', 'accepted_tokens'):
            r = read('long-tool-s1-1.json')
            r['metrics'][key] += 1
            with self.assertRaisesRegex(ValueError, 'counters disagree'):
                s.steady(r, 5, 40)

    def test_variable_rounds_fail_old_fixed_k_control(self):
        r = sampled()
        self.assertEqual(s.steady(r, 12, 40)['steady_ms_per_round_D'], 30)
        original = Path('/home/xanewok/repos/graph-reuse/codex/gpu_harness.py')
        node = next(node for node in ast.parse(original.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'steady')
        scope = {'statistics': statistics}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(original), 'exec'), scope)
        with self.assertRaisesRegex(ValueError, 'full fixed-K'):
            scope['steady'](r, 12, 40)

    def test_trace_counter_coverage(self):
        r = sampled()
        self.assertEqual(len(s.trace_rounds(r['round_log'], r['metrics'])), 70)
        r['round_log'] = r['round_log'].split('\n', 1)[1]
        with self.assertRaisesRegex(ValueError, 'counter coverage'):
            s.steady(r, 12, 40)

    def test_trace_real_uninstrumented_log_is_rejected(self):
        text = (FIXTURES / 'llama-sse/server.log').read_text()
        r = read('long-tool-s1-1.json', FIXTURES / 'llama-sse')
        with self.assertRaisesRegex(ValueError, 'counter coverage'):
            s.trace_rounds(text, r['metrics'])

    def test_trace_checkpoint_replay(self):
        log = 'accepted 2/5 draft tokens (restore checkpoint)\naccepted 2/3 draft tokens\n'
        self.assertEqual(s.trace_rounds(log, {'drafts': 1, 'draft_tokens': 8, 'accepted_tokens': 1}), {8: 1})

    def test_missing_interior_round_is_excluded(self):
        r = sampled()
        before = s.steady(r, 12, 40)['steady_rounds_M']
        del r['chunks'][35]
        self.assertEqual(s.steady(r, 12, 40)['steady_rounds_M'], before - 2)

    def test_too_few_variable_rounds(self):
        with self.assertRaisesRegex(ValueError, 'insufficient steady'):
            s.steady(sampled(count=25), 12, 40)

    def test_all_zero_proposal_arm(self):
        result = s.steady(sampled(zero=True), 12, 40)
        self.assertTrue(result['all_zero_proposal'])
        self.assertEqual(result['verified_tokens_per_round_D'], 1)
        self.assertEqual(result['steady_ms_per_round_D'], 30)

    def test_cut_followup_tool_call_rejected(self):
        r = read('cold-tool-1.json', FIXTURES / 'llama-sse')
        s.tool_gate(r)
        r['response']['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{"cmd":'
        with self.assertRaisesRegex(ValueError, 'cut tool call'):
            s.tool_gate(r)

    def test_real_server_log_and_negative_health_controls(self):
        text = (FIXTURES / 'llama-sse/server.log').read_text()
        s.server_gate(text)
        for bad in [text.replace('n_slots = 1', 'n_slots = 2'), text + '\nbackend offload failed; using CPU sampler']:
            with self.assertRaises(ValueError):
                s.server_gate(bad)

    def test_journal_real_artifact_and_word_boundaries(self):
        guard = h.Harness.__new__(h.Harness)
        guard.env = {'XID_SINCE': '2026-10-03 00:00:00'}
        guard.run = lambda *_a, **_k: SimpleNamespace(stdout=(FIXTURES / 'xid-lines.txt').read_text())
        with self.assertRaises(RuntimeError):
            guard.xid()
        guard.run = lambda *_a, **_k: SimpleNamespace(stdout='stop_xid_hash_s_G=10\n-- No entries --')
        guard.xid()
        guard.run = lambda *_a, **_k: SimpleNamespace(stdout='')
        with self.assertRaisesRegex(RuntimeError, '^empty/unreadable kernel journal$'):
            guard.xid()

    def test_metrics_real_fixture(self):
        record = read('long-tool-s1-1.json', FIXTURES / 'llama-sse')
        guard = h.Harness.__new__(h.Harness)
        guard.http = lambda *_a, **_k: '\n'.join(f'llamacpp:spec_decode_num_{name}_total {record["metrics"][key]}' for name, key in [('drafts', 'drafts'), ('draft_tokens', 'draft_tokens'), ('accepted_tokens', 'accepted_tokens')])
        self.assertEqual(guard.metrics(), record['metrics'])
        guard.http = lambda *_a, **_k: ''
        with self.assertRaisesRegex(RuntimeError, 'missing authoritative metric'):
            guard.metrics()

    def test_unsafe_plan_and_execute_cannot_launch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            with patch.object(h.subprocess, 'run', side_effect=AssertionError('subprocess')), patch.object(h.subprocess, 'Popen', side_effect=AssertionError('launch')), patch.object(h.socket, 'socket', side_effect=AssertionError('network')):
                manifest, cost = s.plan(self.cfg, MATRIX, Path(temp))
                self.assertTrue(manifest.is_file())
                self.assertLessEqual(cost['total_budget_s_D'], 9000)
                self.assertEqual(cost['canary_budget_s_D'], 600)
                self.assertLessEqual(cost['canary_est_s_D'], 420)
                args = SimpleNamespace(out=Path(temp) / 'never', canary=False, matrix=MATRIX)
                with patch.dict(os.environ, {'RUN_GPU': '0', 'CUDA_VISIBLE_DEVICES': ''}):
                    with self.assertRaisesRegex(RuntimeError, 'parent execution'):
                        s.execute(args, self.cfg)
                self.assertFalse(args.out.exists())

    def test_order_and_budget_negative_control(self):
        names = [arm['name'] for arm in self.cfg['arms']]
        self.assertEqual(s.order(self.cfg), names + names[::-1])
        self.assertTrue(all(s.order(self.cfg).count(name) == 2 for name in names))
        self.cfg['cost']['boot_s_G'] = 1000
        with self.assertRaisesRegex(ValueError, 'stage estimates'):
            s.costs(self.cfg)

    def test_frozen_snapshot_bytes_and_negative_control(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            snap = s.snapshot(self.cfg, Path(temp))
            h.frozen(snap)
            self.assertFalse((snap / 'bin/llama-server').exists())
            with (snap / 'bin/sweep-srv').open('ab') as stream:
                stream.write(b'broken')
            with self.assertRaisesRegex(ValueError, 'frozen bytes changed'):
                h.frozen(snap)

    def test_suite_preserves_followup_error_and_records_all_long_cases(self):
        calls = []
        def request(_dest, case, _number, body, stream=False):
            calls.append((case, copy.deepcopy(body)))
            if case == 'short-tool-2':
                return {'error': {'message': 'HTTP 500: cut tool call'}, 'case': case}
            return read('cold-tool-1.json', FIXTURES / 'llama-sse')
        fake = SimpleNamespace(args=SimpleNamespace(canary=False), request=request, deep_body=lambda body, _target: body)
        result = s.suite(fake, Path('.'), self.cfg)
        self.assertEqual(len(result), 7)
        self.assertEqual(sum(case.startswith('long-') for case in result), 4)
        self.assertIn('error', result['short-tool-2'])
        self.assertGreater(len(calls[-1][1]['messages']), len(calls[-2][1]['messages']))

    def test_canary_suite_replay_and_execution_guard(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            out = Path(temp) / 'canary'
            args = SimpleNamespace(out=out, canary=True, matrix=MATRIX, port=8398)
            def boot(_root, label):
                dest = out / label
                dest.mkdir()
                (dest / 'server.log').write_text((FIXTURES / 'llama-sse/server.log').read_text())
                return dest
            def request(_dest, case, _number, body, stream=False):
                return read('long-tool-s1-1.json' if case.startswith('long-') else 'cold-tool-1.json', FIXTURES / 'llama-sse')
            fake = SimpleNamespace(args=args, request=request, boot=boot, stop=lambda: None, xid=lambda: None, deep_body=lambda body, _target: body)
            with patch.dict(os.environ, {'RUN_GPU': '1', 'CUDA_VISIBLE_DEVICES': '', 'XID_SINCE': 'fixture'}), patch.object(s, 'MatrixHarness', return_value=fake), patch.object(s, 'snapshot'), patch.object(s, 'pins', return_value=[]), patch.object(h, 'frozen'), patch.object(h.subprocess, 'Popen', side_effect=AssertionError('launch')):
                # The empty device selection must still block the mocked execution entry.
                with self.assertRaisesRegex(RuntimeError, 'parent execution'):
                    s.execute(args, self.cfg)
            # Replay the suite and gates directly; no execution authorization or subprocess is needed.
            out.mkdir()
            results = s.suite(fake, out, self.cfg)
            self.assertEqual(len(results), 7)
            errors = s.validate(results, out, self.cfg['arms'][0], self.cfg)
            self.assertTrue(errors)
            self.assertTrue((out / 'boot-validation.json').is_file())

    def test_real_gpu_memory_parser_and_negative_controls(self):
        text = (FIXTURES / 'nvidia-smi-query.csv').read_text()
        self.assertEqual(h.gpu_memory(text), [108, 15])
        for bad in [text.replace('108 MiB', 'unknown'), '\n'.join(text.splitlines()[:2])]:
            with self.assertRaises(RuntimeError):
                h.gpu_memory(bad)

    def test_same_arm_identity_gate_on_real_recordings(self):
        r = read('cold-tool-1.json', FIXTURES / 'llama-sse')
        boots = [{'arm': arm, 'requests': {'cold-tool': copy.deepcopy(r)}} for arm in s.order(self.cfg)]
        s.repeat_gate(boots, self.cfg)
        boots[-1]['requests']['cold-tool']['response']['choices'][0]['message']['content'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'same-arm identity failed'):
            s.repeat_gate(boots, self.cfg)

    def test_top_q_stop_preserves_one_step_law(self):
        p, q = [0.2, 0.3, 0.5], [0.8, 0.2, 0.0]
        residual = [max(0, x - y) for x, y in zip(p, q)]
        total = sum(residual)
        def verified(proposal):
            accepted = [z * min(1, x / y) if y else 0 for x, y, z in zip(p, q, proposal)]
            rejected = 1 - sum(accepted)
            return [a + rejected * r / total for a, r in zip(accepted, residual)]
        for threshold in [0, 0.7, 0.9]:
            law = verified(q) if max(q) >= threshold else p
            for actual, expected in zip(law, p):
                self.assertAlmostEqual(actual, expected)
        # Conditioning proposals on their sampled-token probability changes q without changing recorded q.
        self.assertNotEqual(verified([1, 0, 0]), p)

    def test_summary_pairing_and_signed_fixed_cost(self):
        boots = []
        for index, arm in enumerate(s.order(self.cfg), 1):
            r = sampled()
            r['steady'] = s.steady(r, 12, 40)
            r['end_ns'] = 2000000000
            if arm != self.cfg['baseline']:
                r['end_ns'] -= 100000000
            boots.append({'arm': arm, 'boot': index, 'requests': {'long-tool-d8192': r}})
        result = s.summarize(boots, self.cfg)
        self.assertEqual(len(result['paired']), 6)
        self.assertEqual(len(result['per_arm']), 7)
        self.assertLess(result['requests'][0]['request_fixed_ms_estimate_D'], 0)
        self.assertEqual(result['paired'][0]['mean_difference_D']['request_fixed_ms_estimate_D'], -100)


if __name__ == '__main__':
    unittest.main()
