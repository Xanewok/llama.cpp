"""Parser controls use recorded GPU bytes; no model or GPU process starts here."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import analyze
import item
import meta_header
import render


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=item.HERE)
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        item.preflight(self.out)
        self.nodes = analyze.read_trace(self.out / 'recorded-logits-trace.jsonl', draft=False)

    def test_frozen_meta_layouts(self):
        layouts = []
        for name in ('stack9-p55', 'stack10'):
            source = subprocess.check_output(['git', '-C', str(item.TREE), 'show',
                                              item.base.COMMITS[name] + ':ggml/src/ggml-backend-meta.cpp'], text=True)
            bodies = '\n'.join(meta_header.extract(source, struct_name) for struct_name in
                               ('ggml_backend_meta_simple_tensor_container', 'ggml_backend_meta_buffer_context'))
            header = item.TREE / f'build-history-{name}/meta-layout.h'
            self.assertEqual(header.read_text(), '#pragma once\n' + bodies + '\n')
            layouts.append(bodies)
        self.assertEqual(*layouts)

    def test_first_node_and_source_control(self):
        source = {'name': 'weight', 'op': 'NONE', 'type': 'f32', 'shape': [1]*4,
                  'strides': [4]*4, 'op_params': [], 'status': 'frozen_weight'}
        self.nodes[2]['inputs'] = [source]
        changed = copy.deepcopy(self.nodes)
        changed[2]['output']['slices'][0]['data']['sha256'] = '0'*64
        finding = analyze.compare(self.nodes, changed)['first']
        self.assertEqual(finding['phase'], 'decode-1')
        self.assertEqual(finding['class'], 'arithmetic_identical_observed_inputs')
        missing = {'name': 'state', 'op': 'NONE', 'type': 'f32', 'shape': [1]*4,
                   'strides': [4]*4, 'op_params': [], 'status': 'unavailable_before_plan'}
        self.nodes[2]['inputs'] = [missing]
        changed[2]['inputs'] = [missing]
        self.assertEqual(analyze.compare(self.nodes, changed)['first']['class'], 'arithmetic_inputs_differ_or_unobserved')
        self.assertIsNone(analyze.compare(self.nodes, self.nodes)['first'])

    def test_layout_control(self):
        changed = copy.deepcopy(self.nodes)
        changed[0]['output']['strides'][1] += 4
        self.assertEqual(analyze.compare(self.nodes, changed)['first']['class'], 'graph_or_layout_difference')
        changed[0]['index'] = 1
        path = self.out / 'changed.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in changed))
        with self.assertRaises(ValueError):
            analyze.read_trace(path, draft=False)

    def test_final_request_bytes_equal(self):
        def final(path):
            data = path.read_bytes()
            count, draft_steps = struct.unpack_from('<2i', data, 4)
            offset = 12
            for _ in range(count):
                depth, steps = struct.unpack_from('<2i', data, offset)
                end = offset + 8 + (depth + steps + draft_steps)*4
                last = data[offset:end]
                offset = end
            self.assertEqual(offset, len(data))
            return last
        self.assertEqual(final(self.out / 'short.bin'), final(self.out / 'long.bin'))

    def test_repeats_and_reading(self):
        plan = [('a', 'short', 0), ('a', 'long', 0), ('a', 'long', 1), ('a', 'short', 1)]
        real = (self.out / 'recorded-logits-trace.jsonl').read_text()
        raw = item.base.np.zeros((4, 248320), dtype='<f4')
        for i, (arm, order, repeat) in enumerate(plan):
            dest = self.out / f'{i}-{arm}-{order}-r{repeat}'
            dest.mkdir()
            (dest / 'trace.jsonl').write_text(real)
            (dest / 'logits.bin').write_bytes(b'KLDL' + struct.pack('<3i', 90112, 4, 248320) + raw.tobytes())
            (dest / 'tool.log').write_text('history-root: complete nodes=5 traced=1 requests=2\n')
        with patch.object(item, 'validate', side_effect=lambda dest, arm: analyze.read_trace(dest / 'trace.jsonl', draft=False)):
            with patch.dict(item.ARMS, {k: item.ARMS['a'] for k in ('b', 'c', 'd', 'e', 'f', 'untraced')}):
                # Repeat checks must run before missing-arm interpretation.
                dest = self.out / '3-a-short-r1/logits.bin'
                changed = raw.copy()
                changed[0, 0] = 1
                dest.write_bytes(b'KLDL' + struct.pack('<3i', 90112, 4, 248320) + changed.tobytes())
                with self.assertRaises(item.lib.Fail):
                    item.summarize(self.out, plan)

    def test_pre_registered_reading_on_real_logits(self):
        with item.base.REAL_KLF.open('rb') as stream:
            stream.seek(4)
            vocab, _ = struct.unpack('<2i', stream.read(8))
            stream.seek(20)
            values = item.base.np.tile(item.base.np.frombuffer(stream.read(vocab*4), '<f4'), (4, 1))
        changed = values.copy()
        changed[0, int(values[0].argmax())] += 1
        def logits(path):
            _, arm, order, _ = path.parent.name.split('-')
            return (90112, 4, vocab), changed if order == 'long' and arm not in ('b', 'd', 'e') else values
        with patch.object(item, 'validate', return_value=self.nodes), patch.object(item.base, 'read_logits', side_effect=logits):
            result = item.summarize(self.out, item.PLAN)
        self.assertTrue(result['arms']['a']['logits_differ'])
        self.assertFalse(result['arms']['b']['logits_differ'])
        self.assertTrue(result['arms']['c']['logits_differ'])
        self.assertEqual(len(result['reading']), 3)
        self.assertIn('stale-memory-byte', result['reading'][0])
        self.assertIn('MTP/draft-context', result['reading'][1])
        self.assertIn('CUDA graph', result['reading'][2])

    def _summarize(self, differs, nodes_for=None):
        with item.base.REAL_KLF.open('rb') as stream:
            stream.seek(4)
            vocab, _ = struct.unpack('<2i', stream.read(8))
            stream.seek(20)
            values = item.base.np.tile(item.base.np.frombuffer(stream.read(vocab*4), '<f4'), (4, 1))
        changed = values.copy()
        changed[0, int(values[0].argmax())] += 1
        def logits(path):
            _, arm, order, _ = path.parent.name.split('-')
            return (90112, 4, vocab), changed if order == 'long' and arm in differs else values
        nodes_for = nodes_for or (lambda dest, arm: self.nodes)
        with patch.object(item, 'validate', side_effect=nodes_for), patch.object(item.base, 'read_logits', side_effect=logits):
            return item.summarize(self.out, item.PLAN)

    def test_traced_only_dependence_is_inconclusive(self):
        result = self._summarize({'a', 'b', 'c', 'd', 'e', 'f'})
        self.assertEqual(len(result['reading']), 1)
        self.assertIn('callback induced', result['reading'][0])
        self.assertTrue(result['arms']['a']['logits_differ'])
        self.assertFalse(result['arms']['untraced']['logits_differ'])

    def test_draft_only_difference_reading(self):
        def nodes(dest, arm):
            rows = copy.deepcopy(self.nodes)
            if arm != 'untraced' and '-long-' in dest.name:
                row = copy.deepcopy(rows[0])
                row['domain'], row['phase'] = 'catchup', 'prefill'
                rows.append(row)
                other = copy.deepcopy(row)
                other['output']['slices'][0]['data']['sha256'] = '1'*64
                rows[-1] = other
            elif arm != 'untraced':
                row = copy.deepcopy(rows[0])
                row['domain'], row['phase'] = 'catchup', 'prefill'
                rows.append(row)
            return rows
        result = self._summarize(set(), nodes)
        self.assertIn('catchup', result['arms']['a']['draft_first'])
        self.assertEqual(len(result['reading']), 2)
        self.assertIn('target logits equal, draft/catchup differ', result['reading'][1])

    def test_partial_summary_records_repeat_failure(self):
        plan = [('a', 'short', 0), ('a', 'long', 0), ('a', 'long', 1)]
        base_values = item.base.np.zeros((4, 248320), dtype='<f4')
        changed = base_values.copy()
        changed[0, 0] = 1
        def logits(path):
            return (90112, 4, 248320), changed if path.parent.name.startswith('2-') else base_values
        with patch.object(item, 'validate', return_value=self.nodes), patch.object(item.base, 'read_logits', side_effect=logits):
            result = item.summarize(self.out, plan, partial=True)
            self.assertTrue(result['partial'])
            self.assertEqual(len(result['repeat_failures']), 1)
            self.assertIsNone(result['reading'])
            with self.assertRaises(item.lib.Fail):
                item.summarize(self.out, plan)

    def test_manifest_and_environment(self):
        import tomllib
        manifest = tomllib.loads(render.render())
        self.assertEqual(manifest['id'], 'history-root')
        self.assertEqual(manifest['priority'], 23)
        self.assertLessEqual(manifest['budget_s'], 9600)
        self.assertLessEqual(len(manifest['note']), 600)
        self.assertLessEqual(manifest['est_s'], .7*manifest['budget_s'])
        for stage in ('run', 'canary'):
            self.assertIn(str(item.WT / 'work/aa-load/ccx01.sh'), manifest[stage]['cmd'])
        for arm in item.ARMS:
            env = item.environment('stack9-p55', self.out / 'short.bin', self.out, arm)
            self.assertEqual(env['HISTORY_CLEAR'], item.ARMS[arm][1])
            self.assertEqual(env['HISTORY_DRAFT'], str(int(item.ARMS[arm][2])))
            self.assertEqual(env['HISTORY_TRACE'], str(int(item.ARMS[arm][3])))
            if arm == 'e':
                self.assertEqual(env['GGML_CUDA_DISABLE_GRAPHS'], '1')
            else:
                self.assertNotIn('GGML_CUDA_DISABLE_GRAPHS', env)
        self.assertEqual(len(item.PLAN), 28)


if __name__ == '__main__':
    unittest.main()
