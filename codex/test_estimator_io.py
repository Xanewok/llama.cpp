import tempfile
import unittest
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spec_tf import array, draft_probs, validate_rows
from check_vocab import compare_rows, main as check_vocab


class EstimatorIOTest(unittest.TestCase):
    def test_invalid_probability_rows_refused(self):
        for ids, probs in (([0, 1], [np.nan, 0.5]), ([0, 1], [0.4, 0.4]),
                           ([0, -1], [0.5, 0.5]), ([0, 0], [0.5, 0.5])):
            with self.assertRaises(ValueError):
                validate_rows(np.array([ids]), np.array([probs]), 128)
        validate_rows(np.array([[0, -1]]), np.array([[1.0, 0.0]]), 128)

    def test_subset_and_coverage(self):
        ids = np.arange(128).reshape(1, 1, -1)
        logits = -ids.astype(float)
        result, q = draft_probs(ids, logits, np.arange(20, 40))
        np.testing.assert_array_equal(result[0, 0], np.arange(20, 30))
        expected = np.exp(-np.arange(10))
        np.testing.assert_allclose(q[0, 0], expected / expected.sum())
        with self.assertRaises(ValueError):
            draft_probs(ids, logits, np.arange(125, 150))

    def test_boundary_tie_refused(self):
        ids = np.arange(128).reshape(1, 1, -1)
        logits = -ids.astype(float)
        logits[..., 10] = logits[..., 9]
        with self.assertRaisesRegex(ValueError, 'boundary tie'):
            draft_probs(ids, logits, np.arange(20))
        with self.assertRaisesRegex(ValueError, 'top-128 boundary'):
            draft_probs(ids, -ids.astype(float), np.arange(118, 128))
        with self.assertRaises(ValueError):
            draft_probs(ids, np.full(ids.shape, -np.inf), np.arange(20))

    def test_control_reports_low_coverage_and_detects_mismatch(self):
        ids = np.stack((np.arange(128), np.arange(100, 228)))
        logits = -np.arange(128, dtype=float)[None, :].repeat(2, axis=0)
        subset = np.arange(20)
        real_ids = np.arange(10)[None, :].repeat(2, axis=0)
        real_q = np.exp(-np.arange(10, dtype=float))[None, :].repeat(2, axis=0)
        real_q /= real_q.sum(axis=1, keepdims=True)
        covered, reasons, error = compare_rows(ids, logits, real_ids, real_q, subset, 1, (), 256)
        np.testing.assert_array_equal(covered, [True, False])
        np.testing.assert_array_equal(reasons['insufficient'], [False, True])
        self.assertLess(error, 1e-12)
        # A covered-row mismatch must still fail when another row is excluded.
        real_q[0] = np.roll(real_q[0], 1)
        _, _, error = compare_rows(ids, logits, real_ids, real_q, subset, 1, (), 256)
        self.assertGreater(error, 1e-3)
        covered, _, _ = compare_rows(ids[1:], logits[1:], real_ids[1:], real_q[1:], subset, 1, (), 256)
        self.assertFalse(covered.any())

    def test_control_excludes_unproven_boundaries(self):
        ids = np.arange(128)[None, :].repeat(2, axis=0)
        logits = -ids.astype(float)
        logits[1, 127] = logits[1, 126]
        real_ids = np.arange(118, 128)[None, :].repeat(2, axis=0)
        real_q = np.full((2, 10), 0.1)
        covered, reasons, _ = compare_rows(ids, logits, real_ids, real_q, np.arange(118, 128), 1, (), 256)
        self.assertFalse(covered.any())
        self.assertTrue(reasons['boundary'].all())

    def test_partial_control_cli_and_no_observations(self):
        build_tmp = Path(__file__).resolve().parents[1] / 'build-tmp'
        build_tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=build_tmp) as directory:
            root = Path(directory)
            h = {'format': 'spec-tf-v1', 'p_width': 10, 'n_vocab': 256,
                 'streams': [{'id': 'partial', 'n_tokens': 4, 'n_prompt': 1, 'n_continuation': 3}]}
            b = {'format': 'spec-tf-v1', 'stage': 'B', 'target_header': h, 'k': 1,
                 'q_width': 128, 'temperature': 1, 'suppress_ids': []}
            q = np.exp(-np.arange(10, dtype=float))
            q = (q / q.sum()).astype('<f4')
            for name, header in (('target', h), ('full', b), ('sliced', b)):
                (root / name / '0').mkdir(parents=True)
                (root / name / 'header.json').write_text(json.dumps(header))
            np.zeros(4, dtype='<i4').tofile(root / 'target/0/tokens.bin')
            np.tile(np.arange(10, dtype='<i4'), (4, 1)).tofile(root / 'target/0/p_ids.bin')
            np.tile(q, (4, 1)).tofile(root / 'target/0/p_probs.bin')
            for name in ('full', 'sliced'):
                np.array([1, 2], dtype='<i4').tofile(root / name / '0/starts.bin')
                np.tile(np.arange(10, dtype='<i4'), (2, 1)).tofile(root / name / '0/q_sample_ids.bin')
                np.tile(q, (2, 1)).tofile(root / name / '0/q_probs.bin')
                np.stack((np.arange(128), np.arange(100, 228))).astype('<i4').tofile(root / name / '0/q_ids.bin')
                np.tile(-np.arange(128, dtype='<f4'), (2, 1)).tofile(root / name / '0/q_logits.bin')
            vocab, output = root / 'vocab.json', root / 'result.json'
            argv = ['check_vocab', '--target', str(root / 'target'), '--full', str(root / 'full'),
                    '--sliced', str(root / 'sliced'), '--vocab', str(vocab), '--output', str(output)]
            cases = ((list(range(20)), False, 0, 1), (list(range(20)), True, 1, 1), ([250], False, 1, 0))
            for subset, broken, expected_rc, covered_rows in cases:
                real_q = np.tile(q, (2, 1))
                if broken:
                    real_q[0] = np.roll(real_q[0], 1)
                real_q.tofile(root / 'sliced/0/q_probs.bin')
                vocab.write_text(json.dumps(subset))
                with patch('sys.argv', argv), redirect_stdout(io.StringIO()):
                    self.assertEqual(check_vocab(), expected_rc)
                result = json.loads(output.read_text())
                self.assertEqual(result['rows_total_M'], 2)
                self.assertEqual(result['rows_excluded_M'], 2 - covered_rows)
                self.assertEqual(result['passed'], expected_rc == 0)
                self.assertEqual(result['starts_covered_M'], covered_rows)

    def test_truncated_binary_rejected(self):
        build_tmp = Path(__file__).resolve().parents[1] / 'build-tmp'
        build_tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=build_tmp) as directory:
            root = Path(directory)
            (root / 'x.bin').write_bytes(b'\x00' * 3)
            with self.assertRaises(ValueError):
                array(root, 'x', '<f4', [1])


if __name__ == '__main__':
    unittest.main()
