import tempfile
import unittest
from pathlib import Path

import numpy as np

from spec_tf import array, draft_probs, validate_rows


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
