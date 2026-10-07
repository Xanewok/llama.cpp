import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from spec_tf import array, draft_probs


class EstimatorIOTest(unittest.TestCase):
    def test_subset_and_coverage(self):
        ids = np.arange(128).reshape(1, 1, -1)
        logits = -ids.astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result, q = draft_probs(ids, logits, np.arange(20, 40))
        np.testing.assert_array_equal(result[0, 0], np.arange(20, 30))
        expected = np.exp(-np.arange(10))
        np.testing.assert_allclose(q[0, 0], expected / expected.sum())
        with self.assertRaises(ValueError):
            draft_probs(ids, logits, np.arange(125, 150))

    def test_truncated_binary_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'x.bin').write_bytes(b'\x00' * 3)
            with self.assertRaises(ValueError):
                array(root, 'x', '<f4', [1])


if __name__ == '__main__':
    unittest.main()
