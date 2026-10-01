import unittest

from gpu_ab import decode_result, metrics, within


class Tests(unittest.TestCase):
    def test_decode_uses_metric_delta(self):
        before = metrics('vllm:request_decode_time_seconds_sum{model="m"} 10\nvllm:generation_tokens_total{model="m"} 20\n')
        after = metrics('vllm:request_decode_time_seconds_sum{model="m"} 12\nvllm:generation_tokens_total{model="m"} 30\n')
        self.assertEqual(decode_result(before, after), {'decode_s_M': 2., 'tokens_M': 10., 'decode_tps_D': 5.})

    def test_missing_metrics_fail(self):
        with self.assertRaises(RuntimeError):
            decode_result({}, {})

    def test_paths_cannot_escape(self):
        with self.assertRaises(ValueError):
            within('/tmp/outside-worktree')


if __name__ == '__main__':
    unittest.main()
