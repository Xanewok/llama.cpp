import unittest
from pathlib import Path

from import_streams import recorded_stream


class RecordedParserTest(unittest.TestCase):
    def test_real_depth_artifact(self):
        path = Path('/home/xanewok/repos/qwen38-eval/results/2026-10-06-nh6-nextn-lowbit-j5q/a2/0-base/depth-1500.json')
        stream = recorded_stream(path)
        self.assertEqual(len(stream['prompt']), 1500)
        self.assertEqual(len(stream['continuation']), 1536)
        self.assertEqual(stream['continuation'][:3], [760, 1156, 6587])

    def test_real_server_fixture(self):
        path = Path('/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/llama-sse/long-tool-s1-1.json')
        data = __import__('json').loads(path.read_text())
        # This fixture has a string prompt; a teacher-forced tool must refuse it.
        with self.assertRaises((ValueError, KeyError, TypeError)):
            recorded_stream(path)


if __name__ == '__main__':
    unittest.main()
